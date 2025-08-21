@echo off
REM ================================================================
REM COMPLETE DEFECT DETECTION PIPELINE
REM End-to-end training for YOLOX models with P2 feature level
REM Optimized for tiny chip/check detection on high-resolution images
REM ================================================================

setlocal enabledelayedexpansion
set START_TIME=%TIME%

echo.
echo ================================================================
echo  DEFECT DETECTION PIPELINE - STARTING
echo ================================================================
echo  Target: D:\Photomask\final_dataset
echo  Models: SV (1024x500) + EV (2048x1460)
echo  GPU: GTX 1080Ti (11GB VRAM)
echo  Classes: chip (80%%) + check (20%%)
echo ================================================================
echo.

REM Check if we're in the right directory
if not exist "D:\Photomask\final_dataset" (
    echo ERROR: Project directory not found!
    echo Please ensure D:\Photomask\final_dataset exists with SV_dataset and EV_dataset
    pause
    exit /b 1
)

cd /d "D:\Photomask\final_dataset"

REM ================================================================
REM STEP 1: DATA PREPARATION
REM ================================================================
echo [STEP 1/5] PREPARING DATASETS...
echo   - Combining train/test/val splits
echo   - Creating random 80/20 train/val split
echo   - Handling clean images (no defects)
echo.

python 01_prepare_data.py
if %errorlevel% neq 0 (
    echo ERROR: Data preparation failed!
    pause
    exit /b 1
)

echo ✓ Data preparation completed successfully!
echo.

REM ================================================================
REM STEP 2: YOLO TO COCO CONVERSION
REM ================================================================
echo [STEP 2/5] CONVERTING LABELS TO COCO FORMAT...
echo   - Converting YOLO txt files to COCO json
echo   - Handling class mapping: 0=chip, 1=check
echo   - Creating annotation statistics
echo.

python 02_convert_to_coco.py
if %errorlevel% neq 0 (
    echo ERROR: COCO conversion failed!
    pause
    exit /b 1
)

echo ✓ COCO conversion completed successfully!
echo.

REM ================================================================
REM STEP 3: TRAINING SV MODEL
REM ================================================================
echo [STEP 3/5] TRAINING SV MODEL (1024x500)...
echo   - Architecture: YOLOX-S with P2 feature level
echo   - Batch size: 3 (optimized for 1080Ti)
echo   - Loss: Focal Loss for class imbalance
echo   - Epochs: 150 with validation every 5 epochs
echo.

python tools/train.py model_configs/sv_model.py
if %errorlevel% neq 0 (
    echo ERROR: SV model training failed!
    pause
    exit /b 1
)

echo ✓ SV model training completed successfully!
echo.

REM ================================================================
REM STEP 4: TRAINING EV MODEL
REM ================================================================
echo [STEP 4/5] TRAINING EV MODEL (2048x1460)...
echo   - Architecture: YOLOX-S with P2 feature level
echo   - Batch size: 2 (optimized for larger images)
echo   - Loss: Focal Loss for class imbalance
echo   - Epochs: 150 with validation every 5 epochs
echo.

python tools/train.py model_configs/ev_model.py
if %errorlevel% neq 0 (
    echo ERROR: EV model training failed!
    pause
    exit /b 1
)

echo ✓ EV model training completed successfully!
echo.

REM ================================================================
REM STEP 5: ONNX EXPORT
REM ================================================================
echo [STEP 5/5] EXPORTING MODELS TO ONNX...
echo   - Format: ONNX v11 (compatible with edge devices)
echo   - Outputs: boxes, labels, scores (matching your requirements)
echo   - Fixed shapes for optimal inference performance
echo.

REM Find the best SV checkpoint
for /f "delims=" %%i in ('dir "work_dirs\sv_model\best_coco_bbox_mAP_s_epoch_*.pth" /b /o-n 2^>nul') do (
    set SV_CHECKPOINT=work_dirs\sv_model\%%i
    goto :found_sv
)
echo WARNING: Best SV checkpoint not found, using latest...
for /f "delims=" %%i in ('dir "work_dirs\sv_model\epoch_*.pth" /b /o-n 2^>nul') do (
    set SV_CHECKPOINT=work_dirs\sv_model\%%i
    goto :found_sv
)
echo ERROR: No SV checkpoint found!
pause
exit /b 1
:found_sv

REM Find the best EV checkpoint
for /f "delims=" %%i in ('dir "work_dirs\ev_model\best_coco_bbox_mAP_s_epoch_*.pth" /b /o-n 2^>nul') do (
    set EV_CHECKPOINT=work_dirs\ev_model\%%i
    goto :found_ev
)
echo WARNING: Best EV checkpoint not found, using latest...
for /f "delims=" %%i in ('dir "work_dirs\ev_model\epoch_*.pth" /b /o-n 2^>nul') do (
    set EV_CHECKPOINT=work_dirs\ev_model\%%i
    goto :found_ev
)
echo ERROR: No EV checkpoint found!
pause
exit /b 1
:found_ev

echo   - SV Checkpoint: !SV_CHECKPOINT!
echo   - EV Checkpoint: !EV_CHECKPOINT!
echo.

REM Export SV model
echo Exporting SV model to sv_model.onnx...
python tools/deployment/pytorch2onnx.py ^
    model_configs/sv_model.py ^
    "!SV_CHECKPOINT!" ^
    --output-file sv_model.onnx ^
    --shape 500 1024 ^
    --opset-version 11 ^
    --input-names input ^
    --output-names boxes labels scores ^
    --dynamic-axes "{\"boxes\": {0: \"num_boxes\"}, \"labels\": {0: \"num_labels\"}, \"scores\": {0: \"num_scores\"}}"

if %errorlevel% neq 0 (
    echo ERROR: SV ONNX export failed!
    pause
    exit /b 1
)

REM Export EV model
echo Exporting EV model to ev_model.onnx...
python tools/deployment/pytorch2onnx.py ^
    model_configs/ev_model.py ^
    "!EV_CHECKPOINT!" ^
    --output-file ev_model.onnx ^
    --shape 1460 2048 ^
    --opset-version 11 ^
    --input-names input ^
    --output-names boxes labels scores ^
    --dynamic-axes "{\"boxes\": {0: \"num_boxes\"}, \"labels\": {0: \"num_labels\"}, \"scores\": {0: \"num_scores\"}}"

if %errorlevel% neq 0 (
    echo ERROR: EV ONNX export failed!
    pause
    exit /b 1
)

echo ✓ ONNX export completed successfully!
echo.

REM ================================================================
REM PIPELINE COMPLETED
REM ================================================================
set END_TIME=%TIME%

echo ================================================================
echo  PIPELINE COMPLETED SUCCESSFULLY!
echo ================================================================
echo.
echo ✓ Models trained with P2 feature level for tiny object detection
echo ✓ Class imbalance handled with Focal Loss
echo ✓ Optimized for GTX 1080Ti memory constraints
echo ✓ ONNX models ready for edge device deployment
echo.
echo DELIVERABLES:
echo   📁 SV_dataset_processed/     - Processed SV dataset
echo   📁 EV_dataset_processed/     - Processed EV dataset
echo   📁 work_dirs/sv_model/       - SV training logs and checkpoints
echo   📁 work_dirs/ev_model/       - EV training logs and checkpoints
echo   📄 sv_model.onnx            - SV model for 1024x500 images
echo   📄 ev_model.onnx            - EV model for 2048x1460 images
echo.
echo NEXT STEPS:
echo   1. Test the ONNX models with your edge device runtime
echo   2. Adjust score thresholds if needed (currently 0.01)
echo   3. Monitor performance on your specific defect patterns
echo.
echo Start time: %START_TIME%
echo End time:   %END_TIME%
echo ================================================================

REM Create a simple test script
echo Creating inference test script...
(
echo import onnxruntime as ort
echo import numpy as np
echo from PIL import Image
echo import sys
echo.
echo def test_model^(model_path, image_size^):
echo     """Test ONNX model with dummy input"""
echo     try:
echo         session = ort.InferenceSession^(model_path^)
echo         print^(f"✓ Model loaded: {model_path}"^)
echo         print^(f"  Input shape: {image_size}"^)
echo         print^(f"  Inputs: {[inp.name for inp in session.get_inputs()]}"^)
echo         print^(f"  Outputs: {[out.name for out in session.get_outputs()]}"^)
echo         
echo         # Test with dummy input
echo         dummy_input = np.random.rand^(1, 3, image_size[0], image_size[1]^).astype^(np.float32^)
echo         outputs = session.run^(None, {"input": dummy_input}^)
echo         print^(f"  Output shapes: {[out.shape for out in outputs]}"^)
echo         print^(f"  ✓ Model inference successful!\n"^)
echo         return True
echo     except Exception as e:
echo         print^(f"✗ Model test failed: {e}\n"^)
echo         return False
echo.
echo if __name__ == "__main__":
echo     print^("Testing exported ONNX models...\n"^)
echo     
echo     sv_ok = test_model^("sv_model.onnx", ^(500, 1024^)^)
echo     ev_ok = test_model^("ev_model.onnx", ^(1460, 2048^)^)
echo     
echo     if sv_ok and ev_ok:
echo         print^("🎉 All models passed basic tests!"^)
echo         sys.exit^(0^)
echo     else:
echo         print^("⚠ Some models failed tests. Check the error messages above."^)
echo         sys.exit^(1^)
) > test_models.py

echo ✓ Created test_models.py for basic model validation
echo.
echo To test your models, run: python test_models.py
echo.

pause

