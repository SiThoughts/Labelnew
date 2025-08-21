@echo off
REM ================================================================
REM Complete MMDetection Installation Script for Windows 10
REM Optimized for GTX 1080Ti (11GB VRAM)
REM ================================================================

echo [STEP 1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [STEP 2/6] Installing PyTorch for CUDA 11.8 (GTX 1080Ti compatible)...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo [STEP 3/6] Installing OpenMMLab dependencies...
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

echo [STEP 4/6] Installing MMDetection...
mim install "mmdet>=3.0.0"

echo [STEP 5/6] Installing additional required packages...
pip install tqdm pillow

echo [STEP 6/6] Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import mmdet; print(f'MMDetection version: {mmdet.__version__}')"

echo.
echo ================================================================
echo Installation completed successfully!
echo Your system is ready for defect detection training.
echo ================================================================
pause

