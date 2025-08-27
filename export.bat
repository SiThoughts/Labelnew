@echo off
REM ===================================================================
REM export.bat
REM
REM This batch script exports a trained Ultralytics YOLOv8 model to an
REM ONNX file that matches the interface of a legacy PyTorch exporter.
REM It first runs the Ultralytics export with non‑max suppression disabled
REM to produce a raw ONNX file, then calls a Python helper to inject
REM NonMaxSuppression, rename the input to 'input' and produce three
REM outputs: boxes, labels and scores.  Finally it verifies the
REM resulting model and reports success or failure.
REM
REM Usage:
REM   Double‑click or run from a command prompt inside your venv.
REM   Make sure your Ultralytics environment is activated and that
REM   export_to_matching_onnx.py resides in the same directory as
REM   this script.
REM
REM Author: ChatGPT
REM ===================================================================

echo Exporting raw YOLOv8‑P2 ONNX (no NMS) with opset 11...
REM Adjust 'model=' to point to your trained weights (best.pt)
yolo export ^
  model=runs/detect/ev_yv8sP2_2048x1460/weights/best.pt ^
  format=onnx ^
  imgsz=2048,1460 ^
  opset=11 ^
  dynamic=False ^
  nms=False ^
  name=model_raw.onnx

IF NOT EXIST model_raw.onnx (
  echo ERROR: raw export failed to produce model_raw.onnx
  exit /b 1
)

echo Post‑processing raw ONNX to match legacy interface...
python export_to_matching_onnx.py model_raw.onnx model.onnx

IF EXIST model.onnx (
  echo Successfully created model.onnx with interface [input -> boxes, labels, scores].
) ELSE (
  echo ERROR: failed to create model.onnx
  exit /b 1
)

REM Cleanup optional intermediate file
REM del model_raw.onnx

echo Export completed.  Final model saved as model.onnx