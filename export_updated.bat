@echo off
echo Exporting raw YOLOv8-P2 ONNX (no NMS), opset 11...

yolo export ^
  model=runs/detect/ev_yv8sP2_2048x1460/weights/best.pt ^
  format=onnx ^
  imgsz=2048,1460 ^
  dynamic=False ^
  opset=11 ^
  nms=False ^
  name=model_raw.onnx

IF NOT EXIST model_raw.onnx (
  echo ERROR: model_raw.onnx not created.
  exit /b 1
)

echo Post-processing ONNX to match legacy interface: input='input', outputs=['boxes','labels','scores'] with NMS...
python post_export_match_old.py model_raw.onnx model.onnx

IF NOT EXIST model.onnx (
  echo ERROR: post-processing failed; model.onnx missing.
  exit /b 1
)

echo Verifying interface...
python verify_match.py model.onnx

echo DONE. Final artifact: model.onnx

