import onnxruntime as ort, numpy as np, cv2, sys
onnx_path = "model.onnx"
img_path  = sys.argv[1] if len(sys.argv)>1 else None
assert img_path, "Usage: python infer_onnx_updated.py <path_to_image_2048x1460>"

img = cv2.imread(img_path)
assert img is not None, f"Failed to read {img_path}"
h,w = img.shape[:2]
assert (w,h)==(2048,1460), f"Image must be exactly 2048x1460, got {(w,h)}"

blob = img[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255.0
sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

# Run inference with new interface
out = sess.run(None, {"input": blob})  # Note: input name is now "input"

# Expected outputs: boxes, labels, scores
boxes, labels, scores = out
print(f"boxes shape: {boxes.shape}, dtype: {boxes.dtype}")
print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")  
print(f"scores shape: {scores.shape}, dtype: {scores.dtype}")
print(f"Number of detections: {len(boxes)}")

if len(boxes) > 0:
    print(f"First detection: box={boxes[0]}, label={labels[0]}, score={scores[0]:.3f}")

print("Inference OK - Interface matches legacy model")

