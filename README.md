Write a PyTorch implementation of a high-recall industrial defect segmentation model based on HRNet-W18 with a 3-class segmentation head.

Requirements:

1. Overall goal
- I want a fully convolutional segmentation model for industrial surface defects (ceramic cracks + reflectance/color anomalies).
- The model must take arbitrary-resolution RGB images (no fixed input size assumptions) and output per-pixel logits for 3 classes:
  - class 0: background
  - class 1: reflectance/color defect
  - class 2: crack/structural defect
- The core backbone should be HRNet-W18 (high-resolution branch preserved).
- The head should be a simple 3-class segmentation head on top of the backbone features.

2. Backbone (HRNet-W18)
- Use `timm` if available: `create_model("hrnet_w18", features_only=True, pretrained=True)`.
- The model should expose a single high-resolution feature map from the last stage (e.g. the highest-resolution output of HRNet) as the input to the segmentation head.
- The model must remain fully convolutional and support arbitrary H×W (not restricted to multiples of 32).
- No classification head – only feature extraction.

3. Segmentation head (3-class)
- Implement a small segmentation decoder as a separate module, e.g. `SegHead3Class`.
- It should:
  - Take the HRNet feature map of shape `[B, C, H, W]`.
  - Apply:
    - `Conv2d(C, C, kernel_size=3, padding=1)`
    - `BatchNorm2d(C)`
    - `ReLU`
    - `Conv2d(C, 3, kernel_size=1)`  # 3 output channels
  - Return logits of shape `[B, 3, H, W]`.
- Do NOT use any fancy control flow, only standard Conv/BN/ReLU so it is ONNX-exportable.

4. Top-level model class
- Create a module `HRNetW18Seg3Class(nn.Module)` with:
  - `self.backbone` as the HRNet-W18 feature extractor.
  - `self.head` as the 3-class segmentation head.
- `forward(x)` should:
  - Accept `x` of shape `[B, 3, H, W]` in `float32`.
  - Pass x through the backbone, take the highest-resolution feature map (last stage).
  - Pass that into the segmentation head.
  - Return logits `[B, 3, H, W]` at (or close to) input resolution.
- Do not hard-code image size anywhere; use whatever H,W come from the input.

5. Loss helper
- Provide a simple helper function `make_segmentation_loss()` that returns a loss function combining:
  - `nn.CrossEntropyLoss` with class weights (for example: `bg=0.25, reflectance=1.2, crack=2.0`).
- The loss should accept:
  - logits `[B, 3, H, W]`
  - target mask `[B, H, W]` with values {0,1,2}
  and return a scalar loss.

6. Example usage
- At the bottom, include a short example showing:
  - Creating the model.
  - Creating a dummy input tensor `[1, 3, 512, 512]`.
  - Forward pass to get logits.
  - Creating a dummy target mask `[1, 512, 512]`.
  - Computing the loss with the helper.
- Do not run training loop, just show how it would be called.

7. ONNX-export friendly
- Ensure no Python control flow in `forward` that depends on tensor values.
- Only use standard ops: Conv, BatchNorm, ReLU, possibly simple interpolation (`F.interpolate` with mode="bilinear", align_corners=False) if needed, but avoid anything exotic.
- The model should be exportable with:
  - `torch.onnx.export(model, dummy_input, "hrnet_w18_seg3.onnx", opset_version=17, input_names=["input"], output_names=["logits"], dynamic_axes={"input": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}})`

Implement all of this in a single Python file with clear class and function definitions and type hints where reasonable.
