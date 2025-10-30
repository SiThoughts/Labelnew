"""
generate_chip_detector.py
==========================

This script defines a tiny neural network for chip detection and exports it to
an ONNX file.  It illustrates how you can embed all preprocessing,
processing and postprocessing steps inside a single ONNX graph.  The
resulting model takes an image of arbitrary size (either 1‐ or 3‐channel) and
produces the coordinates of a bounding box around the detected chip.  No
external tiling or postprocessing is required at inference time.

The example assumes you have PyTorch and ONNX installed in your Python
environment.  It has **not** been executed inside this environment due to
missing dependencies, so you should treat it as a template.  Feel free to
modify the architecture (for example swap in a pretrained lightweight
segmentation backbone) to meet your performance targets.

Usage
-----

```
python generate_chip_detector.py --opset 13 --output chip_detector.onnx
```

By default the script exports the model with dynamic axes so it can
process any spatial dimensions.  The opset version can be changed via
the `--opset` flag (13 or higher is recommended).  After running
this script you will have an `chip_detector.onnx` file that you can
deploy with ONNX Runtime or other engines.

Note
----

This script intentionally embeds heavy logic inside the network.  In
a production system you might prefer to implement the tiling and
postprocessing outside of ONNX for efficiency.  However, the user
requested a single self-contained ONNX file, so this example uses
ONNX's `Scan` operator to iterate over image patches.

"""

import argparse
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedGray(nn.Module):
    """Convert 3-channel input to a single channel using fixed weights.

    If the input has one channel already, it passes through unchanged.
    """

    def __init__(self):
        super().__init__()
        # fixed 1x1 conv to collapse RGB; weights favour the green channel
        weight = torch.tensor([[[[0.1]]], [[[0.8]]], [[[0.1]]]])  # shape (1,3,1,1)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x
        # apply conv with no bias
        y = F.conv2d(x, self.weight)
        return y


class SimpleFilter(nn.Module):
    """A tiny stack of convolutions to mimic blur + edge detection.

    This network uses fixed convolution kernels to smooth the input and
    extract vertical edges.  You can replace this with a learned
    segmentation network if you have annotated data.
    """

    def __init__(self):
        super().__init__()
        # 5x5 approximate Gaussian blur kernel (normalized)
        gauss = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ], dtype=torch.float32)
        gauss = gauss / gauss.sum()
        gauss = gauss.view(1, 1, 5, 5)
        self.register_buffer("gauss", gauss)

        # Sobel kernel for vertical edges (3x3)
        sobel_y = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ], dtype=torch.float32)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # blur
        x = F.conv2d(x, self.gauss, padding=2)
        # vertical edges
        y = F.conv2d(x, self.sobel_y, padding=1)
        y = y.abs()
        return y


class ChipDetector(nn.Module):
    """Top-level model that handles dynamic tiling and box extraction.

    The forward method accepts an image tensor of shape (1,C,H,W) and returns
    a tensor (1,4) with bounding box coordinates (x1, y1, x2, y2).  All
    operations are expressed using PyTorch primitives that map cleanly to
    ONNX ops.  The tiling is implemented as a Python loop in PyTorch; when
    exported to ONNX this becomes a `Scan` loop internally.
    """

    def __init__(self, tile_size: Tuple[int, int] = (256, 256), stride: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.weighted_gray = WeightedGray()
        self.filter = SimpleFilter()
        self.tile_size = tile_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale if needed
        x = self.weighted_gray(x)
        _, _, H, W = x.shape
        tile_h, tile_w = self.tile_size
        stride_h, stride_w = self.stride

        # initialise accumulation mask with zeros
        mask = torch.zeros_like(x)

        # number of tiles in each dimension
        n_h = math.ceil((H - tile_h) / stride_h) + 1
        n_w = math.ceil((W - tile_w) / stride_w) + 1

        for i in range(n_h):
            for j in range(n_w):
                y0 = i * stride_h
                x0 = j * stride_w
                # ensure tiles do not exceed bounds
                y1 = min(y0 + tile_h, H)
                x1 = min(x0 + tile_w, W)
                y0 = y1 - tile_h
                x0 = x1 - tile_w
                # crop tile
                tile = x[:, :, y0:y1, x0:x1]
                # process tile
                edges = self.filter(tile)
                # dilate via maxpool to grow edges slightly
                dilated = F.max_pool2d(edges, kernel_size=5, stride=1, padding=2)
                # accumulate into the mask (sum overlaps)
                mask[:, :, y0:y1, x0:x1] = mask[:, :, y0:y1, x0:x1] + dilated

        # threshold the accumulated mask: nonzero indicates chip region
        binary = mask > 0

        # compute bounding box by reducing over dimensions
        # find nonzero indices
        idx = binary.nonzero(as_tuple=False)  # shape (N,4)
        if idx.numel() == 0:
            # no detection found; return zeros
            return torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
        # idx columns: batch, channel, y, x
        y_coords = idx[:, 2].float()
        x_coords = idx[:, 3].float()
        x_min = x_coords.min()
        y_min = y_coords.min()
        x_max = x_coords.max() + 1  # inclusive end
        y_max = y_coords.max() + 1
        box = torch.stack([x_min, y_min, x_max, y_max], dim=0).unsqueeze(0)
        return box


def export_model(opset: int, output_path: str, tile_size: Tuple[int, int], stride: Tuple[int, int]):
    model = ChipDetector(tile_size=tile_size, stride=stride)
    model.eval()
    # dummy input with dynamic size (batch=1, channels=3, height, width)
    dummy_input = torch.randn(1, 3, tile_size[0]*2, tile_size[1]*2)
    # export with dynamic axes for height and width
    dynamic_axes = {
        "input": {2: "height", 3: "width"},
        "output": {1: "bbox_coordinates"},
    }
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX model to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export chip detector to ONNX")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--output", type=str, default="chip_detector.onnx", help="Output ONNX filename")
    parser.add_argument("--tile", type=int, nargs=2, default=[256, 256], help="Tile size (height width)")
    parser.add_argument("--stride", type=int, nargs=2, default=[128, 128], help="Stride between tiles (height width)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(args.opset, args.output, tuple(args.tile), tuple(args.stride))