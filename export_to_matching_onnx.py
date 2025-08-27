"""
export_to_matching_onnx
=======================

This helper script post‑processes a raw ONNX model exported from
Ultralytics YOLOv8 so that the resulting ONNX file matches the
interface of a legacy PyTorch/torchvision exporter.  The legacy
format expects a single input tensor named ``input`` with shape
[1, 3, 1460, 2048] and three dynamic outputs named ``boxes``,
``labels`` and ``scores``.  It also embeds a non‑max suppression
(NMS) operation into the graph so that the model returns only
filtered detections rather than per‑class score tensors.

Usage::

    python export_to_matching_onnx.py <raw_model.onnx> <final_model.onnx>

The script performs the following steps:

    1. Loads the raw ONNX file produced by ``yolo export`` with
       ``nms=False`` and a fixed image size (2048×1460).
    2. Renames the single input to ``input``.
    3. Detects the raw boxes and per‑class score tensors.
    4. Adds constant tensors for score and IoU thresholds and the
       maximum number of detections.
    5. Transposes the score tensor to the layout required by
       ``NonMaxSuppression``.
    6. Appends a ``NonMaxSuppression`` node to the graph.
    7. Gathers the selected boxes, class indices and scores into
       three flattened outputs (``boxes``, ``labels``, ``scores``).
    8. Replaces the original graph outputs with the new outputs.
    9. Saves the modified ONNX graph to the second filename provided.

After running this script, the exported model should be compatible
with tooling or pipelines that expect the older exporter’s
signature.

Example::

    # export raw model (no NMS) with Ultralytics
    yolo export model=best.pt format=onnx imgsz=2048,1460 \ 
      dynamic=False opset=11 nms=False name=model_raw.onnx

    # post‑process to legacy interface
    python export_to_matching_onnx.py model_raw.onnx model.onnx

Authors: ChatGPT
"""

import sys
import onnx
from onnx import helper, TensorProto


def find_raw_outputs(graph):
    """Identify the raw boxes and scores tensors in the ONNX graph.

    Ultralytics exports the detector head as two tensors when
    ``nms=False``: boxes with shape [1, B, 4] and scores with shape
    [1, B, C], where B is the number of candidate boxes and C is
    the number of classes.  This function scans the graph’s
    outputs and value_info entries to locate the first pair of
    tensors matching these signatures.  Returns a tuple
    (boxes_name, scores_name).

    Raises:
        RuntimeError: if suitable tensors cannot be found.
    """
    boxes_name = None
    scores_name = None

    def get_shape(value_info):
        return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

    # Check declared outputs first
    for v in graph.output:
        shape = get_shape(v)
        # Expect shape [1, B, 4] for boxes
        if len(shape) == 3 and shape[-1] == 4 and boxes_name is None:
            boxes_name = v.name
        # Expect shape [1, B, C] for scores (C > 4)
        elif len(shape) == 3 and shape[-1] > 4 and scores_name is None:
            scores_name = v.name

    # If not found, search in value_info
    if boxes_name is None or scores_name is None:
        for v in graph.value_info:
            shape = get_shape(v)
            if len(shape) == 3:
                if shape[-1] == 4 and boxes_name is None:
                    boxes_name = v.name
                elif shape[-1] > 4 and scores_name is None:
                    scores_name = v.name

    if boxes_name is None or scores_name is None:
        raise RuntimeError(
            f"Could not locate raw boxes/scores tensors in the graph. "
            f"Found boxes={boxes_name}, scores={scores_name}."
        )
    return boxes_name, scores_name


def build_legacy_outputs(model, raw_boxes, raw_scores):
    """Append NMS and output gathering operations to the ONNX graph.

    This helper function modifies the provided model in place.  It
    appends nodes for score transposition, non‑maximum suppression,
    splitting indices, gathering selected boxes/scores, and
    reassigns graph outputs to three tensors: ``boxes``, ``labels``
    and ``scores``.

    Args:
        model: an ``onnx.ModelProto`` loaded from the raw export.
        raw_boxes: name of the boxes tensor (shape [1,B,4]).
        raw_scores: name of the scores tensor (shape [1,B,C]).

    Returns:
        None.  The model is modified in place.
    """
    g = model.graph

    # 1. Add constant tensors for NMS thresholds and max detections
    score_thresh = helper.make_tensor("score_thresh", TensorProto.FLOAT, [1], [0.25])
    iou_thresh = helper.make_tensor("iou_thresh", TensorProto.FLOAT, [1], [0.60])
    max_det = helper.make_tensor("max_det", TensorProto.INT64, [1], [300])
    g.initializer.extend([score_thresh, iou_thresh, max_det])

    # 2. Transpose scores from [1,B,C] to [1,C,B] for NMS
    transpose_scores_node = helper.make_node(
        "Transpose",
        inputs=[raw_scores],
        outputs=["scores_chw"],
        perm=[0, 2, 1],
        name="TransposeScores"
    )
    g.node.append(transpose_scores_node)

    # 3. Add NonMaxSuppression node
    nms_node = helper.make_node(
        "NonMaxSuppression",
        inputs=[raw_boxes, "scores_chw", "max_det", "iou_thresh", "score_thresh"],
        outputs=["nms_indices"],
        name="NMS",
        center_point_box=0
    )
    g.node.append(nms_node)

    # 4. Split NMS output indices into class indices and box indices
    # nms_indices shape: [N,3] -> [batch_index, class_index, box_index]
    split_node = helper.make_node(
        "Split",
        inputs=["nms_indices"],
        outputs=["b_idx", "c_idx", "b_idx2"],
        axis=1,
        name="SplitIdx"
    )
    g.node.append(split_node)

    # 5. Squeeze to flatten class and box indices
    sq_labels = helper.make_node(
        "Squeeze", inputs=["c_idx"], outputs=["labels"], axes=[1], name="SqueezeLabels"
    )
    sq_boxids = helper.make_node(
        "Squeeze", inputs=["b_idx2"], outputs=["box_ids"], axes=[1], name="SqueezeBoxIds"
    )
    g.node.append(sq_labels)
    g.node.append(sq_boxids)

    # 6. Gather selected boxes (shape [1,N,4] -> squeeze to [N,4])
    gather_boxes = helper.make_node(
        "Gather",
        inputs=[raw_boxes, "box_ids"],
        outputs=["boxes_tmp"],
        axis=1,
        name="GatherBoxes"
    )
    squeeze_boxes = helper.make_node(
        "Squeeze", inputs=["boxes_tmp"], outputs=["boxes"], axes=[0], name="SqueezeBoxes"
    )
    g.node.append(gather_boxes)
    g.node.append(squeeze_boxes)

    # 7. Gather selected scores for each (class, box)
    # Transposed scores (scores_chw) is [1, C, B].
    gather_scores_by_class = helper.make_node(
        "Gather",
        inputs=["scores_chw", "labels"],
        outputs=["scores_cls_tmp"],
        axis=1,
        name="GatherScoresByClass"
    )
    gather_scores_by_box = helper.make_node(
        "Gather",
        inputs=["scores_cls_tmp", "box_ids"],
        outputs=["scores_tmp2"],
        axis=2,
        name="GatherScoresByBox"
    )
    squeeze_scores = helper.make_node(
        "Squeeze",
        inputs=["scores_tmp2"],
        outputs=["scores"],
        axes=[0, 2],
        name="SqueezeScores"
    )
    g.node.extend([gather_scores_by_class, gather_scores_by_box, squeeze_scores])

    # 8. Replace graph outputs
    g.output.clear()
    g.output.extend([
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [None, 4]),
        helper.make_tensor_value_info("labels", TensorProto.INT64, [None]),
        helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None]),
    ])


def convert_model(raw_path: str, out_path: str) -> None:
    """Post‑process a raw ONNX export to match the legacy interface.

    This function loads the raw model, renames its input, injects
    non‑max suppression and output gathering nodes, rewires the
    outputs, and saves the modified model to ``out_path``.

    Args:
        raw_path: Path to the raw ONNX file exported with ``nms=False``.
        out_path: Path where the post‑processed ONNX will be saved.
    """
    model = onnx.load(raw_path)
    graph = model.graph
    # Rename input to 'input'
    if len(graph.input) != 1:
        raise RuntimeError(f"Expected 1 input tensor but found {len(graph.input)}")
    graph.input[0].name = "input"
    # Locate raw outputs
    boxes_name, scores_name = find_raw_outputs(graph)
    # Append NMS and gather operations
    build_legacy_outputs(model, boxes_name, scores_name)
    # Save
    onnx.save(model, out_path)


def verify_model(path: str) -> None:
    """Verify that the post‑processed ONNX matches the desired signature.

    This function loads the model at ``path`` and asserts that:
        * It has exactly one input named ``input`` with shape [1,3,1460,2048].
        * It has exactly three outputs: ``boxes``, ``labels``, ``scores``.
        * The ``NonMaxSuppression`` node exists in the graph.

    Raises:
        AssertionError if the model does not match these expectations.
    """
    model = onnx.load(path)
    g = model.graph
    # Check input
    assert len(g.input) == 1, f"Expected 1 input, got {len(g.input)}"
    inp = g.input[0]
    assert inp.name == "input", f"Input name is {inp.name}, expected 'input'"
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    assert dims == [1, 3, 1460, 2048], f"Input shape is {dims}, expected [1,3,1460,2048]"
    # Check outputs
    outs = [o.name for o in g.output]
    assert outs == ["boxes", "labels", "scores"], f"Output names are {outs}, expected ['boxes','labels','scores']"
    # Check presence of NMS
    has_nms = any(n.op_type == "NonMaxSuppression" for n in g.node)
    assert has_nms, "NonMaxSuppression node not found in the graph"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_to_matching_onnx.py <raw_model.onnx> <final_model.onnx>")
        sys.exit(1)
    raw_model_path = sys.argv[1]
    final_model_path = sys.argv[2]
    convert_model(raw_model_path, final_model_path)
    # Perform a quick verification
    verify_model(final_model_path)
    print(
        f"Successfully converted {raw_model_path} to {final_model_path} with input 'input' "
        f"and outputs ['boxes','labels','scores']."
    )