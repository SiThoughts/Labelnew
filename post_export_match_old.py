# post_export_match_old.py
# Usage: python post_export_match_old.py model_raw.onnx model.onnx
import sys, onnx
from onnx import helper, TensorProto

src = sys.argv[1] if len(sys.argv) > 1 else "model_raw.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else "model.onnx"

m = onnx.load(src)
g = m.graph

# ---- rename input to 'input'
assert len(g.input) == 1, f"Expected 1 input, found {len(g.input)}"
g.input[0].name = "input"

# ---- find raw boxes & scores
# Common Ultralytics nms=False: first output boxes [1, B, 4], second output scores [1, B, C]
def _shape(v):
    return [d.dim_value for d in v.type.tensor_type.shape.dim]

boxes_name, scores_name = None, None

# Prefer declared outputs; fall back to value_info if needed
candidates = list(g.output) + list(g.value_info)

for v in candidates:
    shp = _shape(v)
    if len(shp) == 3 and shp[0] in (0,1):
        if shp[2] == 4 and boxes_name is None:
            boxes_name = v.name
        elif shp[2] != 4 and scores_name is None:
            # likely [1, B, C]
            scores_name = v.name

assert boxes_name and scores_name, f"Could not locate raw boxes/scores. boxes={boxes_name}, scores={scores_name}"

# ---- constants for NMS
score_thresh = helper.make_tensor("score_thresh", TensorProto.FLOAT, [1], [0.25])  # adjust if needed
iou_thresh   = helper.make_tensor("iou_thresh",   TensorProto.FLOAT, [1], [0.60])
max_det      = helper.make_tensor("max_det", TensorProto.INT64, [1], [300])
m.graph.initializer.extend([score_thresh, iou_thresh, max_det])

# ---- transpose scores to [1, C, B] for NMS
t_scores = helper.make_node("Transpose", inputs=[scores_name], outputs=["scores_chw"], perm=[0,2,1], name="TransposeScores")
g.node.append(t_scores)

# ---- NonMaxSuppression: inputs = boxes [1,B,4], scores [1,C,B]
nms = helper.make_node(
    "NonMaxSuppression",
    inputs=[boxes_name, "scores_chw", "max_det", "iou_thresh", "score_thresh"],
    outputs=["nms_indices"],
    name="NMS",
    center_point_box=0
)
g.node.append(nms)

# nms_indices shape: [N,3] => [batch_idx, class_idx, box_idx]
# split indices
split = helper.make_node("Split", inputs=["nms_indices"], outputs=["b_idx","c_idx","b_idx2"], axis=1, name="SplitIdx")
g.node.append(split)

# squeeze to 1D
sq_labels = helper.make_node("Squeeze", inputs=["c_idx"], outputs=["labels"], name="SqueezeLabels", axes=[1])
sq_boxids = helper.make_node("Squeeze", inputs=["b_idx2"], outputs=["box_ids"], name="SqueezeBoxIds", axes=[1])
g.node.extend([sq_labels, sq_boxids])

# Gather selected boxes -> [N,4]
gath_boxes = helper.make_node("Gather", inputs=[boxes_name, "box_ids"], outputs=["boxes_sel_tmp"], axis=1, name="GatherBoxes")
sq_boxes   = helper.make_node("Squeeze", inputs=["boxes_sel_tmp"], outputs=["boxes"], name="SqueezeBoxes", axes=[0])
g.node.extend([gath_boxes, sq_boxes])

# Gather scores for (class,box) pair
# scores_chw: [1, C, B] -> gather C with labels -> [1, N, B]? No, Gather(axis=1) with labels [N] yields [1, N, B]
gath_scores_c = helper.make_node("Gather", inputs=["scores_chw", "labels"], outputs=["scores_cls_tmp"], axis=1, name="GatherScoresByClass")
# gather boxes along axis=2 using box_ids -> [1, N, 1]
gath_scores_b = helper.make_node("Gather", inputs=["scores_cls_tmp", "box_ids"], outputs=["scores_tmp2"], axis=2, name="GatherScoresByBox")
# squeeze to [N]
sq_scores     = helper.make_node("Squeeze", inputs=["scores_tmp2"], outputs=["scores"], name="SqueezeScores", axes=[0,2])
g.node.extend([gath_scores_c, gath_scores_b, sq_scores])

# ---- set graph outputs: boxes [N,4], labels [N], scores [N]
g.output.clear()
g.output.extend([
    helper.make_tensor_value_info("boxes",  TensorProto.FLOAT,  [None, 4]),
    helper.make_tensor_value_info("labels", TensorProto.INT64,  [None]),
    helper.make_tensor_value_info("scores", TensorProto.FLOAT,  [None]),
])

onnx.save(m, dst)
print(f"OK: wrote {dst} with input 'input' and outputs ['boxes','labels','scores']")

