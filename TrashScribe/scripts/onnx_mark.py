"""Add intermediate tensors of an ONNX graph as extra graph outputs (inferred type and shape).

    python scripts/onnx_mark.py in.onnx out.onnx --regex '/detr_encoder/layers\\.\\d+/Add_3_output_0$' ...

Marking a tensor as an output keeps TensorRT from fusing across it: the way the TensorRT 10.3 fusion bugs
were located and worked around (TrashScribe's hypernetwork, A21; SAM 3's decoder, A22). The runtime may
ignore the extra outputs. --list prints the tensors that match instead.
"""
import argparse
import re

import onnx
from onnx import shape_inference

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst", nargs="?")
ap.add_argument("--regex", nargs="+", default=[], help="node output names to mark (re.search)")
ap.add_argument("--tensors", nargs="+", default=[], help="exact tensor names to mark")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

m = onnx.load(a.src)
pats = [re.compile(r) for r in a.regex]
names = [o for n in m.graph.node for o in n.output if o in a.tensors or any(p.search(o) for p in pats)]
if a.list or not a.dst:
    print("\n".join(names))
    raise SystemExit(0)
have = {o.name for o in m.graph.output}
inferred = {v.name: v for v in shape_inference.infer_shapes(m).graph.value_info}
missing = [t for t in names if t not in inferred]
if missing:
    raise SystemExit(f"no inferred shape for {missing[:5]}: not marked")
for t in names:
    if t not in have:
        m.graph.output.append(inferred[t])
onnx.checker.check_model(m)
onnx.save(m, a.dst)
print(f"{len(names)} tensors marked -> {a.dst}")
