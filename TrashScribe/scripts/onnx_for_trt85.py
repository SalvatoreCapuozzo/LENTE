"""Prepare the exported ONNX graph for TensorRT 8.5 (JetPack 5.1.x).

TensorRT 8.5 rejects the opset-17 graph of `export_onnx.py` twice:

1. no importer for LayerNormalization ("Plugin not found"): each node is rewritten as
   the sequence torch.onnx emits below opset 17, which TensorRT 8.x pattern-matches
       ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul -> Add
2. the HF mask decoder computes shapes with OneHot ("an IIOneHotLayer cannot be used to
   compute a shape tensor"): with the static 1x3x640x640 input every shape is a constant,
   so the graph is constant-folded (onnx-graphsurgeon, shipped with JetPack; it replaces
   onnxsim, which has no aarch64 wheel for Python 3.8) until it stops shrinking.

TensorRT >= 8.6 imports the original graph directly; `build_trt.sh` calls this script only
on older versions. Outputs are compared with ONNX Runtime on the original graph.

    python scripts/onnx_for_trt85.py in.onnx out.onnx [--no-check]
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--no-check", action="store_true", help="skip the ONNX Runtime comparison")
ap.add_argument("--max-passes", type=int, default=30)
a = ap.parse_args()


def decompose_layernorm(m):
    g = m.graph
    two = "ln_decomp_two"
    g.initializer.append(numpy_helper.from_array(np.array(2.0, np.float32), two))
    new_nodes, n_ln = [], 0
    for node in g.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue
        att = {x.name: helper.get_attribute_value(x) for x in node.attribute}
        axis, eps = att.get("axis", -1), att.get("epsilon", 1e-5)
        if axis != -1 or any(node.output[1:]):
            raise SystemExit(f"{node.name}: only axis=-1 with a single output is handled")
        x, scale = node.input[0], node.input[1]
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        p = f"{node.name or f'ln{n_ln}'}_decomp"
        t = lambda s: f"{p}/{s}"  # noqa: E731
        g.initializer.append(numpy_helper.from_array(np.array(eps, np.float32), t("eps")))
        y = node.output[0]
        new_nodes += [
            helper.make_node("ReduceMean", [x], [t("mean")], name=t("ReduceMean"), axes=[-1], keepdims=1),
            helper.make_node("Sub", [x, t("mean")], [t("d")], name=t("Sub")),
            helper.make_node("Pow", [t("d"), two], [t("d2")], name=t("Pow")),
            helper.make_node("ReduceMean", [t("d2")], [t("var")], name=t("ReduceMean_1"), axes=[-1], keepdims=1),
            helper.make_node("Add", [t("var"), t("eps")], [t("var_eps")], name=t("Add")),
            helper.make_node("Sqrt", [t("var_eps")], [t("std")], name=t("Sqrt")),
            helper.make_node("Div", [t("d"), t("std")], [t("xhat")], name=t("Div")),
            helper.make_node("Mul", [t("xhat"), scale], [t("scaled") if bias else y], name=t("Mul")),
        ]
        if bias:
            new_nodes.append(helper.make_node("Add", [t("scaled"), bias], [y], name=t("Add_1")))
        n_ln += 1
    del g.node[:]
    g.node.extend(new_nodes)
    print(f"{n_ln} LayerNormalization nodes decomposed")
    return m


def fold(m, max_passes):
    import onnx_graphsurgeon as gs
    for i in range(max_passes):
        del m.graph.value_info[:]                 # stale annotations of the HF export break inference
        for data_prop in (True, False):           # data propagation fails on the empty prompt tensors
            try:
                m = shape_inference.infer_shapes(m, data_prop=data_prop)
                break
            except Exception:  # noqa: BLE001
                continue
        g = gs.import_onnx(m)
        n0 = len(g.nodes)
        g.fold_constants(fold_shapes=True, partitioning="basic").cleanup().toposort()
        m = gs.export_onnx(g)
        print(f"fold pass {i}: {n0} -> {len(g.nodes)} nodes")
        if len(g.nodes) == n0:
            break
    return m


m = fold(decompose_layernorm(onnx.load(a.src)), a.max_passes)
onnx.checker.check_model(m)
onnx.save(m, a.dst)
print("written", a.dst)

if not a.no_check:
    import onnxruntime as ort
    x = np.random.default_rng(0).standard_normal((1, 3, 640, 640)).astype(np.float32)
    outs = []
    for path in (a.src, a.dst):                   # one session at a time: 8 GB shared memory
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        outs.append(s.run(None, {s.get_inputs()[0].name: x}))
        del s
    d = [float(np.abs(u - v).max()) for u, v in zip(*outs)]
    print("max |original - prepared| per output [waste, contour, desc]:", d)
    if max(d) > 1e-3:
        raise SystemExit("prepared graph differs from the original: do not build from it")
