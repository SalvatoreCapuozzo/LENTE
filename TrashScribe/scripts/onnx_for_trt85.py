"""Prepare the exported ONNX graph for TensorRT 8.5 (JetPack 5.1.x).

TensorRT 8.5 rejects the opset-17 graph of `export_onnx.py` twice:

1. no importer for LayerNormalization ("Plugin not found"): each node is rewritten as
   the sequence torch.onnx emits below opset 17, which TensorRT 8.x pattern-matches
       ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul -> Add
2. the HF mask decoder computes shapes with OneHot ("an IIOneHotLayer cannot be used to
   compute a shape tensor"): with the static 1x3x640x640 input every shape is a constant,
   so the graph is constant-folded (onnx-graphsurgeon, shipped with JetPack; it replaces
   onnxsim, which has no aarch64 wheel for Python 3.8) until it stops shrinking.

TensorRT >= 8.6 imports LayerNormalization, but TensorRT 10.3 (JetPack 6.2) still stops at the
OneHot shape computation, so `build_trt.sh` runs this script on every version without onnxsim:
the full rewrite below TensorRT 8.6, the constant folding alone (--keep-layernorm) from 8.6 on.
Outputs are compared with ONNX Runtime on the original graph.

3. TensorRT 10.3 mis-fuses the first layer of the mask decoder's hypernetwork MLPs with what follows
   (24 Nov 2026): the FP32 and FP16 engines then give a different waste mask (max |difference| 0.94 on
   real frames, waste-mask IoU 0.83 with ONNX Runtime) while contour and descriptor maps agree to
   1e-5. Found by marking intermediate tensors with Polygraphy and bisecting: marking the output of
   output_hypernetworks_mlps.0/proj_in/MatMul alone makes the engine agree (1.9e-5). With
   --keep-layernorm these outputs are therefore added as extra graph outputs (256 floats each; the
   runtime ignores them), which keeps them out of the faulty fused kernel.

    python scripts/onnx_for_trt85.py in.onnx out.onnx [--keep-layernorm] [--no-check]
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
ap.add_argument("--keep-layernorm", action="store_true",
                help="only constant-fold (TensorRT >= 8.6 imports LayerNormalization natively)")
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


m = onnx.load(a.src)
m = fold(m if a.keep_layernorm else decompose_layernorm(m), a.max_passes)
if a.keep_layernorm:     # see 3. above
    marks = [n.output[0] for n in m.graph.node
             if n.op_type == "MatMul" and "output_hypernetworks_mlps" in n.name and "/proj_in/" in n.name]
    have = {o.name for o in m.graph.output}
    inferred = {v.name: v for v in shape_inference.infer_shapes(m).graph.value_info}
    for t in marks:
        if t not in have:      # with the inferred type and shape (the checker requires a shape)
            m.graph.output.append(inferred[t] if t in inferred else
                                  helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, [1, 1, 1, 256]))
    print(f"{len(marks)} hypernetwork proj_in outputs marked (TensorRT 10.3 fusion workaround)")
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
