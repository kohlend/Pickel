"""
Convert a MobileFaceNet ONNX model to a Core ML .mlpackage (ML Program, iOS 16+).

Pipeline:  ONNX --(onnx2torch)--> torch.nn.Module --(jit.trace)--> coremltools

The preprocessing (RGB, normalize to [-1, 1]) is baked into the model as an
ImageType input, so the Swift side passes a raw 112x112 CGImage/CVPixelBuffer
and gets a 512-d embedding back — no manual pixel math in Swift.

  normalized = (pixel - mean) / std        with mean = std = 127.5  ->  [-1, 1]
  Core ML ImageType applies:  y = x * scale + bias
      scale = 1 / std          = 1 / 127.5
      bias  = -mean / std      = -1.0   (per channel)

Usage:
  python convert.py --onnx models/w600k_mbf.onnx --out models/FaceEmbedding.mlpackage
  python convert.py --onnx models/w600k_mbf.onnx --precision both
"""
import argparse
import os
import shutil

import numpy as np
import onnx
import torch
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert


def dir_size_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e6
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def _simplify(onnx_model):
    """Fold Constant nodes into initializers so onnx2torch converts cleanly.
    No-op (with a warning) if onnxsim isn't installed."""
    try:
        from onnxsim import simplify
        simplified, ok = simplify(onnx_model)
        if ok:
            return simplified
        print("      onnxsim could not fully simplify; using original graph.")
    except ImportError:
        print("      onnxsim not installed; skipping simplify (may fail on folded biases).")
    return onnx_model


def load_as_traced(onnx_path: str):
    """ONNX -> torch module -> traced graph, plus the ONNX input tensor name."""
    onnx_model = _simplify(onnx.load(onnx_path))
    input_name = onnx_model.graph.input[0].name
    torch_model = onnx2torch_convert(onnx_model).eval()
    example = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, example)
    return traced, input_name


def convert(traced, out_path, in_name, out_name, mean, std, precision):
    scale = 1.0 / std
    bias = [-mean / std] * 3
    prec = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name=in_name,
            shape=(1, 3, 112, 112),
            scale=scale,
            bias=bias,
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.TensorType(name=out_name, dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=prec,
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.short_description = (
        "MobileFaceNet face embedding. Input: RGB face crop 112x112. "
        "Output: 512-d embedding. Preprocessing (RGB, [-1,1]) baked in."
    )
    mlmodel.input_description[in_name] = "Aligned RGB face crop, 112x112 pixels."
    mlmodel.output_description[out_name] = "512-dimensional face embedding (not L2-normalized)."

    if os.path.exists(out_path):
        shutil.rmtree(out_path) if os.path.isdir(out_path) else os.remove(out_path)
    mlmodel.save(out_path)
    return mlmodel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/w600k_mbf.onnx")
    ap.add_argument("--out", default="models/FaceEmbedding.mlpackage")
    ap.add_argument("--input-name", default="face_image")
    ap.add_argument("--output-name", default="embedding")
    ap.add_argument("--mean", type=float, default=127.5)
    ap.add_argument("--std", type=float, default=127.5)
    ap.add_argument("--precision", choices=["fp32", "fp16", "both"], default="both")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(
            f"ONNX not found: {args.onnx}\n"
            "Provide w600k_mbf.onnx (see README), or generate the reference model:\n"
            "  python make_reference_onnx.py --out models/reference_mbf.onnx --seed 0\n"
            "  python convert.py --onnx models/reference_mbf.onnx"
        )

    print(f"[1/3] loading ONNX + tracing: {args.onnx}")
    traced, _ = load_as_traced(args.onnx)

    targets = ["fp32", "fp16"] if args.precision == "both" else [args.precision]
    for prec in targets:
        out = args.out if len(targets) == 1 else args.out.replace(
            ".mlpackage", f"_{prec}.mlpackage")
        print(f"[2/3] converting -> {out}  (precision={prec})")
        convert(traced, out, args.input_name, args.output_name,
                args.mean, args.std, prec)
        print(f"[3/3] saved {out}   size = {dir_size_mb(out):.2f} MB")


if __name__ == "__main__":
    main()
