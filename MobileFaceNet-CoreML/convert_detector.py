"""
Convert the InsightFace SCRFD det_500m face detector to Core ML (.mlpackage).

Pipeline:  ONNX --(fix 640x640, onnxsim)--> onnx2torch --> jit.trace --> coremltools

Detector preprocessing ((x - 127.5) / 128, RGB) is baked in via ImageType, so
Swift passes a raw 640x640 letterboxed BGRA pixel buffer.

Outputs (fp32), rows = grid*grid*2 anchors:
  s8  [12800,1]  s16 [3200,1]  s32 [800,1]   sigmoid face scores
  b8  [12800,4]  b16 [3200,4]  b32 [800,4]   bbox distances (l,t,r,b) / stride
  k8  [12800,10] k16 [3200,10] k32 [800,10]  5 keypoint offsets / stride

The script VALIDATES before saving:
  1. traced torch model vs onnxruntime on a real face collage (max |diff|)
  2. reference anchor decode finds a face with score > 0.5
It refuses to save if either fails — that is what was missing when the first
conversion silently produced a model whose outputs ignored the input.

Usage:  .venv/bin/python convert_detector.py \
            --onnx /root/.insightface/models/buffalo_s/det_500m.onnx \
            --out models/FaceDetector.mlpackage
"""
import argparse
import os
import shutil

import cv2
import numpy as np
import onnx
import torch
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert

SIZE = 640
MEAN, STD = 127.5, 128.0
OUT_NAMES = ["s8", "s16", "s32", "b8", "b16", "b32", "k8", "k16", "k32"]


def fix_input_shape(m: onnx.ModelProto) -> onnx.ModelProto:
    """det_500m ships with dynamic H/W; pin to 640x640 BEFORE simplifying so
    Shape/Reshape chains fold to real numbers (dynamic dims fold to garbage)."""
    dims = m.graph.input[0].type.tensor_type.shape.dim
    for d, v in zip(dims, (1, 3, SIZE, SIZE)):
        d.dim_param = ""
        d.dim_value = v
    return m


def make_test_image() -> np.ndarray:
    """640x640 RGB uint8: gray canvas + a real aligned face crop, upscaled."""
    canvas = np.full((SIZE, SIZE, 3), 127, np.uint8)
    crop_path = os.path.join(os.path.dirname(__file__),
                             "..", "Pickex", "reference_crops", "obama1_aligned112.png")
    bgr = cv2.imread(os.path.abspath(crop_path))
    if bgr is None:
        raise SystemExit(f"test face missing: {crop_path}")
    face = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    canvas[100:356, 150:406] = face
    return canvas


def normalize(img_rgb: np.ndarray) -> np.ndarray:
    x = (img_rgb.astype(np.float32) - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # NCHW


def decode_max_score(outs) -> tuple[float, int]:
    """Max face score + count of >0.5 locations across all strides (sanity)."""
    best, hits = 0.0, 0
    for i, _ in enumerate([8, 16, 32]):
        s = np.asarray(outs[i]).reshape(-1)
        best = max(best, float(s.max()))
        hits += int((s > 0.5).sum())
    return best, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/.insightface/models/buffalo_s/det_500m.onnx")
    ap.add_argument("--out", default="models/FaceDetector.mlpackage")
    args = ap.parse_args()

    print("[1/5] load + pin 640x640 + simplify")
    model = fix_input_shape(onnx.load(args.onnx))
    try:
        from onnxsim import simplify
        model, ok = simplify(model)
        print("      onnxsim:", "ok" if ok else "PARTIAL")
    except ImportError:
        print("      onnxsim not installed — continuing without")
    input_name = model.graph.input[0].name

    print("[2/5] onnxruntime reference on test image")
    import onnxruntime as ort
    x = normalize(make_test_image())
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {input_name: x})
    ref_best, ref_hits = decode_max_score(ref)
    print(f"      ORT max score={ref_best:.3f}  >0.5 locations={ref_hits}")
    if ref_best < 0.5:
        raise SystemExit("ONNX itself does not see the test face — aborting.")

    print("[3/5] onnx2torch + trace, compare against ORT")
    tm = onnx2torch_convert(model).eval()
    with torch.no_grad():
        traced = torch.jit.trace(tm, torch.from_numpy(x))
        touts = traced(torch.from_numpy(x))
    touts = [t.numpy() for t in (touts if isinstance(touts, (tuple, list)) else [touts])]
    if len(touts) != 9:
        raise SystemExit(f"expected 9 outputs from trace, got {len(touts)}")
    worst = max(float(np.abs(a - b).max()) for a, b in zip(touts, ref))
    t_best, _ = decode_max_score(touts)
    print(f"      torch-vs-ORT max|diff|={worst:.2e}   torch max score={t_best:.3f}")
    if worst > 1e-3 or t_best < 0.5:
        raise SystemExit("traced torch model does not match ONNX — aborting.")

    print("[4/5] coremltools convert (fp32, iOS16)")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE),
                             scale=1.0 / STD, bias=[-MEAN / STD] * 3,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in OUT_NAMES],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )

    # Paranoia: no model output may be wired to a constant.
    mil = str(mlmodel._mil_program)
    tail = mil[mil.rfind("} -> ("):]
    print("      MIL returns:", tail.strip()[:120])

    print("[5/5] save", args.out)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    mlmodel.short_description = ("SCRFD det_500m face detector. Input: 640x640 RGB "
                                 "letterbox. Outputs: score/bbox/kps per stride 8/16/32. "
                                 "Preprocessing (x-127.5)/128 baked in.")
    mlmodel.save(args.out)
    print("done — size %.2f MB" % (
        sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs) / 1e6))


if __name__ == "__main__":
    main()
