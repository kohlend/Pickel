"""
Verify that the Core ML .mlpackage reproduces the original ONNX model.

Two checks:

  A) CONVERSION FIDELITY  (same input -> both models)
       - runs the ORIGINAL onnx via onnxruntime
       - runs the CONVERTED .mlpackage via coremltools  (macOS only!)
       - reports cosine similarity; expected > 0.999
     On Linux, Core ML prediction is unavailable, so this falls back to
     comparing onnxruntime vs the onnx2torch graph that is actually fed to
     coremltools — i.e. it validates the exact graph the converter sees.

  B) IDENTITY DISCRIMINATION  (needs REAL weights + REAL aligned face crops)
       - embeds every image in --faces, L2-normalizes, prints the pairwise
         cosine-similarity matrix. Same person -> high, different -> low.

Usage:
  # fidelity, comparing onnx vs mlpackage (run on macOS):
  python verify.py --onnx models/w600k_mbf.onnx --mlpackage models/FaceEmbedding_fp32.mlpackage
  # fidelity fallback (Linux) — onnx vs the traced torch graph:
  python verify.py --onnx models/w600k_mbf.onnx
  # discrimination test:
  python verify.py --onnx models/w600k_mbf.onnx --faces a1.png a2.png b1.png
"""
import argparse
import os

import numpy as np
from PIL import Image

MEAN, STD = 127.5, 127.5


def cosine(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def preprocess_chw(img: Image.Image) -> np.ndarray:
    """RGB, 112x112, normalized to [-1,1], returned as [1,3,112,112] (what the
    ONNX graph expects). Mirrors the preprocessing baked into the Core ML model."""
    img = img.convert("RGB").resize((112, 112), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32)          # HWC, 0..255
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))[None]         # 1,3,112,112


def onnx_embed(sess, chw):
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: chw.astype(np.float32)})[0].ravel()


def coreml_embed(mlmodel, out_name, img112):
    # Core ML ImageType wants a PIL image; preprocessing is inside the model.
    pred = mlmodel.predict({_coreml_input_name(mlmodel): img112})
    key = out_name if out_name in pred else list(pred.keys())[0]
    return np.asarray(pred[key]).ravel()


def _coreml_input_name(mlmodel):
    return mlmodel._spec.description.input[0].name


def check_fidelity(onnx_path, mlpackage):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # a deterministic pseudo-image in [0,255]
    img = Image.fromarray(rng.integers(0, 256, (112, 112, 3), dtype=np.uint8))
    chw = preprocess_chw(img)
    emb_onnx = onnx_embed(sess, chw)

    if mlpackage:
        import coremltools as ct
        try:
            mlmodel = ct.models.MLModel(mlpackage)
            out_name = mlmodel._spec.description.output[0].name
            emb_cml = coreml_embed(mlmodel, out_name, img.resize((112, 112)))
            cs = cosine(emb_onnx, emb_cml)
            print(f"[fidelity] ONNX vs Core ML cosine = {cs:.6f}  "
                  f"({'PASS' if cs > 0.999 else 'CHECK'} threshold 0.999)")
            return
        except Exception as e:
            print(f"[fidelity] Core ML predict unavailable here ({type(e).__name__}: {e}).")
            print("[fidelity] Falling back to ONNX vs traced-torch graph.")

    # Linux fallback: compare onnxruntime against the graph coremltools traces.
    import onnx
    import torch
    from onnx2torch import convert as onnx2torch_convert
    graph = onnx.load(onnx_path)
    try:
        from onnxsim import simplify
        graph, ok = simplify(graph)
    except ImportError:
        pass
    tmodel = onnx2torch_convert(graph).eval()
    with torch.no_grad():
        emb_torch = tmodel(torch.from_numpy(chw)).numpy().ravel()
    cs = cosine(emb_onnx, emb_torch)
    print(f"[fidelity] ONNX vs traced-torch cosine = {cs:.6f}  "
          f"({'PASS' if cs > 0.999 else 'CHECK'} threshold 0.999)")
    print("           (run this on macOS with --mlpackage for the final ONNX-vs-CoreML check)")


def check_discrimination(onnx_path, faces):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    embs, names = [], []
    for p in faces:
        e = onnx_embed(sess, preprocess_chw(Image.open(p)))
        e = e / np.linalg.norm(e)
        embs.append(e); names.append(os.path.basename(p))
    print("\n[discrimination] pairwise cosine similarity (L2-normalized):")
    hdr = "            " + "  ".join(f"{n[:10]:>10}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        row = "  ".join(f"{cosine(embs[i], embs[j]):>10.3f}" for j in range(len(names)))
        print(f"{n[:10]:>10}  {row}")
    print("\nInterpretation: same-person pairs should be markedly higher than "
          "different-person pairs (typical decision threshold ~0.3-0.4 for ArcFace).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/w600k_mbf.onnx")
    ap.add_argument("--mlpackage", default=None,
                    help="Path to .mlpackage (enables ONNX-vs-CoreML check; macOS only)")
    ap.add_argument("--faces", nargs="*", default=None,
                    help="Aligned 112x112 face crops for the discrimination test")
    args = ap.parse_args()

    check_fidelity(args.onnx, args.mlpackage)
    if args.faces:
        check_discrimination(args.onnx, args.faces)


if __name__ == "__main__":
    main()
