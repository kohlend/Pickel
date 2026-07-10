"""
Fetch the production InsightFace MobileFaceNet weights (w600k_mbf.onnx) and
place them at models/w600k_mbf.onnx.

Run this on a machine with unrestricted network access (e.g. your Mac). The
InsightFace `buffalo_s` pack is downloaded from GitHub releases / the
insightface package; here we reuse the `insightface` library to fetch it, then
copy just the recognition backbone out of the pack.

    pip install insightface onnxruntime
    python download_model.py

Note: these weights are trained on WebFace600K and are licensed for
NON-COMMERCIAL research use. For commercial use, obtain a license from
InsightFace (recognition-oss-pack@insightface.ai). See README.
"""
import os
import shutil


def main():
    dst = os.path.join("models", "w600k_mbf.onnx")
    os.makedirs("models", exist_ok=True)

    # The buffalo_s pack bundles det_500m.onnx + w600k_mbf.onnx.
    from insightface.utils import storage
    pack_dir = storage.ensure_available("models", "buffalo_s")
    src = os.path.join(pack_dir, "w600k_mbf.onnx")
    if not os.path.exists(src):
        raise SystemExit(f"w600k_mbf.onnx not found in downloaded pack: {pack_dir}")
    shutil.copy(src, dst)
    print(f"OK -> {dst}  ({os.path.getsize(dst)/1e6:.2f} MB)")
    print("Next:  python convert.py --onnx models/w600k_mbf.onnx --precision both")


if __name__ == "__main__":
    main()
