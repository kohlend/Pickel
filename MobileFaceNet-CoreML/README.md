# MobileFaceNet → Core ML (Pickex)

On-device face-embedding model for the Pickex iOS app. Converts a MobileFaceNet
model to a Core ML **ML Program** (`.mlpackage`, iOS 16+) that takes a raw
112×112 RGB face crop and returns a 512-d embedding. Preprocessing is baked into
the model, so Swift passes a `CVPixelBuffer`/`CGImage` and gets the embedding —
no pixel math on the Swift side.

```
Input   "face_image"   Image, RGB, 112×112
Output  "embedding"    MultiArray Float32 [1, 512]
Baked   y = x/127.5 − 1   (RGB order, range [−1, 1])
Target  mlprogram, iOS 16+ (spec v7), Core ML Tools ≥ 7
```

---

## 1. Model source & license  ⚠️ read this before shipping

Chosen source: **InsightFace `w600k_mbf`** (the `buffalo_s` recognition
backbone) — a MobileFaceNet (MBF) trained on **WebFace600K**. It is the
canonical MobileFaceNet and matches the spec exactly (112×112 → 512, ONNX).

**Licensing is the important caveat.** For face-recognition models the *code*
license and the *weights* license differ:

| Layer | InsightFace | Meaning |
|-------|-------------|---------|
| Code  | MIT | free, incl. commercial |
| **Weights** | **non-commercial research only** (trained on WebFace600K) | **not free for commercial use** |

Every high-quality public MobileFaceNet has the same problem — the weights are
trained on research-only datasets (MS-Celeb-1M, CASIA-WebFace, VGGFace2,
Glint360K, WebFace260M/600K). Alternatives evaluated, all **non-commercial**:
EdgeFace (CC-BY-NC-SA-4.0), GhostFaceNets (CC-BY-NC-ND-4.0), synthetic-data
models (DigiFace-1M / SynFace — datasets themselves non-commercial).

**Before commercial launch, do one of:**
1. Obtain a commercial license from InsightFace — they sell one for the buffalo
   packs: `recognition-oss-pack@insightface.ai`.
2. Train MobileFaceNet yourself on a commercially-licensed dataset.
3. Swap in a differently-licensed embedding model (this pipeline is
   model-agnostic — any `[1,3,112,112]→[1,512]` ONNX drops in).

This repo builds the full pipeline now so development isn't blocked; the
licensing decision is orthogonal to the code.

---

## 2. Preprocessing (must match on the Swift side, 1:1)

The original model expects, per pixel:

```
RGB order (not BGR)
normalized = (pixel − 127.5) / 127.5      →  range [−1, 1]
tensor layout NCHW = [1, 3, 112, 112]
```

This is baked into the `.mlpackage` as an `ImageType` input with
`scale = 1/127.5` and `bias = −1` per channel, so **Swift does not reimplement
it** — just hand Core ML a 112×112 RGB image. Keep the crop/alignment identical
to how faces were aligned at training time (InsightFace uses 5-point similarity
alignment to a 112×112 template); use the same alignment before inference.

---

## 3. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Get the production weights

On any machine, via the InsightFace package:

```bash
pip install insightface onnxruntime
python download_model.py          # -> models/w600k_mbf.onnx  (~13 MB)
```

The exact file is content-addressed:
`sha256 = 9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f`,
13,616,099 bytes, ONNX opset 11, `input.1 [N,3,112,112] → 516 [1,512]`.

(The production model in this repo was already fetched and converted — see
Results below. The only step that still needs your Mac is the Core ML *numeric*
verification, since `predict()` is macOS-only.)

## 5. Convert

```bash
python convert.py --onnx models/w600k_mbf.onnx --precision both
# -> models/FaceEmbedding_fp32.mlpackage
# -> models/FaceEmbedding_fp16.mlpackage
```

Flags: `--input-name` (default `face_image`), `--output-name` (default
`embedding`), `--mean/--std` (default 127.5/127.5), `--precision fp32|fp16|both`.

## 6. Verify (do not skip)

```bash
# Fidelity — same input through ONNX and Core ML (run on macOS):
python verify.py --onnx models/w600k_mbf.onnx \
                 --mlpackage models/FaceEmbedding_fp32.mlpackage
#   expected: ONNX vs Core ML cosine > 0.999

# Identity discrimination — real detection + alignment + embedding:
python discrimination_test.py personA_1.jpg personA_2.jpg personB_1.jpg
#   expected: same-person cosine >> different-person cosine
```

Recorded discrimination run (two photos of Obama = same person, Biden =
different), production `w600k_mbf`:

|        | obama1 | obama2 | biden |
|--------|-------:|-------:|------:|
| obama1 | 1.000  | **0.719** | −0.018 |
| obama2 | 0.719  | 1.000  | 0.074 |
| biden  | −0.018 | 0.074  | 1.000 |

Same person **0.719**, different people **≈0** — cleanly separable (ArcFace
threshold ~0.3–0.4). `verify.py --faces` also exists but uses naive resize;
prefer `discrimination_test.py` for aligned, production-accurate numbers.

On Linux (no Core ML runtime) `verify.py` without `--mlpackage` falls back to
comparing ONNX vs the traced-torch graph that coremltools converts — validating
the exact graph the converter consumes.

---

## Offline reference model (bonus)

A **reference MobileFaceNet** (`mobilefacenet.py` + `make_reference_onnx.py`,
deterministic random weights, correct `[1,3,112,112]→[1,512]` I/O) lets anyone
reproduce and inspect a real `.mlpackage` with zero downloads — useful for CI or
wiring up Xcode before the production weights are in place:

```bash
python make_reference_onnx.py --out models/reference_mbf.onnx --seed 0
python convert.py --onnx models/reference_mbf.onnx --out models/FaceEmbedding.mlpackage
python verify.py --onnx models/reference_mbf.onnx
```

For production, replace `reference_mbf.onnx` with `w600k_mbf.onnx` — nothing else
changes. See `verification_report.txt` for the recorded run.

---

## Results (production w600k_mbf, this repo)

| Artifact | Size |
|----------|------|
| `FaceEmbedding_fp32.mlpackage` | 13.69 MB |
| `FaceEmbedding_fp16.mlpackage` | 6.90 MB (~50% smaller) |

Fidelity (ONNX vs the traced-torch graph coremltools converts):
**cosine = 1.000000** (PASS > 0.999) — the converted graph is numerically
identical to the original ONNX. The final ONNX-vs-Core ML cosine and the
identity-discrimination matrix are produced on macOS with the commands in §6.
Full log: `verification_report.txt`.

## Files

| File | Purpose |
|------|---------|
| `convert.py` | ONNX → `.mlpackage` (baked preprocessing, fp32/fp16) |
| `verify.py` | fidelity + identity-discrimination checks |
| `download_model.py` | fetch `w600k_mbf.onnx` (run where network is open) |
| `mobilefacenet.py` | reference MobileFaceNet architecture |
| `make_reference_onnx.py` | emit the reference ONNX for pipeline validation |
| `verification_report.txt` | recorded verification run |
| `requirements.txt` | pinned toolchain |

## Swift integration (sketch)

```swift
// Xcode auto-generates `FaceEmbedding` from the .mlpackage.
let model = try FaceEmbedding(configuration: MLModelConfiguration())
// pixelBuffer: 112×112 BGRA/RGB CVPixelBuffer of the aligned face crop
let out = try model.prediction(face_image: pixelBuffer)
let embedding = out.embedding          // MLMultiArray, 512 floats
// L2-normalize, then compare with cosine similarity for matching.
```
