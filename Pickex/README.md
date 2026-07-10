# Pickex — Face preprocessing

`FacePreprocessor.swift` turns a photo into the exact **112×112 CVPixelBuffer**
the MobileFaceNet Core ML model expects, for the step between *"Vision detected
a face"* and *"hand the crop to Core ML"*.

## Model contract (from `MobileFaceNet-CoreML/README.md` §2)

| | |
|---|---|
| Input | `face_image` — **Image, RGB, 112×112** |
| Output | `embedding` — Float32 `[1, 512]` |
| Normalization | `y = x/127.5 − 1` is **baked into the `.mlpackage`** (ImageType `scale=1/127.5`, `bias=−1`) |
| Alignment at training | InsightFace **5-point similarity alignment** to the ArcFace 112 template |

Two consequences drive the whole implementation:

1. **Do NOT normalize in Swift.** The scale/bias live inside the model. We hand
   Core ML a plain 112×112 image buffer; it does RGB conversion + `[-1,1]`
   scaling itself. Normalizing again would silently produce garbage embeddings
   (model still runs, no error). → This file does **zero** pixel math.
2. **Align, don't just crop.** The model was trained on similarity-aligned
   faces (eyes/nose/mouth mapped to fixed template positions). A raw bounding
   box gives rotated/off-center faces → degraded embeddings. So we align.

## Pipeline (maps 1:1 to the code)

1. **Detect + landmarks** — `VNDetectFaceLandmarksRequest`; pick the **largest**
   face by bounding-box area. Derive 5 points: eye centers (pupils when Vision
   provides them, else eye-region centroid), nose-region centroid, outer-lip
   min-x / max-x mouth corners. Eyes & mouth are ordered by image-x so the
   template mapping is **roll-invariant**.
2. **Align (primary)** — least-squares 2-D **similarity transform** (rotation +
   uniform scale + translation, no reflection) mapping the 5 points onto the
   ArcFace 112 template. This is the same transform InsightFace uses.
   **Fallback** — if landmarks are missing: expand the bounding box by
   `faceMarginFraction` (**default 0.25 = 25% per side**, configurable), make it
   **square** (avoids distortion), **clamp** to the image so edge faces don't
   crash, then scale to 112.
3. **Resize** — done by the alignment/scale transform, resampled on the GPU via
   a reused `CIContext` (not UIKit) to stay fast when scanning thousands of
   photos.
4. **Normalize** — *none* (baked into the model; see above).
5. **CVPixelBuffer** — `kCVPixelFormatType_32BGRA` (the canonical Core ML
   ImageType buffer; Core ML maps BGRA→RGB). Errors are returned as typed
   `FacePreprocessError` values — **no crashes**; "no face on this photo" is the
   normal case for most library photos.

## Usage

```swift
let pre = FacePreprocessor()                 // default margin 0.25
// PHAsset -> CGImage (resolve EXIF orientation, or pass it in):
do {
    let buffer = try pre.makeFaceInput(from: cgImage, orientation: .up)
    let out = try FaceEmbedding().prediction(face_image: buffer)   // no normalization!
    let embedding = out.embedding            // MLMultiArray, 512 floats
} catch FacePreprocessError.noFaceFound {
    // skip this photo — expected for most images
} catch {
    // log other cases (landmarksUnavailable, invalidCrop, …)
}
// Or non-throwing: pre.makeFaceInputOrNil(from: cgImage)
```

Tune the fallback margin:

```swift
var cfg = FacePreprocessorConfig()
cfg.faceMarginFraction = 0.30
cfg.allowBoundingBoxFallback = true          // false => require landmarks
let pre = FacePreprocessor(config: cfg)
```

## Verification (Step 6)

`reference_crops/` holds five 112×112 aligned crops produced by
`make_reference_crops.py`, which applies the **identical** InsightFace 5-point
alignment the Swift code targets — use them as the ground truth to eyeball the
on-device output against.

| crop | photo | check |
|------|-------|-------|
| `obama1`, `obama2` | same person, two photos | face centered, eyes level |
| `biden`, `alex` | different people / lighting | not distorted |
| `t1` | largest face from a group photo | correct face selected |

Embedding plausibility through the recognition model (same run): **no NaNs,
0% zeros**, embedding norms ~22–26 for every crop. `mask_black.jpg` (heavily
occluded) yields **no detected face** — exercising the "no face" path cleanly.

Reproduce:

```bash
# in the MobileFaceNet-CoreML venv, with ~/.insightface/models/buffalo_s present
python Pickex/make_reference_crops.py photo1.jpg photo2.jpg ...
```

> **Note (environment):** Vision + Core ML are macOS/iOS-only, so
> `FacePreprocessor.swift` itself can't be executed on the Linux conversion
> host. The reference crops above are generated with the equivalent Python
> alignment as ground truth; run the Swift pipeline on-device/macOS to produce
> device-side PNGs and confirm they match these.

## Assumptions (where the model doc left room)

- **Template**: the published ArcFace 5-point 112 template is used as the
  alignment target (InsightFace de-facto standard).
- **Vision → 5 points**: pupils/eye-region centroid, nose-region centroid, outer
  lip extremes. Vision's landmark set differs slightly from InsightFace's
  detector, but the least-squares similarity fit is robust to small differences.
- **CIContext render orientation**: assumed to yield an upright buffer (standard
  behavior). If an on-device crop ever appears vertically mirrored, map the
  template with `ty` directly instead of `outputSize − ty` in `templateCI()`.
