# Pickex — Face pipeline (Steps A–C)

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

---

# Step B — Reference profile from 1–4 user photos

`ReferenceProfileBuilder.swift` turns the user's picked reference photos into
ONE L2-normalized 512-d embedding (`ReferenceProfile`) that the library scan
(Step C) matches against. `FaceEmbedder.swift` is the extracted single-image
→ embedding unit (preprocess → Core ML → `[Float]`), reused by Step C.

Key decisions (documented in code):

- **Aggregation**: normalize-each → mean → normalize, behind the
  `EmbeddingAggregator` protocol so a "keep the set, best-match at query time"
  strategy can be swapped in without touching callers.
- **Consistency threshold `0.4`** (`defaultConsistencyThreshold`): from the
  measured data below — same-person pairs bottom out at ~0.63, impostor pairs
  top out at ~0.06 — a soft warning, never a hard failure.
- **Skips are visible**: photos without a detectable face land in
  `ReferenceProfile.skipped` with a reason for the UI; only if *all* photos
  fail does `build` throw `.noFaceInAnyPhoto`.
- **Parallel**: the 1–4 photos are embedded concurrently via `TaskGroup`.
- The PHPicker layer (`ReferenceProfileBuilder+PHPicker.swift`) is a thin
  wrapper over the testable `build(from: [ReferenceImage])` core.

## Step-B verification (real LFW photos, production model)

Mirror harness: `verify_reference_profile.py` (same aggregation, same threshold).

**Scenario 1 — 4 photos, same person (G. W. Bush):** all 4 used, pairwise
cosine **0.627–0.758**, no warning. ✅

**Scenario 2 — photo 3 swapped for a different person (C. Powell):** warning
fires; exactly the three Powell pairs are flagged (0.034 / 0.051 / 0.058 —
all far below 0.4), Bush pairs stay 0.658–0.746. The flagged indices single
out the odd photo for the UI. ✅

**Scenario 3 — photo 3 is a landscape (no face):** skipped as
`(index 3, noFaceDetected)`, profile built from the remaining 3, no warning. ✅

Final embedding in every scenario: dim 512, L2-norm 1.000.

---

# Step C — Library scan against the reference profile

`LibraryScanner.swift` + `ScanCache.swift`: scan the whole library (10k+
photos) against the `ReferenceProfile`, stream matches + throttled progress,
support cancellation, and never recompute photos already in the cache.

**Public API**

```swift
let scanner = LibraryScanner(embedder: embedder, cache: try ScanCache())

// Streaming (primary): matches appear in the UI as they're found.
for try await event in scanner.scanEvents(against: profile) {
    switch event {
    case .progress(let p):  // p.fraction, p.matchCount, p.skippedNotLocal, p.servedFromCache
    case .match(let m):     // m.assetLocalIdentifier, m.similarity
    }
}

// Collect-style convenience:
let matches = try await scanner.scan(reference: profile) { progress in ... }

// New reference person, existing cache -> instant, no photo access:
let quick = scanner.rematchFromCache(against: newProfile)
```

**Design decisions** (rationale in the file headers):

| Decision | Value | Why |
|---|---|---|
| Scope | `fetchAssets(with: .image)` | videos excluded by the fetch; Live Photos scanned via their still frame |
| iCloud | network **OFF** by default | no surprise cellular download of thousands of originals; skipped assets counted in `skippedNotLocal` for the UI ("X photos not on this device"); opt-in flag for Wi-Fi deep scans |
| Decode | 640 px, `.fastFormat`, `.fast` | plenty for detection + 112 px crop; Photos can serve existing thumbnails |
| Concurrency | bounded TaskGroup, 4 in flight | this IS the memory cap (~4 decoded images at once) + `autoreleasepool` per photo; Vision/Core ML saturate below 4-way anyway |
| Matching | every face, max 8, best score wins | target person is rarely the largest face on group photos |
| Threshold | **0.40**, config knob | calibrated below; the task draft's ~0.70 would sit *inside* the measured positive range and drop half the true matches |
| Progress | every 20 assets + final | no per-asset UI flood |
| Cancel | end the for-await loop | delivered matches stay; cache keeps all computed work → an aborted scan loses nothing, the next one resumes from cache |

**Cache** (`ScanCache.swift`): built-in SQLite3 (no dependency), WAL.
`assets(local_id PK, modified_at, face_count, embeddings BLOB)` — raw
per-face embeddings (`face_count × 512 × Float32`), *including* `face_count=0`
("scanned, no faces" — most photos, and exactly the rows that make re-scans
fast). Invalidation: edited photo → `modified_at` mismatch → recomputed;
deleted photo → `prune(keeping:)` at scan start. Because raw embeddings (not
match results) are stored, a **new** reference profile re-matches from cache
in milliseconds (`rematchFromCache`). Size ≈ 2 KB/photo → ~25 MB at 10k photos.

## Step-C verification (real 19-photo LFW library, production model)

**Matching accuracy** (`verify_library_scan.py` — all-faces + any-match).
Library: 8 Bush photos (disjoint from the 4 reference photos), 10 other
identities, 1 six-face group photo without Bush.

| set | best-face score | result |
|-----|-----------------|--------|
| 8 positives (Bush) | 0.651 – 0.779 | all matched ✅ |
| 10 single-face negatives | −0.033 – 0.078 | none matched ✅ |
| 6-face group photo (no Bush) | 0.071 | not matched ✅ |

**Recall 8/8, false positives 0/11 at threshold 0.40** (in fact anywhere in
0.30–0.50 — the gap is +0.57). Real libraries will be harder (profile views,
occlusion, aging); the threshold is a config knob for exactly that reason.

**Cache semantics** (`verify_scan_cache.py` — identical SQLite schema +
lookup/invalidate/prune/rematch logic, real photos, real model):

| check | result |
|---|---|
| cold scan | 19 model runs, correct 8 matches, baseline |
| warm scan | 19/19 cache hits, **0 model runs**, identical results, ~2600× faster ✅ |
| 1 photo "edited" (modified_at bumped) | exactly 1 recompute, 18 hits ✅ |
| 3 photos "deleted" | `prune` removes exactly 3 rows ✅ |
| new person, cache only (`rematchFromCache`) | correct match, 0 model runs, 0.3 ms ✅ |
| rematch at scale | 10,000 cached embeddings in ~21 ms ✅ |

## Performance (Step-C §6) — honest status

**No iOS device is available in this environment (Linux container), so real
iPhone numbers could not be measured.** What exists instead:

- **Measured here (Linux, x86 CPU, no ANE):** ~53 ms/photo for detect+embed
  → ~53 s/1000 photos single-threaded. An iPhone's ANE runs the fp16 model
  several times faster, but PHImageManager decode becomes the bottleneck;
  a reasonable expectation is **tens of seconds per 1000 photos** on device,
  dominated by image loading.
- **`ScanBenchmark.swift`** is included: call `ScanBenchmark.run(scanner:profile:)`
  on your iPhone (debug button) — it reports wall time, s/1000 photos, cache
  hits, iCloud skips, and peak memory (phys_footprint). Run twice: cold vs
  warm shows the cache payoff.
- **Background suspension:** without a `BGProcessingTask`, iOS suspends the
  app shortly after backgrounding and the scan pauses (cache makes resume
  cheap, so nothing is lost). Decision needed for v1: "keep app open while
  scanning" (simplest) vs. BGProcessingTask continuation — pick after seeing
  real device numbers from the benchmark.

## Assumptions (where the model doc left room)

- **Template**: the published ArcFace 5-point 112 template is used as the
  alignment target (InsightFace de-facto standard).
- **Vision → 5 points**: pupils/eye-region centroid, nose-region centroid, outer
  lip extremes. Vision's landmark set differs slightly from InsightFace's
  detector, but the least-squares similarity fit is robust to small differences.
- **CIContext render orientation**: assumed to yield an upright buffer (standard
  behavior). If an on-device crop ever appears vertically mirrored, map the
  template with `ty` directly instead of `outputSize − ty` in `templateCI()`.
