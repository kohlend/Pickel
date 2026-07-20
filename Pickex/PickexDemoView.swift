//
//  PickexDemoView.swift
//  Pickex
//
//  Minimal end-to-end test screen: pick 1–4 reference photos → build the
//  ReferenceProfile → scan the library → live progress + matched thumbnails.
//  Drop into a fresh SwiftUI iOS-16 app as the root view to smoke-test the
//  whole pipeline. Not production UI — just the wiring.
//
//  Uses the SwiftUI PhotosPicker with the decoupled
//  `build(from: [ReferenceImage])` core API (no PHPickerViewController needed).
//

#if canImport(SwiftUI) && canImport(PhotosUI)
import SwiftUI
import PhotosUI
import Photos
import CoreML

struct PickexDemoView: View {
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var status = "Wähle 1–4 Referenzfotos."
    @State private var progress: ScanProgress?
    @State private var matches: [MatchResult] = []
    @State private var thumbnails: [String: UIImage] = [:]
    @State private var scanTask: Task<Void, Never>?
    // Debug: the actual 112×112 crops fed to the model + embedding fingerprints.
    @State private var debugCrops: [UIImage] = []
    @State private var debugInfo = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                PhotosPicker(selection: $pickerItems, maxSelectionCount: 4,
                             matching: .images) {
                    Label("Referenzfotos wählen", systemImage: "person.crop.circle.badge.plus")
                }
                .disabled(scanTask != nil)

                if scanTask == nil {
                    Button("Profil erstellen & Bibliothek scannen") {
                        scanTask = Task { await runPipeline() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(pickerItems.isEmpty)
                } else {
                    Button("Abbrechen", role: .destructive) {
                        scanTask?.cancel()
                    }
                }

                Text(status).font(.footnote).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                if !debugInfo.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("DEBUG — Crops, wie sie das Modell sieht:").font(.caption2)
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 4) {
                                ForEach(Array(debugCrops.enumerated()), id: \.offset) { _, img in
                                    Image(uiImage: img)
                                        .resizable().frame(width: 120, height: 120)
                                        .border(.red)
                                }
                            }
                        }
                        Text(debugInfo).font(.system(size: 9, design: .monospaced))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if let p = progress {
                    ProgressView(value: p.fraction) {
                        Text(Self.progressLabel(p)).font(.caption)
                    }
                }

                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 90), spacing: 4)], spacing: 4) {
                        ForEach(matches, id: \.assetLocalIdentifier) { match in
                            ZStack(alignment: .bottomTrailing) {
                                if let img = thumbnails[match.assetLocalIdentifier] {
                                    Image(uiImage: img)
                                        .resizable().scaledToFill()
                                        .frame(width: 90, height: 90).clipped()
                                } else {
                                    Color.gray.opacity(0.2).frame(width: 90, height: 90)
                                }
                                Text(String(format: "%.2f", match.similarity))
                                    .font(.caption2).padding(2)
                                    .background(.black.opacity(0.6))
                                    .foregroundStyle(.white)
                            }
                        }
                    }
                }
            }
            .padding()
            .navigationTitle("Pickex Demo")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Cache leeren") {
                        (try? ScanCache())?.clear()
                        status = "Scan-Cache geleert — nächster Scan rechnet alles neu."
                    }
                    .disabled(scanTask != nil)
                }
            }
        }
    }

    /// Built in a plain function (not inline in the View) so the SwiftUI
    /// type-checker doesn't choke on a long `+`/ternary string expression.
    private static func progressLabel(_ p: ScanProgress) -> String {
        var parts = ["\(p.processed)/\(p.total)", "\(p.matchCount) Treffer"]
        if p.bestSimilarity > -1 { parts.append(String(format: "best %.2f", p.bestSimilarity)) }
        if p.skippedNotLocal > 0 { parts.append("\(p.skippedNotLocal) nicht lokal") }
        if p.servedFromCache > 0 { parts.append("\(p.servedFromCache) aus Cache") }
        if p.failedToProcess > 0 { parts.append("⚠️ \(p.failedToProcess) fehlgeschlagen") }
        return parts.joined(separator: " · ")
    }

    // MARK: pipeline

    private func runPipeline() async {
        defer { scanTask = nil }
        matches = []; thumbnails = [:]; progress = nil
        do {
            // 1. Load picked photos into the decoupled core input type.
            status = "Lade Referenzfotos…"
            var refs: [ReferenceImage] = []
            for item in pickerItems {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let ui = UIImage(data: data), let cg = ui.cgImage {
                    refs.append(ReferenceImage(cgImage: cg,
                                               orientation: CGImagePropertyOrientation(ui.imageOrientation)))
                }
            }

            // 2. Model + pipeline objects.
            let embedder = try Self.makeEmbedder()
            let builder = ReferenceProfileBuilder(embedder: embedder)

            // DEBUG: show the exact 112×112 crops + embedding fingerprints for
            // the reference photos, to localize "everything scores 1.00" bugs.
            let pre = FacePreprocessor()
            let ciCtx = CIContext()
            var crops: [UIImage] = []
            var prints: [String] = []
            prints.append("Preprocessor \(FacePreprocessor.debugVersion)")
            for (i, r) in refs.enumerated() {
                // The 640×640 detector input with box (green) + keypoints (red)
                // drawn on it — shows whether the downscaled image is sharp or
                // aliased, and whether the keypoints land on the eyes.
                if let ann = pre.debugAnnotatedInput(from: r.cgImage, orientation: r.orientation) {
                    crops.append(UIImage(cgImage: ann))
                }
                // ...plus the actual 112×112 crop the recognition model eats.
                if let res = try? pre.makeFaceInputDetailed(from: r.cgImage, orientation: r.orientation),
                   let cg = ciCtx.createCGImage(CIImage(cvPixelBuffer: res.pixelBuffer),
                                                from: CIImage(cvPixelBuffer: res.pixelBuffer).extent) {
                    crops.append(UIImage(cgImage: cg))
                }
                prints.append("F\(i + 1): " + pre.debugEyeInfo(from: r.cgImage, orientation: r.orientation))
                if let e = try? embedder.embedding(from: r.cgImage, orientation: r.orientation) {
                    let v = e.vector
                    let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
                    prints.append(String(format: "    emb norm=%.2f [%+.2f %+.2f %+.2f]",
                                         norm, v[0], v[1], v[2]))
                }
            }
            debugCrops = crops
            debugInfo = prints.joined(separator: "\n")

            // 3. Reference profile (Step B) — surface warnings.
            status = "Erstelle Referenzprofil…"
            let profile = try await builder.build(from: refs)
            var notes: [String] = ["Profil aus \(profile.usedPhotoCount) Foto(s)."]
            if !profile.skipped.isEmpty {
                notes.append("\(profile.skipped.count) Foto(s) ohne erkennbares Gesicht übersprungen.")
            }
            if !profile.multipleFaceWarnings.isEmpty {
                notes.append("Mehrere Gesichter auf Foto(s) \(profile.multipleFaceWarnings.map(String.init).joined(separator: ", ")) — größtes verwendet.")
            }
            if let w = profile.consistencyWarning { notes.append("⚠️ " + w.message) }
            status = notes.joined(separator: " ")

            // 4. Photo-library permission.
            let auth = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            guard auth == .authorized || auth == .limited else {
                status = "Kein Foto-Zugriff erlaubt."; return
            }

            // 5. Scan (Step C) — matches appear live. Threshold 0.35: with
            // clean 5-point alignment + the keypoint gate (garbage crops can't
            // reach the matcher) different people sit ~0.1 and the same person
            // 0.35–0.99, so 0.35 recovers imperfect/turned shots while staying
            // far above the different-person ceiling.
            var scanConfig = LibraryScannerConfig()
            scanConfig.matchThreshold = 0.35
            // Also scan iCloud-only photos: downloads a 1280px version per
            // photo on demand (needs network; first full scan takes a while,
            // results are cached so re-scans stay fast).
            scanConfig.allowNetworkAccess = true
            // iCloud fetches are network-bound, not CPU-bound — more photos in
            // flight keeps the pipe full while others download.
            scanConfig.concurrentPhotos = 8
            let scanner = LibraryScanner(embedder: embedder, cache: try? ScanCache(),
                                         config: scanConfig)
            for try await event in scanner.scanEvents(against: profile) {
                switch event {
                case .progress(let p): progress = p
                case .match(let m):
                    matches.append(m)
                    matches.sort { $0.similarity > $1.similarity }
                    loadThumbnail(for: m.assetLocalIdentifier)
                }
            }
            // DEBUG: render the SCAN-side crop of the TOP matches so we can see
            // whether false positives (wrong person at 0.9) come from corrupted
            // crops (alignment bug) or a bad reference. A garbage crop that
            // matches everything shows up here as a warped/nonsense thumbnail.
            debugCrops = []   // replace the reference crops with the match crops
            var lines = ["MATCH-Crops v\(FacePreprocessor.debugVersion):"]
            for m in matches.prefix(8) {
                guard let scanCG = scanner.debugLoadImage(assetID: m.assetLocalIdentifier) else { continue }
                if let res = try? pre.makeFaceInputDetailed(from: scanCG),
                   let cg = ciCtx.createCGImage(CIImage(cvPixelBuffer: res.pixelBuffer),
                                                from: CIImage(cvPixelBuffer: res.pixelBuffer).extent) {
                    debugCrops.append(UIImage(cgImage: cg))
                }
                var line = String(format: "%.2f ", m.similarity) + pre.debugEyeInfo(from: scanCG)
                if let e = try? embedder.embedding(from: scanCG) {
                    let v = e.vector; let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
                    line += String(format: " norm=%.1f", n)
                }
                lines.append(line)
            }
            debugInfo = lines.joined(separator: "\n")
            status = "Fertig: \(matches.count) Treffer."
        } catch ReferenceProfileError.noFaceInAnyPhoto(let skipped) {
            let details = skipped.map { s -> String in
                switch s.reason {
                case .noFaceDetected: return "Foto \(s.index + 1): kein Gesicht erkannt"
                case .processingError(let e) where e.contains("degenerateLandmarks"):
                    return "Foto \(s.index + 1): Gesicht zu schräg/liegend — bitte frontales Foto wählen"
                case .processingError(let e): return "Foto \(s.index + 1): FEHLER: \(e)"
                }
            }.joined(separator: "\n")
            status = "Kein Profil möglich:\n\(details)"
        } catch is CancellationError {
            status = "Abgebrochen — \(matches.count) Treffer behalten, Cache bleibt."
        } catch {
            status = "Fehler: \(error)"
        }
    }

    /// Loads the .mlpackage regardless of whether it was added as
    /// FaceEmbedding.mlpackage or FaceEmbedding_fp16.mlpackage (no dependency
    /// on the Xcode-generated class name).
    private static func makeEmbedder() throws -> FaceEmbedder {
        // Prefer the larger, more accurate R50 model when present.
        let candidates = ["FaceEmbeddingR50", "FaceEmbedding",
                          "FaceEmbedding_fp16", "FaceEmbedding_fp32"]
        for name in candidates {
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                return try FaceEmbedder(modelURL: url)
            }
        }
        throw FaceEmbeddingError.outputMissing(
            "FaceEmbedding*.mlpackage nicht im App-Bundle — Datei ins Xcode-Projekt ziehen (Target-Membership setzen).")
    }

    private func loadThumbnail(for identifier: String) {
        guard thumbnails[identifier] == nil,
              let asset = PHAsset.fetchAssets(withLocalIdentifiers: [identifier],
                                              options: nil).firstObject else { return }
        let opts = PHImageRequestOptions()
        opts.deliveryMode = .opportunistic
        PHImageManager.default().requestImage(for: asset,
                                              targetSize: CGSize(width: 180, height: 180),
                                              contentMode: .aspectFill,
                                              options: opts) { image, _ in
            if let image { thumbnails[identifier] = image }
        }
    }
}
#endif
