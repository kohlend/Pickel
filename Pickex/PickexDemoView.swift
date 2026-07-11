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

                if let p = progress {
                    ProgressView(value: p.fraction) {
                        Text("\(p.processed)/\(p.total) · \(p.matchCount) Treffer"
                             + (p.skippedNotLocal > 0 ? " · \(p.skippedNotLocal) nicht lokal" : "")
                             + (p.servedFromCache > 0 ? " · \(p.servedFromCache) aus Cache" : ""))
                        .font(.caption)
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
        }
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

            // 5. Scan (Step C) — matches appear live.
            let scanner = LibraryScanner(embedder: embedder, cache: try? ScanCache())
            for try await event in scanner.scanEvents(against: profile) {
                switch event {
                case .progress(let p): progress = p
                case .match(let m):
                    matches.append(m)
                    matches.sort { $0.similarity > $1.similarity }
                    loadThumbnail(for: m.assetLocalIdentifier)
                }
            }
            status = "Fertig: \(matches.count) Treffer."
        } catch ReferenceProfileError.noFaceInAnyPhoto(let skipped) {
            let details = skipped.map { s -> String in
                switch s.reason {
                case .noFaceDetected: return "Foto \(s.index + 1): kein Gesicht erkannt"
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
        let candidates = ["FaceEmbedding", "FaceEmbedding_fp16", "FaceEmbedding_fp32"]
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
