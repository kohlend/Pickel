# Pickex — Übergabe an den App-Bau

Der **Erkennungs-Motor ist fertig und funktioniert** (on-device Gesichtserkennung,
findet Fotos einer bestimmten Person in der Foto-Bibliothek — kein Backend, alles
lokal, iOS 16+). Getestet an einer echten 9000-Foto-Bibliothek: saubere Treffer
(0.6–0.99 für dieselbe Person, Fremde <0.15), keine Fehltreffer.

Diese Datei ist die Spezifikation, um daraus die **richtige App** zu bauen:
Optik + kundentauglicher Scan. Das bisherige `PickexDemoView.swift` ist nur ein
Smoke-Test (Debug-Ausgaben, ein Screen) — es wird durch echte UI ersetzt.

Branch: `claude/mobilefacenet-core-ml-vijmfn`.

---

## Der Motor (fertig, nicht mehr anfassen außer nötig)

Alle Typen liegen in `Pickex/`. Reihenfolge der Pipeline:

**1. Referenzprofil bauen** (1–4 Fotos der gesuchten Person → ein Vergleichs-Embedding)
```swift
let embedder = try FaceEmbedder(modelURL: <FaceEmbeddingR50.mlmodelc-URL>)
let builder  = ReferenceProfileBuilder(embedder: embedder)
let refs = pickedPhotos.map { ReferenceImage(cgImage: $0.cg, orientation: $0.orient) }
let profile = try await builder.build(from: refs)   // ReferenceProfile
// profile.skipped / .multipleFaceWarnings / .consistencyWarning fürs UI auswerten
```

**2. Bibliothek scannen** (streamt Treffer live)
```swift
var cfg = LibraryScannerConfig()          // matchThreshold 0.35, targetSize 1280
cfg.allowNetworkAccess = true             // iCloud-Fotos mitscannen (s.u.)
let scanner = LibraryScanner(embedder: embedder, cache: try? ScanCache(), config: cfg)
for try await event in scanner.scanEvents(against: profile) {
    switch event {
    case .progress(let p): // p.processed/p.total/p.matchCount/p.failedToProcess…
    case .match(let m):    // m.assetLocalIdentifier, m.similarity  → Thumbnail via PHImageManager
    }
}
```
- `ScanCache` (SQLite, WAL): cached rohe Face-Embeddings pro Foto. Zweiter Scan =
  Millisekunden. Neues Referenzprofil gegen Cache matchen: `scanner.rematchFromCache(against:)`.
- Cancel: den umgebenden `Task` canceln — Fortschritt bleibt im Cache.

### Wichtige Fakten
- **Erkennung läuft über Apple Vision** (`VNDetectFaceLandmarksRequest`), nicht mehr
  über das SCRFD-Core-ML-Modell (dessen Keypoints waren auf dem Gerät nicht
  deterministisch). `FacePreprocessor` = `v25-vision`.
- **Nur EIN Modell muss ins Bundle**: `FaceEmbeddingR50.mlpackage`
  (in `MobileFaceNet-CoreML/models/`). `FaceDetectorModel.mlpackage` wird **nicht mehr
  gebraucht** und kann raus.
- Alle Qualitäts-Gates aktiv: Mindest-Gesichtsgröße 48px, Anatomie-Check (kein
  gedrehter/kaputter Crop), Norm-Filter (≥14, gegen Matsch-Embeddings),
  Rotation-Retry (liegende Fotos), Flip-Augmentation, Best-of-Set-Matching.

---

## Was die App braucht (der eigentliche Auftrag)

### A. Optik / Screens
1. **Onboarding/Start**: kurz erklären, Foto-Berechtigung anfragen (`.readWrite`).
2. **Referenz wählen**: `PhotosPicker` (1–4 Fotos), Vorschau, Warnungen aus
   `ReferenceProfile` anzeigen („Foto 2 zeigt evtl. andere Person", „kein Gesicht" …).
3. **Scan-Screen**: Fortschritt + Treffer-Galerie, die sich **live** füllt.
4. **Ergebnis-Galerie**: Grid der Treffer, Tap → Vollbild, Sprung in Fotos-App,
   Mehrfachauswahl/Teilen. Sortierung nach Ähnlichkeit oder Datum.
5. Debug-Ausgaben raus, echte Optik rein.

### B. Kundentauglicher Scan (der Grund für diese Übergabe)
Ein Kunde scannt **einmal** und darf **nicht** minutenlang auf einen Balken starren.
Das Herunterladen tausender iCloud-Fotos ist bandbreitenlimitiert (mehrere Minuten,
unvermeidbar) — die Lösung ist NICHT „schneller laden", sondern **sofort Wert zeigen**:

1. **Phase 1 – lokal zuerst**: Nur on-device Fotos scannen (`allowNetworkAccess=false`
   in einem ersten Durchlauf, ODER die iCloud-Fotos ans Ende sortieren). Das ist in
   Sekunden/wenigen Minuten fertig → erste Treffer sofort sichtbar, App ist nutzbar.
2. **Phase 2 – iCloud im Hintergrund**: die iCloud-only-Fotos nachladen und scannen,
   während der Kunde schon die Treffer durchblättert. Live-Status („4.200/9.000
   durchsucht, läuft weiter…"). Ergebnisse tropfen ein.
3. Scan als abbrechbarer Hintergrund-Job; Cache macht Wiederaufnahme trivial.

Technischer Hinweis für Phase 2: die aktuelle `requestWithTimeout` in
`LibraryScanner` blockiert pro Foto einen Worker (Semaphore). Für hohe
Download-Parallelität ohne Thread-Starvation den Bild-Abruf auf echtes
`async/await` umstellen (`withCheckedContinuation` um die PHImageManager-Callback-API),
dann können viele Downloads gleichzeitig laufen (Netz ist latenz-, nicht CPU-limitiert).
Concurrency für lokale/CPU-Arbeit bei ~4 lassen.

### C. Aufräumen
- `FaceDetector.swift` (SCRFD-Wrapper) wird vom `FacePreprocessor` nicht mehr benutzt —
  kann entfernt werden (dann auch `FaceDetectorModel.mlpackage` aus dem Target).
- Demo-Debug-Pfade (`debugAnnotatedInput`, `debugEyeInfo`, `debugDetectorDiagnostics`,
  `debugVersion`) sind nur fürs Smoke-Test — in der App weg.

---

## Setup-Kontext (Xcode)
- Xcode-Projekt „Pickex", Ziel iPhone (Developer Mode). Modelle per Drag&Drop ins
  Target, Target-Membership setzen. `FaceEmbeddingR50.mlpackage` als
  `FaceEmbeddingR50` referenziert (`Bundle.main.url(forResource:"FaceEmbeddingR50",
  withExtension:"mlmodelc")`).
- Der Nutzer arbeitet am Mac + echtem iPhone; Dateien wurden bisher als
  raw.githubusercontent-Rohtext in Xcode eingefügt. Für den App-Bau sinnvoller:
  sauber committen und der Nutzer macht `git pull`.
