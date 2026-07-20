# Pickex — Übergabe an den App-Bau

## Konzept
Pickex findet alle Fotos **deiner/deines Ex** in der Foto-Bibliothek — damit du
sie in einem Rutsch **löschen** kannst. Emotionaler Aufhänger: „Mach Schluss,
auch auf dem Handy." Breakup-/Digital-Cleanup-Thema. Kernflow: Ex-Foto(s) wählen →
Bibliothek scannen → Treffer-Galerie → auswählen & löschen.

Der **Erkennungs-Motor ist fertig und funktioniert** (on-device Gesichtserkennung,
findet Fotos einer bestimmten Person in der Foto-Bibliothek — kein Backend, alles
lokal, iOS 16+). Getestet an einer echten 9000-Foto-Bibliothek: saubere Treffer
(0.6–0.99 für dieselbe Person, Fremde <0.15), keine Fehltreffer.

## Design-Richtung (vom Nutzer vorgegeben)
Visuelle Sprache wie die **„Win Life"-Onboarding-Screens** (der Nutzer hängt sie
als Referenzbilder an — bitte anschauen):
- **Vollflächige Farb-Hintergründe**, die die Stimmung je Schritt tragen
  (Problem-Screens vs. Lösung/Fortschritt-Screens).
- **Große fette Headline** oben, freundliches Maskottchen/Emoji.
- **Abgerundete Karten** (Icon + Text) mit klarem Auswahl-State (Häkchen/Rahmen).
- **Dicker Pill-Button** unten („Continue/Next/Weiter", weiß mit Pfeil-Kreis).
- **Storytelling-Flow**: Problem → Diagnose → Potenzial → Commitment → Aktion.
- Freundlich, spielerisch, emotional — nicht klinisch.

**Farbschema: aus dem Figma des Nutzers** (nicht raten!). Datei-Key
`oHe0VGQYar1AWzw3G2uRvd`. Farben über die Figma-MCP-Tools ziehen:
`get_variable_defs` / `get_design_context` mit einem **node-spezifischen
`/design/`-Link** (der Nutzer liefert ihn per „Copy link to selection"). Erst die
echten Tokens holen, dann als SwiftUI-`Color`-Palette/Theme anlegen. Bis die Tokens
da sind, KEINE finalen Farben festklopfen.

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

### A. Optik / Screens (im Win-Life-Stil, Farben aus Figma)
1. **Onboarding-Flow** (mehrere Screens, Storytelling): „Trennung hinter dir?" →
   „So viele Erinnerungen liegen noch auf deinem Handy" → „Pickex findet sie alle" →
   Foto-Berechtigung anfragen (`.readWrite`). Maskottchen/Emoji, Karten, Pill-Button.
2. **Ex wählen**: `PhotosPicker` (1–4 Fotos der/des Ex), Vorschau, Warnungen aus
   `ReferenceProfile` anzeigen („Foto 2 zeigt evtl. andere Person", „kein Gesicht" …).
3. **Scan-Screen**: Fortschritt + Treffer-Galerie, die sich **live** füllt
   („142 Fotos gefunden, scanne weiter…"). Lokal zuerst → sofort Ergebnisse.
4. **Ergebnis-Galerie & LÖSCHEN** (der Kern-Payoff): Grid der Treffer,
   Mehrfachauswahl (alle/keine), Tap → Vollbild. Primäraktion **„X Fotos löschen"**
   via `PHPhotoLibrary.shared().performChanges { PHAssetChangeRequest.deleteAssets(...) }`
   — iOS zeigt den System-Löschdialog, die Fotos wandern in „Zuletzt gelöscht".
   Optional: „In Album verschieben" / Teilen als sanftere Alternative.
5. **Erfolg/Abschluss**: „Du hast X Erinnerungen losgelassen 💚" (Lösung-Screen,
   grüner Hintergrund). Storytelling schließen.
6. Debug-Ausgaben raus, echte Optik rein.

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
