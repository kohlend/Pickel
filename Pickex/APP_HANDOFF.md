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

## ⚠️ RELEASE-BLOCKER: Modell-Lizenz (vor App-Store-Launch klären)
Die App hat eine **Paywall (€6.99)** → kommerzielle Nutzung. Das aktuell
verbaute Erkennungsmodell darf so **nicht kommerziell** ausgeliefert werden:

| Bestandteil | Lizenz | kommerziell? |
|---|---|---|
| InsightFace-Code | MIT | ✅ |
| **`FaceEmbeddingR50.mlpackage`** (aus `w600k_r50`, trainiert auf WebFace600K) | non-commercial research only | ❌ **Blocker** |
| Gesichtserkennung (Apple Vision) | Apple SDK | ✅ |

Details + geprüfte Alternativen: `MobileFaceNet-CoreML/README.md` §1. Alle
frei verfügbaren Face-Embedding-Gewichte (EdgeFace, GhostFaceNets, synthetische
Modelle) sind ebenfalls non-commercial — das ist der Zustand des Feldes, kein
Rechercheversäumnis.

**Vor Launch eine dieser Optionen — in dieser Reihenfolge prüfen:**
1. **AuraFace (fal.ai) testen — gratis, wahrscheinlich bester Weg.**
   `fal/AuraFace-v1` auf HuggingFace: ArcFace-Architektur, **Apache 2.0**, trainiert
   auf einem kommerziell lizenzierten Datensatz, liegt als **ONNX** vor → passt
   direkt in `convert.py`, kein App-/Engine-Code ändert sich.
   *Einschränkung laut Autoren:* erreicht nicht die Genauigkeit des Original-ArcFace
   (kleinerer Datensatz), Ethnien-Abdeckung ungleichmäßig. **Also messen, nicht
   annehmen**: konvertieren, ins Bundle statt `FaceEmbeddingR50`, neu scannen und
   die Scores gegen die bisherigen Ergebnisse halten (gleiche Person soll ≥0.6
   bleiben, Fremde <0.2). Lizenz/Modellkarte vor Launch selbst gegenlesen.
2. Kommerzielle Lizenz bei InsightFace kaufen (`recognition-oss-pack@insightface.ai`)
   — kein öffentlicher Preis, individuelles Angebot; als on-device-Einzelprodukt
   anfragen. Nur nötig, wenn AuraFace qualitativ abfällt.
3. Eigenes Training auf kommerziell lizenzierten Daten (teuer, langsam).
4. dlib-ResNet (Public Domain, aber 128-d, älter, kein ONNX → Portierungsaufwand).

**Für den App-Bau blockiert das nichts**: die Pipeline ist modell-agnostisch —
jedes ONNX `[1,3,112,112] → [1,512]` lässt sich per `convert.py` eintauschen,
ohne dass App- oder Engine-Code sich ändert. Also weiterbauen, aber die Lizenz
vor Veröffentlichung klären.

## Design-System (aus dem Figma des Nutzers extrahiert — verbindlich)
FigJam-Board `oHe0VGQYar1AWzw3G2uRvd`, Section „Pickex — Brand". Look:
dunkle Lila-Verläufe, Magenta/Coral-Akzente, einfühlsam-emotional. Der Win-Life-
Screenshot ist nur allgemeine Qualitäts-Referenz; **maßgeblich ist dieses Figma.**

**Farben** → als SwiftUI-`Color`-Theme anlegen (Namen beibehalten):
| Token | Hex | Rolle |
|---|---|---|
| Deep Purple | `#3B1566` | BG oben |
| Mid Purple | `#241041` | BG mitte |
| Near Black | `#100A18` | BG unten |
| Magenta | `#E63C9E` | Akzent |
| Coral | `#F26B4E` | Akzent 2 |
| Surface | `#271B37` | Karten / Chat-Bubbles |
| Ink | `#F5F0FA` | Primärtext |
| Muted Lavender | `#A99BC4` | Sekundärtext |
| Delete Red | `#F04D53` | nur Löschen-Aktion |
| Restore Green | `#4F9F7C` | Wiederherstellen/Erfolg |

**Verläufe**: Akzent `#E63C9E → #F26B4E` (90°) · Hintergrund
`#3B1566 → #241041 → #100A18` (vertikal, auf fast allen Screens).

**Typografie**: Headlines/Card-Titel = **New York** (iOS-Serif, fallback Georgia),
Display 28–30pt Semibold / Title 21–22pt Semibold, line-height 1.2, tracking −1%.
Chat & UI = **SF Pro** (fallback system-ui), Body/Bubble 14.5–15pt Regular,
Button 15–16pt Semibold, Caption 11–12pt.

**Markenstimme**: „pickex" ist ein **Chat-Begleiter** („online · here for you") —
warm, aber ehrlich/tough-love. Beispiele aus dem Flow unten.

## Screen-Flow (13 Screens, exakt aus dem Figma)
Header-Chip „pickex · online · here for you" auf den Chat-Screens.
1. **00 Intro (Chat)**: Bubbles „Hey. 👋" · „Honestly? I don't love that you need
   an app like this. 😢" · „Breakups are hard enough." · „But I'm here to make one
   part easier. First — what should I call you?" → Namens-Eingabefeld.
2. **01 Name + Video (Chat)**: „Don't be sad, {name}. Better times are coming. 💜" ·
   „Quick thing before we start — watch this 👇" · Video-Teaser („80%", 4 sec).
3. **02 Stat (Fullscreen)**: groß „80%" · „of people who break up still keep their
   ex's photos." · „Let's change that." · Continue.
4. **03 Foto-Zugriff**: „Access to your photos" · „So we can find and remove the
   photos, we need access… Everything is processed only on your iPhone." · Bullets:
   No photo ever leaves your device / No account, no cloud / Revoke access anytime ·
   **Allow access** → `PHPhotoLibrary.requestAuthorization(for: .readWrite)`.
5. **04 Referenzfotos**: „Whose photos should go?" · „Pick 1–4 photos with a clearly
   visible face…" · Grid mit Foto-Slots (PhotosPicker) · Tipp: verschiedene Winkel/
   Licht · **Start search** → `ReferenceProfileBuilder.build`.
6. **05 Scanning**: Kreis-Progress „74%" · „918 of 1,240 photos" · „Searching your
   library" · „Running entirely on your device — no upload, no cloud." · Cancel.
   → `LibraryScanner.scanEvents` (lokal zuerst, iCloud im Hintergrund).
7. **06 Choice**: „Found {n} photos of them. How do you want to do this?" →
   **Delete them all — I trust you** / **Let me check first**.
8. **07 Msg – blind** (nach „trust you"): „Don't worry — they'll stay in your
   Recently Deleted for 30 days. Just in case." · **Delete them**.
9. **08 Msg – review** (nach „check first"): „Alright — just this once. But don't
   keep anything. You need to let go." · **Show me the photos**.
10. **09 Results**: „{n} photos found" · „Deselect all" · Foto-Grid (4 Spalten,
    Mehrfachauswahl) · **Delete {k} photos** · „Unlock once to remove photos".
11. **10 Paywall**: „Pay once. Use forever." · Bullets: Unlimited searches & people /
    Find every photo in seconds / No subscription, no hidden fees · **Unlock for
    €6.99** · Restore purchase · „One-time purchase, billed to your App Store account."
    (StoreKit 2, Non-Consumable. Gate: Löschen erst nach Kauf.)
12. **11 Confirm (Sheet)**: „Delete {k} photos? They go to 'Recently Deleted' for
    30 days, then they're gone for good." · **Delete** / Cancel → `PHPhotoLibrary`
    `deleteAssets`.
13. **12 Done**: „Space for something new." · „{k} photos are gone from your library.
    Onward, {name}." · **Start over**.

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
Der komplette Screen-Flow steht oben („Screen-Flow, 13 Screens") mit Texten und
Aktionen. Umsetzen im Figma-Look (Lila-Verlauf-Hintergrund, Magenta/Coral-Akzente,
New-York-Headlines, SF-Pro-Body, Chat-Bubbles). Debug-Ausgaben des Demos raus.
Technische Ankerpunkte:
- **Löschen** (Screen 11): `PHPhotoLibrary.shared().performChanges {
  PHAssetChangeRequest.deleteAssets(assets as NSFastEnumeration) }` — iOS zeigt den
  System-Löschdialog, Fotos wandern in „Zuletzt gelöscht" (30 Tage).
- **Paywall** (Screen 10): StoreKit 2, ein Non-Consumable „Unlock" (€6.99). Löschen
  erst nach erfolgreichem Kauf freischalten; Restore anbieten.
- **Live-Ergebnisse**: Scan-Screen (05) zeigt Fortschritt; Treffer fließen live rein,
  Results (09) ist das gefüllte Grid.

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
