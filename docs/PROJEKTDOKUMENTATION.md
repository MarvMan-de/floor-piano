# Floor-Piano — Entwicklungsdokumentation

> **Studienprojekt** · Stand: **2026-07-02**
> Dieses Dokument protokolliert die Entwicklungsphase rund um die
> Kamera-Integration: was analysiert, entschieden und geändert wurde, welche
> Erkenntnisse dabei entstanden und was noch offen ist. Es ergänzt den
> technischen Review-Bericht in [`CODE_REVIEW.md`](CODE_REVIEW.md).

---

## 1. Management Summary (Kurzfassung)

Das Floor-Piano erkennt über eine 3D-Tiefenkamera Fußtritte auf einer bemalten
2-m-Klaviermatte und spielt die entsprechenden Töne. Ziel dieser Phase war zu
klären, **ob und wie das System live auf einem Raspberry Pi 5 (8 GB) mit der
realen Kamera läuft**, und die Software dafür vorzubereiten.

Wichtigste Ergebnisse:

1. **Rechenleistung ist nicht der Engpass.** Gemessen: die Erkennungs-Pipeline
   braucht ~1 ms/Frame (Entwicklungs-PC), hochgerechnet ~8–10 ms auf dem Pi 5 →
   ~100 FPS möglich, während die Kamera nur 30 FPS liefert. **3-facher Puffer.**
   8 GB RAM sind massiv überdimensioniert (Bedarf: wenige hundert MB).
2. **Die eigentliche Herausforderung war die Kamera-/Treiber-Integration** — und
   die hing entscheidend am **exakten Kameramodell**, das sich im Projektverlauf
   zweimal geändert hat (siehe Abschnitt 4).
3. **Die finale Kamera (Orbbec Gemini 335) ist der Idealfall:** modern, vom
   aktuellen SDK (`pyorbbecsdk`) direkt unterstützt — der ursprüngliche
   Standard-Code-Pfad funktioniert damit ohne Treiber-Bastelei.
4. Die Software wurde robuster gemacht, um **beliebige** Kameras zu unterstützen
   (austauschbare Kameraquellen), und um die **Bedienung zu vereinfachen**
   (automatische Platzierungs-Hilfe / „Placement-Coach").

---

## 2. Ausgangslage

- **Hardware-Plattform:** Raspberry Pi 5, 8 GB RAM.
- **Matte:** bedruckte Klaviermatte, **2 m lang**, 14 weiße + 10 schwarze Tasten
  (2 Oktaven, C4–B5).
- **Prinzip:** Kamera von oben → Tiefenbild → Matte per Perspektiv-Entzerrung
  („Warp") auf ein festes Tastatur-Raster projizieren → Fuß = Pixel näher als
  der Boden → Taste löst Ton aus.
- **Vorheriger Stand (Code-Review, siehe `CODE_REVIEW.md`):** drei Blocker —
  (1) Kamera/SDK, (2) RGB↔Tiefe-Registrierung, (3) defekte Audio-Samples.

---

## 3. Performance-Analyse: Läuft es live auf dem Pi 5?

### Vorgehen
Die Kernfunktion `process_frame` (Warp → Blob-Erkennung → Debounce → Audio)
wurde mit synthetischen, realitätsnahen Tiefenframes (zwei „Füße" + 5 %
Messlöcher) über 500 Durchläufe gemessen.

### Messergebnisse
| Umgebung | Zeit/Frame | entspricht |
|---|---|---|
| Entwicklungs-PC (Intel i5-1235U), mehrere Threads | ~1,0 ms | ~1000 FPS |
| Entwicklungs-PC, **1 Thread** | ~1,9 ms | ~530 FPS |
| **Raspberry Pi 5 (hochgerechnet, ~4–5× langsamer/Kern)** | **~8–10 ms** | **~100–125 FPS** |

Die Kamera liefert 30 FPS (33 ms pro Frame). Selbst pessimistisch bleibt damit
**mehr als das Dreifache an Reserve**.

### Warum so günstig
- Die gesamte Bildarbeit läuft auf einer festen Leinwand von **1400×200 Pixeln**
  (280 000 px). Die teuren Schritte (`warpPerspective`, morphologisches Schließen,
  `connectedComponentsWithStats`) sind hochoptimierter C-/NEON-Code in OpenCV;
  Python macht nur leichte Mengenlogik.
- Im Betrieb läuft das Programm **headless** (ohne Fenster) — das Zeichnen/Anzeigen
  entfällt komplett.

### Erkenntnisse
- **Gelernt:** Der oft vermutete Flaschenhals „Bildverarbeitung auf dem Pi" ist
  hier keiner. RAM ist völlig unkritisch (ein 2-GB-Pi würde reichen).
- **Gelernt:** Das echte Risiko lag nie bei der Rechenleistung, sondern bei der
  **Kamera-/Treiber-Anbindung**. Genau dort lag der Fokus der weiteren Arbeit.
- **Bestätigung steht aus:** Die Zahlen sind vom Entwicklungs-PC skaliert, nicht
  auf echter Pi-5-Hardware gemessen. Das Diagnose-Skript `tools/probe_gemini.py`
  zeigt am Pi die reale Bildrate (Feld `fps`).

---

## 4. Kamera-Hardware: die Modell-Klärung (zentrale Lernkurve)

Die wichtigste Erkenntnis der Phase: **das exakte Kameramodell entscheidet über
den kompletten Software-Weg.** Die Annahme dazu änderte sich zweimal:

| Angenommenes Modell | Konsequenz | Reaktion im Projekt |
|---|---|---|
| **Astra Pro** (zuerst angenommen) | Legacy-Gerät, Tiefe nur über **OpenNI2**, RGB separat über UVC; `pyorbbecsdk` bedient sie **nicht** | OpenNI2-Backend recherchiert & gebaut |
| **Original Astra** (Zwischenstand) | Ebenfalls Legacy/OpenNI2, aber besserer Raspberry-Pi-Track-Record; RGB **auch** über OpenNI2 → Registrierung einfacher | Doku/Backend auf original Astra korrigiert |
| **Gemini 335** (finale, echte Kamera) | **Modern, von OrbbecSDK v2 / `pyorbbecsdk` nativ unterstützt** → Standard-Code-Pfad funktioniert direkt | OpenNI2-Weg wird nur noch Fallback |

### Die finale Kamera: Orbbec Gemini 335 (Modellnr. G40155-170)
Recherchierte technische Daten (Quelle: Orbbec-Produktseite/Datenblatt):

| Merkmal | Wert |
|---|---|
| Tiefen-Technologie | Aktives Stereo (Infrarot), 50 mm Basisbreite |
| **Tiefen-Sichtfeld (FOV)** | **90° horizontal / 65° vertikal** |
| Tiefen-Reichweite | 0,10–10 m (optimal **0,26–3 m**) |
| Tiefen-Auflösung | bis 1280×800 @ 30 FPS |
| RGB | bis 1920×1080 @ 30 FPS, FOV 86°/55° |
| Genauigkeit | < 1,5 % RMSE bei 2 m |
| Anschluss | **USB 3.0 Type-C** |

### Software-Unterstützung (recherchiert)
- **`pyorbbecsdk`** (OrbbecSDK v2) unterstützt die Gemini-330-Serie nativ.
- Das PyPI-Paket heißt **`pyorbbecsdk2`**, importiert sich aber als
  **`pyorbbecsdk`** — genau das, was unser Code erwartet, **keine Code-Änderung nötig**.
- Es gibt **fertige Wheels für Linux x86_64 UND ARM64** (Python 3.8–3.13).
  → Installation auf Laptop und Pi ist ein einfaches `pip install pyorbbecsdk2`.
  → Der alte Hinweis in `requirements.txt` („kein arm64-Wheel") ist damit **veraltet**.

### Erkenntnisse
- **Gelernt:** „Orbbec Astra" ist keine Kamera, sondern eine ganze Familie mit
  grundverschiedener Software-Anbindung (Legacy-OpenNI2 vs. modernes OrbbecSDK v2).
  **Ohne exaktes Modell keine belastbare Aussage.**
- **Gelernt:** Die investierte OpenNI2-Arbeit war mit der damaligen Info die
  richtige Entscheidung und ist nicht verloren — sie bleibt als **Fallback** für
  Legacy-Kameras im Repo, schläft aber im Normalbetrieb.

---

## 5. Kamera-Datenformat & Übertragung (Grundlagen)

Zur Klarstellung einer im Projekt aufgekommenen Frage („streamt die Kamera zwei
MP4s?"): **Nein.** Eine Tiefenkamera liefert **rohe, unkomprimierte Frame-Puffer
in Echtzeit**, keine Videodateien.

- **Tiefen-Stream:** je Frame ein 2D-Feld aus **16-Bit-Ganzzahlen** (`uint16`),
  ein Wert pro Pixel = **Entfernung in Millimetern**. `0` = „keine Messung".
  (Im Code geparst von `decode_depth`.)
- **Farb-Stream (RGB):** getrenntes, normales Farbbild.
- Die **Tiefe wird auf dem Kamera-Chip berechnet** (Structured-Light bei Astra,
  aktives Stereo bei der Gemini) — der Pi bekommt **fertige mm-Werte** über USB,
  muss also nichts rechnen. Das erklärt die niedrige CPU-Last.
- Ein normales Kamera-Programm kann nur das **Farbbild** anzeigen (UVC-Webcam),
  **nicht** das Tiefenbild — dafür braucht es das SDK bzw. den `OrbbecViewer`.

---

## 6. Software-Architektur: ein Kern, mehrere Kameraquellen

Wichtige Klarstellung: Es gibt **nicht zwei Programme** (MP4 vs. Kamera), sondern
**einen** Erkennungskern mit **austauschbaren Eingabequellen** (alle mit gleicher
Schnittstelle `start()/read_depth()/stop()`):

| Quelle (in `src/depth_camera.py`) | Zweck |
|---|---|
| `DepthCamera` (pyorbbecsdk) | **Gemini 335** & andere OrbbecSDK-v2-Kameras — **Standardweg** |
| `OpenNI2DepthCamera` | Legacy-Kameras (Astra) über OpenNI2 — **Fallback** |
| `VideoDepthCamera` | MP4 vom Handy — **Entwicklungs-Krücke** (fälscht Tiefe aus Helligkeit/Bewegung) |
| `MockDepthCamera` | synthetische Frames für die Tests |

Die Auswahl erfolgt über die Umgebungsvariable **`FLOOR_PIANO_CAMERA`**
(Standard `orbbec`; `openni2` für Legacy). Umgesetzt durch die neue Factory
`make_depth_camera()`, auf die `main.py` umgestellt wurde. → **Kamerawechsel =
eine Umgebungsvariable, kein Code-Eingriff.**

---

## 7. Durchgeführte Code-Änderungen (mit Commits)

Alle Änderungen dieser Phase liegen auf dem Branch `master`:

| Commit | Inhalt |
|---|---|
| `6cd8971` | **Detection-Rework** abgesichert (war unversioniert): Blob-basierte Treffer (ein Fuß = ein Blob = genau eine Taste, Grenz-Hysterese), `MAX_PRESS_HEIGHT`-Band (nur bodennahe Pixel triggern), Debounce, Matten-Auto-Kalibrierung ohne Marker, diverse Härtungen. |
| `7c302d3` | **OpenNI2-Backend** (`OpenNI2DepthCamera`) + Factory `make_depth_camera()` + Diagnose-Skript `tools/probe_astra.py` + Setup-Runbook + Tests. |
| `6b58c7e` | **Korrektur auf original Astra** (statt Pro): Doku/Kommentare, Runbook umbenannt zu `docs/ASTRA_PI5_SETUP.md`. |
| `b0740a0` | **Placement-Coach** (`src/placement.py`) + Einbindung in `calibrate.py` + Tests. |
| `56e54f0` | **`tools/probe_gemini.py`** — Live-Zahlen-Ausgabe des Tiefenstreams (Realtime-Check). |
| `a602d98` | **`tools/view_gemini.py`** — grafischer Live-Viewer (eingefärbte Tiefe + HUD). |

**Teststand:** 129 automatisierte Tests (pytest), alle grün. Alle hardware-freien
Teile sind damit abgesichert; hardware-abhängige Stellen sind im Code als
`TODO(hardware)` markiert.

---

## 8. Kalibrierung & Ease-of-Use („Placement-Coach")

### Anforderung (vom Team priorisiert)
Kamera **grob** aufstellen → System kalibriert automatisch → sagt **„passt"**
oder **„so nicht — häng sie höher / nach links"**. Kein millimetergenaues Suchen.

### Was `calibrate.py` tut
Findet automatisch die Mattenecken (per ArUco-Marker **oder** markerlos aus den
aufgedruckten Tasten), misst in der Mitte den Bodenabstand, sichert gegen
Störungen ab (jemand läuft durchs Bild, gespiegelte/entartete Erkennung), wartet
auf einen **stabilen Blick** (15 Frames) und speichert dann `config.json`
(Ecken, Bodenabstand, Tastenzahl, Leinwandgröße). Headless: Auto-Speichern +
Timeout; mit Monitor: Taste `s`.

### Neu: der Placement-Coach (`src/placement.py`)
Reine, testbare Geometrie: `assess_placement(...)` liefert **strukturierte**
Rückmeldung statt eines Ja/Nein — Statuscodes `no_mat / no_depth / clipped /
near_edge / too_small / ok` plus konkreter Richtungshinweis („links angeschnitten
→ Kamera nach links schwenken"; gegenüberliegende Kanten → „höher / weiter weg").

Eingebunden in `calibrate.py`: speist gleichzeitig das Headless-Log, ein
GUI-Overlay und eine **`placement_status.json`**.

### Bezug zum Web-UI
Ein **Web-UI über den Pi-Hotspot** ist geplant (wird von einer anderen Person
gebaut). Der Coach liefert deshalb **maschinenlesbare** Daten (Statuscode +
Messwerte + JSON), damit das Web-UI keine Log-Texte parsen muss, sondern
`assess_placement` direkt aufruft **oder** `placement_status.json` pollt.

---

## 9. Geometrie: Sichtfeld & Montagehöhe

Für die 2-m-Matte bei senkrechter Deckenmontage (2 m entlang des horizontalen
Sichtfelds). Formel: `Höhe = (Breite/2) / tan(FOV_h / 2)`.

| Kamera | FOV (H) | Exakt-Fit | **Empfohlene Höhe** |
|---|---|---|---|
| (alt) Astra | 58,4° | 1,79 m | ~2,0 m |
| **Gemini 335** | **90°** | **1,00 m** | **~1,1–1,2 m** (10–20 cm Rand je Seite) |

- **Erkenntnis:** Das weite Sichtfeld der Gemini erlaubt eine **deutlich
  niedrigere Montage** (~1,1 m statt ~2 m) — passt unter jede normale Decke und
  liegt im **optimalen** Tiefenbereich der Kamera (0,26–3 m).
- `floor_depth` ist dann auf **~1100–1200 mm** zu kalibrieren (der Standardwert
  1000 mm im Code wird beim Kalibrieren ohnehin überschrieben).
- Grenze: Software kann Physik nicht ersetzen — eine zu niedrige Montage deckt
  die 2 m nicht ab; der Coach meldet das dann korrekt.

---

## 10. Test- & Diagnose-Werkzeuge

| Werkzeug | Zweck | Braucht Kalibrierung? |
|---|---|---|
| **`OrbbecViewer`** (offiziell) | Live Tiefe+Farbe+IR ansehen, Kamera prüfen | nein |
| **`tools/view_gemini.py`** | eigener grafischer Live-Viewer: eingefärbte Tiefe + HUD (FPS, Auflösung, Format, Distanz mittig/unter Maus, min/med/max) | nein |
| **`tools/probe_gemini.py`** | Live-Zahlen im Terminal (headless-tauglich): Format, Depth-Scale, FPS, Distanzen | nein |
| **`tools/probe_astra.py`** | Phase-0-Probe für Legacy-Astra (OpenNI2) | nein |

Empfohlene Reihenfolge bei Inbetriebnahme:
1. `OrbbecViewer` **oder** `tools/view_gemini.py` → „streamt Tiefe? sinnvoll?"
2. `pip install pyorbbecsdk2`
3. `python3 src/calibrate.py` → erzeugt `config.json`
4. `python3 src/main.py` → das Piano

---

## 11. Repository-Zustand & Git-Workflow

- **Alle Arbeit liegt auf `master`** — bewusst konsolidiert, damit MP4-Testpfad
  **und** Kamera-Pfad zusammen im selben Stand testbar sind.
- Es gibt zusätzlich den (inzwischen im `master` enthaltenen, damit veralteten)
  Branch `astra-openni2` auf GitHub.
- **Offen:** `master` ist **6 Commits vor GitHub** (`origin/master`). Der Push auf
  den Default-Branch wurde bewusst zurückgehalten und wartet auf ausdrückliche
  Freigabe. → **Diese 6 Commits sind aktuell nur lokal gesichert.**
- Merkregel für das Team: **Getestetes/Fertiges → `master`; unverifizierter
  Hardware-Code → eigener Branch, bis die Hardware ihn bestätigt.** (Ein Branch
  ist reines lokales Git und braucht keine GitHub-Funktionen.)

---

## 12. Blocker-Status (Bezug zu `CODE_REVIEW.md`)

| Blocker | Status | Anmerkung |
|---|---|---|
| **#1 Kamera/SDK** | **Gelöst** (durch Gemini 335) | pyorbbecsdk-nativ; Standard-`DepthCamera`-Pfad. Legacy-Fallback (OpenNI2) existiert. |
| **#2 RGB↔Tiefe-Registrierung** | **Offen, aber lösbar** | Gemini kann Hardware-**D2C-Alignment** (im SDK) — noch nicht verdrahtet. Alternativ Kalibrierung direkt über Tiefe/IR (`--source mat`). |
| **#3 Audio-Samples** | **Behoben** | 24 gültige WAV-Dateien vorhanden. |

---

## 13. Offene Punkte / Nächste Schritte

1. **Am Pi:** `pip install pyorbbecsdk2`, Kamera an USB-3-Port → `tools/probe_gemini.py`
   ausführen und **reale FPS bestätigen** (Live-Beweis der Echtzeit-Fähigkeit).
2. **`calibrate.py` an die Gemini anpassen:** RGB kommt bei der Gemini über den
   SDK/UVC-Weg, nicht zwingend über den bisherigen V4L2-Pfad (`find_rgb_camera`).
3. **Registrierung (#2) fertigstellen:** D2C-Alignment aktivieren **oder** Matte
   direkt im Tiefen-/IR-Bild kalibrieren.
4. **`requirements.txt` aktualisieren:** `pyorbbecsdk2` ist jetzt per Wheel
   (auch arm64) installierbar — der „kein arm64-Wheel"-Hinweis ist überholt.
5. **HUD-Erweiterung (optional):** Kamera-Temperatur/Seriennummer/Firmware ins
   Viewer-HUD, sobald am Gerät geprüft ist, was das SDK dafür anbietet.
6. **`master` nach GitHub pushen** (nach Freigabe), damit alles gesichert ist.
7. **Web-UI** (andere Person) an `placement_status.json` / `assess_placement` anbinden.

---

## 14. Lessons Learned (Zusammenfassung)

1. **Exakte Hardware zuerst klären.** Das Kameramodell hat den gesamten
   Software-Weg bestimmt; falsche Annahmen kosteten Umwege. „Orbbec Astra" ≠ ein
   eindeutiges Gerät.
2. **Performance war nie das Problem** — vermutete Engpässe erst messen, dann
   optimieren. Der Pi 5 hat für diese Aufgabe reichlich Reserve.
3. **Abstraktion zahlt sich aus.** Weil die Kameraquelle austauschbar ist, kostete
   der Wechsel Astra → Gemini praktisch keine Änderung am Erkennungskern.
4. **Für die Bedienung mitdenken.** Der Placement-Coach macht aus „exakten Punkt
   suchen" ein „Hinweisen folgen" — und liefert die Daten gleich maschinenlesbar
   für das kommende Web-UI.
5. **Ehrlich dokumentieren, was ungetestet ist.** Hardware-abhängige Stellen sind
   als `TODO(hardware)` markiert; die reine Logik ist per Tests abgesichert.
6. **Moderne Hardware vereinfacht.** Die (unerwartet neuere) Gemini 335 löste den
   größten Blocker praktisch von selbst — neuer ist hier tatsächlich besser.

---

## 15. Phase 2: Kamera-Inbetriebnahme, Web-UI & Play-Modus (Juli 2026)

### 15.1 Inbetriebnahme der Gemini 335 (bestanden)
- Kamera erkannt als `2bc5:0800` (USB 3), Treiber per `pip install pyorbbecsdk2`.
  Einmalig nötig: **udev-Regeln** aus dem SDK-Paket installieren + Kamera neu
  einstecken (sonst `openUsbDevice failed`).
- `tools/probe_gemini.py`: **stabile 30 FPS**, 848×480, Format Y16, Depth-Scale
  1.0 (= Millimeter) → die Echtzeit-Annahme aus Abschnitt 3 ist **live bestätigt**.
- Eigener Live-Viewer `tools/view_gemini.py` (Tiefe eingefärbt + HUD; Umschalten
  Tiefe/RGB/D2C-Overlay). Befund: `65535`-Werte sind der **Ungültig-Marker**
  (0xFFFF), kein 65-m-Messwert — inzwischen wird ≥60000 überall als „keine
  Messung" behandelt, damit weder Boden-Median noch Oberflächen-Referenz
  vergiftet werden.
- Beobachtung Stereo-Tiefe: **Abschattung/Okklusion** an Objektkanten (ein
  Punkt braucht beide IR-Kameras) erklärt „doppelte" Ränder im Tiefenbild; im
  senkrechten Aufbau über einer flachen Matte weitgehend unkritisch.

### 15.2 Web-UI: Integration und Umbau (jetzt Herzstück des Projekts)
Das vom Teamkollegen gebaute FastAPI-Web-UI (Tiles konfigurieren) wurde in
`master` gemergt (ein einziger Entwicklungsbranch) und an die reale Nutzung
angepasst:
- **Kameraquelle:** Gemini-**Farbbild** über pyorbbecsdk (Tasten sind nur in
  Farbe sichtbar), seitenverhältnis-treu auf 640×480 (Letterbox statt Verzerrung).
- **Zuverlässige Tastenplatzierung:** Die Helligkeits-Auto-Erkennung scheiterte
  in echten, schrägen Szenen (legte 24 Streifen übers ganze Bild). Neuer Weg:
  **Mensch setzt/zieht die 4 Mattenecken → die 24 Tasten werden perspektivisch
  projiziert** (`getPerspectiveTransform` + Tastenraster) — korrekt bei jedem
  Kamerawinkel; „Auto-Ecken" liefert nur noch einen Startwert.
- **Bugfixes:** doppelt gezeichnete Tiles (Server brannte sie zusätzlich in den
  MJPEG-Stream), Shutdown-Tracebacks (async-Stream + CancelledError-Logfilter),
  Klick auf eine gemalte Taste spielt ihre Note (Layout-Check per Ohr).

### 15.3 Play-Modus: der Erkennungskern (Finger/Fuß → Ton)
Getestet mit Tablet (Tastenbild) + Finger; das Tablet stand **schräg** → ein
globaler Bodenabstand taugt nicht. Drei Iterationen bis zur robusten Lösung:
1. **Hintergrundsubtraktion:** einmal „Oberfläche erfassen" (Median mehrerer
   D2C-ausgerichteter Tiefenframes) → Pro-Pixel-Referenz; funktioniert bei jeder
   Neigung. (D2C: Tiefe wird per SDK-`AlignFilter` aufs Farbbild gerechnet und
   identisch geletterboxt → Tiefen-Pixel == Tile-Koordinaten.)
2. **Problem „Hand drüber löst aus"** → **Kontakt-Band**: ein Druck ist nur, was
   ~5–30 mm **an** der Fläche liegt (Fingerspitze); Schwebendes ist höher → stumm.
3. **Problem „alle Tasten feuern"** (verrauschte/verschobene Referenz driftet
   ins Band) → **Blob-Erkennung** wie im Tiefen-Piano: nur fingergroße
   zusammenhängende Flächen zählen (zu klein = Rauschen, zu groß = Drift →
   verworfen), jeder Blob feuert genau **eine** Taste.

Trigger laufen server-seitig, der Browser pollt sie, blinkt das Tile und spielt
die Note (`/sounds`-Samples). Edge-Trigger + Release-Debounce verhindern
Dauerfeuer. Stellschrauben: `webui/depth_detect.py`
(`DEFAULT_CONTACT_MIN/MAX_MM`, `DEFAULT_MIN_BLOB_PX`, `DEFAULT_MAX_BLOB_FRAC`).

### 15.4 Projekt-Hygiene
- **Ein** Branch (`master`), alle Feature-Branches gemergt und gelöscht.
- **Eine** Abhängigkeitsliste: `requirements.txt` konsolidiert (Core + Kamera +
  Web-UI + Tests); `pyproject.toml` gefüllt → `uv sync` / `uv run` funktionieren.
- Teststand: **174 pytest**, alle grün (inkl. Kontakt-Band-, Hover-, Drift- und
  Blob-Fällen mit synthetischen Tiefenbildern).

### 15.5 Lessons Learned (Phase 2)
1. **Auto-Erkennung braucht den Menschen im Loop:** 4 Ecken manuell setzen ist
   robuster als jede Helligkeits-Heuristik — und die Mathematik (perspektivische
   Projektion) erledigt den Rest.
2. **„Vor der Fläche" ≠ „auf der Fläche":** Erst das Kontakt-Band macht aus
   Tiefen-Daten eine Berührung; erst Blobs machen daraus einen Finger.
3. **Referenzen altern:** Verrutscht Kamera oder Fläche, muss die
   Oberflächen-Referenz neu erfasst werden — im UI bewusst ein Ein-Klick-Schritt.
4. **Harte Prozess-Kills vermeiden:** `kill -9` während des USB-Streams warf die
   Kamera vom Bus; sauberes Ctrl-C genügt.

### 15.6 Offen für den Abschluss
- Schwellen-Feintuning an der **echten Matte** (statt Tablet).
- Deployment am Pi: systemd-Service für `webui.server` + Hotspot-Einrichtung.
- Optional: Standalone-Pfad (`src/main.py`) an die Web-UI-Tiles anbinden, falls
  Pi-Lautsprecher-Ausgabe ohne Browser gewünscht ist.

---

## Anhang: Glossar

- **Tiefenbild / Depth Map:** Bild, dessen Pixel Entfernungen (mm) statt Farben enthalten.
- **FOV (Field of View):** Sichtwinkel der Kamera in Grad (horizontal/vertikal).
- **Warp / Perspektiv-Entzerrung:** Rechenschritt, der die schräg gesehene Matte
  auf ein gerades, festes Tastatur-Raster projiziert (`cv2.warpPerspective`).
- **Blob:** zusammenhängende Fläche „über dem Boden" — ein Fuß.
- **Structured Light / Aktives Stereo:** zwei Verfahren, mit denen 3D-Kameras
  Tiefe berechnen (Astra bzw. Gemini) — beides passiert **im Kamera-Chip**.
- **OpenNI2:** älteres Treiber-Framework für Legacy-Tiefenkameras (z. B. Astra).
- **pyorbbecsdk / OrbbecSDK v2:** aktuelles Orbbec-SDK; unterstützt moderne
  Kameras wie die Gemini 335 nativ.
- **UVC:** USB-Standard für Webcams (Farbbild); von normalen Kamera-Apps lesbar.
- **D2C (Depth-to-Color) Alignment:** rechnet Tiefen- und Farbbild deckungsgleich
  — nötig, damit „richtige Taste = richtige Note" stimmt.
- **Headless:** Betrieb ohne Monitor/Fenster (Ziel-Betriebsart auf dem Pi).
- **Debounce:** Entprellen — verhindert, dass Messrauschen einen Ton mehrfach auslöst.
