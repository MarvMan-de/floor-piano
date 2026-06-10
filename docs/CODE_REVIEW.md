# Konsolidierter Review-Bericht: Floor Piano v2.0

> Erstellt 2026-06-03 · Multi-Agent-Analyse (5 Perspektiven: correctness, hardware,
> quality, docs, architecture) mit adversarischer Verifikation pro Finding.
> 51 Findings gesammelt, alle als real bestätigt. Überlappungen wurden zusammengeführt.

## ✅ Bearbeitungsstand (2026-06-03)

Alles, was **ohne Hardware** machbar war, ist erledigt:
- **Behoben:** #3 (Audio: offline-Generator + Härtung), #4 (Y16-Decode-Guard via `decode_depth`),
  #5/#6 (Doku-Ehrlichkeit + Pfade), #8 (config-Validierung), #10 (Hailo-Fusion-Doku),
  Hailo komplett entfernt, sowie die Low/Info-Punkte close()/cleanup(), Magic Numbers
  (`constants.py`), Tonleiter-Duplikat, FPS-Throttle, print→`logging`, Integer-Division der
  Tastenbreite (`key_bounds`), Orbbec-Init-Duplikat (`depth_camera.py`), RGB-Handle-Leak,
  27W-PSU-Begründung, pyorbbecsdk-Installationshinweis.
- **#9 (Architektur):** Kernlogik in hardwarefreie Module (`detection.py`, `depth_camera.py`,
  `constants.py`) ausgelagert; Hardware-Init aus dem Konstruktor in `start()`; Exceptions
  statt `sys.exit`; `MockDepthSource`; **43 pytest-Tests** (grün).
- **#7 (Headless):** `waitKey`/GUI DISPLAY-gegated, Headless-Auto-Save in `calibrate.py`,
  SIGTERM-Shutdown, systemd-Vorlage (`docs/floor-piano.service`).
- **#11 (Kamera-Physik):** als Placement-Hinweise in SETUP.md dokumentiert; Lochfüllung offen.

**Laufzeit-Härtung (2. Multi-Agent-Review der Refaktorierung):** numpy-2.x/Buffer-Ownership in
`decode_depth` (`.copy()`); `pygame.mixer.init()` abgesichert (stumm statt Crash-Loop unter
systemd); `pyorbbecsdk` aus `requirements.txt` entfernt (kein arm64-Wheel → `pip install` brach
sonst komplett ab); `warpPerspective` auf `INTER_NEAREST` (keine 0-Loch-Interpolation);
Depth-Scale wird gelesen/geloggt/angewandt (mm-Schwelle stimmt geräteunabhängig); RGB-Kamera
per Probe-Frame validiert + `FLOOR_PIANO_RGB_INDEX`-Override; einmaliger Sanity-Check, ob die
Kalibrier-Ecken ins Tiefenbild passen (deckt #2 vor Ort auf); None-Depth-Profil + nicht-numerische
config sauber als Fehler.

**Verbleibt — braucht echte Hardware:** **#1** (Astra Pro ↔ pyorbbecsdk: liefert die Kamera ein
Depth-Profil? sonst OpenNI2) und **#2** (RGB→Depth-Registrierung — das Kernfeature; Notbehelf via
Auflösungsskalierung beseitigt die Parallaxe nicht → echte Lösung ist D2C-Alignment vor Ort).
Optional später: GPIO-LED/Reset-Button, RANSAC (Phase 2), Tiefen-Lochfüllung.

## ✅ Detektions-Rework + 3. Multi-Agent-Review (2026-06-09)

Das Erkennungssystem wurde überarbeitet und **gegen die echten Mat-Videos verifiziert**
(`videos/source/besser.mp4` + WhatsApp-Clip; Frame-für-Frame-Abgleich der Trigger gegen die
sichtbare Fußposition auf der bemalten Matte):

- **Blob-Detektion** (`detect_hits_blobs`): Ein Fuß = ein zusammenhängender Blob = genau EINE
  Taste (Mehrheits-Overlap statt Pixelzählung pro Taste) → kein Doppel-Trigger mehr auf
  Tastengrenzen; zwei Füße = zwei Blobs = Akkorde funktionieren. Boundary-Hysterese
  (`sticky`): ein 50/50 auf der Grenze stehender Fuß flattert nicht mehr zwischen zwei Noten.
- **Press-Height-Band** (`MAX_PRESS_HEIGHT` 250mm): nur Pixel nahe am Boden drücken Tasten —
  ein schwingender Fuß/Knie/Oberkörper über der Matte löst nichts mehr aus (war Blocker im
  3. Review).
- **HitTracker-Debounce**: Release erst nach 3 Frames ohne Erkennung → Rauschen kann gehaltene
  Noten nicht mehr maschinengewehrartig neu triggern. `suppress_white_under_black` aus dem
  Laufzeitpfad entfernt (zerstörte legitime Schwarz+Weiß-Akkorde; Blob-argmax übernimmt das).
- **Matten-Auto-Kalibrierung** (`mat_calibration.py`): Ecken, Orientierung (alle 4 Flips,
  auch gespiegelte Clips) und Sub-Pixel-Verfeinerung direkt aus den aufgedruckten Tasten —
  ohne ArUco. Auf besser.mp4: Raster-Residuum 9.2 → **0.5 px**. In `calibrate.py` als
  Fallback integriert (`--source auto|aruco|mat`).
- **Video-Testpfad**: `--motion` nutzt jetzt einen festen Median-Hintergrund statt MOG2
  (kein "Einfrieren" stehender Füße, keine Ghosts); Render zeigt die gewarpte Matte in
  Farbe + Maskenkontur → Fehlausrichtung wäre sofort sichtbar.
- **Härtung aus dem 3. Review (25 bestätigte Findings):** `MIN_HIT_PIXELS` skaliert mit der
  Warp-Vergrößerung; Kamera-Stall-Watchdog (Exit nach ~5s ohne Frames → systemd-Restart);
  Ecken werden bei RGB≠Depth-Auflösung umskaliert (`canvas_size`); Mixer auf 32 Kanäle +
  Retry wenn USB-Audio spät kommt; `sample_floor_depth` liefert None statt stillem Default;
  Boden-Median über das Stabilitätsfenster + Spread-Check; Headless-Kalibrierung mit Timeout;
  RGB-Probe erkennt IR/Mono-Nodes; `validate_config` prüft Korner-Geometrie (Winding/NaN);
  SIGTERM vor run() geht nicht mehr verloren; FPS-Log misst echte Fenster; DISPLAY-tot-Guard.
- **108 pytest-Tests** (vorher 69), inkl. End-to-End-Pipeline-Tests (FloorPiano mit
  injizierter Kamera/Audio) und synthetischer Matten-Kalibrierung in allen Orientierungen.

### Vor dem ersten Start auf der Hardware (manuell)
1. **Astra-Depth verifizieren (entscheidend):**
   `python3 -c "from pyorbbecsdk import Pipeline,OBSensorType; p=Pipeline(); print(len(p.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)))"`
   → 0 oder Fehler ⇒ Astra Pro wird nicht bedient ⇒ OpenNI2-Backend/anderes Gerät nötig.
2. **pyorbbecsdk aus Source bauen** (+ udev-Rules); danach `pip3 install -r requirements.txt`.
3. **Kalibrieren** (`python3 src/calibrate.py`), damit `config.json` existiert.
4. **Audio**: USB-Gerät per `aplay -l` als ALSA-Default setzen.
5. **RGB-Index** bei Bedarf per `v4l2-ctl --list-devices` ermitteln, sonst `FLOOR_PIANO_RGB_INDEX`.

---

---

## 🔴 Kritisch / Blocker

### 1. Astra Pro wird von pyorbbecsdk nicht unterstützt — Tiefenpfad scheitert auf echter Hardware
`src/main.py:13,51-61` · `src/calibrate.py:10,20-30`
Beide Module öffnen die Tiefe ausschließlich über `pyorbbecsdk` (`OBSensorType.DEPTH_SENSOR`).
Die originale Astra Pro ist ein Legacy-OpenNI/OpenNI2-Gerät (Tiefe via OpenNI, RGB als UVC)
und steht nicht in der OrbbecSDK-v2-Liste; die Enumeration liefert kein Depth-Profil, beide
Programme brechen mit `sys.exit(1)` ab. `VERSION_2_PLAN.md:36` markiert diese Migration
fälschlich als "erledigt".
**Fix:** Auf ein tatsächlich von OrbbecSDK v2 unterstütztes Gerät wechseln (Astra 2, Femto,
Gemini) ODER Tiefe per OpenNI2-Backend einlesen; vorher mit `lsusb` + Enumerationstest auf
Zielhardware verifizieren.

---

## 🟠 Hoch

### 2. RGB/Depth-Koordinatenraum-Mismatch — Tastenraster sitzt im Tiefenbild falsch
`src/calibrate.py:73-84,95-96,118` · `src/main.py:39-44,94`
*(Konsolidiert aus 4 Findings: correctness, hardware, architecture, docs.)* ArUco-Ecken werden
im RGB-Frame (separater UVC-Sensor) detektiert und als rohe RGB-Pixelkoordinaten gespeichert.
`main.py` baut daraus die Warp-Matrix und wendet sie ungescaled auf das DEPTH-Array an.
Bezeichnend: für das Floor-Sampling skaliert `calibrate.py:95-96` dieselben Koordinaten korrekt
(`dx = center_x * dw/rw`), beim Warp fehlt jede Übertragung. Da beide Sensoren verschiedenes
FOV/Origin/Auflösung haben, ist das Kernfeature (richtige Taste = richtige Note) gebrochen.
**Fix:** ArUco direkt auf dem IR-/Tiefen-registrierten Stream detektieren ODER SDK-Depth-to-Color-
Alignment (D2C) aktivieren; reine Auflösungsskalierung der Ecken ist nur ein Notbehelf.

### 3. Audio-Samples sind HTML-Fehlerseiten statt WAV — Crash beim Start
`src/sounds/download_samples.sh:9,20` · `src/audio.py:20-21`
Die `BASE_URL` (shanealder/Piano) liefert HTTP 404; `curl -L -s` ohne `-f` speichert die
HTML-Seite still als `<note>.wav`. Alle 7 eingecheckten `*.wav` sind tatsächlich GitHub-"Page
not found"-HTML (~301KB, per `file` bestätigt). `pygame.mixer.Sound()` wirft darauf eine
ungefangene Exception → `PianoAudio.__init__` und damit der Start brechen ab.
**Fix:** `BASE_URL` auf reale Quelle korrigieren, `curl -f -L` + Statusprüfung verwenden,
`Sound()`-Aufruf in try/except kapseln und nur bei Erfolg registrieren.

---

## 🟡 Mittel

### 4. Annahme uint16/Y16-Reshape des Tiefenframes ist formatabhängig
`src/main.py:90-92` · `src/calibrate.py:69-71`
`np.frombuffer(...).reshape((h,w))` setzt exakt `h*w*2` Bytes (Y16) voraus, aber
`get_default_video_stream_profile()` garantiert kein Y16. Bei abweichendem Format wirft
`reshape` einen `ValueError` im while-Loop, dessen einziges `except` nur `KeyboardInterrupt`
fängt → Crash statt `continue`.
**Fix:** Depth-Profil explizit als `OB_FORMAT_Y16` anfordern und Bytegröße/`get_format()`
vor dem reshape prüfen.

### 5. Doku-Code-Mismatch: 90 FPS / USB 3.0 / <10ms / Hailo-NPU werden als Ist-Zustand behauptet
`README.md:7,9,57,58` · `docs/SETUP.md:3,11,67-68` · `docs/SHOPPING_LIST.md:3,7,12`
Die Astra Pro ist USB 2.0 mit max. 640x480@30 FPS — 90 FPS sind physikalisch unmöglich, das
USB-3.0-/Bandbreiten-Argument ist falsch. `<10ms`/`<2ms` Latenz ist bei ~33ms Frameperiode +
pygame-Buffer nicht erreichbar (realistisch 40-80ms). Hailo wird nur importiert + als "DETECTED"
geloggt; keine Inferenz, kein Modell, kein Pose-Tracking — die NPU trägt zur Funktion nichts bei.
`VERSION_2_PLAN.md` markiert RANSAC/Hailo ehrlich als offen, README/SETUP widersprechen dem.
**Fix:** Performance-/Hardware-Angaben auf reale Werte (30 FPS, USB 2.0, 40-80ms) korrigieren
und RANSAC/Hailo/NPU konsistent als "geplant" kennzeichnen.

### 6. Falscher Pfad und inkonsistente Skript-Aufrufe in SETUP.md
`docs/SETUP.md:34,51,58`
`cd workspace/projects/floor-piano/sounds` existiert nicht (Skript liegt unter `src/sounds/`)
→ Setup-Schritt schlägt fehl. Zusätzlich `python3 calibrate.py`/`main.py` ohne `src/`-Präfix,
inkonsistent zur README.
**Fix:** Auf `cd src/sounds` korrigieren und Skriptaufrufe mit `src/`-Präfix angeben.

### 7. Headless-Anspruch wird durch erzwungene cv2-GUI unterlaufen
`src/calibrate.py:112,114` · `src/main.py:121-126`
README/SETUP versprechen monitorlosen Betrieb. `calibrate.py` speichert config.json
AUSSCHLIESSLICH bei Tastendruck `s` via `cv2.imshow`/`waitKey` (kein Auto-Save-Pfad). In
`main.py` ist `imshow` korrekt DISPLAY-gegated, `waitKey(1)` aber nicht — `q` (Quit) und `r`
(Re-Level) sind headless funktionslos, womit das beworbene "Auto-Recovery" nur manuell im GUI
existiert.
**Fix:** `waitKey` in den DISPLAY-Guard ziehen, headless einen Auto-Save-Kalibrierpfad +
SIGTERM/GPIO-Steuerung einführen ODER Doku auf "Kalibrierung erfordert einmalig Monitor"
korrigieren.

### 8. config.json wird geschrieben, aber nicht validiert (canvas_size ungenutzt, KeyError-Risiko)
`src/calibrate.py:116-123` · `src/main.py:31-48`
`json.load` ohne try/except, direkter Zugriff auf `config["corners"]`/`["keys"]` → roher
`KeyError`/`JSONDecodeError` bei beschädigter/unvollständiger Datei; `corners` ungeprüft an
`getPerspectiveTransform`. Das geschriebene Feld `canvas_size` wird nirgends gelesen — genau die
Information, die zur RGB→Depth-Skalierung nötig wäre (vgl. Finding 2).
**Fix:** `validate_config()` einführen (Pflichtfelder, `len(corners)==4`, `keys` nicht leer,
`floor_depth > threshold`) und `canvas_size` entweder nutzen oder entfernen.

### 9. Architektur trägt die Roadmap nicht: monolithischer Loop, keine Abstraktion, keine Testbarkeit
`src/main.py:25-64 (__init__)`, `74-133 (run)`
Capture, Warp, Hit-Detektion, Audio und Rendering stecken in einer while-True-Schleife;
Hardware-Init und `sys.exit(1)` liegen im Konstruktor, sodass die Klasse ohne Hardware/SDK nicht
instanziierbar/testbar ist. Phasen 2-4 (RANSAC, Hailo, Headless-Service) müssten jeweils mitten in
diese Schleife eingreifen. Es fehlt eine `DepthSource`-Abstraktion + MockSource, weshalb die Logik
ohne Kamera gar nicht entwickelbar ist. Die Roadmap-Reihenfolge priorisiert zudem AI vor den
tragenden Fundamenten (Registrierung, Headless).
**Fix:** Verantwortlichkeiten trennen (`DepthSource`/`KeyMapper`/`AudioEngine`), `MockSource` für
hardwarefreie Tests einführen, Hardware-Init aus dem Konstruktor in `start()` auslagern, Exceptions
statt `sys.exit`, und Registrierung + Headless vor RANSAC/Hailo ziehen.

### 10. Hailo-Astra-Fusion technisch unstimmig formuliert
`docs/VERSION_2_PLAN.md:31,38`
Der Plan will den 16-bit-Tiefenstream in die Hailo-Pose-Pipeline leiten ("100% Vision/AI-Offload").
Pose-Modelle (z.B. YOLOv8-pose) erwarten 3-Kanal-RGB, nicht rohe Single-Channel-Tiefe;
Warp/Decode/Fusion bleiben ohnehin CPU-Last.
**Fix:** Konzept präzisieren — Hailo betreibt RGB-Pose zur Personen-/Fußlokalisierung, Astra-Tiefe
liefert separat Z zur Trigger-Bestätigung; "100% Offload" streichen.

### 11. Downward-Mount kollidiert mit 0.6m Mindestabstand / Strukturlicht-Geometrie
`docs/SETUP.md:62` · `src/main.py:95`
Astra Pro hat ~0.6m Mindest-Tiefenbereich und ist empfindlich gegen Sonnenlicht-IR sowie
dunkle/glänzende Böden; bei 1m Montage deckt das FOV nur ~1.1m Breite für 7 Tasten + Füße ab.
Loch-/0-Pixel werden ungefüllt als Nicht-Treffer behandelt → False Negatives im kritischen
Nahbereich. In den Docs gar nicht dokumentiert.
**Fix:** Montagehöhe/FOV konkret berechnen und dokumentieren, IR-Störquellen als Risiko nennen,
morphologisches Schließen/Lochfüllung in die Trigger-Logik einbauen.

---

## 🔵 Niedrig / Info

- **ArUco-Markerzentrum statt Mattenecke** (`calibrate.py:80`): `np.mean(corners[i][0])` liefert das Marker-Zentrum → ~halbe Markerbreite Versatz nach innen. *Fix:* zur Mattenecke zeigende Marker-Ecke verwenden oder Platzierungs-Konvention dokumentieren.
- **auto_level_floor ohne Plausibilitätsfilter** (`main.py:66-72`): Median ohne Schutz gegen Fuß im Bild; durch Median + EMA praktisch robust. *Fix:* nur Pixel nahe bisherigem floor_depth / RANSAC (Phase 2).
- **Magic Numbers verstreut** (`main.py:40-41,70,72,85,101`): 700/200/150/1000/0.9/0.1 als nackte Literale; `MIN_HIT_PIXELS=150` hängt implizit an der Auflösung. *Fix:* benannte Konstanten, Pixel-Schwelle relativ zur Zonenfläche.
- **Tastenbreite via Integer-Division** (`main.py:79`): bei nicht teilbarem Raster fallen Restspalten stumm. *Fix:* `round(i*W/n)`.
- **Code-Duplikation Orbbec-Init** (`main.py:51-61` vs. `calibrate.py:20-30`). *Fix:* gemeinsame `depth_camera.py`.
- **Hartkodierte Tonleiter dupliziert** (`audio.py:18`, `calibrate.py:119`): geänderte Keys blieben still stumm. *Fix:* `keys` aus Config an `PianoAudio` übergeben.
- **Fragiles FPS-Print-Throttle** (`main.py:116-119`): `int(fps) % 30 == 0` ist keine zeitbasierte Drosselung; bei <1 FPS Spam. *Fix:* zeitbasiert + `max(dt, 1e-6)`.
- **print() statt logging** (durchgängig): ungünstig für geplanten systemd/journald-Betrieb. *Fix:* `logging`-Modul.
- **sys.exit in Klassen-/Modulcode** (`main.py:15-16,32-33,60-61`): macht Bibliothek untestbar. *Fix:* Exceptions werfen, `sys.exit` nur im `__main__`.
- **Redundante close()/cleanup()** (`audio.py:32-36`): `cleanup()` toter Code. *Fix:* eine Methode bzw. Context-Manager.
- **Keine automatisierten Tests** (Projektweit). *Fix:* `detect_hits()` extrahieren + pytest mit synthetischen numpy-Arrays.
- **Tastenraster geometrisch starr** (`main.py:79,99-102`). *Fix:* Layout als Polygon-Liste modellieren.
- **RGB-Capture-Handle-Leak bei Frühfehler** (`calibrate.py:33-43`). *Fix:* `candidate.release()` + try/finally.
- **27W-PSU-Begründung schief** (`SHOPPING_LIST.md:16`, `SETUP.md:9,69`): real ist das Pi-5-USB-Strombudget (600mA), nicht "Astra-Hochleistung". *Fix:* Begründung sachlich fassen.
- **pyorbbecsdk als nacktes pip-Paket** (`requirements.txt:4`, `SETUP.md:29`): `pip install pyorbbecsdk` ist auf arm64 nicht zuverlässig (Source-Build/Wheel + udev nötig). *Fix:* Installationsweg dokumentieren.

---

## Gesamturteil: Passt das so?

**Nein — im aktuellen Zustand passt es nicht.** Das Projekt ist als Konzept und Plan kohärent,
aber in zwei tragenden Säulen nicht lauffähig: (1) Die spezifizierte Hardware (Astra Pro) wird vom
gewählten SDK gar nicht angesprochen, und (2) selbst wenn Tiefe käme, sitzt das Tastenraster wegen
des RGB/Depth-Koordinatenraum-Fehlers an der falschen Stelle — das Kernfeature "richtige Taste =
richtige Note" funktioniert nicht. Hinzu kommt, dass die Audio-Samples defekt sind (HTML statt WAV)
und der Start crasht. Die Dokumentation verkauft das System zudem als fertiges
"Professional/Headless/90 FPS/NPU"-Produkt, obwohl Code und der ehrlichere VERSION_2_PLAN das nicht
hergeben — erhebliches Overclaiming. Für ein Studienprojekt ist die Codebasis kompakt und lesbar,
aber nicht "v2.0-fertig".

**TOP-3-Maßnahmen als nächste Schritte:**
1. **Hardware-/SDK-Frage klären (Blocker #1):** Auf der echten Kamera enumerieren, ob
   `pyorbbecsdk` ein Depth-Profil liefert. Falls nicht: entweder auf OpenNI2 umstellen oder eine
   OrbbecSDK-v2-fähige Kamera wählen — und die "erledigt"-Markierung im VERSION_2_PLAN zurückziehen.
2. **RGB/Depth-Registrierung lösen (#2):** D2C-Alignment aktivieren oder ArUco direkt im IR-/Tiefen-
   registrierten Bild detektieren. Bis das steht, sind RANSAC und Hailo nutzlos — diese Aufgabe
   gehört vor Phase 2.
3. **Audio reparieren + Doku ehrlich machen (#3, #5):** Funktionierende WAV-Samples beschaffen,
   `curl -f` + try/except absichern; parallel README/SETUP/SHOPPING_LIST auf reale Werte (30 FPS,
   USB 2.0, Median statt RANSAC, Hailo/Headless = "geplant") angleichen.

**Begleitmaßnahme vor allen Phasen:** einen `MockSource`-Pfad + erste pytest-Logiktests einführen,
damit Detektionslogik ohne Hardware entwickelbar und regressionssicher wird.
