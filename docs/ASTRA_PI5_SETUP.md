# Orbbec Astra (Original) auf dem Raspberry Pi 5 — Setup-Runbook

> Ziel: Sobald die Kamera da ist, wird das hier ein **Plug-and-Test** statt einer
> Recherche-Odyssee. Alle Schritte einmal sauber durchgehen, dann läuft die Tiefe.
>
> Gilt für die **original Orbbec Astra** (die klassische, erste Astra) — nicht die
> Astra Pro (separate UVC-Cam) und nicht die Astra 2 (OrbbecSDK v2). Bei der
> original Astra ist die Lage angenehmer als bei der Pro (siehe unten).

## Die entscheidende Tatsache zuerst

Die **original Astra** ist eine *Legacy*-Kamera: Tiefe (und Farbe) laufen über
**OpenNI2** + den Orbbec-Treiber — **nicht** über `pyorbbecsdk` (OrbbecSDK v2).
Der Standard-Pfad des Projekts (`DepthCamera`, pyorbbecsdk) bekommt also kein
Tiefenprofil. Deshalb gibt es das zweite Backend `OpenNI2DepthCamera` in
`src/depth_camera.py`, aktiviert per Umgebungsvariable:

```bash
FLOOR_PIANO_CAMERA=openni2 python3 src/main.py
```

Es liefert dieselben `uint16`-Millimeter-Frames wie der Standard-Pfad — die
Erkennung/Audio-Logik bleibt unverändert.

**Warum die original Astra der angenehme Fall ist:**
- **Bewährt auf dem Pi:** Die original Astra ist seit Jahren *die* Standard-
  Tiefenkamera in der ROS-Robotik (`astra_camera` über OpenNI2). Sehr guter
  Track-Record auf ARM/Raspberry Pi.
- **Registrierung (Blocker #2) einfacher:** Anders als bei der Pro kommt das
  **RGB ebenfalls über OpenNI2** (kein separater UVC-Sensor). Damit funktioniert
  OpenNI2s eingebaute Depth-to-Color-Registrierung — oder ihr nehmt sowieso
  Tiefe/IR direkt (siehe unten).

> Ehrlich: Der fummelige Teil bleibt, OpenNI2 + den Orbbec-Treiber auf arm64
> sauber installiert/gebaut zu bekommen. Der bekannte „USB endpoint not found"-
> Fehler aus den Foren war **Pro-spezifisch** (anderes Gerät) — bei der original
> Astra ist er nicht zu erwarten.

---

## Schritt 1 — Kamera am USB-Bus prüfen

```bash
lsusb | grep -i 2bc5
```
Erwartet: ein Eintrag mit Vendor-ID `2bc5` (Orbbec). Kein Eintrag → Kabel/Port
prüfen (USB-A-Port direkt am Pi, kein Hub), neu einstecken. Notiere die
vollständige ID `2bc5:XXXX` — die bestimmt das genaue Modell.

## Schritt 2 — OpenNI2 + Orbbec-Treiber installieren

Zwei Wege — der zweite ist auf dem Pi oft der bequemere:

**A) Orbbec OpenNI SDK direkt:**
1. SDK für **Linux arm64** laden: <https://www.orbbec.com/developers/openni-sdk/>
   - **Version:** Zuerst **2.3.0.63** versuchen (laut OpenCV-Doku „last known good").
   - Falls kein arm64-Build verfügbar: OpenNI2 aus Source bauen, dabei den
     `arm64_support`-Branch nutzen (<https://github.com/occipital/OpenNI2>), und
     die Orbbec-Treiberbibliothek nach `OpenNI2/Drivers/` legen.
2. Env-Script sourcen (setzt `OPENNI2_INCLUDE` / `OPENNI2_REDIST`):
   ```bash
   cd <entpackter-sdk-ordner> && source OpenNIDevEnvironment
   ```
3. **udev-Regeln** installieren (liegen im SDK bei), dann Kamera neu einstecken:
   ```bash
   sudo cp orbbec-usb.rules /etc/udev/rules.d/   # Dateiname kann variieren
   sudo udevadm control --reload && sudo udevadm trigger
   ```

**B) Fertiges `astra_camera` / ROS-Paket** (wenn Weg A zickt): Für die original
Astra existieren erprobte, vorgebaute OpenNI2-Stacks aus der ROS-Welt. Die
bringen libOpenNI2 + den Orbbec-Treiber + udev-Regeln für ARM gleich mit — oft
der schnellste Weg zu einem funktionierenden Treiber-Stack auf dem Pi.

> Für `systemd`-Betrieb: `OPENNI2_REDIST` fest in die Service-Unit schreiben
> (`Environment=OPENNI2_REDIST=/opt/openni2/redist`). Das Backend liest die
> Variable und übergibt sie an `openni2.initialize()`.

## Schritt 3 — Python-Bindings

```bash
pip install openni numpy        # in der venv des Projekts
```

## Schritt 4 — Probe ausführen (der eigentliche Test)

```bash
python3 tools/probe_astra.py
```
- **GREEN** → die Tiefe läuft. Weiter mit Schritt 5.
- **RED** → Troubleshooting unten. Die App ist *nicht* das Problem, der
  Treiber-Stack ist es.

## Schritt 5 — Floor-Piano mit der echten Kamera starten

```bash
FLOOR_PIANO_CAMERA=openni2 python3 src/calibrate.py   # einmalig kalibrieren
FLOOR_PIANO_CAMERA=openni2 python3 src/main.py        # live
```

---

## Registrierung (Blocker #2) — bei der original Astra entschärft

Zwei gangbare Wege, beide besser als bei der Pro:
- **OpenNI2 Depth-to-Color:** Da das RGB der original Astra ein OpenNI2-Stream
  ist, kann OpenNI2 die Tiefe hardwareseitig aufs Farbbild registrieren
  (`setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR)`). Dann passen
  RGB-detektierte ArUco-Ecken zur Tiefe.
- **Direkt über Tiefe/IR (empfohlen, am robustesten):** `src/mat_calibration.py`
  findet die Mattenecken aus den aufgedruckten Tasten ohne ArUco; das IR-Bild der
  Astra ist pixelgleich zur Tiefe. Damit braucht der Warp gar kein RGB.
  → TODO bei der Kamera: `calibrate.py --source mat` gegen den Tiefen-/IR-Stream prüfen.

## Troubleshooting

| Symptom | Ursache / Maßnahme |
|---|---|
| `openni2.initialize()` schlägt fehl | OpenNI2 findet `libOpenNI2.so`/Treiber nicht. `OPENNI2_REDIST` auf den Redist-Ordner setzen (enthält `libOpenNI2.so` + `OpenNI2/Drivers/`). |
| `open_any()` findet kein Gerät, `lsusb` aber schon | udev-Regeln fehlen/ohne Reload → Schritt 2.3 wiederholen, Kamera neu einstecken. |
| Tiefe läuft, aber Tasten sitzen falsch | Registrierung über RGB statt Tiefe/IR — siehe Abschnitt oben (`--source mat`). |
| Build/Treiber auf arm64 zickt | Weg B (vorgebauter `astra_camera`/ROS-Stack) probieren; oder SDK-Version 2.3.0.63 statt neuerer. |
| `USB endpoint not found` | War ein **Pro-spezifischer** Bug — bei der original Astra unerwartet. Falls doch: anderes OS-Image / SDK 2.3.0.63 / an die Hochschule eskalieren. |

## Wenn OpenNI2 partout nicht will

- OpenCV aus Source mit `WITH_OPENNI2=ON` bauen und `cv2.VideoCapture(cv2.CAP_OPENNI2_ASTRA)`
  nutzen (umgeht die `openni`-Bindings, braucht aber denselben Treiber-Stack).
- Notfalls eine OrbbecSDK-v2-fähige Kamera (Astra 2 / Femto / Gemini) anfragen —
  dann funktioniert der Standard-`pyorbbecsdk`-Pfad ohne all das hier.

## Quellen

- OpenCV — Using Orbbec Astra 3D cameras (OpenNI2): <https://docs.opencv.org/4.x/d0/db6/tutorial_orbbec_astra_openni.html>
- Orbbec-Forum — Astra & OpenNI2 unter ARM / Raspberry Pi: <https://3dclub.orbbec3d.com/t/astra-openni2-under-arm-platform-raspberry-pi-2-ubuntu-14-04/282>
- Orbbec-Forum — Compatibility Astra and OpenNI2: <https://3dclub.orbbec3d.com/t/compatibility-astra-pro-and-openni2/108>
- OpenNI2 arm64 build support: <https://github.com/occipital/OpenNI2/issues/63>
