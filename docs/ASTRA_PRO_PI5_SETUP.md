# Astra Pro auf dem Raspberry Pi 5 — Setup-Runbook

> Ziel: Sobald die Kamera da ist, wird das hier ein **Plug-and-Test** statt einer
> Recherche-Odyssee. Alle Schritte einmal sauber durchgehen, dann läuft die Tiefe.

## Die entscheidende Tatsache zuerst

Die **Astra Pro** ist eine *Legacy*-Kamera. Ihr Tiefensensor wird **nur von
OpenNI2** bedient — **nicht** von `pyorbbecsdk` (OrbbecSDK v2) und **nicht** über
die UVC-Schnittstelle. Über UVC kommt ausschließlich das **RGB**-Bild; die Tiefe
läuft über OpenNI2 + den Orbbec-Treiber.

Deshalb hat dieses Projekt ein zweites Kamera-Backend (`OpenNI2DepthCamera` in
`src/depth_camera.py`), das per Umgebungsvariable aktiviert wird:

```bash
FLOOR_PIANO_CAMERA=openni2 python3 src/main.py
```

Es liefert dieselben `uint16`-Millimeter-Frames wie der Standard-Pfad — die
Erkennung/Audio-Logik bleibt unverändert.

> **Risiko, ehrlich benannt:** OpenNI2 + Astra Pro auf arm64 ist erfahrungsgemäß
> zickig. Es gibt einen ungelösten Bericht (Okt. 2025), dass die Tiefe auf
> Pi 4 / Ubuntu 22.04 / SDK **2.3.0.86** mit *"USB endpoint not found"* gar nicht
> initialisiert. Plane Pufferzeit ein und arbeite die Troubleshooting-Sektion ab.

---

## Schritt 1 — Kamera am USB-Bus prüfen

```bash
lsusb | grep -i 2bc5
```
Erwartet: ein bis zwei Einträge mit Vendor-ID `2bc5` (Orbbec) — der Tiefensensor
**und** eine separate UVC-RGB-Cam. Kein Eintrag → Kabel/Port prüfen
(USB-A-Port direkt am Pi, kein Hub), neu einstecken.

## Schritt 2 — OpenNI2 SDK + Orbbec-Treiber installieren

1. **Orbbec OpenNI SDK** für **Linux arm64** herunterladen:
   <https://www.orbbec.com/developers/openni-sdk/>
   - **Version:** Zuerst **2.3.0.63** versuchen — das ist die „last known good"-
     Version (laut OpenCV-Doku). **2.3.0.86** hat den oben genannten arm64-USB-Bug.
   - Falls kein fertiger arm64-Build verfügbar ist: OpenNI2 aus Source bauen,
     dabei den `arm64_support`-Branch verwenden
     (<https://github.com/occipital/OpenNI2>), und die Orbbec-Treiberbibliothek
     (`libORBBEC.so`) nach `OpenNI2/Drivers/` legen.
2. SDK entpacken, dann das mitgelieferte Env-Script sourcen — das setzt
   `OPENNI2_INCLUDE` und `OPENNI2_REDIST`:
   ```bash
   cd <entpackter-sdk-ordner>
   source OpenNIDevEnvironment
   ```
3. **udev-Regeln** installieren (liegt im SDK-Ordner bei), damit der Zugriff ohne
   root klappt, danach Kamera neu einstecken:
   ```bash
   sudo cp orbbec-usb.rules /etc/udev/rules.d/   # Dateiname kann variieren
   sudo udevadm control --reload && sudo udevadm trigger
   ```

> Für `systemd`-Betrieb (kein interaktives Shell-Env): `OPENNI2_REDIST` fest in
> die Service-Unit schreiben (`Environment=OPENNI2_REDIST=/opt/openni2/redist`).
> Der Backend-Code liest die Variable und übergibt sie an `openni2.initialize()`.

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
# einmalig kalibrieren (siehe Registrierungs-Hinweis unten)
FLOOR_PIANO_CAMERA=openni2 python3 src/calibrate.py
# dann live
FLOOR_PIANO_CAMERA=openni2 python3 src/main.py
```

---

## Registrierung (Blocker #2) — über Tiefe/IR, nicht RGB

Beim Astra Pro sind RGB (UVC) und Tiefe (OpenNI2) **getrennte Sensoren** mit
unterschiedlichem Blickwinkel. OpenNI2s Depth-to-Color-Alignment hilft hier
**nicht**, weil das RGB gar kein OpenNI2-Stream ist. ArUco-Ecken im RGB-Bild zu
suchen und aufs Tiefenbild anzuwenden ergibt also ein **falsch sitzendes
Tastenraster**.

**Sauberer Weg:** Die Matte direkt im **Tiefen-** oder **IR-Bild** kalibrieren.
`src/mat_calibration.py` findet die Mattenecken aus den aufgedruckten Tasten ohne
ArUco; das IR-Bild der Astra ist pixelgleich zur Tiefe. Damit fällt RGB für den
Warp komplett weg und Blocker #2 löst sich auf.
→ TODO bei der Kamera: `calibrate.py --source mat` gegen den Tiefen-/IR-Stream
prüfen.

---

## Troubleshooting

| Symptom | Ursache / Maßnahme |
|---|---|
| `openni2.initialize()` schlägt fehl | OpenNI2 findet `libOpenNI2.so`/Treiber nicht. `OPENNI2_REDIST` auf den Redist-Ordner setzen (enthält `libOpenNI2.so` + `OpenNI2/Drivers/`). |
| `USB endpoint not found` / `USB transfer timeout` beim Tiefen-Open | Bekannter arm64-Bug mit SDK 2.3.0.86. → SDK **2.3.0.63** probieren; anderes OS-Image (Pi OS Bookworm ↔ Ubuntu) testen; an die Hochschule eskalieren (Firmware/anderes Exemplar). |
| RGB geht, Tiefe nicht | Genau das obige USB-Endpoint-Thema — RGB ist UVC und immer unabhängig verfügbar. |
| `open_any()` findet kein Gerät, `lsusb` aber schon | udev-Regeln fehlen/ohne Reload → Schritt 2.3 wiederholen, Kamera neu einstecken. |
| Tiefe läuft, aber Tasten sitzen falsch | Registrierung über RGB statt Tiefe/IR — siehe Abschnitt oben (`--source mat`). |

## Wenn OpenNI2 partout nicht will

Fallback-Optionen, falls der Treiber-Stack auf dem Pi 5 nicht zu zähmen ist:
- OpenCV aus Source mit `WITH_OPENNI2=ON` bauen und `cv2.VideoCapture(cv2.CAP_OPENNI2_ASTRA)`
  nutzen (umgeht die `openni`-Bindings, braucht aber denselben OpenNI2-Treiber-Stack).
- Notfalls eine OrbbecSDK-v2-fähige Kamera (Astra 2 / Femto / Gemini) anfragen —
  dann funktioniert der Standard-`pyorbbecsdk`-Pfad ohne all das hier.

## Quellen

- OpenCV — Using Orbbec Astra 3D cameras (OpenNI2): <https://docs.opencv.org/4.x/d0/db6/tutorial_orbbec_astra_openni.html>
- Orbbec-Forum — Astra Pro init failure on ARM64 (USB endpoint): <https://3dclub.orbbec3d.com/t/astra-pro-fails-to-initialize-on-linux-arm64-with-openni2-sdk-usb-endpoint-mismatch/4544>
- Orbbec-Forum — Compatibility Astra Pro and OpenNI2: <https://3dclub.orbbec3d.com/t/compatibility-astra-pro-and-openni2/108>
- OpenNI2 arm64 build support: <https://github.com/occipital/OpenNI2/issues/63>
