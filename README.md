# FERPlus Realtime (Local Webcam)

Real-time facial-expression classification using FERPlus (ONNX Runtime).

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

Controls:
- q = quit
- s = save snapshot to ./captures/

Environment variables:
- VIDEO_BACKEND (set explicitly when camera feed is black; e.g. macOS: CAP_AVFOUNDATION, Windows: CAP_DSHOW or CAP_MSMF)
- SMOOTH_WINDOW (default 10)
- CONF_THRESH (default 0.45)
- CAM_INDEX (default 0)
- TARGET_FRAME_MS (default 0)

Ethics note: expression classification is not a diagnosis; use only with consent.
