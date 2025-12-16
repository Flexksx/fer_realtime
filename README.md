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

Config files:
- `.env` (git-ignored): optional environment overrides (loaded automatically)
- `config.json`: JSON defaults; environment variables override these

Emotion tuning:
- `config.json` supports `emotion_weights` (per-emotion scalars) to scale how much each emotion contributes to the Ollama trigger delta; `neutral` defaults to `0.0` (ignored) but students can raise it to include it.

Example (ngrok Ollama server):
```bash
OLLAMA_ENABLED=1 \
OLLAMA_URL="https://341f48ced197.ngrok-free.app" \
OLLAMA_MODEL="ollama/deepseek-r1:1.5b" \
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
- OLLAMA_ENABLED (default 0; set to 1 to enable Ollama calls)
- OLLAMA_URL (default http://localhost:11434; you can also pass a full endpoint like .../api/generate)
- OLLAMA_MODEL (default llama3.1; accepts CrewAI-style `ollama/<model>` too)
- OLLAMA_PROMPT (inline prompt text; overrides file)
- OLLAMA_PROMPT_FILE (default ollama_prompt.txt; students edit this; lines starting with # are ignored)
- OLLAMA_CHANGE_THRESHOLD (default 0.15 = 15% probability change trigger)
- (Note: trigger delta ignores `neutral` to avoid it dominating changes)
- OLLAMA_MIN_SECONDS_BETWEEN (default 5; cooldown to avoid spamming)
- OLLAMA_MIN_CONF (default = CONF_THRESH; only trigger when confident)
- OLLAMA_TIMEOUT_S (default 30)
- OLLAMA_OVERLAY (default 1; show last Ollama response on video and keep it until the next trigger succeeds)

Ethics note: expression classification is not a diagnosis; use only with consent.
