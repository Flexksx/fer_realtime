import json
import os
from dataclasses import dataclass


def _as_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _load_dotenv(path: str = ".env") -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if not key or key in os.environ:
                continue
            os.environ[key] = value


def _load_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _get(config: dict, dotted_key: str, default=None):
    cur = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _env(name: str):
    v = os.getenv(name, None)
    if v is None:
        return None
    return v.strip()


@dataclass(frozen=True)
class OllamaSettings:
    requested: bool
    overlay: bool
    url: str
    model: str
    prompt: str
    prompt_file: str
    change_threshold: float
    min_seconds_between: float
    min_conf: float | None
    timeout_s: float


@dataclass(frozen=True)
class AppSettings:
    smooth_window: int
    conf_thresh: float
    cam_index: int
    target_frame_ms: float
    video_backend: str
    emotion_weights: dict
    ollama: OllamaSettings


def load_settings(config_path: str = "config.json", env_path: str = ".env") -> AppSettings:
    _load_dotenv(env_path)
    cfg = _load_json(config_path)

    smooth_window = int(_env("SMOOTH_WINDOW") or _get(cfg, "smooth_window", 10))
    conf_thresh = float(_env("CONF_THRESH") or _get(cfg, "conf_thresh", 0.45))
    cam_index = int(_env("CAM_INDEX") or _get(cfg, "cam_index", 0))
    target_frame_ms = float(_env("TARGET_FRAME_MS") or _get(cfg, "target_frame_ms", 0))
    video_backend = _env("VIDEO_BACKEND")
    if video_backend is None:
        video_backend = str(_get(cfg, "video_backend", "") or "").strip()

    emotion_weights = _get(cfg, "emotion_weights", {}) or {}
    if not isinstance(emotion_weights, dict):
        emotion_weights = {}
    parsed_weights = {}
    for k, v in emotion_weights.items():
        try:
            parsed_weights[str(k)] = float(v)
        except Exception:
            continue

    ollama_requested = _as_bool(_env("OLLAMA_ENABLED"), _as_bool(_get(cfg, "ollama.enabled", False), False))
    ollama_overlay = _as_bool(_env("OLLAMA_OVERLAY"), _as_bool(_get(cfg, "ollama.overlay", True), True))
    ollama_url = str(_env("OLLAMA_URL") or _get(cfg, "ollama.url", "http://localhost:11434")).strip()
    ollama_model = str(_env("OLLAMA_MODEL") or _get(cfg, "ollama.model", "llama3.1")).strip()
    ollama_prompt = str(_env("OLLAMA_PROMPT") or _get(cfg, "ollama.prompt", "")).strip()
    ollama_prompt_file = str(_env("OLLAMA_PROMPT_FILE") or _get(cfg, "ollama.prompt_file", "ollama_prompt.txt")).strip()
    ollama_change_threshold = float(_env("OLLAMA_CHANGE_THRESHOLD") or _get(cfg, "ollama.change_threshold", 0.15))
    ollama_min_seconds_between = float(_env("OLLAMA_MIN_SECONDS_BETWEEN") or _get(cfg, "ollama.min_seconds_between", 5))
    ollama_timeout_s = float(_env("OLLAMA_TIMEOUT_S") or _get(cfg, "ollama.timeout_s", 30))

    min_conf_raw = _env("OLLAMA_MIN_CONF")
    if min_conf_raw is not None and min_conf_raw != "":
        ollama_min_conf = float(min_conf_raw)
    else:
        cfg_min_conf = _get(cfg, "ollama.min_conf", None)
        ollama_min_conf = float(cfg_min_conf) if cfg_min_conf is not None else None

    return AppSettings(
        smooth_window=smooth_window,
        conf_thresh=conf_thresh,
        cam_index=cam_index,
        target_frame_ms=target_frame_ms,
        video_backend=video_backend,
        emotion_weights=parsed_weights,
        ollama=OllamaSettings(
            requested=ollama_requested,
            overlay=ollama_overlay,
            url=ollama_url,
            model=ollama_model,
            prompt=ollama_prompt,
            prompt_file=ollama_prompt_file,
            change_threshold=ollama_change_threshold,
            min_seconds_between=ollama_min_seconds_between,
            min_conf=ollama_min_conf,
            timeout_s=ollama_timeout_s,
        ),
    )
