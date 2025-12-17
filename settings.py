import json
import os
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        case_sensitive=False,
        populate_by_name=True,
    )

    requested: bool = Field(default=False, alias="enabled")
    overlay: bool = Field(default=True)
    url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama3.1")
    prompt: str = Field(default="")
    prompt_file: str = Field(default="ollama_prompt.txt")
    change_threshold: float = Field(default=0.15)
    min_seconds_between: float = Field(default=5)
    min_conf: float | None = Field(default=None)
    timeout_s: float = Field(default=30)

    @field_validator("url", "model", "prompt", "prompt_file", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    smooth_window: int = Field(default=10)
    conf_thresh: float = Field(default=0.45)
    cam_index: int = Field(default=0)
    target_frame_ms: float = Field(default=0)
    video_backend: str = Field(default="")
    emotion_weights: dict[str, float] = Field(default_factory=dict)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)

    @field_validator("emotion_weights", mode="before")
    @classmethod
    def parse_emotion_weights(cls, v: Any) -> dict[str, float]:
        if not isinstance(v, dict):
            return {}
        parsed: dict[str, float] = {}
        for k, weight in v.items():
            try:
                parsed[str(k)] = float(weight)
            except (ValueError, TypeError):
                continue
        return parsed

    @field_validator("video_backend", mode="before")
    @classmethod
    def strip_video_backend(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        return ""


def load_settings(
    config_path: str = "config.json", env_path: str = ".env"
) -> AppSettings:
    settings = AppSettings()

    if config_path and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f) or {}
            settings = _merge_json_config(settings, config_data)

    return settings


def _merge_json_config(settings: AppSettings, config: dict[str, Any]) -> AppSettings:
    updates: dict[str, Any] = {}

    simple_fields = [
        "smooth_window",
        "conf_thresh",
        "cam_index",
        "target_frame_ms",
        "video_backend",
        "emotion_weights",
    ]
    for field in simple_fields:
        if field in config:
            updates[field] = config[field]

    ollama_config = config.get("ollama", {})
    if ollama_config:
        ollama_updates = _prepare_ollama_updates(settings.ollama, ollama_config)
        if ollama_updates:
            updates["ollama"] = OllamaSettings(**ollama_updates)

    if updates:
        return settings.model_copy(update=updates)

    return settings


def _prepare_ollama_updates(
    current_ollama: OllamaSettings, config: dict[str, Any]
) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    if "enabled" in config:
        updates["requested"] = config["enabled"]

    ollama_fields = [
        "overlay",
        "url",
        "model",
        "prompt",
        "prompt_file",
        "change_threshold",
        "min_seconds_between",
        "min_conf",
        "timeout_s",
    ]
    for field in ollama_fields:
        if field in config:
            updates[field] = config[field]

    return updates
