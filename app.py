#!/usr/bin/env python3
"""
Real-time webcam facial expression classification using FERPlus ONNX (emotion-ferplus-8).

- Captures frames from the local camera (OpenCV VideoCapture)
- Detects the largest face (Haar cascade)
- Crops -> grayscale -> resize 64x64 -> (1,1,64,64) float32 (0..255)
- Runs ONNX Runtime inference
- Applies softmax + temporal smoothing over the last N frames
- Overlays top label + confidence on the live video

Controls:
- Press 'q' to quit.
- Press 's' to save a snapshot (frame + crop) into ./captures/

Notes:
- This is facial-expression classification, not a measure of internal emotion.
- Use only with explicit consent.
"""

import os
import time
from collections import deque
import urllib.request
import json

import cv2
import numpy as np
import onnxruntime as ort

from settings import load_settings
from ollama_client import (
    OllamaWorker,
    normalize_ollama_model,
    normalize_ollama_url,
    read_prompt_file,
)

EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
HF_MODEL_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def ensure_model(model_path: str) -> str:
    """Download the ONNX model if missing."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:
        return model_path
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"[model] Downloading FERPlus ONNX model to: {model_path}")
    urllib.request.urlretrieve(HF_MODEL_URL, model_path)
    print(f"[model] Done. Size: {os.path.getsize(model_path)/1024/1024:.2f} MB")
    return model_path


def init_runtime(model_path: str):
    """Initialize face detector + ONNX session once."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return face_cascade, sess, in_name, out_name


def detect_largest_face(gray: np.ndarray, face_cascade, min_size=(60, 60)):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
    if len(faces) == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return max(faces, key=lambda b: b[2] * b[3])


def preprocess_ferplus(gray: np.ndarray, box):
    x, y, w, h = box
    face_gray = gray[y : y + h, x : x + w]
    face_64 = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    inp = face_64.astype(np.float32).reshape(1, 1, 64, 64)  # keep 0..255 range
    return inp, face_64


def overlay_text(frame_bgr, lines, x=10, y=22, line_h=24):
    for i, s in enumerate(lines):
        cv2.putText(frame_bgr, s, (x, y + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, s, (x, y + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def format_emotion_lines(labels, probs):
    """Return all emotions sorted by probability descending."""
    order = np.argsort(probs)[::-1]
    return [f"{labels[i]}: {probs[i]:.2f}" for i in order]


def resolve_backends():
    env_backend = os.getenv("VIDEO_BACKEND", "").strip()
    order = []
    if env_backend:
        order.append(env_backend)
    order.extend(["CAP_ANY", "CAP_AVFOUNDATION", "CAP_DSHOW", "CAP_MSMF", "CAP_V4L2"])

    seen = set()
    backends = []
    for name in order:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        if name.upper() in ("ANY", "CAP_ANY"):
            backends.append((cv2.CAP_ANY, "CAP_ANY"))
            continue

        cv_name = name.upper()
        if not cv_name.startswith("CAP_"):
            cv_name = "CAP_" + cv_name

        backend_id = getattr(cv2, cv_name, None)
        if backend_id is not None:
            backends.append((backend_id, cv_name))
    return backends


def open_camera(cam_index: int):
    attempts = []
    for backend_id, backend_name in resolve_backends():
        cap = cv2.VideoCapture(cam_index, backend_id)
        if not cap.isOpened():
            attempts.append(backend_name)
            cap.release()
            continue

        ok, frame = cap.read()
        if ok and frame is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[cam] Opened camera {cam_index} with backend {backend_name}")
            return cap

        attempts.append(backend_name)
        cap.release()

    raise RuntimeError(
        f"Cannot open camera index {cam_index}. "
        f"Tried backends: {', '.join(attempts)}. "
        "Adjust CAM_INDEX or VIDEO_BACKEND."
    )


def main():
    cfg = load_settings()
    if cfg.video_backend:
        os.environ["VIDEO_BACKEND"] = cfg.video_backend

    weights_by_emotion = getattr(cfg, "emotion_weights", {}) or {}
    weights_vec = np.array([float(weights_by_emotion.get(e, 1.0)) for e in EMOTIONS], dtype=np.float32)

    model_path = ensure_model(os.path.join("model", "emotion-ferplus-8.onnx"))
    face_cascade, sess, in_name, out_name = init_runtime(model_path)

    smooth_window = cfg.smooth_window
    conf_thresh = cfg.conf_thresh
    probs_hist = deque(maxlen=smooth_window)

    cam_index = cfg.cam_index
    cap = open_camera(cam_index)

    target_ms = cfg.target_frame_ms  # 0 = no sleep
    os.makedirs("captures", exist_ok=True)

    ollama_requested = cfg.ollama.requested
    ollama_active = False
    ollama_worker = None
    ollama_threshold = float(cfg.ollama.change_threshold)
    ollama_cooldown_s = float(cfg.ollama.min_seconds_between)
    ollama_min_conf = float(cfg.ollama.min_conf) if cfg.ollama.min_conf is not None else conf_thresh
    ollama_last_submit_ts = 0.0
    prev_mean_probs = None
    prev_top_label = ""
    prev_top_conf = 0.0

    ollama_overlay_enabled = bool(ollama_requested and cfg.ollama.overlay)
    ollama_overlay_text = ""
    ollama_overlay_ts = 0.0

    if ollama_requested:
        prompt = (cfg.ollama.prompt or "").strip()
        prompt_path = (cfg.ollama.prompt_file or "ollama_prompt.txt").strip()
        if not prompt:
            prompt = read_prompt_file(prompt_path).strip()
        if not prompt:
            msg = f"(set prompt in {prompt_path} or OLLAMA_PROMPT)"
            print(f"[ollama] enabled but no prompt found {msg}")
            ollama_overlay_text = msg
        else:
            ollama_url = normalize_ollama_url(cfg.ollama.url)
            ollama_model = normalize_ollama_model(cfg.ollama.model)
            ollama_timeout_s = float(cfg.ollama.timeout_s)
            ollama_worker = OllamaWorker(ollama_url, ollama_model, prompt, timeout_s=ollama_timeout_s)
            ollama_active = True
            print(
                f"[ollama] enabled model={ollama_model} url={ollama_url} "
                f"threshold={ollama_threshold:.2f} cooldown={ollama_cooldown_s:.1f}s"
            )

    fps_t0 = time.time()
    fps_frames = 0
    fps = 0.0

    print("[run] Press 'q' to quit. Press 's' to save snapshot.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = detect_largest_face(gray, face_cascade)

        inp, face_64 = preprocess_ferplus(gray, (x, y, w, h))
        scores = sess.run([out_name], {in_name: inp})[0]
        probs = softmax(np.squeeze(scores)).astype(np.float32)

        probs_hist.append(probs)
        mean_probs = np.mean(np.stack(probs_hist, axis=0), axis=0)
        top = int(np.argmax(mean_probs))
        top_label = EMOTIONS[top]
        top_conf = float(mean_probs[top])

        if ollama_active and ollama_worker is not None and prev_mean_probs is not None:
            abs_delta = np.abs(mean_probs - prev_mean_probs) * weights_vec
            max_abs_change = float(np.max(abs_delta))
            now = time.time()
            if (
                max_abs_change >= ollama_threshold
                and top_conf >= ollama_min_conf
                and (now - ollama_last_submit_ts) >= ollama_cooldown_s
            ):
                probs_map = {EMOTIONS[i]: float(mean_probs[i]) for i in range(len(EMOTIONS))}
                prev_map = {EMOTIONS[i]: float(prev_mean_probs[i]) for i in range(len(EMOTIONS))}
                deltas = {k: probs_map[k] - prev_map[k] for k in probs_map.keys()}
                weights_map = {EMOTIONS[i]: float(weights_vec[i]) for i in range(len(EMOTIONS))}
                changed = sorted(
                    [
                        {
                            "emotion": k,
                            "weight": weights_map[k],
                            "prev": prev_map[k],
                            "cur": probs_map[k],
                            "delta": deltas[k],
                            "weighted_delta": weights_map[k] * deltas[k],
                        }
                        for k in probs_map.keys()
                    ],
                    key=lambda d: abs(d["weighted_delta"]),
                    reverse=True,
                )

                ctx = {
                    "top_label": top_label,
                    "top_conf": f"{top_conf:.3f}",
                    "top_conf_pct": f"{top_conf * 100:.1f}",
                    "prev_top_label": prev_top_label,
                    "prev_top_conf": f"{prev_top_conf:.3f}",
                    "prev_top_conf_pct": f"{prev_top_conf * 100:.1f}",
                    "delta_max": f"{max_abs_change:.3f}",
                    "delta_max_pct": f"{max_abs_change * 100:.1f}",
                    "probs_json": json.dumps(probs_map, ensure_ascii=False),
                    "prev_probs_json": json.dumps(prev_map, ensure_ascii=False),
                    "weights_json": json.dumps(weights_map, ensure_ascii=False),
                    "changed_json": json.dumps(changed, ensure_ascii=False),
                }

                if ollama_worker.submit(ctx):
                    ollama_last_submit_ts = now

        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 80), 2)

        if top_conf < conf_thresh:
            shown = f"unknown (top={top_label}, {top_conf:.2f} < {conf_thresh:.2f})"
        else:
            shown = f"{top_label} ({top_conf:.2f})"

        fps_frames += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_frames / dt
            fps_frames = 0
            fps_t0 = time.time()

        if ollama_overlay_enabled and ollama_worker is not None:
            if ollama_worker.last_text and ollama_worker.last_ts > ollama_overlay_ts:
                ollama_overlay_text = str(ollama_worker.last_text).strip()
                ollama_overlay_ts = float(ollama_worker.last_ts)
            elif ollama_worker.last_error and ollama_worker.last_ts > ollama_overlay_ts and not ollama_overlay_text:
                ollama_overlay_text = f"(ollama error: {ollama_worker.last_error})"
                ollama_overlay_ts = float(ollama_worker.last_ts)

        ollama_line = None
        if ollama_overlay_enabled:
            if ollama_overlay_text:
                msg = ollama_overlay_text
            elif not ollama_active:
                msg = "(ollama disabled)"
            else:
                msg = "(waiting for first trigger...)"
            if len(msg) > 100:
                msg = msg[:97].rstrip() + "..."
            ollama_line = f"Ollama: {msg}"

        emotion_lines = format_emotion_lines(EMOTIONS, mean_probs)
        overlay_text(frame, [
            f"FERPlus (smoothed {len(probs_hist)}/{smooth_window})",
            f"Top: {shown}",
            f"FPS: {fps:.1f}",
            *( [ollama_line] if ollama_line else [] ),
            "Keys: q=quit, s=save",
            "All emotions:",
            *emotion_lines,
        ])

        cv2.imshow("FERPlus Realtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(os.path.join("captures", f"{ts}_frame.jpg"), frame)
            cv2.imwrite(os.path.join("captures", f"{ts}_face64.jpg"), face_64)
            print(f"[save] captures/{ts}_frame.jpg and captures/{ts}_face64.jpg")

        prev_mean_probs = mean_probs.copy()
        prev_top_label = top_label
        prev_top_conf = top_conf

        if target_ms > 0:
            time.sleep(target_ms / 1000.0)

    cap.release()
    cv2.destroyAllWindows()
    if ollama_worker is not None:
        ollama_worker.close()


if __name__ == "__main__":
    main()
