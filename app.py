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

import cv2
import numpy as np
import onnxruntime as ort

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
    model_path = ensure_model(os.path.join("model", "emotion-ferplus-8.onnx"))
    face_cascade, sess, in_name, out_name = init_runtime(model_path)

    smooth_window = int(os.getenv("SMOOTH_WINDOW", "10"))
    conf_thresh = float(os.getenv("CONF_THRESH", "0.45"))
    probs_hist = deque(maxlen=smooth_window)

    cam_index = int(os.getenv("CAM_INDEX", "0"))
    cap = open_camera(cam_index)

    target_ms = float(os.getenv("TARGET_FRAME_MS", "0"))  # 0 = no sleep
    os.makedirs("captures", exist_ok=True)

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

        emotion_lines = format_emotion_lines(EMOTIONS, mean_probs)
        overlay_text(frame, [
            f"FERPlus (smoothed {len(probs_hist)}/{smooth_window})",
            f"Top: {shown}",
            f"FPS: {fps:.1f}",
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

        if target_ms > 0:
            time.sleep(target_ms / 1000.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
