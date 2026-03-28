"""
Obstacle Detection Pipeline
----------------------------
Uses YOLOv8n for object detection + bounding-box-based depth proxy
to detect obstacles in your walking path and trigger alerts.

Requirements:
    pip install ultralytics opencv-python pyttsx3 numpy

Run:
    python detector.py                  # uses webcam (index 0)
    python detector.py --source 1       # use camera index 1
    python detector.py --source video.mp4  # use a video file
"""

import cv2
import numpy as np
import pyttsx3
import threading
import time
import argparse
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

# Classes from COCO that are real walking obstacles
OBSTACLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
    13: "bench",
    14: "bird",       # low-flying obstacle
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    66: "keyboard",
    67: "phone",
    73: "book",
    77: "clock",
}

# Danger zone: center horizontal band of frame (fraction of width)
CENTER_ZONE_LEFT  = 0.25
CENTER_ZONE_RIGHT = 0.75

# Depth proxy thresholds (based on bounding box height as % of frame height)
# The taller the box, the closer the object
DANGER_THRESHOLD  = 0.40   # >40% of frame height → imminent
WARNING_THRESHOLD = 0.20   # >20% of frame height → approaching

# Minimum confidence for a detection to count
CONF_THRESHOLD = 0.45

# Minimum seconds between consecutive voice alerts (avoid spam)
ALERT_COOLDOWN = 2.5

# ──────────────────────────────────────────────
# ALERT ENGINE  (runs in a background thread)
# ──────────────────────────────────────────────

class AlertEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)   # words per minute
        self.engine.setProperty("volume", 1.0)
        self._lock = threading.Lock()
        self._last_alert_time = 0
        self._queue = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def alert(self, message: str, priority: str = "warning"):
        """Queue a voice alert if cooldown has passed."""
        now = time.time()
        with self._lock:
            if now - self._last_alert_time >= ALERT_COOLDOWN:
                self._queue.append(message)
                self._last_alert_time = now

    def _worker(self):
        while True:
            if self._queue:
                with self._lock:
                    msg = self._queue.pop(0)
                self.engine.say(msg)
                self.engine.runAndWait()
            time.sleep(0.05)


# ──────────────────────────────────────────────
# DEPTH PROXY
# ──────────────────────────────────────────────

def estimate_risk(box, frame_h, frame_w):
    """
    Returns: ('danger'|'warning'|'safe', relative_depth_score)

    Depth proxy = bounding box height / frame height
    The bigger the object in frame, the closer it is.
    """
    x1, y1, x2, y2 = box
    box_h = (y2 - y1) / frame_h
    box_cx = ((x1 + x2) / 2) / frame_w   # center x as fraction

    # Only care about objects in the center walking corridor
    in_path = CENTER_ZONE_LEFT < box_cx < CENTER_ZONE_RIGHT

    if not in_path:
        return "safe", box_h

    if box_h >= DANGER_THRESHOLD:
        return "danger", box_h
    elif box_h >= WARNING_THRESHOLD:
        return "warning", box_h
    return "safe", box_h


# ──────────────────────────────────────────────
# DRAW OVERLAY
# ──────────────────────────────────────────────

COLORS = {
    "danger":  (0, 0, 255),    # red
    "warning": (0, 165, 255),  # orange
    "safe":    (0, 255, 0),    # green
}

def draw_overlay(frame, detections):
    """Draw bounding boxes, labels, risk level, and center corridor."""
    h, w = frame.shape[:2]

    # Draw center corridor
    cx1 = int(CENTER_ZONE_LEFT  * w)
    cx2 = int(CENTER_ZONE_RIGHT * w)
    cv2.rectangle(frame, (cx1, 0), (cx2, h), (255, 255, 0), 1)
    cv2.putText(frame, "WALK PATH", (cx1 + 4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    for (label, risk, depth_score, box) in detections:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS[risk]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        tag = f"{label} | {risk.upper()} | {depth_score:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Status banner
    dangers  = [d for d in detections if d[1] == "danger"]
    warnings = [d for d in detections if d[1] == "warning"]

    if dangers:
        banner_text  = f"⚠ STOP — {dangers[0][0].upper()} DIRECTLY AHEAD"
        banner_color = (0, 0, 200)
    elif warnings:
        banner_text  = f"CAUTION — {warnings[0][0].upper()} APPROACHING"
        banner_color = (0, 100, 200)
    else:
        banner_text  = "PATH CLEAR"
        banner_color = (0, 150, 0)

    cv2.rectangle(frame, (0, h - 40), (w, h), banner_color, -1)
    cv2.putText(frame, banner_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def run(source=0):
    print("[INFO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")   # auto-downloads on first run (~6 MB)

    print("[INFO] Starting alert engine...")
    alert_engine = AlertEngine()

    print(f"[INFO] Opening camera/video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    print("[INFO] Running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── Run YOLO inference ──
        results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in OBSTACLE_CLASSES:
                continue

            label      = OBSTACLE_CLASSES[cls_id]
            xyxy       = box.xyxy[0].cpu().numpy()
            risk, depth = estimate_risk(xyxy, h, w)

            detections.append((label, risk, depth, xyxy))

        # ── Trigger alerts ──
        dangers  = [d for d in detections if d[1] == "danger"]
        warnings = [d for d in detections if d[1] == "warning"]

        if dangers:
            obj = dangers[0][0]
            alert_engine.alert(f"Warning! {obj} directly in your path. Stop!")
        elif warnings:
            obj = warnings[0][0]
            alert_engine.alert(f"Caution. {obj} ahead.")

        # ── Draw & show ──
        frame = draw_overlay(frame, detections)
        cv2.imshow("Obstacle Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time obstacle detector")
    parser.add_argument(
        "--source", default=0,
        help="Camera index (0, 1, …) or path to a video file"
    )
    args = parser.parse_args()

    # Convert to int if it's a digit string (e.g. "0" or "1")
    source = int(args.source) if str(args.source).isdigit() else args.source
    run(source)