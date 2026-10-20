#!/usr/bin/env python3
"""
Smart Wheelchair AI Vision & Obstacle Avoidance - Main Runner.
Graduation Project - Mechatronics Engineering.
"""

import os
import sys

# Auto-resolve ~/packages on Mini PC if not inside a venv
_pkg_dir = os.path.expanduser("~/packages")
if os.path.isdir(_pkg_dir) and _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Silence C++ NNPACK stderr spam at OS file-descriptor level so terminal doesn't lag
try:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
except Exception:
    pass

import time
import argparse
import threading
import torch

from wheelchair_vision.config import config
from wheelchair_vision.vision.camera import ZeroLagCamera
from wheelchair_vision.vision.yolo_detector import YOLODetector
from wheelchair_vision.vision.floor_detector import detect_unknown_floor_boxes
from wheelchair_vision.navigation.danger_zone import ZoneEvaluator, estimate_distance_meters
from wheelchair_vision.navigation.smoother import CommandSmoother
from wheelchair_vision.web.streamer import FrameStreamer
from wheelchair_vision.web.app import create_app

torch.set_num_threads(config.NUM_THREADS)


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
        self.blocked_l = False
        self.blocked_c = False
        self.blocked_r = False
        self.closest_dist = 9.9
        self.obj_count = 0
        self.ai_fps = 0.0


def ai_worker_loop(camera, shared_state, control_settings):
    """
    Background worker running YOLO and Floor-Obstacle segmentation in parallel
    so the live video stream never waits for inference.
    """
    yolo = YOLODetector()
    zone_evaluator = ZoneEvaluator()
    smoother = CommandSmoother(window_size=3)
    prev_ai_t = time.time()
    ai_fps_smooth = 10.0

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        with shared_state.lock:
            danger_zone_y_ratio = control_settings["danger_zone_y_ratio"]
            use_unknown = control_settings["use_unknown_edge_helper"]

        danger_line_y = int(config.DISPLAY_HEIGHT * danger_zone_y_ratio)

        # 1. YOLO Object Tracking
        yolo_dets, yolo_boxes = yolo.track(frame)
        all_detections = []

        for yd in yolo_dets:
            x1, y1, x2, y2 = yd["xyxy"]
            dist_m = estimate_distance_meters(y2, yd["box_h"], config.DISPLAY_HEIGHT)
            is_danger = (y2 >= danger_line_y) or (yd["box_h"] >= config.DISPLAY_HEIGHT * 0.35)
            all_detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": yd["cx"],
                    "conf": yd["conf"],
                    "label": yd["label"],
                    "track_id": yd["track_id"],
                    "dist_m": dist_m,
                    "is_danger": is_danger,
                    "trail": yd["trail"],
                }
            )

        # 2. Multi-Cue Floor Obstacle Detection (Cardboard boxes, footwear, etc.)
        if use_unknown:
            unk_boxes, _, _ = detect_unknown_floor_boxes(frame, yolo_boxes)
            for x1, y1, x2, y2, cue_label in unk_boxes:
                box_h = y2 - y1
                cx = (x1 + x2) // 2
                dist_m = estimate_distance_meters(y2, box_h, config.DISPLAY_HEIGHT)
                is_danger = (y2 >= danger_line_y) or (box_h >= config.DISPLAY_HEIGHT * 0.35)
                all_detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "conf": 0.88,
                        "label": cue_label,
                        "track_id": -1,
                        "dist_m": dist_m,
                        "is_danger": is_danger,
                        "trail": [],
                    }
                )

        # 3. Zone arbitration & temporal smoothing
        raw_l, raw_c, raw_r, closest_dist = zone_evaluator.evaluate(all_detections, danger_line_y)
        smooth_l, smooth_c, smooth_r = smoother.update(raw_l, raw_c, raw_r)

        # Calculate smoothed AI loop rate
        now = time.time()
        dt = now - prev_ai_t
        prev_ai_t = now
        if dt > 0:
            ai_fps_smooth = 0.8 * ai_fps_smooth + 0.2 * (1.0 / dt)

        with shared_state.lock:
            shared_state.detections = all_detections
            shared_state.blocked_l = smooth_l
            shared_state.blocked_c = smooth_c
            shared_state.blocked_r = smooth_r
            shared_state.closest_dist = closest_dist
            shared_state.obj_count = len(all_detections)
            shared_state.ai_fps = round(ai_fps_smooth, 1)

        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description="Smart Wheelchair AI Vision System")
    parser.add_argument("--host", default=config.HOST, help="Host IP to bind web dashboard")
    parser.add_argument("--port", type=int, default=config.PORT, help="Port for web dashboard")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" SMART WHEELCHAIR AI VISION & OBSTACLE AVOIDANCE SYSTEM")
    print(f" -> Web Dashboard: http://{args.host}:{args.port}")
    print("=" * 65 + "\n")

    camera = ZeroLagCamera()
    shared_state = SharedState()
    control_settings = {
        "danger_zone_y_ratio": config.DEFAULT_DANGER_Y_RATIO,
        "use_unknown_edge_helper": config.USE_UNKNOWN_HELPER,
        "show_edge_overlay": config.SHOW_EDGE_OVERLAY,
    }
    telemetry = {
        "fps": 0.0,
        "ai_fps": 0.0,
        "command": "INITIALIZING...",
        "steer_deg": 0,
        "speed_pct": 0,
        "closest_dist_m": None,
        "left_zone": "CLEAR",
        "center_zone": "CLEAR",
        "right_zone": "CLEAR",
        "objects_count": 0,
    }

    # Start background AI worker thread
    ai_thread = threading.Thread(
        target=ai_worker_loop,
        args=(camera, shared_state, control_settings),
        daemon=True,
    )
    ai_thread.start()

    # Create and run web application
    streamer = FrameStreamer(camera, shared_state, control_settings, telemetry)
    app = create_app(streamer, shared_state, control_settings, telemetry)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
