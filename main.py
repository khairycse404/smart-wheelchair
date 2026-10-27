#!/usr/bin/env python3
"""
Smart Wheelchair AI Vision & 360° LiDAR Perception - Main Runner.
Graduation Project - Mechatronics Engineering.
"""

import os
import sys

# Auto-load ROS 2 Jazzy + ~/lidar_ws environment if present and not yet sourced
if (
    os.path.isdir("/opt/ros/jazzy")
    and os.environ.get("_WHEELCHAIR_ROS2_BOOTSTRAPPED") != "1"
    and "/opt/ros/jazzy" not in os.environ.get("AMENT_PREFIX_PATH", "")
):
    env = os.environ.copy()
    env["_WHEELCHAIR_ROS2_BOOTSTRAPPED"] = "1"
    py_exe = sys.executable
    args_quoted = " ".join(f'"{a}"' for a in sys.argv)
    cmd = (
        "source /opt/ros/jazzy/setup.bash && "
        "[ -f ~/lidar_ws/install/setup.bash ] && source ~/lidar_ws/install/setup.bash; "
        f'exec "{py_exe}" {args_quoted}'
    )
    os.execve("/bin/bash", ["/bin/bash", "-c", cmd], env)

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
from wheelchair_vision.lidar.ros2_scanner import ROS2LidarScanner
from wheelchair_vision.lidar.radar_renderer import LidarRadarStreamer
from wheelchair_vision.fusion.sensor_fusion import SensorFusionEngine
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

        # Subsystem 2: 360° LiDAR state
        self.lidar_connected = False
        self.lidar_hz = 0.0
        self.lidar_closest_dist = 9.9
        self.lidar_closest_angle = 0.0
        self.lidar_points_count = 0
        self.lidar_obstacles = []
        self.lidar_best_heading_deg = 0.0
        self.lidar_best_clearance_m = 3.5
        self.lidar_sectors = {
            name: {"min_dist": 9.9, "blocked": False, "closest_angle": 0.0}
            for name in ["front_left", "front_center", "front_right", "left_flank", "right_flank", "rear"]
        }


def ai_worker_loop(camera, lidar_scanner, fusion_engine, shared_state, control_settings):
    """
    Background worker running YOLO, Floor-Obstacle segmentation, and Camera+LiDAR
    object depth fusion in parallel so the live video stream never waits for inference.
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
            lidar_danger_m = control_settings.get("lidar_danger_m", config.LIDAR_DEFAULT_DANGER_M)

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

        # 3. Sensor Fusion: Associate Camera Bounding Boxes with LiDAR Laser Depth
        all_detections = fusion_engine.fuse_detections(
            all_detections, lidar_scanner, lidar_danger_m
        )

        # 4. Zone arbitration & temporal smoothing
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
    parser = argparse.ArgumentParser(description="Smart Wheelchair AI Vision & LiDAR Fusion System")
    parser.add_argument("--host", default=config.HOST, help="Host IP to bind web dashboard")
    parser.add_argument("--port", type=int, default=config.PORT, help="Port for web dashboard")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print(" SMART WHEELCHAIR: AI VISION + 360° RPLiDAR A1M8 + SENSOR FUSION")
    print(f" -> Web Dashboard: http://localhost:{args.port} (or http://{args.host}:{args.port})")
    print("=" * 72 + "\n")

    camera = ZeroLagCamera()
    shared_state = SharedState()
    control_settings = {
        "danger_zone_y_ratio": config.DEFAULT_DANGER_Y_RATIO,
        "use_unknown_edge_helper": config.USE_UNKNOWN_HELPER,
        "show_edge_overlay": config.SHOW_EDGE_OVERLAY,
        "show_lidar_on_cam": config.SHOW_LIDAR_ON_CAM,
        "lidar_danger_m": config.LIDAR_DEFAULT_DANGER_M,
        "lidar_radar_max_m": config.LIDAR_DEFAULT_RADAR_MAX_M,
        "lidar_angle_offset_deg": config.LIDAR_ANGLE_OFFSET_DEG,
        "show_lidar_clusters": True,
        "estop_active": False,
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
        "estop_active": False,
        "lidar_connected": False,
        "lidar_hz": 0.0,
        "lidar_closest_dist_m": None,
        "lidar_closest_angle_deg": 0.0,
        "lidar_sectors": {},
        "lidar_obstacles": [],
        "lidar_best_heading_deg": 0.0,
        "lidar_best_clearance_m": 3.5,
    }

    # Start Subsystem 2: ROS 2 360° RPLiDAR A1M8 Scanner & Bird's-Eye Radar
    lidar_scanner = ROS2LidarScanner(shared_state, control_settings)
    lidar_streamer = LidarRadarStreamer(lidar_scanner, shared_state, control_settings)

    # Start Subsystem 3: Multi-Sensor Fusion Engine (Camera + LiDAR)
    fusion_engine = SensorFusionEngine()

    # Start Subsystem 1: Background AI Vision + Fusion worker thread
    ai_thread = threading.Thread(
        target=ai_worker_loop,
        args=(camera, lidar_scanner, fusion_engine, shared_state, control_settings),
        daemon=True,
    )
    ai_thread.start()

    # Create and run web application
    streamer = FrameStreamer(
        camera, lidar_scanner, fusion_engine, shared_state, control_settings, telemetry
    )
    app = create_app(streamer, lidar_streamer, shared_state, control_settings, telemetry)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
