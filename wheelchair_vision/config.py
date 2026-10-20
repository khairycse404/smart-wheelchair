"""
Centralized Configuration for the Smart Wheelchair AI Vision System.
"""

import os
from dataclasses import dataclass


@dataclass
class VisionConfig:
    # Camera capture settings (4:3 aspect ratio provides maximum uncropped vertical field of view)
    CAPTURE_WIDTH: int = 640
    CAPTURE_HEIGHT: int = 480
    CAPTURE_FPS: int = 30
    PREFER_DEVICE_INDICES: tuple = (2, 0, 1)

    # Frame dimensions used by AI processing and internal coordinate system
    DISPLAY_WIDTH: int = 640
    DISPLAY_HEIGHT: int = 480

    # Stream display dimensions & JPEG quality
    STREAM_WIDTH: int = 640
    STREAM_HEIGHT: int = 480
    JPEG_QUALITY: int = 80

    # YOLO Detector settings
    MODEL_PATH: str = "yolov8n.pt"
    YOLO_IMGSZ: int = 320
    CONFIDENCE_THRESHOLD: float = 0.35
    NUM_THREADS: int = 2

    # Obstacle avoidance & Danger zone defaults
    DEFAULT_DANGER_Y_RATIO: float = 0.58
    USE_UNKNOWN_HELPER: bool = True
    SHOW_EDGE_OVERLAY: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 5000

    # Distance estimation calibration
    GROUND_Y_MIN_RATIO: float = 0.25
    GROUND_Y_SPAN_RATIO: float = 0.70
    MAX_PROXIMITY_DIST_M: float = 2.5
    MIN_DISTANCE_M: float = 0.20


# Default global instance
config = VisionConfig()
