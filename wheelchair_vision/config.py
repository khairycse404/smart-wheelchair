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

    # LiDAR (RPLiDAR A1M8 / ROS 2 /scan) & Sensor Fusion settings
    LIDAR_ENABLED: bool = True
    LIDAR_TOPIC: str = "/scan"
    LIDAR_SERIAL_PORT: str = "/dev/ttyUSB0"
    LIDAR_MIN_RANGE_M: float = 0.15         # Self-masking radius (ignore wheelchair frame)
    LIDAR_MAX_RANGE_M: float = 6.00         # Max valid processing range in meters
    LIDAR_CRITICAL_STOP_M: float = 0.45     # Tier 1: Immediate automatic Emergency Stop ring (m)
    LIDAR_DEFAULT_DANGER_M: float = 0.85    # Tier 2: Obstacle Avoidance / Rerouting ring (m)
    LIDAR_CAUTION_SLOW_M: float = 1.35      # Tier 3: Proportional Speed Slowdown ring (m)
    LIDAR_DEFAULT_RADAR_MAX_M: float = 3.5  # Default radar view radius in meters
    LIDAR_ANGLE_OFFSET_DEG: float = 180.0   # Default orientation offset (180 deg matches desk/chair setup)
    WHEELCHAIR_WIDTH_M: float = 0.68        # Physical width of electric wheelchair for path planning
    CAMERA_HFOV_DEG: float = 68.0           # Camera horizontal field of view (deg) for LiDAR fusion
    SHOW_LIDAR_ON_CAM: bool = True          # Project frontal LiDAR laser points onto camera video

    # Subsystem 4: Arduino Serial Bridge, DAC & GPS Module defaults
    ARDUINO_SERIAL_PORT: str = "AUTO"       # Auto-detects /dev/ttyACM* or /dev/ttyUSB1+
    ARDUINO_BAUD_RATE: int = 115200
    DEFAULT_PIN_SPEED: int = 9              # Y-Axis Throttle PWM / Aux output pin
    DEFAULT_PIN_STEER: int = 10             # X-Axis Steer PWM / Aux output pin
    DEFAULT_PIN_RELAY: int = 7              # Joystick Spoof Takeover Relay pin
    DEFAULT_PIN_ESTOP: int = 2              # Hardware E-Stop input pin
    DEFAULT_PIN_GPS_RX: int = 4             # Arduino RX (connects to GPS TX)
    DEFAULT_PIN_GPS_TX: int = 3             # Arduino TX (connects to GPS RX)
    DAC_I2C_SPEED_ADDR: str = "0x60"        # MCP4725 #1 (Throttle / Y-Axis)
    DAC_I2C_STEER_ADDR: str = "0x61"        # MCP4725 #2 (Steer / X-Axis)
    DEFAULT_MANUAL_SPEED_PCT: int = 45      # Default manual control speed (%)


# Default global instance
config = VisionConfig()

