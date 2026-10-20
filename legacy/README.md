# Legacy Prototypes & Development Steps

This folder preserves the step-by-step evolutionary milestones developed during the creation of the wheelchair vision system:

1. **`step1_camera_test.py`**:
   - Initial hardware verification of the USB/V4L2 camera.
   - Tests frame capture, resolution settings, and FPS diagnostics.

2. **`step2_yolo_obstacle_detector.py`**:
   - Introduction of YOLOv8 object detection on camera frames.
   - Basic bounding box visualization and object classification.

3. **`step3_advanced_robot_vision.py`**:
   - Addition of real-time object tracking (ByteTrack).
   - Multi-cue floor obstacle segmentation for non-standard floor hazards (boxes, slippers, floor objects).
   - Ground-contact line estimation and dynamic obstacle danger scoring.

4. **`step4_web_dashboard.py`**:
   - Full monolithic prototype combining camera capture, AI inference, navigation command synthesis, and MJPEG web streaming with an HTML5 dark-theme dashboard.

> **Note**: All production code has been modularized into the `wheelchair_vision/` package and is launched via `main.py` in the project root.
