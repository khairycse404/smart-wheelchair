"""
YOLOv8 Real-Time Object Detection and Multi-Object Tracking Module.
"""

from collections import deque, defaultdict
from ultralytics import YOLO
from ..config import config


class YOLODetector:
    """
    Wraps Ultralytics YOLOv8 for persistent tracking, ground contact extraction,
    and robot collision zone evaluation.
    """

    def __init__(self, model_path=None, imgsz=None, conf=None):
        self.model_path = model_path or config.MODEL_PATH
        self.imgsz = imgsz or config.YOLO_IMGSZ
        self.conf = conf or config.CONFIDENCE_THRESHOLD
        self.model = YOLO(self.model_path)
        self.track_history = defaultdict(lambda: deque(maxlen=15))

    def track(self, frame):
        """
        Runs tracking on the frame.
        Returns:
            detections: List of dicts with:
                'xyxy': (x1, y1, x2, y2),
                'cx': center x,
                'ground_y': bottom contact y2,
                'conf': confidence float,
                'label': class name,
                'track_id': integer ID or -1,
                'trail': list of (cx, y2) tuples
            raw_boxes: List of (x1, y1, x2, y2)
        """
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            persist=True,
            verbose=False,
        )[0]

        detections = []
        raw_boxes = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                track_id = int(box.id[0]) if box.id is not None else -1

                cx = (x1 + x2) // 2
                raw_boxes.append((x1, y1, x2, y2))

                trail_pts = []
                if track_id != -1:
                    self.track_history[track_id].append((cx, y2))
                    trail_pts = list(self.track_history[track_id])

                detections.append(
                    {
                        "xyxy": (x1, y1, x2, y2),
                        "cx": cx,
                        "ground_y": y2,
                        "box_h": y2 - y1,
                        "conf": conf_val,
                        "label": label,
                        "track_id": track_id,
                        "trail": trail_pts,
                    }
                )

        return detections, raw_boxes
