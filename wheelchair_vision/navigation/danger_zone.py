"""
Floor Danger Zone and 3-Zone Robot Collision Evaluation Module.
"""

from ..config import config


def estimate_distance_meters(y2, box_h, frame_h):
    """
    Estimates rough obstacle distance in meters from camera perspective geometry:
    obstacles closer to the bottom of the screen (higher y2) or taking up more vertical
    height (higher box_h) have higher proximity.
    """
    ground_ratio = max(
        0.01,
        min(1.0, (y2 - frame_h * config.GROUND_Y_MIN_RATIO) / (frame_h * config.GROUND_Y_SPAN_RATIO)),
    )
    height_ratio = max(0.01, min(1.0, (box_h / frame_h) * 1.4))
    proximity = max(ground_ratio, height_ratio)
    approx_m = max(
        config.MIN_DISTANCE_M,
        config.MAX_PROXIMITY_DIST_M * (1.0 - proximity) + config.MIN_DISTANCE_M,
    )
    return round(approx_m, 2)


class ZoneEvaluator:
    """
    Splits the camera view into LEFT, CENTER, and RIGHT corridors, and tests
    whether any obstacle ground-contact point enters the robot danger zone.
    """

    def __init__(self, frame_w=None, frame_h=None):
        self.w = frame_w or config.DISPLAY_WIDTH
        self.h = frame_h or config.DISPLAY_HEIGHT
        self.left_boundary = self.w // 3
        self.right_boundary = 2 * (self.w // 3)

    def evaluate(self, detections, danger_line_y):
        """
        Evaluates a list of detections against the danger zone.

        Args:
            detections: List of detection tuples/dicts containing (x1, y1, x2, y2, cx, dist_m)
            danger_line_y: Integer pixel height of the horizontal danger line

        Returns:
            raw_left, raw_center, raw_right (booleans), closest_dist (float)
        """
        raw_left = False
        raw_center = False
        raw_right = False
        closest_dist = 9.9

        for det in detections:
            x1, y1, x2, y2, cx, dist_m, is_danger = (
                det["x1"],
                det["y1"],
                det["x2"],
                det["y2"],
                det["cx"],
                det["dist_m"],
                det["is_danger"],
            )
            closest_dist = min(closest_dist, dist_m)

            if is_danger:
                if x1 < self.left_boundary:
                    raw_left = True
                if x1 < self.right_boundary and x2 > self.left_boundary:
                    raw_center = True
                if x2 > self.right_boundary:
                    raw_right = True

        return raw_left, raw_center, raw_right, closest_dist
