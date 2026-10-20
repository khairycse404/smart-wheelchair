"""
Command Smoothing and Steering Decision Generator.
"""

from collections import deque


class CommandSmoother:
    """
    Temporal majority-voting filter to smooth raw boolean obstacle detections
    across successive video frames, preventing decision jitter.
    """

    def __init__(self, window_size=3):
        self.left_hist = deque(maxlen=window_size)
        self.center_hist = deque(maxlen=window_size)
        self.right_hist = deque(maxlen=window_size)

    def update(self, left, center, right):
        self.left_hist.append(1 if left else 0)
        self.center_hist.append(1 if center else 0)
        self.right_hist.append(1 if right else 0)
        smooth_l = sum(self.left_hist) >= (len(self.left_hist) // 2 + 1)
        smooth_c = sum(self.center_hist) >= (len(self.center_hist) // 2 + 1)
        smooth_r = sum(self.right_hist) >= (len(self.right_hist) // 2 + 1)
        return smooth_l, smooth_c, smooth_r


def get_navigation_command(blocked_left, blocked_center, blocked_right, min_dist_m):
    """
    Arbitrates the steering command and speed based on zone blockages and obstacle proximity.

    Returns:
        (command_text, steer_deg, speed_pct, display_color_bgr)
    """
    if not blocked_center:
        if blocked_left and not blocked_right:
            return "SLIGHT RIGHT (AVOID LEFT)", +15, 70, (0, 255, 180)
        elif blocked_right and not blocked_left:
            return "SLIGHT LEFT (AVOID RIGHT)", -15, 70, (0, 255, 180)
        return "CLEAR -> MOVE FORWARD", 0, 100, (0, 255, 0)

    if min_dist_m < 0.45 and blocked_left and blocked_right:
        return "EMERGENCY STOP! (ALL BLOCKED)", 0, 0, (0, 0, 255)
    elif not blocked_left and not blocked_right:
        return "OBSTACLE AHEAD -> STEER LEFT", -35, 40, (0, 165, 255)
    elif not blocked_left:
        return "OBSTACLE AHEAD -> TURN LEFT", -45, 35, (0, 165, 255)
    elif not blocked_right:
        return "OBSTACLE AHEAD -> TURN RIGHT", +45, 35, (0, 165, 255)
    else:
        return "ALL ZONES BLOCKED -> STOP!", 0, 0, (0, 0, 255)
