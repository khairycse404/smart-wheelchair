"""
Command Smoothing, Multi-Tier Safety Arbitrator, and Autonomous Path Decision Generator.
"""

from collections import deque
from ..config import config


class CommandSmoother:
    """
    Temporal majority-voting filter to smooth raw boolean obstacle detections
    across successive frames, preventing decision jitter.
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


def get_navigation_command(
    cam_blocked_l: bool,
    cam_blocked_c: bool,
    cam_blocked_r: bool,
    cam_min_dist_m: float,
    lidar_sectors: dict = None,
    lidar_best_heading_deg: float = 0.0,
    lidar_best_clearance_m: float = 3.5,
    danger_dist_m: float = None,
    estop_active: bool = False,
):
    """
    Arbitrates the wheelchair's steering angle, speed percentage, and safety state
    by combining Camera corridors, 360° LiDAR sectors, VFH path planning, and E-Stop.

    Returns:
        (command_text, steer_deg, speed_pct, display_color_bgr)
    """
    # 0. Instant Manual / Hardware Emergency Stop Override
    if estop_active:
        return "EMERGENCY STOP ENGAGED (E-STOP ACTIVE)", 0, 0, (0, 0, 255)

    danger_m = danger_dist_m or config.LIDAR_DEFAULT_DANGER_M
    crit_m = config.LIDAR_CRITICAL_STOP_M
    caution_m = config.LIDAR_CAUTION_SLOW_M

    # Extract LiDAR sector states if connected
    if lidar_sectors:
        sec_fl = lidar_sectors.get("front_left", {"min_dist": 9.9, "blocked": False})
        sec_fc = lidar_sectors.get("front_center", {"min_dist": 9.9, "blocked": False})
        sec_fr = lidar_sectors.get("front_right", {"min_dist": 9.9, "blocked": False})
        sec_lf = lidar_sectors.get("left_flank", {"min_dist": 9.9, "blocked": False})
        sec_rf = lidar_sectors.get("right_flank", {"min_dist": 9.9, "blocked": False})

        lidar_l_blk = sec_fl["blocked"]
        lidar_c_blk = sec_fc["blocked"]
        lidar_r_blk = sec_fr["blocked"]
        flank_l_blk = sec_lf["blocked"]
        flank_r_blk = sec_rf["blocked"]
        lidar_front_min = min(sec_fl["min_dist"], sec_fc["min_dist"], sec_fr["min_dist"])
    else:
        lidar_l_blk = lidar_c_blk = lidar_r_blk = False
        flank_l_blk = flank_r_blk = False
        lidar_front_min = 9.9

    # Fuse Frontal Corridors (Blocked if EITHER Camera OR LiDAR detects hazard)
    blocked_left = cam_blocked_l or lidar_l_blk
    blocked_center = cam_blocked_c or lidar_c_blk
    blocked_right = cam_blocked_r or lidar_r_blk
    front_min_m = min(cam_min_dist_m, lidar_front_min)

    # 1. Tier 1 Safety: Automatic Critical Proximity Stop (< 0.45m in front)
    if front_min_m <= crit_m:
        return f"CRITICAL OBSTACLE ({front_min_m:.2f}m) -> AUTO-STOP!", 0, 0, (0, 0, 255)

    # 2. Tier 2 Safety: Center Corridor Blocked -> Reroute using Best Free Path + Flank Protection
    if blocked_center:
        can_go_left = (not blocked_left) and (not flank_l_blk)
        can_go_right = (not blocked_right) and (not flank_r_blk)

        if not can_go_left and not can_go_right:
            # Check if VFH path planner found an intermediate passable gap
            if lidar_best_clearance_m >= danger_m and abs(lidar_best_heading_deg) >= 15.0:
                steer = int(lidar_best_heading_deg)
                dir_str = "LEFT" if steer < 0 else "RIGHT"
                return f"REROUTE TO GAP -> STEER {dir_str} ({steer:+d} deg)", steer, 35, (0, 165, 255)
            return "ALL SAFE PATHS BLOCKED -> STOP!", 0, 0, (0, 0, 255)

        if can_go_left and can_go_right:
            # Choose direction favored by LiDAR VFH path planner
            if lidar_best_heading_deg <= 0:
                steer = int(min(-25.0, lidar_best_heading_deg))
                return f"AVOID OBSTACLE -> STEER LEFT ({steer:+d} deg)", steer, 45, (0, 165, 255)
            else:
                steer = int(max(25.0, lidar_best_heading_deg))
                return f"AVOID OBSTACLE -> STEER RIGHT ({steer:+d} deg)", steer, 45, (0, 165, 255)
        elif can_go_left:
            steer = int(min(-30.0, lidar_best_heading_deg)) if lidar_best_heading_deg < 0 else -35
            return f"OBSTACLE AHEAD -> TURN LEFT ({steer:+d} deg)", steer, 40, (0, 165, 255)
        else:
            steer = int(max(30.0, lidar_best_heading_deg)) if lidar_best_heading_deg > 0 else +35
            return f"OBSTACLE AHEAD -> TURN RIGHT ({steer:+d} deg)", steer, 40, (0, 165, 255)

    # 3. Center is Clear, but Left or Right Corridor is Blocked -> Gentle Nudge Away
    if blocked_left and not blocked_right and not flank_r_blk:
        return "SLIGHT RIGHT (AVOID LEFT HAZARD)", +18, 65, (0, 230, 200)
    if blocked_right and not blocked_left and not flank_l_blk:
        return "SLIGHT LEFT (AVOID RIGHT HAZARD)", -18, 65, (0, 230, 200)

    # 4. Tier 3 Safety: Caution Zone (0.85m < dist <= 1.35m) -> Proportional Slowdown
    if front_min_m <= caution_m:
        steer_hint = int(lidar_best_heading_deg * 0.35) if abs(lidar_best_heading_deg) >= 10 else 0
        return f"CAUTION ({front_min_m:.2f}m) -> SLOW FORWARD", steer_hint, 60, (0, 220, 255)

    # 5. All Clear -> Full Safe Forward Cruise
    return "CLEAR PATH -> MOVE FORWARD", 0, 100, (0, 255, 0)
