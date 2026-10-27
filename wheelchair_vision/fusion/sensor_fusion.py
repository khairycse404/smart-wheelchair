"""
Multi-Sensor Fusion Engine: Optical AI Camera (YOLOv8 + Floor Hazards) + 360° RPLiDAR A1M8.

Performs:
  1. Azimuth-to-Pixel Projection: Projects frontal LiDAR laser points (-34° to +34°)
     onto the Camera video stream so laser reflections appear directly on obstacles.
  2. Object-Level Depth Fusion: Matches Camera bounding boxes [x1, x2] to LiDAR rays
     within the corresponding azimuth window to attach millimeter-accurate laser depth,
     while preserving Camera ground-contact depth for low floor hazards (e.g., slippers)
     that sit below the horizontal LiDAR scan plane.
  3. Corridor-Level Complementary Fusion: Combines Camera + LiDAR zone states and
     identifies which modality triggered each corridor blockage (FUSED, LIDAR, or CAM).
"""

import cv2
import numpy as np
from ..config import config


class SensorFusionEngine:
    def __init__(self):
        self.w = config.DISPLAY_WIDTH
        self.h = config.DISPLAY_HEIGHT
        self.hfov_deg = config.CAMERA_HFOV_DEG

    def pixel_to_azimuth_deg(self, x_px: float) -> float:
        """Maps horizontal image pixel x in [0, W] to LiDAR azimuth angle in [-HFOV/2, +HFOV/2]."""
        return ((float(x_px) / self.w) - 0.5) * self.hfov_deg

    def azimuth_to_pixel_x(self, azimuth_deg: float) -> int:
        """Maps frontal LiDAR azimuth angle (deg) to horizontal camera pixel x."""
        norm = (azimuth_deg / self.hfov_deg) + 0.5
        return int(max(0, min(self.w - 1, round(norm * self.w))))

    def dist_to_pixel_y(self, dist_m: float, danger_line_y: int, danger_m: float) -> int:
        """
        Maps metric LiDAR distance (m) to vertical camera pixel y such that:
          - dist_m == danger_m lands exactly on danger_line_y
          - dist_m < danger_m lands below danger_line_y (inside the red Danger Zone)
          - dist_m > danger_m lands above danger_line_y toward the horizon (y = 125)
        """
        horizon_y = 125
        bottom_y = self.h - 28
        if dist_m <= danger_m:
            ratio = (danger_m - max(0.15, dist_m)) / max(0.15, danger_m - 0.15)
            y = danger_line_y + ratio * (bottom_y - danger_line_y)
        else:
            max_far = 3.2
            ratio = min(1.0, (dist_m - danger_m) / max(0.2, max_far - danger_m))
            y = danger_line_y - (ratio ** 0.65) * (danger_line_y - horizon_y)
        return int(max(70, min(self.h - 22, round(y))))

    def fuse_detections(self, detections: list, lidar_scanner, lidar_danger_m: float) -> list:
        """
        Associates each Camera detection (YOLO object or Floor hazard) with matching
        LiDAR rays inside its horizontal azimuth cone.
        """
        if lidar_scanner is None:
            for d in detections:
                d["fusion_source"] = "CAM"
                d["azimuth_deg"] = round(self.pixel_to_azimuth_deg(d["cx"]), 1)
            return detections

        snap = lidar_scanner.get_snapshot()
        lidar_ok = snap["connected"] and len(snap["ranges_m"]) > 0

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cx = det["cx"]
            cam_dist = det["dist_m"]
            az_center = self.pixel_to_azimuth_deg(cx)
            det["azimuth_deg"] = round(az_center, 1)
            det["cam_dist_m"] = cam_dist

            if not lidar_ok:
                det["fusion_source"] = "CAM"
                det["lidar_dist_m"] = None
                continue

            # Use inner 75% of bounding box width to avoid background wall rays at box edges
            box_w = max(8, x2 - x1)
            pad = int(box_w * 0.125)
            deg_min = self.pixel_to_azimuth_deg(x1 + pad)
            deg_max = self.pixel_to_azimuth_deg(x2 - pad)
            if deg_max - deg_min < 3.0:
                deg_min = az_center - 2.0
                deg_max = az_center + 2.0

            lidar_d = lidar_scanner.get_min_distance_in_angle_window(deg_min, deg_max)

            # Determine if LiDAR ray hit this object or passed over a low floor object:
            # If the object is a low floor box/slipper (top edge y1 is low on screen) AND
            # LiDAR reads much farther away (> cam_dist + 0.65m), the laser beam shot over it!
            is_low_floor_object = (y1 > self.h * 0.52) and (det["track_id"] == -1 or (y2 - y1) < self.h * 0.28)

            if lidar_d is not None and lidar_d <= 4.0:
                if is_low_floor_object and (lidar_d > cam_dist + 0.60):
                    # Low floor hazard below the 2D LiDAR plane -> Keep Camera ground distance!
                    det["fusion_source"] = "CAM (LOW FLOOR)"
                    det["lidar_dist_m"] = round(lidar_d, 2)
                else:
                    # True 3D obstacle seen by both Camera & LiDAR -> Fuse with exact LiDAR metric depth!
                    det["fusion_source"] = "FUSED"
                    det["lidar_dist_m"] = round(lidar_d, 2)
                    det["dist_m"] = round(lidar_d, 2)
                    if lidar_d <= lidar_danger_m:
                        det["is_danger"] = True
            else:
                det["fusion_source"] = "CAM"
                det["lidar_dist_m"] = None

        return detections

    @staticmethod
    def fuse_corridors(cam_l: bool, cam_c: bool, cam_r: bool, lidar_sectors: dict, lidar_connected: bool):
        """
        Combines Camera 3-corridor states (Left, Center, Right) with LiDAR frontal
        sectors (front_left, front_center, front_right) and returns fused booleans
        plus human-readable source tags ('CLEAR', 'FUSED', 'LIDAR', 'CAM').
        """
        if lidar_connected and lidar_sectors:
            lid_l = lidar_sectors.get("front_left", {}).get("blocked", False)
            lid_c = lidar_sectors.get("front_center", {}).get("blocked", False)
            lid_r = lidar_sectors.get("front_right", {}).get("blocked", False)
        else:
            lid_l = lid_c = lid_r = False

        def _combine(c_blk: bool, l_blk: bool):
            fused_blk = c_blk or l_blk
            if c_blk and l_blk:
                src = "BLOCKED (FUSED)"
            elif l_blk:
                src = "BLOCKED (LIDAR)"
            elif c_blk:
                src = "BLOCKED (CAM)"
            else:
                src = "CLEAR"
            return fused_blk, src

        f_l, src_l = _combine(cam_l, lid_l)
        f_c, src_c = _combine(cam_c, lid_c)
        f_r, src_r = _combine(cam_r, lid_r)
        return f_l, f_c, f_r, src_l, src_c, src_r

    def draw_lidar_projection_on_camera(
        self, frame: np.ndarray, lidar_snap: dict, danger_line_y: int, danger_m: float
    ):
        """
        Projects frontal LiDAR laser scan points (-34° to +34°) directly onto the
        Camera video stream so the user can visually see the laser beam hitting obstacles.
        """
        if not lidar_snap or not lidar_snap.get("connected"):
            return

        angles = lidar_snap["angles_deg"]
        ranges = lidar_snap["ranges_m"]
        if len(ranges) == 0:
            return

        half_fov = self.hfov_deg * 0.5
        fov_mask = (angles >= -half_fov) & (angles <= half_fov) & (ranges <= 3.5)
        if not np.any(fov_mask):
            return

        f_ang = angles[fov_mask]
        f_rng = ranges[fov_mask]

        # Sort left-to-right across the camera image
        order = np.argsort(f_ang)
        f_ang = f_ang[order]
        f_rng = f_rng[order]

        pts_uv = []
        for a_deg, r_m in zip(f_ang, f_rng):
            u = self.azimuth_to_pixel_x(float(a_deg))
            v = self.dist_to_pixel_y(float(r_m), danger_line_y, danger_m)
            pts_uv.append((u, v, float(r_m)))

        # Draw subtle laser scan line connecting adjacent rays if close in depth
        for i in range(1, len(pts_uv)):
            u1, v1, r1 = pts_uv[i - 1]
            u2, v2, r2 = pts_uv[i]
            if abs(r1 - r2) < 0.25 and abs(u2 - u1) < 40:
                line_col = (0, 0, 255) if min(r1, r2) <= danger_m else (230, 210, 40)
                cv2.line(frame, (u1, v1), (u2, v2), line_col, 1, cv2.LINE_AA)

        # Draw individual projected laser points
        for u, v, r_m in pts_uv:
            if r_m <= config.LIDAR_CRITICAL_STOP_M:
                col = (20, 20, 255)      # Critical Red
                rad = 4
            elif r_m <= danger_m:
                col = (0, 80, 255)       # Danger Red-Orange
                rad = 4
            elif r_m <= danger_m * 1.4:
                col = (0, 215, 255)      # Caution Yellow
                rad = 3
            else:
                col = (255, 240, 50)     # Safe Cyan
                rad = 3
            cv2.circle(frame, (u, v), rad, col, -1, cv2.LINE_AA)
