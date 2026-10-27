"""
2D Bird's-Eye View (BEV) 360-Degree LiDAR Radar Renderer & Streamer.
Displays:
  - 6 Directional Safety Sectors
  - 2 Concentric Safety Perimeters (Critical Stop 0.45m & Avoidance 0.85m)
  - Discrete Obstacle Clusters (#1, #2, ...) with Bounding Boxes, Distance (m), and Width (m)
  - Autonomous Best-Path Trajectory Vector (VFH Gap Finder)
  - Emergency Stop (E-Stop) visual alert
"""

import time
import math
import cv2
import numpy as np

from ..config import config


class LidarRadarStreamer:
    """
    Renders a real-time 360-degree top-down Bird's-Eye View (BEV) radar map
    with obstacle clustering, autonomous path trajectory, and multi-tier safety zones.
    """

    def __init__(self, lidar_scanner, shared_state, control_settings):
        self.lidar = lidar_scanner
        self.shared_state = shared_state
        self.control_settings = control_settings
        self.w = 600
        self.h = 480
        self.cx = self.w // 2
        self.cy = 258
        self.max_radius_px = 185

    def _azimuth_to_cv_angle(self, azimuth_deg: float) -> float:
        return azimuth_deg - 90.0

    def render_frame(self) -> np.ndarray:
        frame = np.full((self.h, self.w, 3), (18, 23, 35), dtype=np.uint8)

        with self.shared_state.lock:
            danger_m = self.control_settings.get("lidar_danger_m", config.LIDAR_DEFAULT_DANGER_M)
            radar_max_m = self.control_settings.get("lidar_radar_max_m", config.LIDAR_DEFAULT_RADAR_MAX_M)
            angle_offset = self.control_settings.get("lidar_angle_offset_deg", config.LIDAR_ANGLE_OFFSET_DEG)
            show_clusters = self.control_settings.get("show_lidar_clusters", True)
            estop_active = self.control_settings.get("estop_active", False)
            cam_detections = list(self.shared_state.detections)

        snap = self.lidar.get_snapshot()
        px_per_m = self.max_radius_px / max(1.0, radar_max_m)
        danger_radius_px = int(min(self.max_radius_px, danger_m * px_per_m))
        crit_radius_px = int(min(self.max_radius_px, config.LIDAR_CRITICAL_STOP_M * px_per_m))

        # 1. Draw shaded sector wedges
        overlay = frame.copy()
        sector_wedges = [
            ("front_left", -45.0, -15.0, True),
            ("front_center", -15.0, 15.0, True),
            ("front_right", 15.0, 45.0, True),
            ("left_flank", -135.0, -45.0, False),
            ("right_flank", 45.0, 135.0, False),
            ("rear", 135.0, 225.0, False),
        ]
        sectors_data = snap["sectors"]

        for sec_name, a_start, a_end, is_frontal in sector_wedges:
            s_info = sectors_data.get(sec_name, {"blocked": False, "min_dist": 9.9})
            cv_start = self._azimuth_to_cv_angle(a_start)
            cv_end = self._azimuth_to_cv_angle(a_end)

            if s_info["blocked"]:
                cv2.ellipse(
                    overlay,
                    (self.cx, self.cy),
                    (self.max_radius_px, self.max_radius_px),
                    0,
                    cv_start,
                    cv_end,
                    (25, 25, 180),
                    -1,
                )
            elif is_frontal:
                cv2.ellipse(
                    overlay,
                    (self.cx, self.cy),
                    (self.max_radius_px, self.max_radius_px),
                    0,
                    cv_start,
                    cv_end,
                    (25, 65, 30),
                    -1,
                )

        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # 2. Draw concentric metric distance rings
        step_m = 0.5 if radar_max_m <= 3.0 else 1.0
        r_m = step_m
        while r_m <= radar_max_m + 0.01:
            r_px = int(r_m * px_per_m)
            cv2.circle(frame, (self.cx, self.cy), r_px, (55, 68, 88), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{r_m:.1f}m",
                (self.cx + 4, self.cy - r_px + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (120, 140, 165),
                1,
                cv2.LINE_AA,
            )
            r_m += step_m

        # 3. Draw Safety Perimeters: Avoidance Ring (Orange) + Critical Stop Ring (Red)
        if danger_radius_px > 0:
            cv2.circle(
                frame,
                (self.cx, self.cy),
                danger_radius_px,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
        if crit_radius_px > 0:
            cv2.circle(
                frame,
                (self.cx, self.cy),
                crit_radius_px,
                (50, 50, 255),
                1,
                cv2.LINE_AA,
            )

        # 4. Draw sector boundary rays
        boundary_angles = [-135.0, -45.0, -15.0, 15.0, 45.0, 135.0]
        for b_deg in boundary_angles:
            rad = math.radians(b_deg)
            x_end = int(self.cx + self.max_radius_px * math.sin(rad))
            y_end = int(self.cy - self.max_radius_px * math.cos(rad))
            is_front_corridor = abs(b_deg) in (15.0, 45.0)
            line_col = (140, 180, 90) if is_front_corridor else (65, 78, 98)
            cv2.line(frame, (self.cx, self.cy), (x_end, y_end), line_col, 1, cv2.LINE_AA)

        # 5. Draw Autonomous Best-Path Trajectory Vector (Green Arrow)
        if snap["connected"] and not estop_active:
            best_hdg = snap["best_heading_deg"]
            best_clr = snap["best_path_clearance_m"]
            if best_clr > config.LIDAR_CRITICAL_STOP_M:
                path_len_m = min(radar_max_m * 0.85, max(0.8, best_clr * 0.85))
                path_px = path_len_m * px_per_m
                h_rad = math.radians(best_hdg)
                px_end = int(self.cx + path_px * math.sin(h_rad))
                py_end = int(self.cy - path_px * math.cos(h_rad))
                cv2.arrowedLine(
                    frame,
                    (self.cx, self.cy),
                    (px_end, py_end),
                    (80, 255, 120),
                    3,
                    cv2.LINE_AA,
                    tipLength=0.16,
                )
                cv2.putText(
                    frame,
                    f"PATH {best_hdg:+.0f}d",
                    (px_end - 28, max(62, py_end - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (100, 255, 140),
                    1,
                    cv2.LINE_AA,
                )

        # 6. Plot 360-degree LaserScan points
        angles_deg = snap["angles_deg"]
        ranges_m = snap["ranges_m"]

        if snap["connected"] and len(ranges_m) > 0:
            vis_mask = ranges_m <= radar_max_m
            v_ang = np.radians(angles_deg[vis_mask])
            v_rng = ranges_m[vis_mask]

            r_px = v_rng * px_per_m
            pts_x = (self.cx + r_px * np.sin(v_ang)).astype(np.int32)
            pts_y = (self.cy - r_px * np.cos(v_ang)).astype(np.int32)

            for px, py, dist in zip(pts_x, pts_y, v_rng):
                if dist <= config.LIDAR_CRITICAL_STOP_M:
                    pt_col = (20, 20, 255)    # Critical Red
                    pt_rad = 3
                elif dist <= danger_m:
                    pt_col = (40, 90, 255)    # Danger Red-Orange
                    pt_rad = 3
                elif dist <= danger_m * 1.5:
                    pt_col = (0, 200, 255)    # Caution Yellow
                    pt_rad = 2
                else:
                    pt_col = (220, 245, 60)   # Safe Cyan
                    pt_rad = 2
                cv2.circle(frame, (int(px), int(py)), pt_rad, pt_col, -1, cv2.LINE_AA)

            # 7. Draw Clustered Obstacles (#1..#4) with clean non-overlapping badges
            if show_clusters:
                placed_label_ys = []
                for obs in snap["obstacles"][:4]:
                    if obs["min_dist_m"] > radar_max_m:
                        continue
                    bx1 = int(self.cx + obs["x1_m"] * px_per_m)
                    bx2 = int(self.cx + obs["x2_m"] * px_per_m)
                    by1 = int(self.cy - obs["y2_m"] * px_per_m)
                    by2 = int(self.cy - obs["y1_m"] * px_per_m)

                    is_obs_danger = obs["min_dist_m"] <= danger_m
                    b_col = (60, 60, 255) if is_obs_danger else (255, 195, 50)
                    cv2.rectangle(
                        frame,
                        (bx1 - 3, by1 - 3),
                        (bx2 + 3, by2 + 3),
                        b_col,
                        1,
                        cv2.LINE_AA,
                    )
                    lbl = f"#{obs['id']} ({obs['min_dist_m']:.2f}m)"
                    lx = max(10, min(self.w - 95, (bx1 + bx2) // 2 - 28))
                    ly = max(64, by1 - 6)
                    for prev_y in placed_label_ys:
                        if abs(ly - prev_y) < 14:
                            ly = prev_y + 15
                    placed_label_ys.append(ly)
                    cv2.putText(
                        frame,
                        lbl,
                        (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        b_col,
                        1,
                        cv2.LINE_AA,
                    )

            # Highlight overall closest obstacle with target ring & ray
            min_d = snap["overall_min_dist"]
            min_a = snap["overall_closest_angle"]
            if min_d <= radar_max_m:
                c_rad = math.radians(min_a)
                cx_pt = int(self.cx + (min_d * px_per_m) * math.sin(c_rad))
                cy_pt = int(self.cy - (min_d * px_per_m) * math.cos(c_rad))
                t_col = (0, 0, 255) if min_d <= danger_m else (0, 255, 255)
                cv2.line(frame, (self.cx, self.cy), (cx_pt, cy_pt), t_col, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx_pt, cy_pt), 7, t_col, 2, cv2.LINE_AA)

        # 7b. Plot Camera AI / Fused Objects on the 2D Radar (even low floor hazards below LiDAR plane)
        for c_det in cam_detections[:4]:
            c_dist = c_det.get("dist_m", 9.9)
            c_az = c_det.get("azimuth_deg", 0.0)
            if c_dist <= radar_max_m:
                c_rad = math.radians(c_az)
                ox = int(self.cx + (c_dist * px_per_m) * math.sin(c_rad))
                oy = int(self.cy - (c_dist * px_per_m) * math.cos(c_rad))
                f_src = c_det.get("fusion_source", "CAM")
                m_col = (255, 110, 255) if f_src == "FUSED" else (0, 215, 255)
                cv2.drawMarker(
                    frame,
                    (ox, oy),
                    m_col,
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=11,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                tag = f"[{f_src.split()[0]}] {c_det['label']} {c_dist:.2f}m"
                cv2.putText(
                    frame,
                    tag,
                    (max(8, min(self.w - 155, ox + 8)), max(62, oy + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.37,
                    m_col,
                    1,
                    cv2.LINE_AA,
                )

        # 8. Draw Wheelchair icon at radar center
        cv2.rectangle(frame, (self.cx - 10, self.cy - 13), (self.cx + 10, self.cy + 13), (220, 230, 240), -1)
        cv2.rectangle(frame, (self.cx - 13, self.cy - 10), (self.cx - 10, self.cy + 10), (50, 180, 255), -1)
        cv2.rectangle(frame, (self.cx + 10, self.cy - 10), (self.cx + 13, self.cy + 10), (50, 180, 255), -1)
        cv2.arrowedLine(
            frame,
            (self.cx, self.cy + 6),
            (self.cx, self.cy - 22),
            (56, 189, 248),
            2,
            cv2.LINE_AA,
            tipLength=0.35,
        )

        # 9. Draw Sector Status Labels around the radar perimeter
        label_positions = [
            ("front_center", "FRONT-C", self.cx - 38, self.cy - self.max_radius_px - 12),
            ("front_left", "FRONT-L", self.cx - 145, self.cy - self.max_radius_px + 12),
            ("front_right", "FRONT-R", self.cx + 72, self.cy - self.max_radius_px + 12),
            ("left_flank", "LEFT FLANK", 14, self.cy),
            ("right_flank", "RIGHT FLANK", self.w - 115, self.cy),
            ("rear", "REAR", self.cx - 32, self.cy + self.max_radius_px + 20),
        ]
        for sec_key, title, lx, ly in label_positions:
            s_info = sectors_data.get(sec_key, {"min_dist": 9.9, "blocked": False})
            d_val = s_info["min_dist"]
            is_blk = s_info["blocked"]
            col = (70, 70, 255) if is_blk else (110, 235, 130)
            dist_str = f"{d_val:.2f}m" if d_val < 9.0 else "CLEAR"
            cv2.putText(
                frame,
                f"{title}: {dist_str}",
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                col,
                1,
                cv2.LINE_AA,
            )

        # 10. Top & Bottom HUD Bars
        cv2.rectangle(frame, (0, 0), (self.w, 48), (15, 18, 26), -1)
        if estop_active:
            cv2.rectangle(frame, (0, 0), (self.w, 48), (0, 0, 180), -1)
            cv2.putText(
                frame,
                "EMERGENCY STOP ENGAGED (PRESS E-STOP OR SPACE TO RESUME)",
                (22, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            status_col = (80, 235, 120) if snap["connected"] else (0, 165, 255)
            header_txt = (
                f"360 LIDAR: {snap['scan_hz']:.1f} Hz | Clusters: {len(snap['obstacles'])} | "
                f"Danger: {danger_m:.2f}m | Rot: {int(angle_offset)}d"
                if snap["connected"]
                else f"LIDAR STATUS: {snap['status_msg']}"
            )
            cv2.putText(
                frame,
                header_txt,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                status_col,
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame, (0, self.h - 30), (self.w, self.h), (15, 18, 26), -1)
        if snap["connected"] and snap["overall_min_dist"] < 9.0:
            near_txt = (
                f"NEAREST: {snap['overall_min_dist']:.2f}m ({snap['overall_closest_angle']:+.0f}d) | "
                f"BEST PATH: {snap['best_heading_deg']:+.0f}d ({snap['best_path_clearance_m']:.2f}m clear)"
            )
            near_col = (60, 60, 255) if snap["overall_min_dist"] <= danger_m else (220, 230, 240)
        else:
            near_txt = "NEAREST 360 OBSTACLE: NONE IN RANGE"
            near_col = (150, 165, 185)

        cv2.putText(
            frame,
            near_txt,
            (14, self.h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            near_col,
            1,
            cv2.LINE_AA,
        )

        return frame

    def generate(self):
        while True:
            t0 = time.time()
            frame = self.render_frame()
            ret, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            elapsed = time.time() - t0
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
