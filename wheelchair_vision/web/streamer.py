"""
MJPEG Video Frame Streaming and Multi-Sensor Fusion HUD Rendering Module.
"""

import time
import cv2
from ..config import config
from ..navigation.smoother import get_navigation_command
from ..vision.floor_detector import detect_unknown_floor_boxes


class FrameStreamer:
    """
    Renders real-time HUD overlays, Fused (Camera + LiDAR) corridor states,
    projected LiDAR laser points, and yields MJPEG multipart chunks for live streaming.
    """

    def __init__(self, camera, lidar_scanner, fusion_engine, shared_state, control_settings, telemetry):
        self.camera = camera
        self.lidar_scanner = lidar_scanner
        self.fusion_engine = fusion_engine
        self.shared_state = shared_state
        self.control_settings = control_settings
        self.telemetry = telemetry
        self.w = config.DISPLAY_WIDTH
        self.h = config.DISPLAY_HEIGHT
        self.left_boundary = self.w // 3
        self.right_boundary = 2 * (self.w // 3)

    def generate(self):
        prev_time = time.time()
        video_fps = 30.0

        while True:
            loop_start = time.time()
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self.shared_state.lock:
                danger_zone_y_ratio = self.control_settings["danger_zone_y_ratio"]
                show_edge_overlay = self.control_settings["show_edge_overlay"]
                show_lidar_on_cam = self.control_settings.get("show_lidar_on_cam", config.SHOW_LIDAR_ON_CAM)
                lidar_danger_m = self.control_settings.get("lidar_danger_m", config.LIDAR_DEFAULT_DANGER_M)
                estop_active = self.control_settings.get("estop_active", False)
                detections = list(self.shared_state.detections)
                cam_blocked_l = self.shared_state.blocked_l
                cam_blocked_c = self.shared_state.blocked_c
                cam_blocked_r = self.shared_state.blocked_r
                closest_dist = self.shared_state.closest_dist
                obj_count = self.shared_state.obj_count
                ai_fps_val = self.shared_state.ai_fps
                lidar_connected = self.shared_state.lidar_connected
                lidar_sectors = dict(self.shared_state.lidar_sectors)
                lidar_best_hdg = self.shared_state.lidar_best_heading_deg
                lidar_best_clr = self.shared_state.lidar_best_clearance_m

            danger_line_y = int(self.h * danger_zone_y_ratio)

            if show_edge_overlay:
                _, mask_bgr, horizon_y = detect_unknown_floor_boxes(frame, [])
                frame[:] = cv2.addWeighted(frame, 0.72, mask_bgr, 0.55, 0)
                cv2.line(frame, (0, horizon_y), (self.w, horizon_y), (255, 0, 255), 1)

            # 1. Project frontal LiDAR laser scan rays onto Camera frame
            if show_lidar_on_cam and self.lidar_scanner is not None:
                lidar_snap = self.lidar_scanner.get_snapshot()
                self.fusion_engine.draw_lidar_projection_on_camera(
                    frame, lidar_snap, danger_line_y, lidar_danger_m
                )

            # 2. Draw Fused AI + Floor Obstacle Bounding Boxes
            for det in detections:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                cx, label = det["cx"], det["label"]
                track_id, dist_m = det["track_id"], det["dist_m"]
                is_danger = det["is_danger"]
                fusion_src = det.get("fusion_source", "CAM")
                trail_pts = det.get("trail", [])

                for i in range(1, len(trail_pts)):
                    cv2.line(frame, trail_pts[i - 1], trail_pts[i], (255, 200, 0), 2)

                if is_danger:
                    color = (0, 0, 255)        # Red when in danger
                elif fusion_src == "FUSED":
                    color = (255, 110, 255)    # Magenta/Neon when Camera + LiDAR depth fused
                elif track_id == -1:
                    color = (0, 215, 255)      # Gold/Orange for unknown floor obstacles
                else:
                    color = (0, 255, 0)        # Green for safe YOLO objects

                id_str = f"#{track_id} " if track_id != -1 else ""
                dist_str = f"{dist_m:.2f}m [FUSED]" if fusion_src == "FUSED" else f"~{dist_m:.1f}m [{fusion_src}]"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, y2), 5, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"{id_str}{label} {dist_str}",
                    (x1, max(65, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                )

            # 3. Fuse Camera + LiDAR Frontal Corridors
            fused_l, fused_c, fused_r, src_l, src_c, src_r = self.fusion_engine.fuse_corridors(
                cam_blocked_l, cam_blocked_c, cam_blocked_r, lidar_sectors, lidar_connected
            )

            # Fast ROI-only semi-transparent floor overlay using Fused Corridor States
            roi_slice = frame[danger_line_y:, :]
            overlay_roi = roi_slice.copy()
            zone_coords = [
                (0, self.left_boundary, fused_l, "LEFT", src_l),
                (self.left_boundary, self.right_boundary, fused_c, "CENTER", src_c),
                (self.right_boundary, self.w, fused_r, "RIGHT", src_r),
            ]
            for x_start, x_end, is_blk, _, _ in zone_coords:
                fill_color = (0, 0, 200) if is_blk else (0, 140, 0)
                cv2.rectangle(
                    overlay_roi,
                    (x_start, 0),
                    (x_end, self.h - danger_line_y),
                    fill_color,
                    -1,
                )
            cv2.addWeighted(overlay_roi, 0.22, roi_slice, 0.78, 0, roi_slice)

            cv2.line(frame, (0, danger_line_y), (self.w, danger_line_y), (0, 255, 255), 2)
            cv2.putText(
                frame,
                "FUSED DANGER ZONE (Camera Ground Contact + Projected LiDAR Laser Rays)",
                (12, danger_line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2,
            )
            cv2.line(frame, (self.left_boundary, 55), (self.left_boundary, self.h), (255, 255, 0), 2)
            cv2.line(frame, (self.right_boundary, 55), (self.right_boundary, self.h), (255, 255, 0), 2)

            for x_start, x_end, is_blk, z_title, z_src in zone_coords:
                z_col = (0, 0, 255) if is_blk else (0, 255, 0)
                cv2.putText(
                    frame,
                    f"{z_title}: {z_src}",
                    (x_start + 10, self.h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    z_col,
                    2,
                )

            cmd_text, steer_deg, speed_pct, cmd_color = get_navigation_command(
                cam_blocked_l,
                cam_blocked_c,
                cam_blocked_r,
                closest_dist,
                lidar_sectors=lidar_sectors if lidar_connected else None,
                lidar_best_heading_deg=lidar_best_hdg,
                lidar_best_clearance_m=lidar_best_clr,
                danger_dist_m=lidar_danger_m,
                estop_active=estop_active,
            )

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                video_fps = 0.85 * video_fps + 0.15 * (1.0 / dt)

            cv2.rectangle(frame, (0, 0), (self.w, 52), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"VIDEO: {video_fps:.0f} FPS (AI: {ai_fps_val:.0f} Hz) | {cmd_text}",
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                cmd_color,
                2,
            )

            with self.shared_state.lock:
                self.telemetry["fps"] = round(video_fps, 1)
                self.telemetry["ai_fps"] = ai_fps_val
                self.telemetry["command"] = cmd_text
                self.telemetry["steer_deg"] = steer_deg
                self.telemetry["speed_pct"] = speed_pct
                self.telemetry["closest_dist_m"] = closest_dist if closest_dist < 9.0 else None
                self.telemetry["left_zone"] = src_l
                self.telemetry["center_zone"] = src_c
                self.telemetry["right_zone"] = src_r
                self.telemetry["objects_count"] = obj_count
                self.telemetry["estop_active"] = estop_active
                self.telemetry["lidar_connected"] = self.shared_state.lidar_connected
                self.telemetry["lidar_hz"] = self.shared_state.lidar_hz
                self.telemetry["lidar_closest_dist_m"] = (
                    self.shared_state.lidar_closest_dist
                    if self.shared_state.lidar_closest_dist < 9.0
                    else None
                )
                self.telemetry["lidar_closest_angle_deg"] = self.shared_state.lidar_closest_angle
                self.telemetry["lidar_sectors"] = dict(self.shared_state.lidar_sectors)
                self.telemetry["lidar_obstacles"] = list(self.shared_state.lidar_obstacles)
                self.telemetry["lidar_best_heading_deg"] = self.shared_state.lidar_best_heading_deg
                self.telemetry["lidar_best_clearance_m"] = self.shared_state.lidar_best_clearance_m

            ret_jpg, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
            )
            if not ret_jpg:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            elapsed = time.time() - loop_start
            if elapsed < 0.034:
                time.sleep(0.034 - elapsed)
