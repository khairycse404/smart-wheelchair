import cv2
import numpy as np
import time
from collections import deque, defaultdict
from ultralytics import YOLO

# Camera Settings
CAMERA_INDEX = 2
# Capture at full 1920x1080 so the webcam uses the ENTIRE sensor (widest view, no center crop!)
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
# Downscale to 1280x720 in software for fast display & AI processing
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.40


class CommandSmoother:
    """Prevents single-frame flicker by requiring consistent zone states over N frames."""

    def __init__(self, window_size=5):
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
    """Returns (command_string, steering_angle_deg, speed_pct, color_bgr)."""
    if not blocked_center:
        if blocked_left and not blocked_right:
            return "SLIGHT RIGHT (AVOID LEFT)", +15, 70, (0, 255, 180)
        elif blocked_right and not blocked_left:
            return "SLIGHT LEFT (AVOID RIGHT)", -15, 70, (0, 255, 180)
        return "CLEAR -> MOVE FORWARD", 0, 100, (0, 255, 0)

    # Center is blocked
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


def estimate_distance_meters(y2, box_h, frame_h):
    """
    Approximate floor distance based primarily on where the bottom of the object (y2)
    touches the ground (lower in frame = closer to camera), plus box height for tall close objects.
    (Note: Exact millimeter distance will come from the LiDAR!)
    """
    ground_ratio = max(0.01, min(1.0, (y2 - frame_h * 0.25) / (frame_h * 0.70)))
    height_ratio = max(0.01, min(1.0, (box_h / frame_h) * 1.4))

    proximity = max(ground_ratio, height_ratio)
    approx_m = max(0.2, 2.5 * (1.0 - proximity) + 0.2)
    return round(approx_m, 2)


def detect_unknown_floor_obstacles(frame_gray, danger_line_y, left_b, right_b):
    """
    Classical CV helper: Checks for strong edges/textures in the bottom danger zone
    so even unknown objects (boxes/crates not in YOLO) can be noticed.
    """
    roi = frame_gray[danger_line_y:, :]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 150)

    l_edges = edges[:, :left_b]
    c_edges = edges[:, left_b:right_b]
    r_edges = edges[:, right_b:]

    l_density = (np.count_nonzero(l_edges) / max(1, l_edges.size)) * 100.0
    c_density = (np.count_nonzero(c_edges) / max(1, c_edges.size)) * 100.0
    r_density = (np.count_nonzero(r_edges) / max(1, r_edges.size)) * 100.0

    return edges, l_density, c_density, r_density


def main():
    print("Loading YOLOv8 Nano model with Multi-Object Tracking...")
    model = YOLO("yolov8n.pt")

    # Auto-detect camera: tries index 2 (Laptop external cam), then index 0 (Mini PC USB cam)
    cap = None
    for idx in (2, 0, 1):
        test_cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                print(f"Opened AUKEY camera at /dev/video{idx}!")
                cap = test_cap
                break
            test_cap.release()

    if cap is None:
        print("Error: Could not open any camera on /dev/video2, 0, or 1.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_ZOOM, 0)

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Sensor Capture: {raw_w}x{raw_h} -> Downscaled to: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

    w = DISPLAY_WIDTH
    h = DISPLAY_HEIGHT

    left_boundary = w // 3
    right_boundary = 2 * (w // 3)

    # Danger zone height ratio (0.55 means bottom 45% of screen is the Danger Zone)
    # Press [+] to expand danger zone further ahead (move line UP), [-] to shrink (move line DOWN)
    danger_zone_y_ratio = 0.58
    use_unknown_edge_helper = False
    show_edge_overlay = False

    smoother = CommandSmoother(window_size=4)
    track_history = defaultdict(lambda: deque(maxlen=20))

    print("\n=== CONTROLS INSIDE VIDEO WINDOW ===")
    print("  [q]       : Quit")
    print("  [+] / [-] : Move yellow DANGER ZONE line UP (further) / DOWN (closer)")
    print("  [u]       : Toggle Unknown-Obstacle Edge Helper (for unlabeled boxes)")
    print("  [e]       : Toggle Canny Edge visual overlay in bottom zone")
    print("====================================\n")

    prev_time = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Resize full 1080p sensor image down to 720p so we keep 100% of the wide field of view
        frame = cv2.resize(raw_frame, (w, h))

        # Yellow line position on screen
        danger_line_y = int(h * danger_zone_y_ratio)

        # 1. Run YOLO with ByteTrack (imgsz=480 boosts CPU FPS significantly!)
        results = model.track(
            frame, imgsz=480, conf=CONFIDENCE_THRESHOLD, persist=True, verbose=False
        )[0]

        raw_left = False
        raw_center = False
        raw_right = False
        closest_dist = 9.9

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                track_id = int(box.id[0]) if box.id is not None else -1

                box_h = y2 - y1
                cx = (x1 + x2) // 2

                dist_m = estimate_distance_meters(y2, box_h, h)
                closest_dist = min(closest_dist, dist_m)

                # ANY object whose yellow bottom dot (y2) touches inside the Danger Zone
                # (below the yellow line) OR takes up >35% of screen height is DANGER!
                is_danger = (y2 >= danger_line_y) or (box_h >= h * 0.35)

                if is_danger:
                    if x1 < left_boundary:
                        raw_left = True
                    if x1 < right_boundary and x2 > left_boundary:
                        raw_center = True
                    if x2 > right_boundary:
                        raw_right = True

                if track_id != -1:
                    track_history[track_id].append((cx, y2))
                    pts = list(track_history[track_id])
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], (255, 200, 0), 2)

                color = (0, 0, 255) if is_danger else (0, 255, 0)
                id_str = f"#{track_id} " if track_id != -1 else ""

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, y2), 6, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"{id_str}{label} ~{dist_m:.1f}m ({conf:.0%})",
                    (x1, max(90, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # 2. Optional Unknown-Obstacle Floor Edge Helper (only run when enabled to save FPS)
        if use_unknown_edge_helper or show_edge_overlay:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges, l_dens, c_dens, r_dens = detect_unknown_floor_obstacles(
                gray, danger_line_y, left_boundary, right_boundary
            )
            if use_unknown_edge_helper:
                if l_dens > 6.5:
                    raw_left = True
                if c_dens > 6.5:
                    raw_center = True
                if r_dens > 6.5:
                    raw_right = True

            if show_edge_overlay:
                edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edge_bgr[:, :, 0] = 0
                frame[danger_line_y:, :] = cv2.addWeighted(
                    frame[danger_line_y:, :], 0.7, edge_bgr, 0.8, 0
                )

        # 3. Smooth zone states over 4 frames to prevent flickering
        blocked_l, blocked_c, blocked_r = smoother.update(
            raw_left, raw_center, raw_right
        )

        # Fast ROI-only semi-transparent overlay (much faster than copying whole frame!)
        roi_slice = frame[danger_line_y:, :]
        overlay_roi = roi_slice.copy()
        zone_coords = [
            (0, left_boundary, blocked_l, "LEFT"),
            (left_boundary, right_boundary, blocked_c, "CENTER"),
            (right_boundary, w, blocked_r, "RIGHT"),
        ]
        for x_start, x_end, is_blk, _ in zone_coords:
            fill_color = (0, 0, 200) if is_blk else (0, 140, 0)
            cv2.rectangle(
                overlay_roi, (x_start, 0), (x_end, h - danger_line_y), fill_color, -1
            )
        cv2.addWeighted(overlay_roi, 0.22, roi_slice, 0.78, 0, roi_slice)

        # Draw Zone Dividers and Danger Horizon Line
        cv2.line(frame, (0, danger_line_y), (w, danger_line_y), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "ROBOT DANGER ZONE (Any yellow dot below this line = BLOCKED | Press +/- to move line)",
            (15, danger_line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.line(frame, (left_boundary, 80), (left_boundary, h), (255, 255, 0), 2)
        cv2.line(frame, (right_boundary, 80), (right_boundary, h), (255, 255, 0), 2)

        for x_start, x_end, is_blk, z_title in zone_coords:
            z_col = (0, 0, 255) if is_blk else (0, 255, 0)
            state_txt = "BLOCKED" if is_blk else "CLEAR"
            cv2.putText(
                frame,
                f"{z_title}: {state_txt}",
                (x_start + 30, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                z_col,
                2,
            )

        # 4. Compute Navigation Command, Steering Angle & Speed %
        cmd_text, steer_deg, speed_pct, cmd_color = get_navigation_command(
            blocked_l, blocked_c, blocked_r, closest_dist
        )

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # Top HUD Banner
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | {cmd_text}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            cmd_color,
            2,
        )
        helper_state = "ON" if use_unknown_edge_helper else "OFF"
        cv2.putText(
            frame,
            f"STEER: {steer_deg:+d} deg | SPEED: {speed_pct}% | Unknown-Box Helper [u]: {helper_state} | Edge View [e]",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Step 3 - Advanced Robot Vision & Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            danger_zone_y_ratio = max(0.30, danger_zone_y_ratio - 0.04)
        elif key == ord("-"):
            danger_zone_y_ratio = min(0.85, danger_zone_y_ratio + 0.04)
        elif key == ord("u"):
            use_unknown_edge_helper = not use_unknown_edge_helper
        elif key == ord("e"):
            show_edge_overlay = not show_edge_overlay

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
