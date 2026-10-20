import os
import sys

# Automatically find packages in ~/packages on the Mini PC without needing PYTHONPATH=~/packages
_pkg_dir = os.path.expanduser("~/packages")
if os.path.isdir(_pkg_dir) and _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import cv2
import numpy as np
import time
import socket
import threading
import torch
from collections import deque, defaultdict
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO

# Silence C++ NNPACK stderr spam at OS file-descriptor level so SSH terminal doesn't lag
devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, 2)

torch.set_num_threads(2)

# Camera & Display Settings (640x480 native 4:3 gives widest field of view and 30 FPS on Mini PC)
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
YOLO_IMGSZ = 320
CONFIDENCE_THRESHOLD = 0.35

app = Flask(__name__)

state_lock = threading.Lock()
control_settings = {
    "danger_zone_y_ratio": 0.58,      # Default yellow line right below floor horizon
    "use_unknown_edge_helper": True,  # ON by default so unknown boxes get detected automatically!
    "show_edge_overlay": False,
}

# Shared AI results updated by the background AI thread (NEVER blocks the video stream!)
ai_state = {
    "detections": [],
    "blocked_l": False,
    "blocked_c": False,
    "blocked_r": False,
    "closest_dist": 9.9,
    "obj_count": 0,
    "ai_fps": 0.0,
}

latest_telemetry = {
    "fps": 0.0,
    "ai_fps": 0.0,
    "command": "INITIALIZING...",
    "steer_deg": 0,
    "speed_pct": 0,
    "closest_dist_m": None,
    "left_zone": "CLEAR",
    "center_zone": "CLEAR",
    "right_zone": "CLEAR",
    "objects_count": 0,
}


class ZeroLagCamera:
    """Continuously grabs the newest frame from the USB webcam in a background thread."""

    def __init__(self):
        self.cap = self._open_best_camera()
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_ZOOM, 0)
        self.cap.set(cv2.CAP_PROP_SHARPNESS, 180)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _open_best_camera(self):
        for idx in (2, 0, 1):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Successfully opened camera at /dev/video{idx}!")
                    return cap
                cap.release()
        raise RuntimeError("Could not open any USB camera on /dev/video2, 0, or 1!")

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                with self.lock:
                    self.latest_frame = resized
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()


class CommandSmoother:
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


def estimate_distance_meters(y2, box_h, frame_h):
    ground_ratio = max(0.01, min(1.0, (y2 - frame_h * 0.25) / (frame_h * 0.70)))
    height_ratio = max(0.01, min(1.0, (box_h / frame_h) * 1.4))
    proximity = max(ground_ratio, height_ratio)
    approx_m = max(0.2, 2.5 * (1.0 - proximity) + 0.2)
    return round(approx_m, 2)


def boxes_overlap(boxA, boxB):
    """Returns True if two (x1, y1, x2, y2) boxes overlap significantly."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return False
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return (inter_area / float(min(areaA, areaB))) > 0.25


def detect_unknown_floor_boxes(frame, existing_boxes):
    """
    3-Cue Floor-Horizon Obstacle Detector:
    Separates Warm/Yellow objects, Cool/Blue boxes, and Dark floor objects into independent
    channels so adjacent floor obstacles never merge with each other or with the background wall/door.
    Returns a list of (x1, y1, x2, y2, label_str) bounding boxes, the BGR debug mask, and horizon_y.
    """
    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 1. Sample lower carpet reference region (y: 0.68h .. 0.90h, x: 0.15w .. 0.85w)
    ref_y1, ref_y2 = int(h * 0.68), int(h * 0.90)
    ref_x1, ref_x2 = int(w * 0.15), int(w * 0.85)
    ref_mask = np.ones((ref_y2 - ref_y1, ref_x2 - ref_x1), dtype=bool)
    for (bx1, by1, bx2, by2) in existing_boxes:
        ox1, oy1 = max(0, bx1 - ref_x1), max(0, by1 - ref_y1)
        ox2, oy2 = min(ref_x2 - ref_x1, bx2 - ref_x1), min(ref_y2 - ref_y1, by2 - ref_y1)
        if ox2 > ox1 and oy2 > oy1:
            ref_mask[oy1:oy2, ox1:ox2] = False

    ref_lab_pixels = lab[ref_y1:ref_y2, ref_x1:ref_x2][ref_mask]
    ref_hsv_pixels = hsv[ref_y1:ref_y2, ref_x1:ref_x2][ref_mask]
    if len(ref_lab_pixels) < 200:
        ref_lab_pixels = lab[ref_y1:ref_y2, ref_x1:ref_x2].reshape(-1, 3)
        ref_hsv_pixels = hsv[ref_y1:ref_y2, ref_x1:ref_x2].reshape(-1, 3)

    ref_l, ref_a, ref_b = np.median(ref_lab_pixels, axis=0)
    ref_s = np.median(ref_hsv_pixels[:, 1])

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]
    S = hsv[:, :, 1]

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    local_grad = cv2.blur(grad, (25, 25))

    # 2. Estimate Floor Horizon Y automatically:
    is_carpet = (np.abs(L - ref_l) < 35) & (np.hypot(A - ref_a, B - ref_b) < 18)
    horizon_y = int(h * 0.46)
    for y in range(int(h * 0.78), int(h * 0.34), -2):
        row_carpet_ratio = np.mean(is_carpet[y, int(w * 0.15):int(w * 0.85)])
        if row_carpet_ratio < 0.25:
            horizon_y = y
            break
    horizon_y = int(np.clip(horizon_y, int(h * 0.38), int(h * 0.54)))

    # 3. Three Separate Cue Masks so adjacent different-colored objects NEVER merge:
    # Cue A: Warm / Saturated Objects (Yellow Box, Orange/Red/Green objects)
    cue_warm = (
        ((B - ref_b > 15.0) & (L > 120) & (S > ref_s + 26)) |
        ((np.abs(A - ref_a) > 16.0) & (L > 100) & (S > ref_s + 32)) |
        ((S > ref_s + 60) & (L > 115))
    ).astype(np.uint8) * 255
    cue_warm[:max(55, horizon_y - 40), :] = 0
    cue_warm[int(h * 0.94):, :] = 0
    cue_warm[:, :8] = 0
    cue_warm[:, w - 8:] = 0

    # Cue B: Cool / Blue / Packaged Box Objects (Blue Box, Bottles, Cartons)
    cue_cool = (
        ((ref_b - B > 13.5) & (L > 105) & (S > 38)) |
        ((ref_b - B > 8.5) & (np.abs(gx) > 32) & (L > 140) & (S > 28))
    ).astype(np.uint8) * 255
    cue_cool[:int(h * 0.28), :] = 0
    cue_cool[int(h * 0.94):, :] = 0
    cue_cool[:, :8] = 0
    cue_cool[:, w - 8:] = 0

    # Cue C: Dark / Footwear / Textured Floor Obstacles (Black Slipper, Shoes, Bags)
    # Using local_grad > 24 eliminates flat dark background walls/beds (local_grad ~ 13)
    # while keeping textured dark objects (like the slipper + logo) and their attached sole!
    dark_core = ((ref_l - L > 50) & (L < 135) & (local_grad > 24)).astype(np.uint8) * 255
    dark_core[:max(55, horizon_y - 60), :] = 0
    dark_core[int(h * 0.94):, :] = 0
    k_sole = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 13))
    near_dark = cv2.dilate(dark_core, k_sole, iterations=1) > 0
    sole_ext = (near_dark & (np.abs(gy) > 35) & (L < 210)).astype(np.uint8) * 255
    cue_dark = np.maximum(dark_core, sole_ext)
    cue_dark[:max(55, horizon_y - 60), :] = 0
    cue_dark[int(h * 0.94):, :] = 0
    cue_dark[:, :8] = 0
    cue_dark[:, w - 8:] = 0

    # Mask out existing YOLO detections in all cues
    for (bx1, by1, bx2, by2) in existing_boxes:
        pad = 10
        x1, y1 = max(0, bx1 - pad), max(0, by1 - pad)
        x2, y2 = min(w, bx2 + pad), min(h, by2 + pad)
        cue_warm[y1:y2, x1:x2] = 0
        cue_cool[y1:y2, x1:x2] = 0
        cue_dark[y1:y2, x1:x2] = 0

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_close_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_close_med = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 11))

    unknown_boxes = []
    debug_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Process Warm and Cool first, then mask their boxes out of cue_dark so they never bridge!
    for cue_label, raw_mask, color_ch, k_cl in [
        ("yellow/warm box", cue_warm, 2, k_close_small),
        ("blue/cool box", cue_cool, 0, k_close_med),
        ("dark floor obj", cue_dark, 1, k_close_med),
    ]:
        if cue_label == "dark floor obj":
            for (ex1, ey1, ex2, ey2, _) in unknown_boxes:
                pad = 10
                raw_mask[max(0, ey1 - pad):min(h, ey2 + pad), max(0, ex1 - pad):min(w, ex2 + pad)] = 0

        m = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_cl)
        debug_mask[:, :, color_ch] = np.maximum(debug_mask[:, :, color_ch], m)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cue_cands = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 450:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            x2, y2 = rx + rw, ry + rh

            if rw < 28 or rh < 24:
                continue
            if rw > w * 0.46 or rh > h * 0.55:
                continue
            aspect = rw / float(max(1, rh))
            if aspect > 4.5 or aspect < 0.25:
                continue

            # Must reach down to the floor/horizon region
            if y2 < max(horizon_y - 12, int(h * 0.40)) or y2 > h - 6:
                continue

            if cue_label == "dark floor obj":
                if rx < 15 or rw < 55 or rh < 36:
                    continue
                patch_l = L[ry:y2, rx:x2]
                patch_g = grad[ry:y2, rx:x2]
                if np.percentile(patch_l, 15) > 80 or np.mean(patch_g) < 32:
                    continue

            # Verify local contrast against the carpet immediately below y2
            below_y1 = min(h - 4, y2 + 3)
            below_y2 = min(h, y2 + 24)
            if below_y2 > below_y1:
                carpet_below = lab[below_y1:below_y2, rx:x2]
                obj_patch = lab[ry:y2, rx:x2]
                diff_below = np.linalg.norm(
                    np.median(obj_patch.reshape(-1, 3), axis=0) -
                    np.median(carpet_below.reshape(-1, 3), axis=0)
                )
                if diff_below < 11.0:
                    continue

            cue_cands.append((rx, ry, x2, y2, area, cue_label))

        cue_cands.sort(key=lambda item: item[4], reverse=True)
        for (rx, ry, x2, y2, area, cl) in cue_cands:
            if any(boxes_overlap((rx, ry, x2, y2), yb) for yb in existing_boxes):
                continue
            if any(boxes_overlap((rx, ry, x2, y2), (ex1, ey1, ex2, ey2)) for (ex1, ey1, ex2, ey2, _) in unknown_boxes):
                continue
            unknown_boxes.append((rx, ry, x2, y2, cl))

    return unknown_boxes, debug_mask, horizon_y


def ai_worker_loop(camera):
    """
    Runs YOLO AI + Floor-Anomaly Box Detector continuously in its own background thread
    so the live video stream NEVER pauses or stutters!
    """
    model = YOLO("yolov8n.pt")
    w, h = DISPLAY_WIDTH, DISPLAY_HEIGHT
    left_boundary = w // 3
    right_boundary = 2 * (w // 3)
    smoother = CommandSmoother(window_size=3)
    track_history = defaultdict(lambda: deque(maxlen=15))

    prev_ai_t = time.time()
    ai_fps_smooth = 10.0

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        with state_lock:
            danger_zone_y_ratio = control_settings["danger_zone_y_ratio"]
            use_unknown_helper = control_settings["use_unknown_edge_helper"]
        danger_line_y = int(h * danger_zone_y_ratio)

        # 1. Run YOLO AI
        results = model.track(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=CONFIDENCE_THRESHOLD,
            persist=True,
            verbose=False,
        )[0]

        raw_left = False
        raw_center = False
        raw_right = False
        closest_dist = 9.9
        new_detections = []
        yolo_xyxy = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                track_id = int(box.id[0]) if box.id is not None else -1

                yolo_xyxy.append((x1, y1, x2, y2))
                box_h = y2 - y1
                cx = (x1 + x2) // 2
                dist_m = estimate_distance_meters(y2, box_h, h)
                closest_dist = min(closest_dist, dist_m)

                is_danger = (y2 >= danger_line_y) or (box_h >= h * 0.35)
                if is_danger:
                    if x1 < left_boundary:
                        raw_left = True
                    if x1 < right_boundary and x2 > left_boundary:
                        raw_center = True
                    if x2 > right_boundary:
                        raw_right = True

                trail_pts = []
                if track_id != -1:
                    track_history[track_id].append((cx, y2))
                    trail_pts = list(track_history[track_id])

                new_detections.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        cx,
                        conf,
                        label,
                        track_id,
                        dist_m,
                        is_danger,
                        trail_pts,
                    )
                )

        # 2. Run 3-Cue Floor Obstacle Detector (catches yellow box, blue box, slippers, etc.)
        if use_unknown_helper:
            unk_boxes, _, _ = detect_unknown_floor_boxes(frame, yolo_xyxy)
            for x1, y1, x2, y2, cue_label in unk_boxes:
                box_h = y2 - y1
                cx = (x1 + x2) // 2
                dist_m = estimate_distance_meters(y2, box_h, h)
                closest_dist = min(closest_dist, dist_m)

                is_danger = (y2 >= danger_line_y) or (box_h >= h * 0.35)
                if is_danger:
                    if x1 < left_boundary:
                        raw_left = True
                    if x1 < right_boundary and x2 > left_boundary:
                        raw_center = True
                    if x2 > right_boundary:
                        raw_right = True

                new_detections.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        cx,
                        0.88,
                        cue_label,
                        -1,
                        dist_m,
                        is_danger,
                        [],
                    )
                )

        obj_count = len(new_detections)
        blocked_l, blocked_c, blocked_r = smoother.update(
            raw_left, raw_center, raw_right
        )

        now = time.time()
        dt = now - prev_ai_t
        prev_ai_t = now
        if dt > 0:
            ai_fps_smooth = 0.8 * ai_fps_smooth + 0.2 * (1.0 / dt)

        with state_lock:
            ai_state["detections"] = new_detections
            ai_state["blocked_l"] = blocked_l
            ai_state["blocked_c"] = blocked_c
            ai_state["blocked_r"] = blocked_r
            ai_state["closest_dist"] = closest_dist
            ai_state["obj_count"] = obj_count
            ai_state["ai_fps"] = round(ai_fps_smooth, 1)

        # Pace background AI thread slightly so Mini PC CPU stays cool and never freezes
        time.sleep(0.02)


# Initialize Camera & Background AI Thread once globally
global_camera = None
ai_thread_started = False
init_lock = threading.Lock()


def ensure_started():
    global global_camera, ai_thread_started
    with init_lock:
        if global_camera is None:
            global_camera = ZeroLagCamera()
        if not ai_thread_started:
            t = threading.Thread(
                target=ai_worker_loop, args=(global_camera,), daemon=True
            )
            t.start()
            ai_thread_started = True


def generate_frames():
    ensure_started()
    w, h = DISPLAY_WIDTH, DISPLAY_HEIGHT
    left_boundary = w // 3
    right_boundary = 2 * (w // 3)

    prev_time = time.time()
    video_fps = 30.0

    while True:
        loop_start = time.time()
        ret, frame = global_camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        with state_lock:
            danger_zone_y_ratio = control_settings["danger_zone_y_ratio"]
            show_edge_overlay = control_settings["show_edge_overlay"]

            detections = list(ai_state["detections"])
            blocked_l = ai_state["blocked_l"]
            blocked_c = ai_state["blocked_c"]
            blocked_r = ai_state["blocked_r"]
            closest_dist = ai_state["closest_dist"]
            obj_count = ai_state["obj_count"]
            ai_fps_val = ai_state["ai_fps"]

        danger_line_y = int(h * danger_zone_y_ratio)

        if show_edge_overlay:
            _, mask_bgr, horizon_y = detect_unknown_floor_boxes(frame, [])
            frame[:] = cv2.addWeighted(frame, 0.72, mask_bgr, 0.55, 0)
            cv2.line(frame, (0, horizon_y), (w, horizon_y), (255, 0, 255), 1)

        # Draw latest AI + Unknown-Box detections instantly (0ms wait!)
        for (
            x1,
            y1,
            x2,
            y2,
            cx,
            conf,
            label,
            track_id,
            dist_m,
            is_danger,
            trail_pts,
        ) in detections:
            for i in range(1, len(trail_pts)):
                cv2.line(frame, trail_pts[i - 1], trail_pts[i], (255, 200, 0), 2)

            if is_danger:
                color = (0, 0, 255)  # Red when inside Danger Zone
            elif track_id == -1:
                color = (0, 215, 255)  # Gold/Orange for unknown floor obstacles outside Danger Zone
            else:
                color = (0, 255, 0)  # Green for safe YOLO objects

            id_str = f"#{track_id} " if track_id != -1 else ""
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, y2), 5, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"{id_str}{label} ~{dist_m:.1f}m",
                (x1, max(65, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
            )

        # Fast ROI-only semi-transparent floor overlay
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

        cv2.line(frame, (0, danger_line_y), (w, danger_line_y), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "ROBOT DANGER ZONE (Yellow dot below line = BLOCKED | Use Line UP/DOWN buttons)",
            (12, danger_line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            2,
        )
        cv2.line(frame, (left_boundary, 55), (left_boundary, h), (255, 255, 0), 2)
        cv2.line(frame, (right_boundary, 55), (right_boundary, h), (255, 255, 0), 2)

        for x_start, x_end, is_blk, z_title in zone_coords:
            z_col = (0, 0, 255) if is_blk else (0, 255, 0)
            state_txt = "BLOCKED" if is_blk else "CLEAR"
            cv2.putText(
                frame,
                f"{z_title}: {state_txt}",
                (x_start + 18, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                z_col,
                2,
            )

        cmd_text, steer_deg, speed_pct, cmd_color = get_navigation_command(
            blocked_l, blocked_c, blocked_r, closest_dist
        )

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            video_fps = 0.85 * video_fps + 0.15 * (1.0 / dt)

        cv2.rectangle(frame, (0, 0), (w, 52), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"VIDEO: {video_fps:.0f} FPS (AI: {ai_fps_val:.0f} Hz) | {cmd_text}",
            (14, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            cmd_color,
            2,
        )

        with state_lock:
            latest_telemetry["fps"] = round(video_fps, 1)
            latest_telemetry["ai_fps"] = ai_fps_val
            latest_telemetry["command"] = cmd_text
            latest_telemetry["steer_deg"] = steer_deg
            latest_telemetry["speed_pct"] = speed_pct
            latest_telemetry["closest_dist_m"] = (
                closest_dist if closest_dist < 9.0 else None
            )
            latest_telemetry["left_zone"] = "BLOCKED" if blocked_l else "CLEAR"
            latest_telemetry["center_zone"] = "BLOCKED" if blocked_c else "CLEAR"
            latest_telemetry["right_zone"] = "BLOCKED" if blocked_r else "CLEAR"
            latest_telemetry["objects_count"] = obj_count

        ret_jpg, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
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


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Wheelchair AI Vision Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0f172a;
            color: #f8fafc;
            font-family: system-ui, -apple-system, sans-serif;
            padding: 16px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #1e293b;
            padding: 14px 20px;
            border-radius: 12px;
            margin-bottom: 16px;
            border: 1px solid #334155;
        }
        .header h1 { font-size: 1.25rem; color: #38bdf8; }
        .badge {
            background: #22c55e;
            color: #052e16;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 16px;
        }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }
        .video-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #334155;
            text-align: center;
        }
        .video-card img {
            width: 100%;
            max-height: 74vh;
            object-fit: contain;
            border-radius: 8px;
            background: #000;
        }
        .telemetry-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #334155;
        }
        .card-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .card-value {
            font-size: 1.35rem;
            font-weight: 700;
        }
        .zones {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 8px;
        }
        .zone-box {
            padding: 12px 8px;
            border-radius: 8px;
            text-align: center;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .clear { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
        .blocked { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid #ef4444; }
        .btn-group {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 8px;
        }
        button {
            background: #334155;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            flex: 1;
        }
        button:hover { background: #475569; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Smart Wheelchair — Mini PC AI Vision GUI</h1>
        <span class="badge">ZERO-LAG STREAM</span>
    </div>
    <div class="grid">
        <div class="video-card">
            <img src="/video_feed" alt="Live AI Camera Stream">
        </div>
        <div class="telemetry-panel">
            <div class="card">
                <div class="card-label">Navigation Decision</div>
                <div class="card-value" id="cmd" style="color: #38bdf8;">LOADING...</div>
            </div>
            <div class="card">
                <div class="card-label">Zone Status (Left / Center / Right)</div>
                <div class="zones">
                    <div id="z_left" class="zone-box clear">LEFT<br>CLEAR</div>
                    <div id="z_center" class="zone-box clear">CENTER<br>CLEAR</div>
                    <div id="z_right" class="zone-box clear">RIGHT<br>CLEAR</div>
                </div>
            </div>
            <div class="card">
                <div class="card-label">Remote Controls (Tune Live on Mini PC)</div>
                <div class="btn-group">
                    <button onclick="sendAction('line_up')">Line UP (+)</button>
                    <button onclick="sendAction('line_down')">Line DOWN (-)</button>
                </div>
                <div class="btn-group">
                    <button onclick="sendAction('toggle_unknown')">Toggle Unknown-Box [u]</button>
                    <button onclick="sendAction('toggle_edges')">Toggle Mask View [e]</button>
                </div>
            </div>
            <div class="card">
                <div class="card-label">Steering & Speed Command</div>
                <div class="card-value" id="motion">Steer: 0° | Speed: 100%</div>
            </div>
            <div class="card">
                <div class="card-label">Mini PC Performance</div>
                <div class="card-value" id="fps">Video: -- FPS | AI: -- Hz</div>
            </div>
        </div>
    </div>
    <script>
        async function sendAction(action) {
            await fetch('/control/' + action, { method: 'POST' });
        }
        setInterval(async () => {
            try {
                const res = await fetch('/telemetry');
                const data = await res.json();
                document.getElementById('cmd').innerText = data.command;
                document.getElementById('motion').innerText = `Steer: ${data.steer_deg > 0 ? '+' : ''}${data.steer_deg}° | Speed: ${data.speed_pct}%`;
                document.getElementById('fps').innerText = `Video: ${data.fps} FPS | AI: ${data.ai_fps} Hz | Objects: ${data.objects_count}`;

                const setZone = (id, name, state) => {
                    const el = document.getElementById(id);
                    el.className = 'zone-box ' + (state === 'BLOCKED' ? 'blocked' : 'clear');
                    el.innerHTML = `${name}<br>${state}`;
                };
                setZone('z_left', 'LEFT', data.left_zone);
                setZone('z_center', 'CENTER', data.center_zone);
                setZone('z_right', 'RIGHT', data.right_zone);
            } catch (e) {}
        }, 200);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/telemetry")
def telemetry():
    with state_lock:
        return jsonify(latest_telemetry)


@app.route("/control/<action>", methods=["POST"])
def control(action):
    with state_lock:
        if action == "line_up":
            control_settings["danger_zone_y_ratio"] = max(
                0.28, control_settings["danger_zone_y_ratio"] - 0.04
            )
        elif action == "line_down":
            control_settings["danger_zone_y_ratio"] = min(
                0.85, control_settings["danger_zone_y_ratio"] + 0.04
            )
        elif action == "toggle_unknown":
            control_settings["use_unknown_edge_helper"] = not control_settings[
                "use_unknown_edge_helper"
            ]
        elif action == "toggle_edges":
            control_settings["show_edge_overlay"] = not control_settings[
                "show_edge_overlay"
            ]
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" MINI PC SMART WHEELCHAIR WEB GUI (YOLO + UNKNOWN BOX DETECTOR)!")
    print(" -> Open in Laptop Browser: http://10.0.0.2:5000")
    print("=" * 65 + "\n")
    ensure_started()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
