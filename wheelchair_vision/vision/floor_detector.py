"""
Multi-Cue Floor-Horizon Obstacle Segmentation Module.
Detects non-YOLO floor obstacles (cardboard boxes, packages, slippers, shoes)
without merging with background walls, doors, or furniture.
"""

import cv2
import numpy as np


def boxes_overlap(boxA, boxB):
    """Returns True if two (x1, y1, x2, y2) bounding boxes overlap significantly."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return False
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return (inter / float(min(areaA, areaB))) > 0.25


def detect_unknown_floor_boxes(frame, existing_boxes):
    """
    3-Cue Floor-Horizon Obstacle Detector:
    Separates Warm/Yellow objects, Cool/Blue boxes, and Dark floor objects into independent
    channels so adjacent floor obstacles never merge with each other or with the background wall/door.

    Args:
        frame: Current BGR image (e.g. 640x480).
        existing_boxes: List of (x1, y1, x2, y2) bounding boxes from YOLO to mask out.

    Returns:
        unknown_boxes: List of (x1, y1, x2, y2, label_str)
        debug_mask: 3-channel BGR visualization mask
        horizon_y: Estimated vertical pixel coordinate of the carpet-to-wall horizon
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
    for bx1, by1, bx2, by2 in existing_boxes:
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

    # 2. Estimate Floor Horizon Y automatically
    is_carpet = (np.abs(L - ref_l) < 35) & (np.hypot(A - ref_a, B - ref_b) < 18)
    horizon_y = int(h * 0.46)
    for y in range(int(h * 0.78), int(h * 0.34), -2):
        row_carpet_ratio = np.mean(is_carpet[y, int(w * 0.15):int(w * 0.85)])
        if row_carpet_ratio < 0.25:
            horizon_y = y
            break
    horizon_y = int(np.clip(horizon_y, int(h * 0.38), int(h * 0.54)))

    # 3. Three Separate Cue Masks
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
    for bx1, by1, bx2, by2 in existing_boxes:
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

    for cue_label, raw_mask, color_ch, k_cl in [
        ("yellow/warm box", cue_warm, 2, k_close_small),
        ("blue/cool box", cue_cool, 0, k_close_med),
        ("dark floor obj", cue_dark, 1, k_close_med),
    ]:
        if cue_label == "dark floor obj":
            for ex1, ey1, ex2, ey2, _ in unknown_boxes:
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
        for rx, ry, x2, y2, area, cl in cue_cands:
            if any(boxes_overlap((rx, ry, x2, y2), yb) for yb in existing_boxes):
                continue
            if any(boxes_overlap((rx, ry, x2, y2), (ex1, ey1, ex2, ey2)) for ex1, ey1, ex2, ey2, _ in unknown_boxes):
                continue
            unknown_boxes.append((rx, ry, x2, y2, cl))

    return unknown_boxes, debug_mask, horizon_y
