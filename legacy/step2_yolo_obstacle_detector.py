import cv2
import time
from ultralytics import YOLO

# 1. Camera Configuration (AUKEY USB Webcam at /dev/video2)
CAMERA_INDEX = 2
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 2. Obstacle Threshold:
# If an object's bounding box height takes up more than 35% of the screen height,
# we consider it "CLOSE / DANGER" in front of the robot.
CLOSE_HEIGHT_RATIO = 0.35
CONFIDENCE_THRESHOLD = 0.45


def get_navigation_command(blocked_left, blocked_center, blocked_right):
    """Decide robot movement based on which camera zones are blocked."""
    if not blocked_center:
        return "CLEAR -> MOVE FORWARD", (0, 255, 0)  # Green
    elif not blocked_left and not blocked_right:
        return "CENTER BLOCKED -> TURN LEFT OR RIGHT", (0, 165, 255)  # Orange
    elif not blocked_left:
        return "OBSTACLE AHEAD -> TURN LEFT", (0, 165, 255)  # Orange
    elif not blocked_right:
        return "OBSTACLE AHEAD -> TURN RIGHT", (0, 165, 255)  # Orange
    else:
        return "ALL ZONES BLOCKED -> STOP!", (0, 0, 255)  # Red


def main():
    print("Loading YOLOv8 Nano AI model (downloads ~6MB automatically on first run)...")
    model = YOLO("yolov8n.pt")

    print(f"Opening AUKEY camera at index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {CAMERA_INDEX}.")
        return

    # Unlock 30 FPS using MJPG hardware compression
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define Left / Center / Right zone boundaries
    left_boundary = w // 3
    right_boundary = 2 * (w // 3)

    print("AI Obstacle Detection started! Press 'q' in the video window to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO AI on the current frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        blocked_left = False
        blocked_center = False
        blocked_right = False

        # Loop through every detected object
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Calculate object center X and relative height on screen
            obj_center_x = (x1 + x2) // 2
            box_height_ratio = (y2 - y1) / h
            is_close = box_height_ratio >= CLOSE_HEIGHT_RATIO

            # Check which zone the object is in if it is close enough
            if is_close:
                # Check overlap with Left, Center, Right zones
                if x1 < left_boundary:
                    blocked_left = True
                if x1 < right_boundary and x2 > left_boundary:
                    blocked_center = True
                if x2 > right_boundary:
                    blocked_right = True

            # Color: Red if close obstacle, Green if far/safe
            color = (0, 0, 255) if is_close else (0, 255, 0)
            status_tag = "CLOSE!" if is_close else "FAR"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.0%} [{status_tag}]",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

        # Draw vertical zone lines (Left | Center | Right)
        cv2.line(frame, (left_boundary, 80), (left_boundary, h), (255, 255, 0), 2)
        cv2.line(frame, (right_boundary, 80), (right_boundary, h), (255, 255, 0), 2)

        # Draw zone status labels at the bottom
        zones = [
            ("LEFT ZONE", blocked_left, 40),
            ("CENTER ZONE", blocked_center, left_boundary + 40),
            ("RIGHT ZONE", blocked_right, right_boundary + 40),
        ]
        for z_name, is_blocked, x_pos in zones:
            z_color = (0, 0, 255) if is_blocked else (0, 255, 0)
            z_state = "BLOCKED" if is_blocked else "FREE"
            cv2.putText(
                frame,
                f"{z_name}: {z_state}",
                (x_pos, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                z_color,
                2,
            )

        # Calculate FPS and get Robot Navigation Command
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        cmd_text, cmd_color = get_navigation_command(
            blocked_left, blocked_center, blocked_right
        )

        # Draw dark top banner for telemetry
        cv2.rectangle(frame, (0, 0), (w, 75), (25, 25, 25), -1)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | ROBOT CMD: {cmd_text}",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            cmd_color,
            2,
        )

        cv2.imshow("Step 2 - YOLO AI Obstacle Detection (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
