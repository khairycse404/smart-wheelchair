import cv2
import time

# Index 2 is your AUKEY USB Webcam (/dev/video2)
# (Index 0 is usually the built-in laptop webcam)
CAMERA_INDEX = 2

def main():
    print(f"Opening camera at index {CAMERA_INDEX} (/dev/video{CAMERA_INDEX})...")
    # Use V4L2 backend on Linux for direct hardware control
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Error: Could not open camera index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX to 0 if testing with the built-in webcam.")
        return

    # IMPORTANT: Enable MJPG compression FIRST so USB 2.0 can stream at 30 FPS
    # (Without MJPG, raw YUYV hits the USB 2.0 bandwidth limit and caps at 10 FPS!)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened successfully! Resolution: {actual_w}x{actual_h}")
    print("Press 'q' inside the video window to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Calculate live FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # Draw FPS and resolution on the top-left of the screen
        cv2.putText(
            frame,
            f"AUKEY Cam | {actual_w}x{actual_h} | FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Show the live video feed
        cv2.imshow("Step 1 - AUKEY Camera Test (Press 'q' to quit)", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed cleanly.")


if __name__ == "__main__":
    main()
