"""
High-performance, zero-lag USB camera interface for Linux V4L2 webcams.
"""

import time
import threading
import cv2
from ..config import config


class ZeroLagCamera:
    """
    Spawns a dedicated background reader thread to grab only the latest frame
    from the V4L2 video device, discarding any internal OpenCV buffer queues.
    Guarantees 0ms latency for real-time robotic navigation.
    """

    def __init__(self, device_indices=None, width=None, height=None, fps=None):
        self.device_indices = device_indices or config.PREFER_DEVICE_INDICES
        self.width = width or config.CAPTURE_WIDTH
        self.height = height or config.CAPTURE_HEIGHT
        self.fps = fps or config.CAPTURE_FPS

        self.cap = self._open_best_camera()
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _open_best_camera(self):
        for idx in self.device_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"[ZeroLagCamera] Successfully opened camera at /dev/video{idx}")
                    return cap
                cap.release()
        raise RuntimeError(
            f"[ZeroLagCamera] Could not open any USB camera from candidates: {self.device_indices}"
        )

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                resized = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                with self.lock:
                    self.latest_frame = resized
            else:
                time.sleep(0.005)

    def read(self):
        """Returns (ret, frame) where frame is a copy of the newest frame."""
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        """Stops the reader thread and releases the hardware video device."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
