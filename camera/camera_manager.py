"""
Camera Manager
──────────────
Captures frames in a daemon thread so the main loop never blocks on I/O.
Provides the latest frame on demand via `get_frame()`.
"""

import cv2
import threading
import time
import logging
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

logger = logging.getLogger(__name__)


class CameraManager:
    """Threaded camera wrapper with FPS counter."""

    def __init__(self, camera_index=None, width=None, height=None):
        self.camera_index = camera_index if camera_index is not None else CAMERA_INDEX
        self.width = width or CAMERA_WIDTH
        self.height = height or CAMERA_HEIGHT

        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # FPS bookkeeping
        self._frame_count = 0
        self._fps = 0.0
        self._fps_timer = time.time()

    # ── public API ──────────────────────────────────────────

    def start(self):
        """Open the camera and begin background capture."""
        logger.info(
            "Opening camera %d  (%dx%d @ %d FPS requested)",
            self.camera_index, self.width, self.height, CAMERA_FPS,
        )

        #self._cap = cv2.VideoCapture(self.camera_index)
        # self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        import platform

        if platform.system() == "Windows":
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera ready: %dx%d", actual_w, actual_h)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def get_frame(self):
        """Return a *copy* of the latest frame, or ``None``."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def fps(self):
        return self._fps

    @property
    def is_running(self):
        return self._running

    def stop(self):
        """Release the camera and join the thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("Camera stopped")

    # ── internals ───────────────────────────────────────────

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                self._tick_fps()
            else:
                logger.warning("Frame grab failed — retrying")
                time.sleep(0.01)

    def _tick_fps(self):
        self._frame_count += 1
        now = time.time()
        dt = now - self._fps_timer
        if dt >= 1.0:
            self._fps = self._frame_count / dt
            self._frame_count = 0
            self._fps_timer = now

    def __del__(self):
        self.stop()