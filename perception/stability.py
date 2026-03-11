"""
Stability / Motion Intelligence
────────────────────────────────
Determines whether the camera view is steady enough to justify
the cost of OCR.  Uses dense optical flow (Farnebäck) with a
frame-difference fallback.
"""

import cv2
import numpy as np
import logging
from collections import deque
from config import (
    STABILITY_THRESHOLD,
    STABILITY_FRAMES_REQUIRED,
    MOTION_HISTORY_SIZE,
)

logger = logging.getLogger(__name__)


class StabilityDetector:

    def __init__(self):
        self._prev_gray = None
        self._history = deque(maxlen=MOTION_HISTORY_SIZE)
        self._stable_streak = 0
        self._last_score = 0.0

    # ── public ──────────────────────────────────────────────

    def update(self, frame):
        """
        Feed a new frame.

        Returns
        -------
        is_stable : bool
        motion_score : float   (lower = calmer)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0.0

        score = self._optical_flow_score(self._prev_gray, gray)
        if score < 0:
            score = self._frame_diff_score(self._prev_gray, gray)

        self._prev_gray = gray
        self._last_score = score
        self._history.append(score)

        if score < STABILITY_THRESHOLD:
            self._stable_streak += 1
        else:
            self._stable_streak = 0

        is_stable = self._stable_streak >= STABILITY_FRAMES_REQUIRED
        return is_stable, score

    @property
    def motion_score(self):
        return self._last_score

    @property
    def average_motion(self):
        return float(np.mean(self._history)) if self._history else 0.0

    def reset(self):
        self._prev_gray = None
        self._history.clear()
        self._stable_streak = 0

    # ── internals ───────────────────────────────────────────

    def _optical_flow_score(self, prev, cur):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            return float(np.mean(mag))
        except Exception:
            return -1.0

    def _frame_diff_score(self, prev, cur):
        return float(np.mean(cv2.absdiff(prev, cur)))