"""
Finger Tracker
──────────────
Detects the index-finger tip via MediaPipe Hands.
When the index finger is extended and pointing, returns its (x, y)
so the intent resolver can switch to FINGER_READ mode.
"""

import cv2
import logging
from config import FINGER_CONFIDENCE, FINGER_TRACKING_ENABLED

logger = logging.getLogger(__name__)


class FingerTracker:

    def __init__(self):
        self._enabled = FINGER_TRACKING_ENABLED
        self._hands = None
        self._last_tip = None
        self._detected = False

        if not self._enabled:
            return

        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=FINGER_CONFIDENCE,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Hands ready")
        except ImportError:
            logger.warning("mediapipe not installed — finger tracking disabled")
            self._enabled = False

    # ── public ──────────────────────────────────────────────

    def detect(self, frame):
        """
        Returns
        -------
        fingertip : (x, y) in pixels, or None
        detected  : bool
        """
        if not self._enabled or self._hands is None:
            return None, False

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            tip = lm[8]   # INDEX_FINGER_TIP
            pip = lm[6]   # INDEX_FINGER_PIP

            # "Extended" heuristic: tip is above PIP in image space
            if tip.y < pip.y:
                px, py = int(tip.x * w), int(tip.y * h)
                self._last_tip = (px, py)
                self._detected = True
                return (px, py), True

        self._last_tip = None
        self._detected = False
        return None, False

    def draw(self, frame):
        """Overlay a marker on the fingertip (debug)."""
        if self._last_tip:
            cv2.circle(frame, self._last_tip, 10, (0, 255, 0), -1)
            cv2.circle(frame, self._last_tip, 16, (0, 255, 0), 2)
        return frame

    @property
    def fingertip(self):
        return self._last_tip

    @property
    def is_detected(self):
        return self._detected

    def release(self):
        if self._hands:
            self._hands.close()