"""
Intent Resolver
───────────────
Central decision logic: given sensor outputs, decide which
system mode should be active.

Modes
-----
IDLE         — nothing happening
GUIDANCE     — user needs directional cues
AUTO_READ    — text is centred & stable → read it
FINGER_READ  — user is pointing at specific text
SPEAKING     — TTS is active (pause perception)
"""

import logging
from config import CENTER_TOLERANCE_X, CENTER_TOLERANCE_Y, FINGER_PROXIMITY_THRESHOLD

logger = logging.getLogger(__name__)


class Mode:
    IDLE = "IDLE"
    GUIDANCE = "GUIDANCE"
    AUTO_READ = "AUTO_READ"
    FINGER_READ = "FINGER_READ"
    SPEAKING = "SPEAKING"


class IntentResolver:

    def __init__(self):
        self._mode = Mode.IDLE

    def resolve(self, *, stable, text_boxes, fingertip,
                frame_shape, is_speaking=False):
        """
        Parameters
        ----------
        stable      : bool
        text_boxes  : list[dict]
        fingertip   : (x,y) | None
        frame_shape : (H, W, C)
        is_speaking : bool

        Returns
        -------
        mode   : str
        target : dict | list | None
        """
        if is_speaking:
            self._mode = Mode.SPEAKING
            return Mode.SPEAKING, None

        has_text = len(text_boxes) > 0

        # finger takes priority
        if fingertip is not None and has_text:
            nearest = self._nearest_box(fingertip, text_boxes)
            if nearest is not None:
                self._mode = Mode.FINGER_READ
                return Mode.FINGER_READ, nearest

        if not has_text:
            self._mode = Mode.IDLE
            return Mode.IDLE, None

        if not stable:
            self._mode = Mode.GUIDANCE
            return Mode.GUIDANCE, None

        h, w = frame_shape[:2]
        if self._text_is_centred(text_boxes, w, h):
            self._mode = Mode.AUTO_READ
            return Mode.AUTO_READ, text_boxes

        self._mode = Mode.GUIDANCE
        return Mode.GUIDANCE, None

    @property
    def current_mode(self):
        return self._mode

    # ── helpers ─────────────────────────────────────────────

    @staticmethod
    def _nearest_box(tip, boxes):
        fx, fy = tip
        best, best_d = None, float("inf")
        for b in boxes:
            bx, by, bw, bh = b["bbox"]
            cx, cy = bx + bw // 2, by + bh // 2
            d = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
            if d < best_d and d < FINGER_PROXIMITY_THRESHOLD:
                best, best_d = b, d
        return best

    @staticmethod
    def _text_is_centred(boxes, fw, fh):
        cx = sum(b["bbox"][0] + b["bbox"][2] / 2 for b in boxes) / len(boxes)
        cy = sum(b["bbox"][1] + b["bbox"][3] / 2 for b in boxes) / len(boxes)
        dx = abs(cx - fw / 2) / fw
        dy = abs(cy - fh / 2) / fh
        return dx < CENTER_TOLERANCE_X and dy < CENTER_TOLERANCE_Y