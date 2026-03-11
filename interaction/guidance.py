"""
Spatial Guidance Engine
───────────────────────
Analyses text-box positions relative to the frame centre and
emits short audio cues ("move left", "tilt down", …).

A cooldown timer prevents the same cue from repeating too fast.
"""

import time
import logging
from config import (
    GUIDANCE_COOLDOWN,
    CENTER_TOLERANCE_X,
    CENTER_TOLERANCE_Y,
    MIN_TEXT_SCALE,
    MAX_TEXT_SCALE,
)

logger = logging.getLogger(__name__)


class GuidanceEngine:

    MOVE_LEFT = "move left"
    MOVE_RIGHT = "move right"
    TILT_UP = "tilt up"
    TILT_DOWN = "tilt down"
    CLOSER = "move closer"
    BACK = "move back"
    HOLD = "hold still"
    TEXT_FOUND = "text detected"
    NO_TEXT = "no text visible"
    GOOD = "good position"

    def __init__(self):
        self._cooldowns = {}
        self._last = None

    def analyze(self, text_boxes, frame_shape, stable=True):
        """Return a guidance cue string, or ``None`` if suppressed."""
        h, w = frame_shape[:2]

        if not text_boxes:
            return self._emit(self.NO_TEXT)
        if not stable:
            return self._emit(self.HOLD)

        # area-weighted centroid
        sw, sy, total = 0.0, 0.0, 0.0
        for b in text_boxes:
            bx, by, bw, bh = b["bbox"]
            a = bw * bh
            sw += (bx + bw / 2) * a
            sy += (by + bh / 2) * a
            total += a
        if total == 0:
            return None
        cx, cy = sw / total, sy / total
        scale = total / (w * h)

        if scale < MIN_TEXT_SCALE:
            return self._emit(self.CLOSER)
        if scale > MAX_TEXT_SCALE:
            return self._emit(self.BACK)

        dx = (cx - w / 2) / w
        dy = (cy - h / 2) / h

        if abs(dx) > CENTER_TOLERANCE_X:
            return self._emit(self.MOVE_RIGHT if dx > 0 else self.MOVE_LEFT)
        if abs(dy) > CENTER_TOLERANCE_Y:
            return self._emit(self.TILT_DOWN if dy > 0 else self.TILT_UP)

        return self._emit(self.GOOD)

    def _emit(self, cue):
        now = time.time()
        if now - self._cooldowns.get(cue, 0) < GUIDANCE_COOLDOWN:
            return None
        self._cooldowns[cue] = now
        self._last = cue
        return cue

    @property
    def last_cue(self):
        return self._last

    def reset_cooldowns(self):
        self._cooldowns.clear()