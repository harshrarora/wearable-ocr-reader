"""
Temporal OCR Fusion  ⭐  Key Innovation
──────────────────────────────────────
Accumulates OCR outputs from N consecutive stable frames and
merges them via word-level majority voting aligned with
difflib.SequenceMatcher.

Why it matters
--------------
Single-frame OCR is noisy under slight motion blur or uneven
lighting.  By voting across frames we dramatically reduce errors
without needing a larger (slower) model.
"""

import difflib
import logging
from collections import deque
from config import FUSION_BUFFER_SIZE, FUSION_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class OCRFusion:

    def __init__(self, buffer_size=None):
        self._size = buffer_size or FUSION_BUFFER_SIZE
        self._buf = deque(maxlen=self._size)

    # ── public ──────────────────────────────────────────────

    def add_result(self, text, confidence=1.0):
        if text and text.strip():
            self._buf.append({"text": text.strip(), "conf": confidence})

    def fuse(self):
        """
        Returns
        -------
        fused_text  : str
        fused_conf  : float
        """
        if not self._buf:
            return "", 0.0
        if len(self._buf) == 1:
            r = self._buf[0]
            return r["text"], r["conf"]

        items = list(self._buf)

        # pick highest-confidence result as reference
        ref = max(items, key=lambda r: r["conf"])
        ref_text = ref["text"]

        # keep only items similar enough to the reference
        similar = [
            r for r in items
            if difflib.SequenceMatcher(None, ref_text, r["text"]).ratio()
               >= FUSION_SIMILARITY_THRESHOLD
        ]
        if not similar:
            best = max(items, key=lambda r: r["conf"])
            return best["text"], best["conf"]

        fused = self._vote(similar, ref)
        avg_c = sum(r["conf"] for r in similar) / len(similar)
        return fused, avg_c

    @property
    def buffer_count(self):
        return len(self._buf)

    @property
    def is_ready(self):
        return len(self._buf) >= self._size

    def clear(self):
        self._buf.clear()

    # ── internals ───────────────────────────────────────────

    @staticmethod
    def _vote(items, ref):
        """Word-level majority vote aligned to *ref*."""
        ref_words = ref["text"].split()
        if not ref_words:
            return ""

        slots = [[] for _ in ref_words]

        for item in items:
            words = item["text"].split()
            conf = item["conf"]
            matcher = difflib.SequenceMatcher(None, ref_words, words)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ("equal", "replace"):
                    for k in range(min(i2 - i1, j2 - j1)):
                        if i1 + k < len(slots):
                            slots[i1 + k].append((words[j1 + k], conf))

        out = []
        for idx, cands in enumerate(slots):
            if not cands:
                out.append(ref_words[idx])
                continue
            scores = {}
            for w, c in cands:
                key = w.lower()
                if key not in scores:
                    scores[key] = {"word": w, "score": 0.0}
                scores[key]["score"] += c
            best = max(scores.values(), key=lambda v: v["score"])
            out.append(best["word"])
        return " ".join(out)