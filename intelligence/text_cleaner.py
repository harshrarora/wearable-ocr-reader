"""
Text Cleaner
────────────
Post-processes raw OCR output so it sounds natural through TTS.
Removes garbage, fixes common OCR artefacts, and chunks long
text at sentence boundaries.
"""

import re
import logging
from config import REMOVE_SPECIAL_CHARS

logger = logging.getLogger(__name__)

_GARBAGE_RE = [
    re.compile(r"^[^a-zA-Z0-9]+$"),
    re.compile(r"^.{1,2}$"),
    re.compile(r"[|\\/{}\[\]<>]{2,}"),
]


class TextCleaner:

    def clean(self, text):
        """Full cleaning pipeline → ready-for-TTS string."""
        if not text:
            return ""
        text = self._strip_control(text)
        lines = text.split("\n")
        lines = [l.strip() for l in lines if self._valid(l.strip())]
        text = " ".join(lines)
        text = self._fix_ocr_errors(text)
        text = self._sentence_structure(text)
        return re.sub(r"\s+", " ", text).strip()

    def chunk_for_speech(self, text, max_len=120):
        """Split cleaned text at sentence boundaries for TTS."""
        if len(text) <= max_len:
            return [text] if text else []
        parts = re.split(r"(?<=[.!?])\s+", text)
        chunks, cur = [], ""
        for s in parts:
            if len(cur) + len(s) + 1 <= max_len:
                cur = f"{cur} {s}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks

    # ── internals ───────────────────────────────────────────

    @staticmethod
    def _strip_control(t):
        t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
        if REMOVE_SPECIAL_CHARS:
            t = re.sub(r"[^\w\s.,!?;:'\-/()&@#$%]", "", t)
        return t

    @staticmethod
    def _valid(line):
        if not line:
            return False
        for pat in _GARBAGE_RE:
            if pat.match(line):
                return False
        alpha = sum(c.isalpha() for c in line)
        return alpha / max(len(line), 1) >= 0.3

    @staticmethod
    def _fix_ocr_errors(t):
        t = re.sub(r"\bl\b", "I", t)       # isolated 'l' → 'I'
        t = re.sub(r"\s{2,}", " ", t)
        return t

    @staticmethod
    def _sentence_structure(t):
        t = re.sub(r"\.(?=[A-Z])", ". ", t)
        words = t.split()
        if len(words) > 15 and not any(c in t for c in ".!?,;:"):
            out = []
            for i, w in enumerate(words):
                out.append(w)
                if (i + 1) % 10 == 0 and i < len(words) - 1:
                    out.append(",")
            t = " ".join(out)
        return t