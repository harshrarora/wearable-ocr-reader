"""
Text Region Detector
────────────────────
Locates bounding boxes of readable text inside a frame.
Primary back-end: EasyOCR (does detection + recognition in one pass).
Fallback: MSER-based region proposal.
"""

import cv2
import numpy as np
import logging
from config import TEXT_DETECTION_CONFIDENCE, TEXT_MIN_AREA

logger = logging.getLogger(__name__)


class TextDetector:

    def __init__(self, reader=None):
        """
        Parameters
        ----------
        reader : easyocr.Reader or None
            Shared EasyOCR reader (initialised once in main).
        """
        self._reader = reader
        self._last_boxes = []

    # ── public ──────────────────────────────────────────────

    def detect(self, frame):
        """
        Returns
        -------
        list[dict]  — each dict has keys:
            bbox       (x, y, w, h)
            polygon    list of 4 corner points (or None)
            confidence float
            text       str or None   (available when EasyOCR is used)
        """
        if self._reader is not None:
            boxes = self._detect_easyocr(frame)
        else:
            boxes = self._detect_mser(frame)
        self._last_boxes = boxes
        return boxes

    @property
    def last_boxes(self):
        return self._last_boxes

    def combined_bbox(self):
        """Bounding rect that encloses *all* detected boxes."""
        if not self._last_boxes:
            return None
        x0 = min(b["bbox"][0] for b in self._last_boxes)
        y0 = min(b["bbox"][1] for b in self._last_boxes)
        x1 = max(b["bbox"][0] + b["bbox"][2] for b in self._last_boxes)
        y1 = max(b["bbox"][1] + b["bbox"][3] for b in self._last_boxes)
        return (x0, y0, x1 - x0, y1 - y0)

    # ── EasyOCR back-end ────────────────────────────────────

    def _detect_easyocr(self, frame):
        try:
            results = self._reader.readtext(frame, detail=1, paragraph=False)
            boxes = []
            for polygon, text, conf in results:
                if conf < TEXT_DETECTION_CONFIDENCE:
                    continue
                pts = np.array(polygon, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                if w * h < TEXT_MIN_AREA:
                    continue
                boxes.append(
                    dict(bbox=(x, y, w, h), polygon=polygon,
                         confidence=conf, text=text)
                )
            return boxes
        except Exception as exc:
            logger.error("EasyOCR detection failed: %s", exc)
            return []

    # ── MSER fallback ───────────────────────────────────────

    def _detect_mser(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        raw = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w * h < TEXT_MIN_AREA:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.1 or aspect > 15:
                continue
            raw.append(dict(bbox=(x, y, w, h), polygon=None,
                            confidence=0.5, text=None))
        return self._nms(raw)

    @staticmethod
    def _nms(boxes, thresh=0.3):
        if not boxes:
            return []
        bbs = np.array([b["bbox"] for b in boxes], dtype=np.float32)
        x1, y1, w, h = bbs[:, 0], bbs[:, 1], bbs[:, 2], bbs[:, 3]
        x2, y2 = x1 + w, y1 + h
        areas = w * h
        order = areas.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            ovr = inter / areas[order[1:]]
            order = order[1:][ovr < thresh]
        return [boxes[i] for i in keep]