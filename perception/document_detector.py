"""
Document Detector
─────────────────
Finds rectangular page boundaries and optionally applies
perspective correction so OCR sees a clean, upright page.
"""

import cv2
import numpy as np
import logging
from config import MIN_DOCUMENT_AREA_RATIO, PERSPECTIVE_CORRECTION

logger = logging.getLogger(__name__)


class DocumentDetector:

    def __init__(self):
        self._last_corners = None

    def detect(self, frame):
        """
        Returns
        -------
        corners : ndarray (4,2) or None
        warped  : ndarray or None   (perspective-corrected image)
        """
        h, w = frame.shape[:2]
        min_area = h * w * MIN_DOCUMENT_AREA_RATIO

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(cnt) < min_area:
                continue
            eps = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                self._last_corners = corners
                warped = self._four_point_warp(frame, corners) if PERSPECTIVE_CORRECTION else None
                return corners, warped
        return None, None

    @property
    def last_corners(self):
        return self._last_corners

    # ── helpers ─────────────────────────────────────────────

    @staticmethod
    def _order_corners(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        rect[0] = pts[np.argmin(s)]   # TL
        rect[2] = pts[np.argmax(s)]   # BR
        rect[1] = pts[np.argmin(d)]   # TR
        rect[3] = pts[np.argmax(d)]   # BL
        return rect

    def _four_point_warp(self, image, pts):
        ordered = self._order_corners(pts.astype(np.float32))
        tl, tr, br, bl = ordered
        max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array(
            [[0, 0], [max_w - 1, 0],
             [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, M, (max_w, max_h))