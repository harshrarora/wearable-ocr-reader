"""
OCR Engine
──────────
Image preprocessing pipeline + EasyOCR execution.
Preprocessing: greyscale → upscale → denoise → CLAHE → adaptive
threshold → deskew.
"""

import cv2
import numpy as np
import logging
from config import OCR_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class OCREngine:

    def __init__(self, reader=None):
        self._reader = reader

    # ── public ──────────────────────────────────────────────

    def read_region(self, frame, bbox):
        """OCR a single (x,y,w,h) region. Returns (text, confidence)."""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 0 or h <= 0:
            return "", 0.0
        crop = frame[y:y + h, x:x + w]
        return self._ocr(self._preprocess(crop))

    def read_boxes(self, frame, text_boxes):
        """
        Read from a list of boxes.  Reuses text already extracted by
        the detector when available.

        Returns (combined_text, average_confidence).
        """
        ordered = sorted(text_boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        texts, confs = [], []
        for box in ordered:
            if box.get("text") and box["confidence"] >= OCR_CONFIDENCE_THRESHOLD:
                texts.append(box["text"])
                confs.append(box["confidence"])
            else:
                t, c = self.read_region(frame, box["bbox"])
                if t and c >= OCR_CONFIDENCE_THRESHOLD:
                    texts.append(t)
                    confs.append(c)
        combined = " ".join(texts)
        avg = float(np.mean(confs)) if confs else 0.0
        return combined, avg

    # ── preprocessing ───────────────────────────────────────

    def _preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # upscale tiny crops
        h, w = gray.shape
        if h < 50 or w < 50:
            s = max(50 / h, 50 / w, 2.0)
            gray = cv2.resize(gray, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)

        gray = cv2.fastNlMeansDenoising(gray, h=10)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # gray = clahe.apply(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # sharpen text edges
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        # binary = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 11, 2,
        # )
        # binary = cv2.medianBlur(binary, 3)
        # #return self._deskew(binary)
        # return gray
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2
        )

        binary = cv2.medianBlur(binary, 3)

        return self._deskew(binary)

    @staticmethod
    def _deskew(img):
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 50:
            return img
        try:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if abs(angle) > 15 or abs(angle) < 0.5:
                return img
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return img

    # ── recognition ─────────────────────────────────────────

    # def _ocr(self, image):
    #     if self._reader is None:
    #         return "", 0.0
    #     try:
    #         results = self._reader.readtext(image, detail=1, paragraph=True)
    #         texts, confs = [], []
    #         # for _, text, conf in results:
    #         #     if conf >= OCR_CONFIDENCE_THRESHOLD:
    #         #         texts.append(text)
    #         #         confs.append(conf)
    #         for item in results:
    #             if len(item) == 3:
    #                 _, text, conf = item
    #             else:
    #                 text = item[1]
    #                 conf = 0.5
    #         return " ".join(texts), float(np.mean(confs)) if confs else 0.0
    #     except Exception as exc:
    #         logger.error("OCR failed: %s", exc)
    #         return "", 0.0
    def _ocr(self, image):

        if self._reader is None:
            return "", 0.0

        try:
            results = self._reader.readtext(image, detail=1, paragraph=True)

            texts = []
            confs = []

            for item in results:

                if len(item) == 3:
                    _, text, conf = item
                else:
                    text = item[1]
                    conf = 0.5

                if conf >= OCR_CONFIDENCE_THRESHOLD:
                    texts.append(text)
                    confs.append(conf)

            combined = " ".join(texts)
            avg = float(np.mean(confs)) if confs else 0.0

            return combined, avg

        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            return "", 0.0

    def read_full(self, frame):

        results = self._reader.readtext(frame, detail=1, paragraph=True)

        texts = []
        confs = []

        for item in results:
            if len(item) == 3:
                _, text, conf = item
                texts.append(text)
                confs.append(conf)

        full_text = " ".join(texts)

        avg_conf = sum(confs) / len(confs) if confs else 0

        return full_text, avg_conf