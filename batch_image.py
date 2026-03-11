#!/usr/bin/env python3
"""
Image Batch Tester - processes all images in test_images/ folder
Optimized for testing OCR + TTS on multiple different images
"""

import cv2
import time
import logging
import pyttsx3
import os

# ── Bootstrap logging ───────────────────────────────────────
from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger("batch_test")

import config
from perception.text_detector import TextDetector
from perception.document_detector import DocumentDetector
from intelligence.ocr_engine import OCREngine
from intelligence.text_cleaner import TextCleaner

# Initialize TTS once globally
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)
tts_engine.setProperty("volume", 1.0)


class BatchImageTester:

    def __init__(self):
        logger.info("=" * 60)
        logger.info("   Batch Image Tester")
        logger.info("=" * 60)

        # Load EasyOCR
        import easyocr
        logger.info("Loading EasyOCR model...")
        self.reader = easyocr.Reader(config.OCR_LANGUAGES,
                                     gpu=config.OCR_GPU,
                                     verbose=False)
        logger.info("EasyOCR ready")

        # Modules
        self.text_detector = TextDetector(reader=self.reader)
        self.doc_detector = DocumentDetector()
        self.ocr_engine = OCREngine(reader=self.reader)
        self.cleaner = TextCleaner()

    def process_folder(self, folder="test_images"):
        """Process all images in folder"""

        if not os.path.exists(folder):
            logger.error(f"Folder '{folder}' not found")
            return

        image_files = sorted([f for f in os.listdir(folder)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not image_files:
            logger.warning(f"No images found in {folder}")
            return

        logger.info(f"Found {len(image_files)} images")
        print()

        for i, img_name in enumerate(image_files, 1):
            print("=" * 60)
            print(f"IMAGE {i}/{len(image_files)}: {img_name}")
            print("=" * 60)

            path = os.path.join(folder, img_name)
            frame = cv2.imread(path)

            if frame is None:
                logger.warning(f"Failed to load {img_name}")
                continue

            # Process this image
            self._process_image(frame, img_name)

            # Pause between images
            print("\n⏸ Waiting 3 seconds before next image...\n")
            time.sleep(3)

        print("=" * 60)
        print("✓ All images processed")
        print("=" * 60)

    def _process_image(self, frame, name):
        """Process a single image"""

        # 1. Optional: Document detection + perspective correction
        working = frame
        if config.DOCUMENT_DETECTION_ENABLED:
            corners, warped = self.doc_detector.detect(frame)
            if warped is not None:
                working = warped
                logger.info("✓ Document boundary detected, perspective corrected")

        # 2. Text detection
        text_boxes = self.text_detector.detect(working)
        logger.info(f"Found {len(text_boxes)} text regions")

        if not text_boxes:
            print("⚠ No text detected in image\n")
            return

        # 3. OCR
        raw, avg_conf = self.ocr_engine.read_boxes(working, text_boxes)
        logger.info(f"OCR confidence: {avg_conf:.2f}")

        # 4. Clean text
        clean = self.cleaner.clean(raw)

        if not clean or len(clean) < 6:
            print("⚠ No valid text after cleaning\n")
            return

        # 5. Display
        print(f"\n📄 EXTRACTED TEXT ({len(clean)} chars):")
        print("-" * 60)
        print(clean)
        print("-" * 60)

        # 6. Speak
        print("\n🔊 Speaking...")
        try:
            global tts_engine
            tts_engine.say(clean)
            tts_engine.runAndWait()
            print("✓ Finished speaking\n")
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            print(f"✗ TTS error: {e}\n")

        # 7. Optional: Show visualization
        if config.DEBUG_DISPLAY:
            self._show(frame, text_boxes)

    def _show(self, frame, boxes):
        """Display frame with bounding boxes"""
        vis = frame.copy()

        for b in boxes:
            x, y, w, h = b["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if b.get("text"):
                cv2.putText(vis, b["text"][:30], (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(f"Batch Tester", vis)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()


def main():
    tester = BatchImageTester()
    tester.process_folder("test_images")


if __name__ == "__main__":
    main()