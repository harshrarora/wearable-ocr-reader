"""
Multi-Shot Capture
──────────────────
Burst mode photography for improved OCR accuracy.

Takes N quick photos, runs OCR on each, fuses results using temporal voting.
Compensates for motion blur, lighting variations, and transient errors.
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiShotCapture:
    """
    Burst photography with OCR fusion.
    
    Workflow:
        1. Capture N frames (with interval between shots)
        2. Run OCR on each frame independently
        3. Fuse results using OCR fusion (majority voting)
        4. Return best combined text
    
    Usage:
        msc = MultiShotCapture(camera, ocr_engine, fusion_engine)
        text, confidence = msc.capture_and_read(num_shots=3)
    """

    def __init__(self, camera_manager, ocr_engine, ocr_fusion, text_cleaner,
                 shot_interval=0.3, save_shots=False, shots_dir="./shots"):
        """
        Args:
            camera_manager: CameraManager instance
            ocr_engine: OCREngine instance
            ocr_fusion: OCRFusion instance (for temporal voting)
            text_cleaner: TextCleaner instance
            shot_interval: Seconds between shots (default 0.3)
            save_shots: If True, save captured frames to disk
            shots_dir: Directory to save shots (if save_shots=True)
        """
        self.camera = camera_manager
        self.ocr_engine = ocr_engine
        self.fusion = ocr_fusion
        self.cleaner = text_cleaner
        
        self.shot_interval = shot_interval
        self.save_shots = save_shots
        self.shots_dir = Path(shots_dir)
        
        if self.save_shots:
            self.shots_dir.mkdir(exist_ok=True)
            logger.info(f"Multi-shot saving to: {self.shots_dir}")

    def capture_and_read(self, num_shots=3, text_detector=None):
        """
        Capture multiple frames and fuse OCR results.
        
        Args:
            num_shots: Number of frames to capture (default 3)
            text_detector: Optional TextDetector to find regions first
        
        Returns:
            (fused_text, avg_confidence) tuple
        """
        logger.info(f"Starting {num_shots}-shot capture...")
        
        frames = []
        ocr_results = []
        confidences = []
        
        # Capture frames
        for i in range(num_shots):
            frame = self._capture_frame()
            if frame is None:
                logger.warning(f"Shot {i+1} failed to capture")
                continue
            
            frames.append(frame)
            
            # Optional: save to disk
            if self.save_shots:
                timestamp = int(time.time() * 1000)
                filepath = self.shots_dir / f"shot_{timestamp}_{i}.jpg"
                cv2.imwrite(str(filepath), frame)
            
            # Wait before next shot (except last)
            if i < num_shots - 1:
                time.sleep(self.shot_interval)
        
        if not frames:
            logger.error("No frames captured")
            return "", 0.0
        
        logger.info(f"Captured {len(frames)} frames, running OCR...")
        
        # Detect text regions in first frame (reuse for all frames)
        text_boxes = None
        if text_detector:
            logger.info("Detecting text regions...")
            text_boxes = text_detector.detect(frames[0])
            if text_boxes:
                logger.info(f"Found {len(text_boxes)} text regions")
            else:
                logger.warning("No text regions detected")
        
        # Run OCR on each frame
        for i, frame in enumerate(frames):
            if text_boxes:
                # Use detected text boxes (much better accuracy)
                text, conf = self.ocr_engine.read_boxes(frame, text_boxes)
            else:
                # Full frame OCR (fallback, lower accuracy)
                text, conf = self.ocr_engine.read_full(frame)
            
            ocr_results.append(text)
            confidences.append(conf)
            logger.info(f"Shot {i+1} OCR [conf={conf:.2f}]: '{text[:80] if text else '(empty)'}'")
        
        # Check if we got ANY text
        if not any(ocr_results):
            logger.warning("All OCR results empty - no text detected in any frame")
            return "", 0.0
        
        # Fuse results using temporal voting
        logger.info("Fusing OCR results...")
        fused_text = self._fuse_results(ocr_results)
        
        # Clean final text
        clean_text = self.cleaner.clean(fused_text)
        
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        logger.info(f"Final text [conf={avg_conf:.2f}]: {clean_text[:80]}")
        return clean_text, avg_conf

    # ── Internal helpers ────────────────────────────────────

    def _capture_frame(self):
        """Capture single frame from camera"""
        # Try multiple times (camera might be busy)
        for attempt in range(3):
            frame = self.camera.get_frame()
            if frame is not None:
                return frame.copy()
            time.sleep(0.05)
        return None

    def _fuse_results(self, ocr_results):
        """
        Fuse multiple OCR results using temporal voting.
        
        Uses the existing OCRFusion module with word-level majority voting.
        """
        # Clear fusion buffer
        self.fusion.clear()
        
        # Add all results to fusion buffer
        for text in ocr_results:
            self.fusion.add_result(text)
        
        # Get fused result (returns tuple: text, confidence)
        fused, conf = self.fusion.fuse()
        return fused


# ══════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.logger import setup_logger
    setup_logger()
    
    from camera.camera_manager import CameraManager
    from intelligence.ocr_engine import OCREngine
    from intelligence.ocr_fusion import OCRFusion
    from intelligence.text_cleaner import TextCleaner
    import config
    
    print("Multi-Shot Capture Test")
    print("Point camera at text and press ENTER to capture...")
    print("-" * 50)
    
    # Initialize modules
    camera = CameraManager()
    camera.start()
    
    # Load EasyOCR
    import easyocr
    reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.OCR_GPU, verbose=False)
    
    ocr_engine = OCREngine(reader=reader)
    fusion = OCRFusion(buffer_size=5)
    cleaner = TextCleaner()
    
    msc = MultiShotCapture(
        camera, ocr_engine, fusion, cleaner,
        shot_interval=0.5,
        save_shots=True,
        shots_dir="./test_shots"
    )
    
    try:
        while True:
            input("Press ENTER to capture 3 shots (or Ctrl+C to quit)...")
            
            text, conf = msc.capture_and_read(num_shots=3)
            print(f"\n✓ RESULT [confidence={conf:.2f}]:")
            print(f"  {text}\n")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        camera.stop()
