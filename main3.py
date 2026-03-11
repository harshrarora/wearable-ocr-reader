#!/usr/bin/env python3
"""

       WEARABLE  AI  READING  CAP  —  Main Pipeline
 Camera → Perception → Intelligence → Interaction

  Press 'q' to quit  |  'r' to reset

"""

import cv2
import time
import hashlib
import logging

import pyttsx3
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)

# ── Bootstrap logging before any other import ──────────────
from utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("main")

import config
from camera.camera_manager import CameraManager
from perception.stability import StabilityDetector
from perception.text_detector import TextDetector
from perception.document_detector import DocumentDetector
from perception.finger_tracker import FingerTracker
from intelligence.intent_resolver import IntentResolver, Mode
from intelligence.ocr_engine import OCREngine
from intelligence.ocr_fusion import OCRFusion
from intelligence.text_cleaner import TextCleaner
from interaction.guidance import GuidanceEngine
from interaction.tts_manager import TTSManager
from interaction.state_machine import StateMachine, SystemState


class WearableReader:
    """Top-level application: wires every module together."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("   Wearable AI Reading Cap — initialising")
        logger.info("=" * 60)

        # ── Shared EasyOCR reader (heavy — load once) ──────
        self._ocr_reader = self._load_easyocr()

        # ── Modules ────────────────────────────────────────
        self.camera           = CameraManager()
        self.stability        = StabilityDetector()
        self.text_detector    = TextDetector(reader=self._ocr_reader)
        self.doc_detector     = DocumentDetector()
        self.finger_tracker   = FingerTracker()
        self.intent           = IntentResolver()
        self.ocr_engine       = OCREngine(reader=self._ocr_reader)
        self.fusion           = OCRFusion()
        self.cleaner          = TextCleaner()
        self.guidance         = GuidanceEngine()
        #self.tts              = TTSManager()
        self.last_spoken      = ""
        self.last_speak_time  = 0
        self.sm               = StateMachine()

        self._frame_n = 0
        self._running = False
        logger.info("All modules ready")

    # ── EasyOCR bootstrap ──────────────────────────────────

    @staticmethod
    def _load_easyocr():
        try:
            import easyocr
            logger.info("Loading EasyOCR model …")
            reader = easyocr.Reader(config.OCR_LANGUAGES,
                                    gpu=config.OCR_GPU, verbose=False)
            logger.info("EasyOCR ready")
            return reader
        except ImportError:
            logger.error("easyocr not installed → OCR disabled")
        except Exception as exc:
            logger.error("EasyOCR init failed: %s", exc)
        return None

    # ══════════════════════════════════════════════════════════
    #  MAIN  LOOP
    # ══════════════════════════════════════════════════════════

    def start(self):
        self.camera.start()
        #self.tts.start()
        #self.tts.say("Wearable reader ready")
        self._running = True
        logger.info("System running")
        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
        finally:
            self.stop()

    def _loop(self):

        import os

        image_folder = "test_images"
        image_files = sorted(os.listdir(image_folder))

        for img_name in image_files:

            t0 = time.time()

            # 1 ── capture (from image instead of webcam) ─────────────
            path = os.path.join(image_folder, img_name)

            frame = cv2.imread(path)

            if frame is None:
                continue

            print("Processing image:", img_name)

            self._frame_n += 1

            # 2 ── state timeouts ──────────────────────────
            tout = self.sm.check_timeouts()
            if tout:
                self.sm.transition(tout)

            # 4 ── stability ──────────────────────────────
            # stable, motion = self.stability.update(frame)
            stable = True
            motion = 0

            # 5 ── skip heavy ops on some frames ──────────
            skip = (self._frame_n % (config.PERCEPTION_SKIP_FRAMES + 1) != 0)
            if skip and not stable:
                self._show(frame, f"MOTION {motion:.1f}")
                self._pace(t0)
                continue

            # 6 ── document detection ─────────────────────
            working = frame
            warped = None

            if config.DOCUMENT_DETECTION_ENABLED:
                corners, warped = self.doc_detector.detect(frame)
                if warped is not None:
                    working = warped

            # 7 ── text detection ─────────────────────────
            text_boxes = self.text_detector.detect(working)

            # 8 ── finger detection ───────────────────────
            fingertip, _ = self.finger_tracker.detect(frame)

            # 9 ── intent ─────────────────────────────────
            # mode, target = self.intent.resolve(
            #     stable=stable,
            #     text_boxes=text_boxes,
            #     fingertip=fingertip,
            #     frame_shape=frame.shape,
            #     is_speaking=False,
            # )
            mode = Mode.AUTO_READ
            target = None

            # 10 ── act ───────────────────────────────────
            if mode == Mode.GUIDANCE:
                self._do_guidance(text_boxes, frame.shape, stable)
            elif mode == Mode.AUTO_READ:
                self._do_auto_read(working, text_boxes)
            elif mode == Mode.FINGER_READ:
                self._do_finger_read(working, target)
            elif mode == Mode.IDLE:
                self.sm.transition(SystemState.IDLE)

            # 11 ── debug overlay ─────────────────────────
            self._show(frame, mode, text_boxes, fingertip, stable, motion)

            # pause so speech finishes before next image
            time.sleep(3)

    # ── mode handlers ──────────────────────────────────────

    def _do_guidance(self, boxes, shape, stable):
        self.sm.transition(SystemState.GUIDANCE)
        cue = self.guidance.analyze(boxes, shape, stable)
        if cue:
            #self.tts.say_now(cue)
            print("GUIDANCE:", cue)


    # def _do_auto_read(self, frame, boxes):
    #     if not boxes:
    #         return
    #
    #     # collect text (re-use detector results when available)
    #     has_text = all(b.get("text") for b in boxes)
    #     if has_text:
    #         ordered = sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    #         raw = " ".join(b["text"] for b in ordered
    #                        if b["confidence"] >= config.OCR_CONFIDENCE_THRESHOLD)
    #         avg_c = sum(b["confidence"] for b in ordered) / len(ordered)
    #     else:
    #         raw, avg_c = self.ocr_engine.read_boxes(frame, boxes)
    #
    #     if not raw:
    #         return
    #
    #     self.fusion.add_result(raw, avg_c)
    #
    #     h = hashlib.md5(raw.encode()).hexdigest()[:8]
    #     if self.fusion.is_ready and self.sm.can_read(h):
    #         fused, fc = self.fusion.fuse()
    #         clean = self.cleaner.clean(fused)
    #         if clean and len(clean) > 3:
    #             logger.info("AUTO READ [%.2f]: %s", fc, clean[:80])
    #             self.sm.transition(SystemState.AUTO_READ)
    #             self.tts.say_now("Reading")
    #             for chunk in self.cleaner.chunk_for_speech(clean):
    #                 self.tts.say(chunk)
    #             self.sm.mark_read(h)
    #             self.fusion.clear()
    # def _do_auto_read(self, frame,boxes):
    #     #raw, avg_c = self.ocr_engine.read_boxes(frame, boxes)
    #     raw, avg_c = self.ocr_engine.read_full(frame)
    #     #raw, avg_c = self.ocr_engine._ocr(frame)
    #
    #     clean = self.cleaner.clean(raw)
    #
    #     if clean and len(clean) > 3 and clean != self.last_spoken:
    #         print("READING:", clean)
    #
    #         self.last_spoken = clean
    #
    #         # tts_engine.say(clean)
    #         # tts_engine.runAndWait()
    #         import threading
    #
    #         # def speak_async(text):
    #         #     tts_engine.say(text)
    #         #     tts_engine.runAndWait()
    #         import threading
    #
    #         tts_lock = threading.Lock()
    #
    #         def speak_async(text):
    #
    #             if not tts_lock.acquire(blocking=False):
    #                 return  # already speaking
    #
    #             try:
    #                 tts_engine.say(text)
    #                 tts_engine.runAndWait()
    #             finally:
    #                 tts_lock.release()
    #
    #         threading.Thread(target=speak_async, args=(clean,), daemon=True).start()

    # def _do_auto_read(self, frame, boxes):
    #     raw, avg_c = self.ocr_engine.read_boxes(frame, boxes)
    #     raw = raw.upper()
    #
    #     clean = self.cleaner.clean(raw)
    #
    #     if not clean or len(clean) < 4:
    #         return
    #
    #     now = time.time()
    #
    #     # wait 4 seconds before speaking again
    #     if now - self.last_speak_time < 1:
    #         return
    #
    #     if clean == self.last_spoken:
    #         return
    #
    #     print("READING:", clean)
    #
    #     self.last_spoken = clean
    #     self.last_speak_time = now
    #
    #     # try:
    #     #     tts_engine.stop()
    #     #     tts_engine.say(clean)
    #     #     tts_engine.runAndWait()
    #     # except RuntimeError:
    #     #     pass
    #     print("READING:", clean)
    #
    #     try:
    #         global tts_engine
    #         tts_engine.stop()
    #         tts_engine = pyttsx3.init()
    #         tts_engine.setProperty("rate", 160)
    #         tts_engine.say(clean)
    #         tts_engine.runAndWait()
    #         time.sleep(1)
    #     except Exception as e:
    #         print("TTS error:", e)

    def _do_auto_read(self, frame, boxes):

        raw, avg_c = self.ocr_engine.read_boxes(frame, boxes)
        raw = raw.upper()

        clean = self.cleaner.clean(raw)

        if not clean or len(clean) < 6:
            return

        now = time.time()

        # prevent speaking too often
        if now - self.last_speak_time < 2:
            return

        # only speak if text changed a lot
        similarity = 0
        if self.last_spoken:
            similarity = sum(
                a == b for a, b in zip(clean, self.last_spoken)
            ) / min(len(clean), len(self.last_spoken))

        if similarity > 0.85:
            return

        print("READING:", clean)

        self.last_spoken = clean
        self.last_speak_time = now

        # FIX: Don't reinitialize engine - just reuse it
        try:
            global tts_engine
            tts_engine.say(clean)
            tts_engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
            # If it fails, reinitialize as fallback
            try:
                tts_engine = pyttsx3.init()
                tts_engine.setProperty("rate", 160)
                tts_engine.say(clean)
                tts_engine.runAndWait()
            except:
                pass


    def _do_finger_read(self, frame, box):
        if box is None:
            return
        self.sm.transition(SystemState.FINGER_READ)

        text = box.get("text") or ""
        conf = box.get("confidence", 0)
        if not text:
            text, conf = self.ocr_engine.read_region(frame, box["bbox"])

        clean = self.cleaner.clean(text)
        if clean:
            h = hashlib.md5(clean.encode()).hexdigest()[:8]
            if self.sm.can_read(h):
                logger.info("FINGER READ [%.2f]: %s", conf, clean[:80])
                self.tts.say(clean)
                self.sm.mark_read(h)

    # ── debug display ──────────────────────────────────────

    def _show(self, frame, mode, boxes=None, tip=None,
              stable=False, motion=0.0):
        if not config.DEBUG_DISPLAY:
            return
        vis = frame.copy()
        h, w = vis.shape[:2]

        if boxes:
            for b in boxes:
                x, y, bw, bh = b["bbox"]
                c = (0, 255, 0) if stable else (0, 255, 255)
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), c, 2)
                if b.get("text"):
                    cv2.putText(vis, b["text"][:25], (x, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        if tip:
            cv2.circle(vis, tip, 12, (0, 0, 255), -1)

        # crosshair
        cx, cy = w // 2, h // 2
        cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
        cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)

        # status bar
        bar = (
            f"Mode: {mode}  Stable: {'Y' if stable else 'N'}  "
            f"Motion: {motion:.1f}  Boxes: {len(boxes) if boxes else 0}  "
            f"FPS: {self.camera.fps:.0f}"
        )
        cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(vis, bar, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        cv2.imshow("Wearable AI Reader", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._running = False
        elif key == ord("r"):
            self.sm.force_state(SystemState.IDLE)
            self.fusion.clear()
            self.guidance.reset_cooldowns()
            self.stability.reset()
            logger.info("Manual reset")

    # ── helpers ────────────────────────────────────────────

    @staticmethod
    def _pace(t0):
        elapsed = time.time() - t0
        time.sleep(max(0, config.MAIN_LOOP_DELAY - elapsed))

    def stop(self):
        logger.info("Shutting down …")
        self._running = False
        #self.tts.say("Shutting down")
        time.sleep(1)
        self.camera.stop()
        #self.tts.stop()
        self.finger_tracker.release()
        cv2.destroyAllWindows()
        logger.info("Goodbye.")


# ══════════════════════════════════════════════════════════
#  ENTRY  POINT
# ══════════════════════════════════════════════════════════

def main():
    print(__doc__)
    WearableReader().start()


if __name__ == "__main__":
    main()