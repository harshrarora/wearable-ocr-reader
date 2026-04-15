#!/usr/bin/env python3
"""

       WEARABLE  AI  READING  CAP  —  Production System
 Camera → Voice Control → Multi-Mode Reading → TTS

  MODES:
    • TRIGGER MODE: Say "capture" for multi-shot burst reading
    • CONTINUOUS MODE: Auto-scan and read when stable

  VOICE COMMANDS:
    • "capture" / "read this" → Multi-shot capture + read
    • "continuous" → Switch to auto-scan mode
    • "stop" / "pause" → Pause reading
    • "repeat" → Re-read last text
    • "exit" / "shutdown" → Quit application

"""

import cv2
import time
import hashlib
import logging
import threading

import pyttsx3
# Don't create global engine - create per-thread instead

# ── Bootstrap logging ──────────────────────────────────────
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
from interaction.voice_controller import VoiceController, VoiceCommand
from camera.multi_shot_capture import MultiShotCapture
from intelligence.currency_detector import CurrencyDetector


class WearableReader:
    """
    Production Wearable AI Reading System

    Two operating modes:
    1. TRIGGER MODE (default): Voice-activated multi-shot capture
    2. CONTINUOUS MODE: Auto-scan when stable + text detected
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("   Wearable AI Reading Cap — Production v2.0")
        logger.info("=" * 60)

        # ── Load OCR model ─────────────────────────────────
        self._ocr_reader = self._load_easyocr()

        # ── Core modules ───────────────────────────────────
        self.camera = CameraManager()
        self.stability = StabilityDetector()
        self.text_detector = TextDetector(reader=self._ocr_reader)
        self.doc_detector = DocumentDetector()
        self.finger_tracker = FingerTracker()
        self.intent = IntentResolver()
        self.ocr_engine = OCREngine(reader=self._ocr_reader)
        self.fusion = OCRFusion(buffer_size=config.FUSION_BUFFER_SIZE)
        self.cleaner = TextCleaner()
        self.guidance = GuidanceEngine()
        self.sm = StateMachine()

        # ── Multi-shot capture ─────────────────────────────
        self.multi_shot = MultiShotCapture(
            camera_manager=self.camera,
            ocr_engine=self.ocr_engine,
            ocr_fusion=self.fusion,
            text_cleaner=self.cleaner,
            shot_interval=config.MULTISHOT_INTERVAL,
            save_shots=config.MULTISHOT_SAVE_SHOTS,
            shots_dir=config.MULTISHOT_SHOTS_DIR
        )

        # ── Voice control ──────────────────────────────────
        self.voice = None
        if config.VOICE_ENABLED:
            try:
                self.voice = VoiceController(
                    language=config.VOICE_LANGUAGE,
                    timeout=config.VOICE_TIMEOUT,
                    phrase_time_limit=config.VOICE_PHRASE_LIMIT
                )
                logger.info("Voice control enabled")
            except Exception as e:
                logger.warning(f"Voice control disabled: {e}")
                self.voice = None

        # ── State tracking ─────────────────────────────────
        self.current_mode = config.DEFAULT_MODE  # "trigger" or "continuous"
        self.last_spoken = ""
        self.last_speak_time = 0
        self.is_speaking = False
        self._frame_n = 0
        self._running = False

        # Currency detector (SAFE)
        self.currency = None
        self.last_currency_time = 0

        if config.CURRENCY_ENABLED:
            try:
                self.currency = CurrencyDetector(api_key=config.CURRENCY_API_KEY)
                logger.info("Currency detection enabled")
            except Exception as e:
                logger.warning(f"Currency disabled: {e}")

        logger.info(f"Default mode: {self.current_mode.upper()}")
        logger.info("All modules ready")

    # ══════════════════════════════════════════════════════════
    #  MAIN  LOOP
    # ══════════════════════════════════════════════════════════

    def start(self):
        """Start system: camera, voice, main loop"""
        self.camera.start()
        cv2.namedWindow("Wearable AI Reader", cv2.WINDOW_NORMAL)

        if self.voice:
            self.voice.start()
            self._announce("System ready. Say capture to read.")
        else:
            self._announce("System ready. Voice control disabled.")

        self._running = True
        logger.info("System running")

        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
        finally:
            self.stop()

    def _loop(self):
        """Main event loop"""
        while self._running:
            t0 = time.time()

            # 1 ── Process voice commands ──────────────────
            self._process_voice_commands()

            # 2 ── Capture frame ───────────────────────────
            frame = self.camera.get_frame()
            if frame is None:
                print("NO FRAME")
                time.sleep(0.01)
                continue
            self._frame_n += 1

            # 3 ── Mode-specific processing ────────────────
            if self.current_mode == "trigger":
                # In trigger mode, just show preview (commands handled above)
                self._show(frame, "TRIGGER MODE - Say 'capture' to read")

            elif self.current_mode == "continuous":
                # Continuous auto-scan mode (original behavior)
                self._continuous_mode_cycle(frame)

            # 4 ── Pace ────────────────────────────────────
            self._pace(t0)

    # ── Mode handlers ──────────────────────────────────────

    def _process_voice_commands(self):
        """Check for voice commands and execute"""
        if not self.voice:
            return

        cmd = self.voice.get_command(block=False)
        if not cmd:
            return

        logger.info(f"Voice command: {cmd}")

        if cmd == VoiceCommand.CAPTURE:
            self._handle_capture_command()

        elif cmd == VoiceCommand.CONTINUOUS:
            self._switch_mode("continuous")

        elif cmd == VoiceCommand.STOP or cmd == VoiceCommand.PAUSE:
            self._switch_mode("trigger")

        elif cmd == VoiceCommand.REPEAT:
            self._repeat_last()

        elif cmd == VoiceCommand.COUNT_MONEY:
            self._count_money()

        elif cmd == VoiceCommand.EXIT or cmd == VoiceCommand.SHUTDOWN:
            self._announce("Shutting down")
            self._running = False

    def _handle_capture_command(self):
        """Execute multi-shot capture + OCR + TTS"""
        logger.info("Executing multi-shot capture...")
        self._announce_sync("Capturing")

        # Run directly on main thread (NOT in background)
        self._do_multishot_read()

    def _do_multishot_read(self):
        """Multi-shot capture with OCR fusion"""
        try:
            # Capture and fuse (pass text_detector for region detection)
            text, conf = self.multi_shot.capture_and_read(
                num_shots=config.MULTISHOT_COUNT,
                text_detector=self.text_detector
            )

            if not text or len(text) < 3:
                logger.warning("No text detected in multi-shot capture")
                self._announce_sync("No text found. Switching to continuous mode.")
                time.sleep(1)
                self._switch_mode("continuous")
                return

            # Speak result (SYNCHRONOUSLY - no threading)
            logger.info(f"READING [conf={conf:.2f}]: {text}")
            self.last_spoken = text
            self.last_speak_time = time.time()

            self._speak_sync(text)  # Use synchronous version

        except Exception as e:
            logger.error(f"Multi-shot read error: {e}")
            self._announce_sync("Reading error. Try continuous mode.")
            time.sleep(1)
            self._switch_mode("continuous")

    def _continuous_mode_cycle(self, frame):
        """Original continuous auto-scan mode"""
        # State timeouts
        tout = self.sm.check_timeouts()
        if tout:
            self.sm.transition(tout)

        # Stability check
        stable, motion = self.stability.update(frame)

        # Skip heavy ops on some frames
        skip = (self._frame_n % (config.PERCEPTION_SKIP_FRAMES + 1) != 0)
        if skip and not stable:
            self._show(frame, f"CONTINUOUS MODE - MOTION {motion:.1f}")
            return

        # Document detection (optional)
        working = frame
        if config.DOCUMENT_DETECTION_ENABLED:
            corners, warped = self.doc_detector.detect(frame)
            if warped is not None:
                working = warped

        # Text detection
        text_boxes = self.text_detector.detect(working)

        # Finger detection (optional)
        fingertip = None
        if config.FINGER_TRACKING_ENABLED:
            fingertip, _ = self.finger_tracker.detect(frame)

        # Intent resolution
        mode, target = self.intent.resolve(
            stable=stable,
            text_boxes=text_boxes,
            fingertip=fingertip,
            frame_shape=frame.shape,
            is_speaking=self.is_speaking,
        )

        # Act on intent
        if mode == Mode.GUIDANCE:
            self._do_guidance(text_boxes, frame.shape, stable)
        elif mode == Mode.AUTO_READ:
            self._do_auto_read(working, text_boxes)
        elif mode == Mode.FINGER_READ:
            self._do_finger_read(working, target)
        elif mode == Mode.IDLE:
            self.sm.transition(SystemState.IDLE)

        # Debug display
        self._show(frame, f"CONTINUOUS MODE - {mode}", text_boxes, fingertip, stable, motion)

    def _do_guidance(self, boxes, shape, stable):
        """Spatial positioning guidance"""
        if not config.GUIDANCE_ENABLED:
            return

        self.sm.transition(SystemState.GUIDANCE)
        cue = self.guidance.analyze(boxes, shape, stable)
        if cue:
            logger.debug(f"GUIDANCE: {cue}")
            # Optionally speak guidance (can be annoying, so disabled by default)
            # self._speak(cue)

    def _do_auto_read(self, frame, boxes):
        """Continuous mode auto-read when stable"""
        raw, avg_c = self.ocr_engine.read_boxes(frame, boxes)
        clean = self.cleaner.clean(raw)

        if not clean or len(clean) < 6:
            return

        now = time.time()

        # Cooldown
        if now - self.last_speak_time < 2:
            return

        # Similarity check (avoid re-reading same content)
        if self.last_spoken:
            similarity = sum(
                a == b for a, b in zip(clean, self.last_spoken)
            ) / min(len(clean), len(self.last_spoken))
            if similarity > 0.85:
                return

        logger.info(f"AUTO-READ [conf={avg_c:.2f}]: {clean}")

        self.last_spoken = clean
        self.last_speak_time = now

        threading.Thread(target=lambda: self._speak(clean), daemon=True).start()

    def _do_finger_read(self, frame, box):
        """Finger-pointing selective read (experimental)"""
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
                logger.info(f"FINGER READ [conf={conf:.2f}]: {clean}")
                self._speak(clean)
                self.sm.mark_read(h)

    # ── Mode switching ─────────────────────────────────────

    def _switch_mode(self, new_mode):
        """Switch between trigger and continuous modes"""
        if new_mode == self.current_mode:
            return

        self.current_mode = new_mode
        logger.info(f"Mode switched to: {new_mode.upper()}")

        if new_mode == "continuous":
            self._announce("Continuous mode")
        else:
            self._announce("Trigger mode")

    def _repeat_last(self):
        """Re-speak last read text"""
        if self.last_spoken:
            logger.info(f"REPEAT: {self.last_spoken}")
            self._speak(self.last_spoken)
        else:
            self._announce("Nothing to repeat")

    # ── TTS helpers ────────────────────────────────────────

    def _speak_sync(self, text):
        """Speak text using TTS (creates new engine each time - WORKS on Windows)"""
        try:
            self.is_speaking = True
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self.is_speaking = False

    def _speak(self, text):
        """Speak text in background thread"""

        def speak_async():
            try:
                self.is_speaking = True
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                self.is_speaking = False

        threading.Thread(target=speak_async, daemon=True).start()

    def _announce_sync(self, message):
        """Quick system announcement (synchronous)"""
        self._speak_sync(message)

    def _announce(self, message):
        """Quick system announcement (async)"""
        self._speak(message)

    # ── Debug display ──────────────────────────────────────

    def _show(self, frame, status, boxes=None, tip=None, stable=False, motion=0.0):
        """Debug visualization window"""
        if not config.DEBUG_DISPLAY:
            return
        # print("SHOW RUNNING")
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw text boxes
        if boxes:
            for b in boxes:
                x, y, bw, bh = b["bbox"]
                c = (0, 255, 0) if stable else (0, 255, 255)
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), c, 2)
                if b.get("text"):
                    cv2.putText(vis, b["text"][:25], (x, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        # Draw fingertip
        if tip:
            cv2.circle(vis, tip, 12, (0, 0, 255), -1)

        # Crosshair
        cx, cy = w // 2, h // 2
        cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
        cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)

        # Status bar
        cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(vis, status, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

        cv2.imshow("Wearable AI Reader", vis)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._running = False
        elif key == ord("c"):
            self._handle_capture_command()
        elif key == ord("m"):
            new_mode = "continuous" if self.current_mode == "trigger" else "trigger"
            self._switch_mode(new_mode)
        elif key == ord("r"):
            self._repeat_last()
        elif key == ord("n"):
            self._count_money()

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _load_easyocr():
        """Load EasyOCR model"""
        try:
            import easyocr
            logger.info("Loading EasyOCR model...")
            reader = easyocr.Reader(config.OCR_LANGUAGES,
                                    gpu=config.OCR_GPU, verbose=False)
            logger.info("EasyOCR ready")
            return reader
        except ImportError:
            logger.error("easyocr not installed → OCR disabled")
        except Exception as exc:
            logger.error(f"EasyOCR init failed: {exc}")
        return None

    @staticmethod
    def _pace(t0):
        """Maintain target frame rate"""
        elapsed = time.time() - t0
        time.sleep(max(0, config.MAIN_LOOP_DELAY - elapsed))

    # def _count_money(self):
    #     if not self.currency:
    #         self._announce("Currency not available")
    #         return
    #
    #     now = time.time()
    #     if now - self.last_currency_time < 3:
    #         return
    #
    #     logger.info("Counting money...")
    #     self._announce("Counting money please wait")
    #
    #     frame = self.camera.get_frame()
    #     if frame is None:
    #         self._announce("Camera error")
    #         return
    #
    #     try:
    #         total, breakdown, _ = self.currency.detect_and_count(frame)
    #
    #         result_text = self.currency.format_result(total, breakdown)
    #
    #         logger.info(f"CURRENCY: {result_text}")
    #
    #         self._announce(result_text)
    #
    #         self.last_currency_time = now
    #
    #     except Exception as e:
    #         logger.error(f"Currency error: {e}")
    #         self._announce("Currency detection failed")

    def _count_money(self):
        if not self.currency:
            self._announce_sync("Currency not available")
            return

        now = time.time()
        if now - self.last_currency_time < 3:
            return

        logger.info("Counting money...")
        self._announce_sync("Counting")

        frame = self.camera.get_frame()
        if frame is None:
            self._announce_sync("Camera error")
            return

        try:
            total, breakdown, _ = self.currency.detect_and_count(frame)

            result_text = self.currency.format_result(total, breakdown)

            logger.info(f"CURRENCY: {result_text}")

            self._announce_sync(result_text)

            self.last_currency_time = now

        except Exception as e:
            logger.error(f"Currency error: {e}")
            self._announce_sync("Currency detection failed")

    # def _count_money(self):
    #     if not self.currency:
    #         self._announce("Currency not available")
    #         return
    #
    #     now = time.time()
    #     if now - self.last_currency_time < 3:
    #         return
    #
    #     logger.info("Counting money...")
    #     self._announce("Checking money")
    #
    #     frame = self.camera.get_frame()
    #     if frame is None:
    #         self._announce("Camera error")
    #         return
    #
    #     def detect():
    #         try:
    #             total, breakdown, _ = self.currency.detect_and_count(frame)
    #
    #             result_text = self.currency.format_result(total, breakdown)
    #
    #             logger.info(f"CURRENCY: {result_text}")
    #             self._announce(result_text)
    #
    #         except Exception as e:
    #             logger.error(f"Currency error: {e}")
    #             self._announce("Currency server error")
    #
    #     # 🔥 Run in background (NON-BLOCKING)
    #     threading.Thread(target=detect, daemon=True).start()
    #
    #     self.last_currency_time = now

    def stop(self):
        """Shutdown sequence"""
        logger.info("Shutting down...")
        self._running = False
        time.sleep(1)

        if self.voice:
            self.voice.stop()

        self.camera.stop()
        self.finger_tracker.release()

        if config.DEBUG_DISPLAY:
            cv2.destroyAllWindows()

        logger.info("Goodbye.")


# ══════════════════════════════════════════════════════════
#  ENTRY  POINT
# ══════════════════════════════════════════════════════════

def main():
    print(__doc__)
    print("\nKEYBOARD SHORTCUTS:")
    print("  'c' = Capture (multi-shot)")
    print("  'm' = Toggle mode (trigger/continuous)")
    print("  'r' = Repeat last text")
    print("  'q' = Quit")
    print("-" * 60)

    WearableReader().start()


if __name__ == "__main__":
    main()
