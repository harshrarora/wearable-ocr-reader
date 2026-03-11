"""
TTS Manager
───────────
Text-to-speech in a dedicated daemon thread.
Exposes ``say()`` (queued) and ``say_now()`` (clears queue first).
The ``is_speaking`` flag lets the main loop pause perception.
"""

import threading
import queue
import time
import logging
from config import TTS_ENGINE, TTS_RATE, TTS_VOLUME

logger = logging.getLogger(__name__)


class TTSManager:

    def __init__(self):
        self._engine = None
        self._q = queue.Queue()
        self._speaking = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._init_engine()

    # ── init ────────────────────────────────────────────────

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", TTS_RATE)
            self._engine.setProperty("volume", TTS_VOLUME)
            voices = self._engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break
            logger.info("pyttsx3 TTS ready")
        except Exception as exc:
            logger.error("TTS init failed: %s", exc)

    # ── public ──────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text, priority=1):
        if not text or not text.strip():
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)

    def say_now(self, text):
        if not text or not text.strip():
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)

    def clear_queue(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    @property
    def is_speaking(self):
        with self._lock:
            return self._speaking

    @property
    def queue_size(self):
        return self._q.qsize()

    def stop(self):
        self._running = False
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        logger.info("TTS stopped")

    # ── worker ──────────────────────────────────────────────

    def _worker(self):
        while self._running:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            text, _ = item
            with self._lock:
                self._speaking = True
            self._speak(text)
            with self._lock:
                self._speaking = False

    def _speak(self, text):
        if self._engine is None:
            logger.info("[TTS-sim] %s", text)
            time.sleep(len(text) * 0.04)
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            logger.error("TTS error: %s", exc)