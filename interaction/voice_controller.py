"""
Voice Controller
────────────────
Hands-free voice command recognition for wearable system.
Listens for trigger words: "capture", "read", "stop", "continuous", etc.

Uses PocketSphinx for OFFLINE speech recognition (no internet required).
"""

import logging
import threading
import queue
import time

try:
    import speech_recognition as sr
except ImportError:
    sr = None

logger = logging.getLogger(__name__)


class VoiceCommand:
    """Voice command enum"""
    CAPTURE = "capture"
    READ = "read"
    CONTINUOUS = "continuous"
    STOP = "stop"
    PAUSE = "pause"
    REPEAT = "repeat"
    EXIT = "exit"
    SHUTDOWN = "shutdown"
    COUNT_MONEY = "count_money"
    UNKNOWN = "unknown"


class VoiceController:
    """
    Continuous voice command listener using background thread.

    Usage:
        voice = VoiceController()
        voice.start()

        while True:
            cmd = voice.get_command()
            if cmd == VoiceCommand.CAPTURE:
                # Handle capture
            elif cmd == VoiceCommand.EXIT:
                break

        voice.stop()
    """

    def __init__(self, language="en-US", timeout=3.0, phrase_time_limit=2.0):
        """
        Args:
            language: Speech recognition language (en-US, hi-IN, etc.)
            timeout: Max seconds to wait for phrase start
            phrase_time_limit: Max seconds for phrase duration
        """
        if sr is None:
            raise ImportError(
                "speech_recognition not installed. "
                "Install: pip install SpeechRecognition pyaudio"
            )

        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Command queue (thread-safe)
        self.command_queue = queue.Queue()

        # Threading
        self._listening_thread = None
        self._stop_listening = False

        # Calibrate for ambient noise
        logger.info("Calibrating microphone for ambient noise (3 seconds)...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
        logger.info("Voice controller ready")

    def start(self):
        """Start listening in background thread"""
        if self._listening_thread and self._listening_thread.is_alive():
            logger.warning("Voice controller already running")
            return

        self._stop_listening = False
        self._listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listening_thread.start()
        logger.info("Voice listening started")

    def stop(self):
        """Stop listening thread"""
        self._stop_listening = True
        if self._listening_thread:
            self._listening_thread.join(timeout=2.0)
        logger.info("Voice listening stopped")

    def get_command(self, block=False, timeout=None):
        """
        Get next command from queue.

        Args:
            block: If True, wait for command. If False, return immediately.
            timeout: Max seconds to wait (only if block=True)

        Returns:
            VoiceCommand or None if queue empty (when block=False)
        """
        try:
            return self.command_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def has_command(self):
        """Check if any commands waiting in queue"""
        return not self.command_queue.empty()

    # ── Internal methods ────────────────────────────────────

    def _listen_loop(self):
        """Background thread: continuously listen for commands"""
        while not self._stop_listening:
            try:
                with self.microphone as source:
                    # Listen for audio
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_time_limit
                    )

                # OFFLINE-ONLY: Use pocketsphinx (no internet required)
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    logger.debug(f"Heard: '{text}'")
                except sr.UnknownValueError:
                    # Could not understand audio
                    continue
                except sr.RequestError as e:
                    logger.error(f"Sphinx error: {e}")
                    continue

                # Parse command
                cmd = self._parse_command(text)
                if cmd != VoiceCommand.UNKNOWN:
                    self.command_queue.put(cmd)
                    logger.info(f"Command recognized: {cmd}")

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except Exception as e:
                logger.error(f"Voice recognition error: {e}")
                time.sleep(1)  # Brief pause before retry

    def _parse_command(self, text):
        """
        Parse recognized text into VoiceCommand.

        Supports variations:
        - "capture" / "take photo" / "read this" → CAPTURE
        - "continuous" / "continuous mode" / "auto read" → CONTINUOUS
        - "stop" / "pause" → STOP
        - "repeat" / "say again" → REPEAT
        - "exit" / "shutdown" / "quit" → EXIT
        """
        text = text.lower().strip()

        # Capture triggers
        if any(word in text for word in ["capture", "take photo", "read this", "scan"]):
            return VoiceCommand.CAPTURE

        # Mode switch
        if any(word in text for word in ["continuous", "auto", "keep reading"]):
            return VoiceCommand.CONTINUOUS

        # Stop/pause
        if any(word in text for word in ["stop", "pause", "halt"]):
            return VoiceCommand.STOP

        # Repeat
        if any(word in text for word in ["repeat", "again", "say again"]):
            return VoiceCommand.REPEAT

        # Exit
        if any(word in text for word in ["exit", "shutdown", "quit", "close"]):
            return VoiceCommand.EXIT

        if any(word in text for word in ["count money", "how much", "money", "cash"]):
            return VoiceCommand.COUNT_MONEY

        return VoiceCommand.UNKNOWN


# ══════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Voice Controller Test")
    print("Say: 'capture', 'continuous', 'stop', 'repeat', or 'exit'")
    print("-" * 50)

    voice = VoiceController()
    voice.start()

    try:
        while True:
            cmd = voice.get_command(block=True, timeout=1.0)
            if cmd:
                print(f">>> COMMAND: {cmd}")
                if cmd == VoiceCommand.EXIT:
                    print("Exiting...")
                    break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        voice.stop()