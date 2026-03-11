import time
import logging

logger = logging.getLogger(__name__)

class SystemState:
    IDLE = "IDLE"
    GUIDANCE = "GUIDANCE"
    AUTO_READ = "AUTO_READ"
    FINGER_READ = "FINGER_READ"
    SPEAKING = "SPEAKING"
    COOLDOWN = "COOLDOWN"


class StateMachine:

    def __init__(self):
        self._state = SystemState.IDLE
        self._entered = time.time()
        self._last_read = 0.0
        self._read_hash = None

    @property
    def state(self):
        return self._state

    def transition(self, new):
        if new == self._state:
            return True
        logger.info(f"State {self._state} -> {new}")
        self._state = new
        self._entered = time.time()
        return True

    # def can_read(self, text_hash=None):
    #     now = time.time()
    #     if now - self._last_read < 3:
    #         return False
    #     return True
    def can_read(self, text_hash=None):

        now = time.time()

        # prevent too frequent speaking
        if now - self._last_read < 8:
            return False

        # prevent repeating same text
        if text_hash and text_hash == self._read_hash:
            return False

        return True

    def mark_read(self, text_hash=None):
        self._last_read = time.time()
        self._read_hash = text_hash

    def check_timeouts(self):
        return None

    def force_state(self, state):
        self._state = state
        self._entered = time.time()