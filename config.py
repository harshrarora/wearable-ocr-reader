"""
Central configuration for the Wearable AI Reading Cap.
Every tunable parameter lives here — nothing is hard-coded in modules.
"""

# ═══════════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════════
CAMERA_INDEX = 1                  # 0 = default webcam / Pi Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ═══════════════════════════════════════════════════════════
#  STABILITY  /  MOTION
# ═══════════════════════════════════════════════════════════
STABILITY_THRESHOLD = 2.0          # mean pixel motion below this → "stable"
STABILITY_FRAMES_REQUIRED = 5      # consecutive stable frames needed
MOTION_HISTORY_SIZE = 10           # rolling window for average

# ═══════════════════════════════════════════════════════════
#  TEXT  DETECTION
# ═══════════════════════════════════════════════════════════
TEXT_DETECTION_CONFIDENCE = 0.3    # minimum EasyOCR box confidence
TEXT_MIN_AREA = 500                # ignore tiny boxes (pixels²)

# ═══════════════════════════════════════════════════════════
#  DOCUMENT  DETECTION
# ═══════════════════════════════════════════════════════════
DOCUMENT_DETECTION_ENABLED = False  # try to find page boundary
PERSPECTIVE_CORRECTION = True      # warp to rectangle if found
MIN_DOCUMENT_AREA_RATIO = 0.10     # page must be ≥ 10 % of frame

# ═══════════════════════════════════════════════════════════
#  FINGER  TRACKING
# ═══════════════════════════════════════════════════════════
FINGER_TRACKING_ENABLED = True
FINGER_CONFIDENCE = 0.7           # MediaPipe detection threshold
FINGER_PROXIMITY_THRESHOLD = 80   # px — max distance finger↔text box

# ═══════════════════════════════════════════════════════════
#  OCR
# ═══════════════════════════════════════════════════════════
OCR_LANGUAGES = ["en"]
OCR_GPU = False                    # True on Jetson / CUDA machines
OCR_CONFIDENCE_THRESHOLD = 0.35

# ═══════════════════════════════════════════════════════════
#  TEMPORAL  OCR  FUSION
# ═══════════════════════════════════════════════════════════
FUSION_BUFFER_SIZE = 1             # frames to accumulate before fusing
FUSION_SIMILARITY_THRESHOLD = 0.6  # min similarity to allow fusion

# ═══════════════════════════════════════════════════════════
#  TEXT  CLEANING
# ═══════════════════════════════════════════════════════════
MIN_WORD_LENGTH = 2
MIN_LINE_LENGTH = 3
REMOVE_SPECIAL_CHARS = True

# ═══════════════════════════════════════════════════════════
#  SPATIAL  GUIDANCE
# ═══════════════════════════════════════════════════════════
GUIDANCE_COOLDOWN = 5.0            # seconds between same cue
CENTER_TOLERANCE_X = 0.15          # ± fraction of frame width
CENTER_TOLERANCE_Y = 0.15
MIN_TEXT_SCALE = 0.05              # text area / frame area
MAX_TEXT_SCALE = 0.80

# ═══════════════════════════════════════════════════════════
#  TTS
# ═══════════════════════════════════════════════════════════
TTS_ENGINE = "pyttsx3"             # "pyttsx3" (offline)
TTS_RATE = 160                     # words-per-minute
TTS_VOLUME = 1.0

# ═══════════════════════════════════════════════════════════
#  STATE  MACHINE
# ═══════════════════════════════════════════════════════════
IDLE_TIMEOUT = 30.0                # seconds of GUIDANCE before → IDLE
SPEAKING_BLOCK_PERCEPTION = True   # freeze perception while speaking
READ_COOLDOWN = 10.0                # seconds before re-reading same region

# ═══════════════════════════════════════════════════════════
#  RUNTIME  /  DEBUG
# ═══════════════════════════════════════════════════════════
MAIN_LOOP_DELAY = 0.03            # target ~30 FPS
PERCEPTION_SKIP_FRAMES = 5        # run heavy ops every Nth frame
DEBUG_DISPLAY = True               # OpenCV window — set False on headless Pi

# ═══════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════
LOG_LEVEL = "INFO"
LOG_FILE = "wearable_reader.log"