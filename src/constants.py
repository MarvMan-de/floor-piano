"""Shared constants for the floor piano.

Pure data only (no logic, no heavy imports) so this is safe to import from
anywhere without side effects.
"""

# --- keyboard layout -------------------------------------------------------

# White keys of an octave, left -> right.
WHITE_SCALE = ["C", "D", "E", "F", "G", "A", "B"]

# Within an octave, a black (sharp) key follows these white-key positions.
# There is NO black key after E (index 2) or B (index 6).
BLACK_AFTER = {0: "C#", 1: "D#", 3: "F#", 4: "G#", 5: "A#"}

# Default keyboard: 14 white + 10 black = 24 keys = 2 octaves (C4..B5).
DEFAULT_NUM_WHITE = 14
START_OCTAVE = 4

# Black key geometry, relative to a white key / the mat depth.
BLACK_WIDTH_RATIO = 0.6     # black key width as a fraction of a white key
BLACK_HEIGHT_RATIO = 0.62   # black key covers this fraction of the mat depth (from the top)

# ArUco marker IDs at the mat corners: 0=Top-Left, 1=Top-Right, 2=Bottom-Right, 3=Bottom-Left.
CORNER_IDS = [0, 1, 2, 3]

# --- warp canvas -----------------------------------------------------------

# Warped "piano" canvas the depth image is projected onto (pixels).
# ~100 px per white key at the default 14-white layout.
TARGET_WIDTH = 1400
TARGET_HEIGHT = 200

# --- triggering ------------------------------------------------------------

# Depth trigger defaults (millimetres). Overridden by config.json at runtime.
DEFAULT_FLOOR_DEPTH = 1000        # camera -> floor distance
DEFAULT_TRIGGER_THRESHOLD = 50    # 5cm safety buffer above the floor

# Only pixels within this many mm ABOVE the floor can press a key. Anything
# higher (a foot swinging mid-step, a knee, a torso) is ignored instead of
# firing notes. Overridable per-config as "max_press_height".
MAX_PRESS_HEIGHT = 250

# A key fires when more than this many of its pixels are above the floor.
# Interpreted at the SOURCE-frame scale; main.py rescales it to warp-canvas
# pixels so the sensitivity doesn't change with camera resolution / mount.
MIN_HIT_PIXELS = 150

# A held key is released only after it has been absent for this many consecutive
# frames (~100ms at 30fps) — keeps depth noise from re-triggering the note.
RELEASE_FRAMES = 3

# Weight of the newest sample when re-leveling the floor (exponential moving average).
FLOOR_EMA_ALPHA = 0.1

# Give up after this many consecutive empty camera reads (~5s at the 100ms
# timeout) — lets systemd restart the service instead of spinning silently.
CAMERA_STALL_LIMIT = 50
