"""Shared constants for the floor piano.

Pure data only (no logic, no heavy imports) so this is safe to import from
anywhere without side effects.
"""

# The 7 white keys of one octave (C major), laid out left -> right across the mat.
DEFAULT_KEYS = ["C", "D", "E", "F", "G", "A", "B"]

# ArUco marker IDs at the mat corners: 0=Top-Left, 1=Top-Right, 2=Bottom-Right, 3=Bottom-Left.
CORNER_IDS = [0, 1, 2, 3]

# Warped "piano" canvas that the depth image is projected onto (pixels).
TARGET_WIDTH = 700
TARGET_HEIGHT = 200

# Depth trigger defaults (millimetres). Overridden by config.json at runtime.
DEFAULT_FLOOR_DEPTH = 1000        # camera -> floor distance
DEFAULT_TRIGGER_THRESHOLD = 50    # 5cm safety buffer above the floor

# A key column must contain more than this many above-floor pixels to fire.
MIN_HIT_PIXELS = 150

# Weight of the newest sample when re-leveling the floor (exponential moving average).
FLOOR_EMA_ALPHA = 0.1
