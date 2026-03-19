"""
face_system/mac/config.py
─────────────────────────
Single source of truth for all tuneable settings.
Edit this file — nothing else needs to change.
"""

# ── Pi connection ─────────────────────────────────────────────────────────────

PI_URL = "http://192.168.2.2:8000"
# If using Tailscale instead of ethernet, swap to your Tailscale IP:
# PI_URL = "http://100.64.0.2:8000"

PI_POLL_INTERVAL = 30           # seconds between automatic Pi status checks
PI_REQUEST_TIMEOUT = 2.0        # seconds before an HTTP call to Pi times out
PI_IDENTIFY_THRESHOLD = 0.4     # cosine distance cutoff — lower = stricter match
                                 # 0.3 → very strict   0.5 → more lenient

# ── Dashboard ─────────────────────────────────────────────────────────────────

DASHBOARD_PORT = 8765           # websocket server port  ws://localhost:8765
DASHBOARD_HOST = "localhost"    # bind address — keep localhost unless you know why

# ── Core allocation ───────────────────────────────────────────────────────────

WORKER_CORES_OVERRIDE = None
# Set to an integer to skip the startup prompt, e.g.:
#   WORKER_CORES_OVERRIDE = 3   →  1 dashboard + 3 workers
# Leave as None to be asked on every launch.

# ── Video processing ──────────────────────────────────────────────────────────

POLL_RATE_SECONDS = 4.0
# How often (in seconds) to sample a video for face detection.
# Lower = more frames processed, slower throughput.
# 2.0 → dense sampling    8.0 → sparse / fast

BATCH_SIZE = 4
# How many video frames to detect in parallel per worker.
# Set to your worker core count for maximum throughput.
# Has no effect on image jobs.

# ── Face matching ─────────────────────────────────────────────────────────────

EMBEDDING_DIVERSITY_THRESHOLD = 0.15
# Minimum cosine distance between a new embedding and all existing ones
# before it is saved as an additional reference.
# Lower = only very different angles saved (fewer refs, leaner DB)
# Higher = more variations saved (better recognition, larger DB)
# Recommended range: 0.10 – 0.25

EMBEDDING_CHECK_INTERVAL = 150
# How many processed frames between diversity checks for a known person.
# Higher = less frequent Pi calls during video processing.
# Has no effect on image jobs.

MAX_EMBEDDINGS_PER_PERSON = 20
# Hard cap on reference embeddings per person.
# Prevents unbounded DB growth for frequently-seen individuals.

# ── New person handling ───────────────────────────────────────────────────────

AUTO_REGISTER_NEW = True
# If True  → unknown faces are automatically registered as Person_XXXXXX
# If False → unknown faces are labelled but NOT written to the DB

DELETE_AFTER_SCAN = False
# If True → source file is deleted from disk after successful processing
# Applies to both images and videos.
# Failed jobs are never deleted regardless of this setting.

# ── Detection model ───────────────────────────────────────────────────────────

INSIGHTFACE_DET_SIZE = (640, 640)
# Input resolution fed to the face detector.
# Larger = more accurate on small/distant faces, slower.
# (320, 320) → fast    (640, 640) → balanced    (960, 960) → slow + accurate

INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]
# ONNX execution providers in priority order.
# If you have a compatible GPU add it first:
#   ["CUDAExecutionProvider", "CPUExecutionProvider"]
#   ["CoreMLExecutionProvider", "CPUExecutionProvider"]  ← Apple Silicon

# ── Scan server (camera identification) ──────────────────────────────────────

SCAN_SERVER_PORT = 8766         # HTTPS port for the camera scan page
SCAN_SERVER_HOST = "0.0.0.0"   # bind to all interfaces so LAN devices can reach it

# ── UI defaults ───────────────────────────────────────────────────────────────

SHOW_PREVIEWS_DEFAULT = True    # worker preview cards visible on launch
SHOW_VIZ_DEFAULT = True         # people visualization visible on launch
DIVERSITY_CHECK_DEFAULT = True  # diversity check toggle default state