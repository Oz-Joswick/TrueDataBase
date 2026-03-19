"""
face_system/mac/detector.py
────────────────────────────
InsightFace wrapper for face detection and embedding generation.
Responsibilities:
  - Load the InsightFace model once per worker process
  - Detect all faces in a frame and return bounding boxes + embeddings
  - Provide both sync (single image) and async (video batch) interfaces
  - Run CPU-heavy detection in a process pool to avoid blocking the event loop
"""

import asyncio
import concurrent.futures
import os
from typing import Optional

import numpy as np

from config import INSIGHTFACE_DET_SIZE, INSIGHTFACE_PROVIDERS


# ── Per-process model instance ────────────────────────────────────────────────
# Each worker process loads its own copy of the model.
# Never shared across processes — InsightFace is not process-safe.

_face_app = None


def init_worker_model():
    """
    Load InsightFace into the current process.
    Called once by worker.py at process startup before any jobs are processed.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _face_app
    if _face_app is not None:
        return  # already loaded in this process

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "insightface is not installed.\n"
            "Run: pip install insightface onnxruntime"
        )

    print(f"  [detector] loading InsightFace model "
          f"(PID {os.getpid()})...")

    _face_app = FaceAnalysis(providers=INSIGHTFACE_PROVIDERS)
    _face_app.prepare(ctx_id=0, det_size=INSIGHTFACE_DET_SIZE)

    print(f"  [detector] model ready  "
          f"det_size={INSIGHTFACE_DET_SIZE}  "
          f"providers={INSIGHTFACE_PROVIDERS}")


def _require_model():
    """Raise a clear error if detect() is called before init_worker_model()."""
    if _face_app is None:
        raise RuntimeError(
            "InsightFace model not loaded. "
            "Call init_worker_model() at process startup."
        )


# ── Core detection function ───────────────────────────────────────────────────

def _run_detection(frame_bytes: bytes, shape: tuple) -> list[dict]:
    """
    Reconstruct a frame from raw bytes and run face detection.
    Runs inside a worker process — safe to call from a ProcessPoolExecutor.

    Returns a list of face dicts:
        {
            "bbox":      [x1, y1, x2, y2],   # floats
            "embedding": [float, ...],         # 512-dim normalised ArcFace vector
            "det_score": float,                # detection confidence 0–1
            "kps":       [[x,y], ...],         # 5 facial keypoints (optional)
        }
    """
    _require_model()

    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)
    faces = _face_app.get(frame)

    results = []
    for face in faces:
        result = {
            "bbox":      face.bbox.tolist(),
            "embedding": face.normed_embedding.tolist(),
            "det_score": round(float(face.det_score), 4),
        }
        # Keypoints are optional — not all model configs return them
        if hasattr(face, "kps") and face.kps is not None:
            result["kps"] = face.kps.tolist()
        results.append(result)

    return results


# ── Sync interface (single image) ─────────────────────────────────────────────

def detect(frame: np.ndarray) -> list[dict]:
    """
    Synchronous face detection. Blocks until complete.
    Used for single images where we don't need concurrency.

    Args:
        frame   BGR numpy array (H, W, 3) as returned by cv2.imread()

    Returns:
        List of face dicts — empty list if no faces found.
    """
    _require_model()
    return _run_detection(frame.tobytes(), frame.shape)


# ── Async interface (video batches) ──────────────────────────────────────────

# Module-level process pool — created once, shared across async calls
# within the same worker process.
_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None


def get_pool(num_workers: int = None) -> concurrent.futures.ProcessPoolExecutor:
    """
    Return the shared ProcessPoolExecutor, creating it if necessary.
    num_workers defaults to os.cpu_count().
    Each pool worker calls init_worker_model() on first use via initializer.
    """
    global _pool
    if _pool is None:
        workers = num_workers or os.cpu_count() or 1
        print(f"  [detector] creating process pool  "
              f"{workers} worker(s)")
        _pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker_model,
        )
    return _pool


async def detect_async(frame: np.ndarray) -> list[dict]:
    """
    Async face detection — does not block the event loop.
    Used for video batch processing where multiple frames are
    detected concurrently via asyncio.gather().

    Args:
        frame   BGR numpy array (H, W, 3)

    Returns:
        List of face dicts — empty list if no faces found.
    """
    loop = asyncio.get_event_loop()
    pool = get_pool()

    return await loop.run_in_executor(
        pool,
        _run_detection,
        frame.tobytes(),
        frame.shape,
    )


# ── Batch async helper ────────────────────────────────────────────────────────

async def detect_batch(frames: list[np.ndarray]) -> list[list[dict]]:
    """
    Detect faces in a batch of frames concurrently.
    Returns a list of face lists, one per input frame.

    Equivalent to:
        results = await asyncio.gather(*[detect_async(f) for f in frames])

    But provided here as a convenience so worker.py doesn't need to
    import asyncio.gather directly.
    """
    tasks = [detect_async(f) for f in frames]
    return list(await asyncio.gather(*tasks, return_exceptions=False))


# ── Utility ───────────────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray,
                    faces: list[dict],
                    color: tuple = (0, 255, 100),
                    thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes and detection scores onto a frame.
    Returns a copy of the frame — does not modify the original.
    Useful for debugging; not called during normal processing.
    """
    import cv2
    out = frame.copy()
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        score = face.get("det_score", 0)
        cv2.putText(out, f"{score:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)
    return out


def shutdown_pool():
    """
    Cleanly shut down the process pool.
    Call this before the process exits to avoid resource warnings.
    """
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
        print("  [detector] process pool shut down")