"""
face_system/mac/upload.py
──────────────────────────
File type detection, image loading, and video frame sampling.
Responsibilities:
  - Validate file extensions against supported types
  - Load images as numpy arrays ready for InsightFace
  - Yield sampled frames from video files at a configurable rate
  - Provide total frame count and duration estimates for progress reporting
"""

import os
from typing import Generator, Tuple

import cv2
import numpy as np


# ── Supported formats ─────────────────────────────────────────────────────────

SUPPORTED_IMAGES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
}

SUPPORTED_VIDEOS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm",
}

SUPPORTED_ALL = SUPPORTED_IMAGES | SUPPORTED_VIDEOS


# ── Type detection ────────────────────────────────────────────────────────────

def get_file_type(path: str) -> str:
    """
    Return 'image' or 'video' based on file extension.
    Raises ValueError for unsupported extensions.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in SUPPORTED_IMAGES:
        return "image"
    if ext in SUPPORTED_VIDEOS:
        return "video"

    raise ValueError(
        f"Unsupported file type: '{ext}'\n"
        f"  Supported images : {', '.join(sorted(SUPPORTED_IMAGES))}\n"
        f"  Supported videos : {', '.join(sorted(SUPPORTED_VIDEOS))}"
    )


def is_supported(path: str) -> bool:
    """Return True if the file extension is supported, False otherwise."""
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_ALL


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and return it as a BGR numpy array.
    InsightFace expects BGR (same as OpenCV default).

    Raises:
        FileNotFoundError  if the path does not exist
        ValueError         if OpenCV cannot decode the file
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    frame = cv2.imread(path)

    if frame is None:
        raise ValueError(
            f"Could not decode image: {path}\n"
            f"File may be corrupt or in an unsupported colour space."
        )

    return frame


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load an image as RGB (for libraries that expect RGB instead of BGR).
    Not used by InsightFace directly, provided for convenience.
    """
    return cv2.cvtColor(load_image(path), cv2.COLOR_BGR2RGB)


# ── Video metadata ────────────────────────────────────────────────────────────

def get_video_info(path: str) -> dict:
    """
    Return metadata about a video file without reading all frames.

    Returns:
        {
            fps:              float,
            total_frames:     int,
            duration_seconds: float,
            width:            int,
            height:           int,
            codec:            str,
        }
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int   = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec        = "".join([chr((fourcc_int >> (8 * i)) & 0xFF)
                             for i in range(4)]).strip()
    cap.release()

    duration = total_frames / fps if fps > 0 else 0.0

    return {
        "fps":              round(fps, 2),
        "total_frames":     total_frames,
        "duration_seconds": round(duration, 2),
        "width":            width,
        "height":           height,
        "codec":            codec,
    }


def estimate_sample_count(path: str, poll_rate_seconds: float) -> int:
    """
    Estimate how many frames will be sampled from a video.
    Used by worker.py to calculate progress percentages.
    """
    try:
        info = get_video_info(path)
        return max(1, int(info["duration_seconds"] / poll_rate_seconds))
    except Exception:
        return 1


# ── Video frame generator ─────────────────────────────────────────────────────

def video_frame_generator(
    path:               str,
    poll_rate_seconds:  float = 4.0,
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    Yield (frame, timestamp_seconds) tuples from a video file,
    sampling one frame every poll_rate_seconds of video time.

    Args:
        path               Path to the video file.
        poll_rate_seconds  Seconds of video time between sampled frames.
                           2.0 = dense,  4.0 = balanced,  8.0 = sparse.

    Yields:
        (frame, timestamp) where frame is a BGR numpy array and
        timestamp is the position in the video in seconds.

    Raises:
        FileNotFoundError  if the path does not exist
        ValueError         if OpenCV cannot open the file
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0   # safe fallback for files with missing metadata

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration       = total_frames / fps if total_frames > 0 else 0.0
    frame_interval = max(1, int(fps * poll_rate_seconds))
    expected       = max(1, int(duration / poll_rate_seconds))

    print(f"  [upload] {os.path.basename(path)}")
    print(f"           {duration:.1f}s  |  {fps:.1f} fps  |  "
          f"sampling every {poll_rate_seconds}s  |  "
          f"~{expected} frame(s)")

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp = frame_index / fps
            yield frame, round(timestamp, 3)

        frame_index += 1

    cap.release()


# ── Batch helpers ─────────────────────────────────────────────────────────────

def collect_files_from_paths(paths: list[str]) -> dict[str, list[str]]:
    """
    Walk a mixed list of files and folders.
    Returns {"images": [...], "videos": [...]} with absolute paths.
    Images are listed before videos (fast jobs first).
    """
    images, videos = [], []

    for path in paths:
        path = os.path.abspath(path)

        if os.path.isfile(path):
            try:
                ft = get_file_type(path)
                (images if ft == "image" else videos).append(path)
            except ValueError:
                pass

        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fname in sorted(files):
                    full = os.path.join(root, fname)
                    try:
                        ft = get_file_type(full)
                        (images if ft == "image" else videos).append(full)
                    except ValueError:
                        pass

    return {"images": images, "videos": videos}