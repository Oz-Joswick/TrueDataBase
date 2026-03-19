"""
face_system/mac/worker.py
─────────────────────────
Runs as a separate process on one worker core.
Responsibilities:
  - Pull jobs from the shared job_queue one at a time
  - Process images and videos (detect → identify → register)
  - Push state updates to state_queue after every meaningful event
  - Never touch the UI or the dashboard directly
"""

import multiprocessing
import os
import time
from datetime import datetime

import cv2
import numpy as np

from config import (
    POLL_RATE_SECONDS,
    BATCH_SIZE,
    AUTO_REGISTER_NEW,
    DELETE_AFTER_SCAN,
    PI_IDENTIFY_THRESHOLD,
    EMBEDDING_DIVERSITY_THRESHOLD,
    EMBEDDING_CHECK_INTERVAL,
    MAX_EMBEDDINGS_PER_PERSON,
)
from upload import load_image, video_frame_generator
from detector import detect, detect_async, init_worker_model
from tracker import get_cached, cache_face, make_temp_name, is_diverse_enough
from client import identify, register, get_embeddings, add_embedding
from queue_manager import JobStatus


# ── State update helpers ──────────────────────────────────────────────────────

def _push(state_queue: multiprocessing.Queue, worker_id: int, **kwargs):
    """
    Push a state snapshot to the dashboard server.
    Always includes worker_id and a timestamp.
    """
    state_queue.put_nowait({
        "worker_id": worker_id,
        "ts":        datetime.now().isoformat(),
        **kwargs,
    })


# ── Face identification logic ─────────────────────────────────────────────────

import asyncio

async def _identify_faces(
    frame:       np.ndarray,
    faces:       list[dict],
    worker_id:   int,
    state_queue: multiprocessing.Queue,
    job_id:      int,
    timestamp:   float = None,
    embed_counter: dict = None,   # {person_id: frames_since_last_check}
) -> list[dict]:
    """
    For every detected face in a frame:
      1. Check session cache — skip Pi call if already seen this session
      2. Ask Pi to identify the embedding
      3. If known  → cache, maybe add diversity embedding
      4. If unknown → auto-register (if enabled), cache

    Returns list of result dicts for state broadcast.
    """
    if not faces:
        return []

    results = []

    # Fire all Pi identify calls concurrently
    import httpx
    identify_tasks = [identify(face["embedding"]) for face in faces]
    matches = await asyncio.gather(*identify_tasks, return_exceptions=True)

    for face, match in zip(faces, matches):
        emb = face["embedding"]

        # Handle network errors gracefully
        if isinstance(match, Exception):
            results.append({"label": "Pi error", "type": "error",
                            "bbox": face["bbox"]})
            continue

        cached = get_cached(emb)

        if cached:
            label     = cached["name"]
            face_type = "session"
            person_id = cached.get("person_id")
            ref_count = cached.get("ref_count", 1)

            # Periodic diversity check
            if person_id and embed_counter is not None:
                embed_counter[person_id] = \
                    embed_counter.get(person_id, 0) + 1
                if (embed_counter[person_id] >= EMBEDDING_CHECK_INTERVAL
                        and ref_count < MAX_EMBEDDINGS_PER_PERSON):
                    embed_counter[person_id] = 0
                    existing = await get_embeddings(person_id)
                    if is_diverse_enough(emb, existing,
                                        EMBEDDING_DIVERSITY_THRESHOLD):
                        res = await add_embedding(person_id, emb)
                        new_total = res.get("total_embeddings", ref_count)
                        cached["ref_count"] = new_total
                        cache_face(emb, cached)

        elif match.get("name") != "Unknown":
            label     = match["name"]
            face_type = "known"
            person_id = match.get("person_id")
            ref_count = match.get("ref_count", 1)
            cache_face(emb, {**match, "ref_count": ref_count})

            # Immediately check diversity for the new angle we just saw
            if (person_id and ref_count < MAX_EMBEDDINGS_PER_PERSON):
                existing = await get_embeddings(person_id)
                if is_diverse_enough(emb, existing,
                                     EMBEDDING_DIVERSITY_THRESHOLD):
                    res = await add_embedding(person_id, emb)
                    new_total = res.get("total_embeddings", ref_count)
                    match["ref_count"] = new_total
                    cache_face(emb, {**match, "ref_count": new_total})

        else:
            # Unknown face
            if AUTO_REGISTER_NEW:
                temp   = make_temp_name()
                result = await register(temp, emb, {
                    "auto_detected": True,
                    "needs_review":  True,
                    "source_job":    job_id,
                })
                pid = result.get("person_id")
                cache_face(emb, {"name": temp, "person_id": pid,
                                 "ref_count": 1})
                label     = temp
                face_type = "new"
            else:
                label     = "Unknown"
                face_type = "unknown"

        results.append({
            "label":     label,
            "type":      face_type,
            "bbox":      face["bbox"],
            "person_id": locals().get("person_id"),
        })

    if timestamp is not None:
        names = [r["label"] for r in results]
        print(f"  [worker-{worker_id}] "
              f"{timestamp:.1f}s — {', '.join(names) if names else 'no faces'}")

    return results


# ── Image processing ──────────────────────────────────────────────────────────

async def _process_image(job, worker_id: int,
                          state_queue: multiprocessing.Queue) -> dict:
    frame = load_image(job.path)

    _push(state_queue, worker_id,
          type="frame",
          job_id=job.id,
          filename=job.filename,
          progress=0,
          faces=[])

    faces   = detect(frame)
    results = await _identify_faces(
        frame, faces, worker_id, state_queue, job.id
    )

    _push(state_queue, worker_id,
          type="frame",
          job_id=job.id,
          filename=job.filename,
          progress=100,
          faces=results)

    return {
        "faces_found": len(results),
        "results":     results,
    }


# ── Video processing ──────────────────────────────────────────────────────────

async def _process_video(job, worker_id: int,
                          state_queue: multiprocessing.Queue,
                          poll_rate: float = POLL_RATE_SECONDS) -> dict:
    all_results    = {}
    batch_frames   = []
    batch_ts       = []
    embed_counter  = {}     # per-person frame counter for diversity checks
    processed      = 0

    # We need total frame count to report progress
    cap = cv2.VideoCapture(job.path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    cap.release()
    expected_samples = max(1, int((total_frames / fps) / poll_rate))

    for frame, timestamp in video_frame_generator(job.path, poll_rate):
        batch_frames.append(frame)
        batch_ts.append(timestamp)

        if len(batch_frames) >= BATCH_SIZE:
            # Detect all frames in batch simultaneously
            detection_tasks = [detect_async(f) for f in batch_frames]
            batch_faces     = await asyncio.gather(*detection_tasks)

            for f, faces, ts in zip(batch_frames, batch_faces, batch_ts):
                results = await _identify_faces(
                    f, faces, worker_id, state_queue,
                    job.id, ts, embed_counter
                )
                all_results[ts] = results
                processed += 1

                _push(state_queue, worker_id,
                      type="frame",
                      job_id=job.id,
                      filename=job.filename,
                      progress=min(99, int(processed / expected_samples * 100)),
                      timestamp=ts,
                      faces=results)

            batch_frames = []
            batch_ts     = []

    # Flush remaining frames under batch size
    if batch_frames:
        detection_tasks = [detect_async(f) for f in batch_frames]
        batch_faces     = await asyncio.gather(*detection_tasks)
        for f, faces, ts in zip(batch_frames, batch_faces, batch_ts):
            results = await _identify_faces(
                f, faces, worker_id, state_queue,
                job.id, ts, embed_counter
            )
            all_results[ts] = results
            processed += 1

    all_faces = [r for results in all_results.values() for r in results]
    unique_people = {r["label"] for r in all_faces
                     if r["type"] in ("known", "new")}

    return {
        "frames_processed": processed,
        "faces_found":      len(all_faces),
        "unique_people":    list(unique_people),
        "results":          all_results,
    }


# ── Worker entry point ────────────────────────────────────────────────────────

def run_worker(worker_id:   int,
               job_queue:   multiprocessing.Queue,
               state_queue: multiprocessing.Queue,
               poll_rate:   float = POLL_RATE_SECONDS):
    """
    Called by worker_pool.py as the target of each worker Process.
    Runs an asyncio event loop that processes jobs until poisoned or killed.
    """
    # Load InsightFace model once in this process
    init_worker_model()
    print(f"  [worker-{worker_id}] ready (PID {os.getpid()})")

    _push(state_queue, worker_id, type="ready", status="idle")

    async def _loop():
        while True:
            # Block until a job arrives (with periodic wakeups to stay alive)
            job = None
            while job is None:
                try:
                    job = job_queue.get(timeout=2.0)
                except Exception:
                    continue

            # Poison pill — clean shutdown
            if job is None or getattr(job, "id", None) == -1:
                print(f"  [worker-{worker_id}] received shutdown signal")
                break

            # Skip jobs removed from registry while queued
            if job.status == JobStatus.PENDING.value or \
               hasattr(job.status, "value") and \
               job.status == JobStatus.PENDING:
                pass    # proceed
            else:
                print(f"  [worker-{worker_id}] skipping job {job.id} "
                      f"— status {getattr(job.status,'value',job.status)}")
                continue

            # ── Process job ───────────────────────────────────────────────────
            print(f"  [worker-{worker_id}] starting job {job.id}: "
                  f"{job.filename}")

            _push(state_queue, worker_id,
                  type="job_start",
                  job_id=job.id,
                  filename=job.filename,
                  file_type=job.file_type,
                  status="processing",
                  started_at=datetime.now().isoformat())

            try:
                if job.file_type == "image":
                    result = await _process_image(job, worker_id, state_queue)
                else:
                    result = await _process_video(
                        job, worker_id, state_queue, poll_rate
                    )

                _push(state_queue, worker_id,
                      type="job_done",
                      job_id=job.id,
                      filename=job.filename,
                      status="done",
                      finished_at=datetime.now().isoformat(),
                      result=result)

                print(f"  [worker-{worker_id}] done  job {job.id} "
                      f"— {result.get('faces_found', 0)} face(s)")

                # Delete source file if configured
                if DELETE_AFTER_SCAN:
                    try:
                        os.remove(job.path)
                        print(f"  [worker-{worker_id}] deleted {job.filename}")
                    except OSError as e:
                        print(f"  [worker-{worker_id}] could not delete "
                              f"{job.filename}: {e}")

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"  [worker-{worker_id}] ERROR job {job.id}: {e}\n{tb}")

                _push(state_queue, worker_id,
                      type="job_failed",
                      job_id=job.id,
                      filename=job.filename,
                      status="failed",
                      finished_at=datetime.now().isoformat(),
                      error=str(e))

            finally:
                _push(state_queue, worker_id,
                      type="idle",
                      status="idle",
                      filename=None,
                      faces=[],
                      progress=0)

    try:
        asyncio.run(_loop())
    except (KeyboardInterrupt, SystemExit):
        pass
    print(f"  [worker-{worker_id}] exiting")