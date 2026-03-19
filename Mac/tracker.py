"""
face_system/mac/tracker.py
───────────────────────────
Per-worker session state.
Responsibilities:
  - Cache faces seen this session to avoid redundant Pi calls
  - Fingerprint embeddings for fast cache lookups
  - Check whether a new embedding is diverse enough to save as a reference
  - Generate temporary names for auto-registered unknown people
  - Track per-person frame counters for throttled diversity checks

One instance of this module's state lives per worker process.
Workers never share tracker state with each other — each builds
its own session picture independently.
"""

import uuid
import numpy as np
from typing import Optional


# ── Session cache ─────────────────────────────────────────────────────────────
# Maps embedding fingerprint → person data dict.
# Populated the first time a face is identified; consulted on every
# subsequent frame to avoid hitting Pi for every face in every frame.

_session_cache: dict[tuple, dict] = {}


def session_key(embedding: list[float]) -> tuple:
    """
    Produce a lightweight fingerprint from an embedding vector.
    Uses the first 12 dimensions rounded to 2 decimal places.
    Collision rate is negligible for face embeddings in practice.

    Fast enough to call on every frame without measurable overhead.
    """
    return tuple(np.round(embedding[:12], 2))


def get_cached(embedding: list[float]) -> Optional[dict]:
    """
    Look up an embedding in the session cache.
    Returns the cached person dict, or None if not seen this session.
    """
    return _session_cache.get(session_key(embedding))


def cache_face(embedding: list[float], data: dict):
    """
    Store person data against an embedding fingerprint.
    data should contain at minimum:
        {name, person_id, ref_count}
    """
    _session_cache[session_key(embedding)] = data


def update_cached_ref_count(embedding: list[float], new_ref_count: int):
    """
    Update the ref_count on an already-cached entry.
    Called after a new diversity embedding is successfully saved to Pi.
    """
    key = session_key(embedding)
    if key in _session_cache:
        _session_cache[key]["ref_count"] = new_ref_count


def clear_session():
    """
    Wipe the session cache.
    Called between jobs if isolation between files is desired,
    or left alone to let recognition improve across a batch.
    """
    _session_cache.clear()


def session_size() -> int:
    return len(_session_cache)


def session_people() -> list[dict]:
    """Return all unique cached people (deduplicated by person_id)."""
    seen_ids = set()
    people   = []
    for data in _session_cache.values():
        pid = data.get("person_id")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            people.append(data)
    return people


# ── Diversity check ───────────────────────────────────────────────────────────

def is_diverse_enough(new_embedding:      list[float],
                       existing_embeddings: list[list[float]],
                       threshold:          float = 0.15) -> bool:
    """
    Return True if new_embedding is different enough from all existing
    embeddings to be worth saving as an additional reference.

    Uses cosine distance: distance = 1 - cosine_similarity.
    Both vectors are assumed to be unit-normalised (InsightFace normed_embedding).

    Args:
        new_embedding        The candidate embedding to evaluate.
        existing_embeddings  All embeddings already stored for this person.
        threshold            Minimum cosine distance to be considered diverse.
                             0.10 = only very different angles
                             0.15 = balanced (default)
                             0.25 = save most variations

    Returns:
        True  → save this embedding as a new reference
        False → too similar to an existing reference, skip
    """
    if not existing_embeddings:
        return True     # no existing refs — always save the first

    new_vec = np.array(new_embedding, dtype=np.float32)

    # Normalise defensively — should already be unit length from InsightFace
    norm = np.linalg.norm(new_vec)
    if norm > 0:
        new_vec = new_vec / norm

    for existing in existing_embeddings:
        existing_vec = np.array(existing, dtype=np.float32)
        e_norm = np.linalg.norm(existing_vec)
        if e_norm > 0:
            existing_vec = existing_vec / e_norm

        cosine_similarity = float(np.dot(new_vec, existing_vec))
        cosine_distance   = 1.0 - cosine_similarity

        if cosine_distance < threshold:
            return False    # too similar to this existing ref

    return True


def most_similar_distance(new_embedding:      list[float],
                           existing_embeddings: list[list[float]]) -> float:
    """
    Return the cosine distance to the closest existing embedding.
    Returns 1.0 (maximum distance) if there are no existing embeddings.
    Useful for logging how novel a new angle is.
    """
    if not existing_embeddings:
        return 1.0

    new_vec = np.array(new_embedding, dtype=np.float32)
    norm    = np.linalg.norm(new_vec)
    if norm > 0:
        new_vec = new_vec / norm

    min_distance = 1.0
    for existing in existing_embeddings:
        ev   = np.array(existing, dtype=np.float32)
        en   = np.linalg.norm(ev)
        if en > 0:
            ev = ev / en
        dist = 1.0 - float(np.dot(new_vec, ev))
        if dist < min_distance:
            min_distance = dist

    return round(min_distance, 4)


# ── Temporary name generation ─────────────────────────────────────────────────

def make_temp_name() -> str:
    """
    Generate a short unique placeholder name for an auto-registered person.
    Format: Person_XXXXXX  (6 uppercase alphanumeric characters)

    Example: Person_A3F2B1

    The person can be renamed later via the dashboard or Pi API.
    """
    suffix = str(uuid.uuid4()).replace("-", "")[:6].upper()
    return f"Person_{suffix}"


# ── Per-person frame counter ──────────────────────────────────────────────────
# Tracks how many frames each known person has been seen since the last
# diversity check. Prevents hammering Pi with embedding requests every frame.

_embed_counters: dict[int, int] = {}   # person_id → frame count


def increment_embed_counter(person_id: int) -> int:
    """
    Increment the frame counter for a person and return the new value.
    """
    _embed_counters[person_id] = _embed_counters.get(person_id, 0) + 1
    return _embed_counters[person_id]


def reset_embed_counter(person_id: int):
    """Reset a person's frame counter after a diversity check."""
    _embed_counters[person_id] = 0


def get_embed_counter(person_id: int) -> int:
    return _embed_counters.get(person_id, 0)


def clear_embed_counters():
    """Wipe all frame counters. Called between jobs if needed."""
    _embed_counters.clear()


# ── Debug helpers ─────────────────────────────────────────────────────────────

def print_session_summary():
    people = session_people()
    print(f"  [tracker] session cache: {session_size()} entries, "
          f"{len(people)} unique people")
    for p in people:
        print(f"    · {p.get('name','?')}  "
              f"id={p.get('person_id','?')}  "
              f"refs={p.get('ref_count','?')}")