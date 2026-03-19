"""
face_system/pi/database.py
───────────────────────────
All PostgreSQL / pgvector interactions.
Responsibilities:
  - Manage an asyncpg connection pool
  - Nearest-neighbour face search via pgvector cosine distance
  - Insert, fetch, and update people records
  - Insert and fetch face embeddings
  - Report database statistics for the /status endpoint

Never import FastAPI here — this module is pure data access.
"""

import json
import os
import time
from typing import Optional

import asyncpg

# ── Configuration ─────────────────────────────────────────────────────────────
# Read from environment so the connection string is never hard-coded.
# Set DATABASE_URL before starting uvicorn, e.g.:
#   export DATABASE_URL="postgresql://postgres:password@localhost/facedb"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost/facedb",
)

# Pool sizing — Pi 4/5 handles 5–10 concurrent asyncpg connections comfortably
POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 10

# ── Pool ──────────────────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def init_pool():
    """
    Create the asyncpg connection pool.
    Called once at FastAPI startup.
    """
    global _pool
    _pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        command_timeout=10,
    )
    print(f"[db] pool ready  min={POOL_MIN_SIZE}  max={POOL_MAX_SIZE}")


async def close_pool():
    """Drain and close the pool. Called at FastAPI shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("[db] pool closed")


def _require_pool():
    if _pool is None:
        raise RuntimeError(
            "Database pool not initialised. "
            "Ensure init_pool() was called at startup."
        )


# ── Embedding serialisation ───────────────────────────────────────────────────
# pgvector expects embeddings as a string like "[0.1, 0.2, ...]"
# asyncpg returns them as strings too — we parse them back to float lists.

def _vec_to_str(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"


def _str_to_vec(raw: str) -> list[float]:
    return [float(x) for x in raw.strip("[]").split(",")]


# ── Search ────────────────────────────────────────────────────────────────────

async def search(embedding: list[float],
                 threshold: float = 0.4) -> dict:
    """
    Find the closest matching person using pgvector cosine distance.

    cosine distance  = 1 - cosine_similarity
    confidence       = 1 - cosine_distance   (higher is better, 0-1)

    The threshold is a cosine distance cutoff — matches with distance
    greater than threshold are treated as Unknown.

    Returns on match:
        {
            name:       str,
            person_id:  int,
            confidence: float,
            metadata:   dict,
            ref_count:  int,
        }

    Returns on no match:
        {"name": "Unknown", "confidence": None}
    """
    _require_pool()
    vec_str = _vec_to_str(embedding)

    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                p.id,
                p.name,
                p.metadata,
                COUNT(fe2.id)                          AS ref_count,
                (fe.embedding <=> $1::vector)          AS distance
            FROM face_embeddings fe
            JOIN people p        ON p.id  = fe.person_id
            LEFT JOIN face_embeddings fe2 ON fe2.person_id = p.id
            GROUP BY p.id, p.name, p.metadata, fe.embedding
            ORDER BY distance ASC
            LIMIT 1
        """, vec_str)

    if row is None:
        return {"name": "Unknown", "confidence": None}

    distance   = float(row["distance"])
    confidence = round(1.0 - distance, 4)

    if distance > threshold:
        return {"name": "Unknown", "confidence": None}

    metadata = row["metadata"]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    return {
        "name":       row["name"],
        "person_id":  row["id"],
        "confidence": confidence,
        "metadata":   metadata or {},
        "ref_count":  int(row["ref_count"]),
    }


# ── Insert person ─────────────────────────────────────────────────────────────

async def insert_person(name:      str,
                         embedding: list[float],
                         metadata:  dict = None) -> int:
    """
    Insert a new person and their first face embedding.
    Returns the new person ID.
    Both inserts are wrapped in a transaction — either both succeed or neither.
    """
    _require_pool()
    vec_str  = _vec_to_str(embedding)
    meta_str = json.dumps(metadata or {})

    async with _pool.acquire() as conn:
        async with conn.transaction():
            person_id = await conn.fetchval("""
                INSERT INTO people (name, metadata)
                VALUES ($1, $2::jsonb)
                RETURNING id
            """, name, meta_str)

            await conn.execute("""
                INSERT INTO face_embeddings (person_id, embedding)
                VALUES ($1, $2::vector)
            """, person_id, vec_str)

    return person_id


# ── Get all people ────────────────────────────────────────────────────────────

async def get_all_people() -> list[dict]:
    """
    Return all people with their embedding counts.
    Ordered by embedding count descending (most-recognised first).
    """
    _require_pool()

    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                p.id,
                p.name,
                p.metadata,
                p.created_at,
                COUNT(fe.id) AS embedding_count
            FROM people p
            LEFT JOIN face_embeddings fe ON fe.person_id = p.id
            GROUP BY p.id
            ORDER BY embedding_count DESC, p.created_at DESC
        """)

    result = []
    for row in rows:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        result.append({
            "id":              row["id"],
            "name":            row["name"],
            "metadata":        metadata or {},
            "embedding_count": int(row["embedding_count"]),
            "created_at":      row["created_at"].isoformat(),
        })

    return result


# ── Get single person ─────────────────────────────────────────────────────────

async def get_person(person_id: int) -> Optional[dict]:
    """
    Return a single person record, or None if not found.
    """
    _require_pool()

    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                p.id,
                p.name,
                p.metadata,
                p.created_at,
                COUNT(fe.id) AS embedding_count
            FROM people p
            LEFT JOIN face_embeddings fe ON fe.person_id = p.id
            WHERE p.id = $1
            GROUP BY p.id
        """, person_id)

    if row is None:
        return None

    metadata = row["metadata"]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    return {
        "id":              row["id"],
        "name":            row["name"],
        "metadata":        metadata or {},
        "embedding_count": int(row["embedding_count"]),
        "created_at":      row["created_at"].isoformat(),
    }


# ── Update person ─────────────────────────────────────────────────────────────

async def update_person(person_id: int,
                         name:      Optional[str]  = None,
                         metadata:  Optional[dict] = None):
    """
    Update name and/or metadata for an existing person.
    Only provided non-None fields are written.
    """
    _require_pool()

    async with _pool.acquire() as conn:
        if name is not None:
            await conn.execute("""
                UPDATE people SET name = $1 WHERE id = $2
            """, name.strip(), person_id)

        if metadata is not None:
            await conn.execute("""
                UPDATE people SET metadata = $1::jsonb WHERE id = $2
            """, json.dumps(metadata), person_id)


# ── Embeddings ────────────────────────────────────────────────────────────────

async def get_embeddings_for_person(person_id: int) -> list[list[float]]:
    """
    Return all stored embedding vectors for a person as float lists.
    Ordered oldest first so diversity checks compare against
    the original registration embedding first.
    """
    _require_pool()

    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT embedding::text
            FROM face_embeddings
            WHERE person_id = $1
            ORDER BY created_at ASC
        """, person_id)

    return [_str_to_vec(row["embedding"]) for row in rows]


async def add_embedding_for_person(person_id: int,
                                    embedding:  list[float]) -> int:
    """
    Insert a new reference embedding for an existing person.
    Returns the updated total embedding count for this person.
    """
    _require_pool()
    vec_str = _vec_to_str(embedding)

    async with _pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                INSERT INTO face_embeddings (person_id, embedding)
                VALUES ($1, $2::vector)
            """, person_id, vec_str)

            total = await conn.fetchval("""
                SELECT COUNT(*) FROM face_embeddings WHERE person_id = $1
            """, person_id)

    return int(total)


# ── Statistics ────────────────────────────────────────────────────────────────

async def get_stats() -> dict:
    """
    Return aggregate database statistics for the /status endpoint.

    Returns:
        {
            total_people:     int,
            total_embeddings: int,
            needs_review:     int,
            db_size_mb:       float,
        }
    """
    _require_pool()

    async with _pool.acquire() as conn:

        total_people = await conn.fetchval(
            "SELECT COUNT(*) FROM people"
        )

        total_embeddings = await conn.fetchval(
            "SELECT COUNT(*) FROM face_embeddings"
        )

        # People flagged for manual review (auto-detected unknowns)
        needs_review = await conn.fetchval("""
            SELECT COUNT(*)
            FROM people
            WHERE metadata->>'needs_review' = 'true'
        """)

        # Approximate DB size on disk
        db_size_bytes = await conn.fetchval("""
            SELECT pg_database_size(current_database())
        """)

    db_size_mb = round(int(db_size_bytes or 0) / (1024 * 1024), 2)

    return {
        "total_people":     int(total_people     or 0),
        "total_embeddings": int(total_embeddings or 0),
        "needs_review":     int(needs_review     or 0),
        "db_size_mb":       db_size_mb,
    }