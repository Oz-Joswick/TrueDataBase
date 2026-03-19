"""
face_system/mac/client.py
──────────────────────────
All HTTP communication with the Raspberry Pi FastAPI server.
Responsibilities:
  - Identify a face embedding against the Pi database
  - Register a new person with their first embedding
  - Fetch and add reference embeddings for known people
  - Retrieve people list and individual person records
  - Update person metadata
  - Fetch Pi system status

Every function is async and uses a shared httpx.AsyncClient session
for connection reuse across calls within the same worker event loop.
All network errors are caught and returned as structured error dicts
so workers never crash on Pi unavailability.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import httpx

from config import PI_URL, PI_REQUEST_TIMEOUT, PI_IDENTIFY_THRESHOLD


# ── Shared client session ─────────────────────────────────────────────────────
# One AsyncClient per worker process event loop.
# Reuses TCP connections across calls — important for throughput
# when processing many faces per second.

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=PI_URL,
            timeout=PI_REQUEST_TIMEOUT,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )
    return _client


async def close_client():
    """Call at worker shutdown to cleanly close the HTTP session."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# ── Error wrapper ─────────────────────────────────────────────────────────────

def _error(msg: str, **extra) -> dict:
    """Consistent error response shape so callers never need to check types."""
    return {"error": True, "message": msg, **extra}


async def _post(endpoint: str, payload: dict) -> dict:
    """POST JSON to Pi. Returns parsed response or error dict."""
    try:
        client = _get_client()
        resp   = await client.post(endpoint, json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return _error("Pi unreachable", endpoint=endpoint)
    except httpx.TimeoutException:
        return _error("Pi timeout", endpoint=endpoint)
    except httpx.HTTPStatusError as e:
        return _error(f"HTTP {e.response.status_code}", endpoint=endpoint)
    except Exception as e:
        return _error(str(e), endpoint=endpoint)


async def _get(endpoint: str, params: dict = None) -> dict | list:
    """GET from Pi. Returns parsed response or error dict."""
    try:
        client = _get_client()
        resp   = await client.get(endpoint, params=params or {})
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return _error("Pi unreachable", endpoint=endpoint)
    except httpx.TimeoutException:
        return _error("Pi timeout", endpoint=endpoint)
    except httpx.HTTPStatusError as e:
        return _error(f"HTTP {e.response.status_code}", endpoint=endpoint)
    except Exception as e:
        return _error(str(e), endpoint=endpoint)


async def _patch(endpoint: str, payload: dict) -> dict:
    """PATCH JSON to Pi. Returns parsed response or error dict."""
    try:
        client = _get_client()
        resp   = await client.patch(endpoint, json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return _error("Pi unreachable", endpoint=endpoint)
    except httpx.TimeoutException:
        return _error("Pi timeout", endpoint=endpoint)
    except httpx.HTTPStatusError as e:
        return _error(f"HTTP {e.response.status_code}", endpoint=endpoint)
    except Exception as e:
        return _error(str(e), endpoint=endpoint)


# ── Face identification ───────────────────────────────────────────────────────

async def identify(embedding:  list[float],
                   threshold:  float = PI_IDENTIFY_THRESHOLD) -> dict:
    """
    Search the Pi database for the closest matching face.

    Returns on match:
        {
            name:       str,
            person_id:  int,
            confidence: float,   # 0–1, higher is better
            metadata:   dict,
            ref_count:  int,
        }

    Returns on no match:
        {"name": "Unknown", "confidence": None}

    Returns on error:
        {"error": True, "message": str, "name": "Unknown"}
    """
    result = await _post("/identify", {
        "embedding": embedding,
        "threshold": threshold,
    })

    if result.get("error"):
        return {**result, "name": "Unknown", "confidence": None}

    return result


async def identify_many(embeddings: list[list[float]],
                         threshold:  float = PI_IDENTIFY_THRESHOLD
                         ) -> list[dict]:
    """
    Identify multiple embeddings concurrently.
    Returns a list of results in the same order as the input.
    """
    tasks = [identify(emb, threshold) for emb in embeddings]
    return list(await asyncio.gather(*tasks, return_exceptions=False))


# ── Person registration ───────────────────────────────────────────────────────

async def register(name:      str,
                   embedding: list[float],
                   metadata:  dict = None) -> dict:
    """
    Register a new person with their first face embedding.

    Returns:
        {"success": True, "person_id": int}
        or error dict on failure.
    """
    return await _post("/register", {
        "name":      name,
        "embedding": embedding,
        "metadata":  metadata or {},
    })


# ── Embeddings ────────────────────────────────────────────────────────────────

async def get_embeddings(person_id: int) -> list[list[float]]:
    """
    Fetch all stored embeddings for a person.
    Returns a list of 512-dim float lists.
    Returns empty list on error so diversity checks degrade gracefully.
    """
    result = await _get(f"/person/{person_id}/embeddings")

    if isinstance(result, dict) and result.get("error"):
        return []

    return result.get("embeddings", [])


async def add_embedding(person_id: int, embedding: list[float]) -> dict:
    """
    Add a new reference embedding to an existing person.

    Returns:
        {"success": True, "person_id": int, "total_embeddings": int}
        or error dict on failure.
    """
    return await _post(f"/person/{person_id}/embeddings", {
        "embedding": embedding,
    })


# ── People ────────────────────────────────────────────────────────────────────

async def get_all_people() -> list[dict]:
    """
    Fetch all people from the Pi database with their embedding counts.
    Used by the dashboard viz panel.

    Returns list of:
        {
            id:              int,
            name:            str,
            metadata:        dict,
            embedding_count: int,
            created_at:      str,
        }
    Returns empty list on error.
    """
    result = await _get("/people")

    if isinstance(result, dict) and result.get("error"):
        return []

    return result if isinstance(result, list) else []


async def get_person(person_id: int) -> dict:
    """
    Fetch a single person record by ID.

    Returns person dict or error dict.
    """
    return await _get(f"/person/{person_id}")


async def update_person(person_id: int,
                         name:      str = None,
                         metadata:  dict = None) -> dict:
    """
    Update a person's name and/or metadata.
    Only provided fields are updated — omitted fields are unchanged.

    Returns:
        {"success": True}
        or error dict.
    """
    payload = {}
    if name     is not None: payload["name"]     = name
    if metadata is not None: payload["metadata"] = metadata

    if not payload:
        return _error("Nothing to update — provide name or metadata")

    return await _patch(f"/person/{person_id}", payload)


# ── Pi system status ──────────────────────────────────────────────────────────

async def get_pi_status() -> dict:
    """
    Fetch system status from Pi.
    Called by dashboard_server on its polling interval.

    Returns:
        {
            total_people:    int,
            total_embeddings: int,
            uptime:          str,
            version:         str,
            db_size_mb:      float,
        }
    or error dict.
    """
    return await _get("/status")


# ── Convenience: fire and forget ──────────────────────────────────────────────

async def maybe_add_embedding(person_id:            int,
                               embedding:            list[float],
                               existing_embeddings:  list[list[float]],
                               threshold:            float,
                               max_refs:             int) -> Optional[int]:
    """
    Fetch existing embeddings, check diversity, add if warranted.
    Returns new total_embeddings count, or None if nothing was added.

    Combines get_embeddings + diversity check + add_embedding into
    one call so worker.py stays concise.
    """
    from tracker import is_diverse_enough

    if len(existing_embeddings) >= max_refs:
        return None

    if not is_diverse_enough(embedding, existing_embeddings, threshold):
        return None

    result = await add_embedding(person_id, embedding)
    if result.get("error"):
        return None

    return result.get("total_embeddings")