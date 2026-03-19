"""
face_system/pi/main.py
───────────────────────
FastAPI application. All HTTP endpoints.
Responsibilities:
  - Identify a face embedding via pgvector nearest-neighbour search
  - Register new people and their first embedding
  - Add reference embeddings to existing people
  - Serve people list and individual records
  - Update person name / metadata
  - Report system status for the dashboard Pi panel

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Or via systemd (see README).
"""

import os
import time
from datetime import datetime
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database as db


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool()
    print("[api] database pool ready")
    yield
    await db.close_pool()
    print("[api] database pool closed")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="face_system Pi API",
    version="1.0.0",
    description="Face recognition database server.",
    lifespan=lifespan,
)

# Allow the Mac (any local origin) to call the API.
# Tighten this in production if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Record startup time for uptime reporting
_started_at = time.monotonic()


# ── Request / response models ─────────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    embedding: list[float]
    threshold: float = 0.4          # cosine distance cutoff

class RegisterRequest(BaseModel):
    name:      str
    embedding: list[float]
    metadata:  dict = {}

class AddEmbeddingRequest(BaseModel):
    embedding: list[float]

class UpdatePersonRequest(BaseModel):
    name:     Optional[str] = None
    metadata: Optional[dict] = None


# ── Identify ──────────────────────────────────────────────────────────────────

@app.post("/identify")
async def identify(req: IdentifyRequest):
    """
    Search the database for the closest matching face.

    Returns on match:
        {name, person_id, confidence, metadata, ref_count}

    Returns on no match:
        {name: "Unknown", confidence: null}
    """
    if len(req.embedding) != 512:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 512-dim embedding, got {len(req.embedding)}"
        )

    match = await db.search(req.embedding, req.threshold)
    return match


# ── Register ──────────────────────────────────────────────────────────────────

@app.post("/register", status_code=201)
async def register(req: RegisterRequest):
    """
    Add a new person to the database with their first face embedding.

    Returns:
        {success: true, person_id: int}
    """
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    if len(req.embedding) != 512:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 512-dim embedding, got {len(req.embedding)}"
        )

    person_id = await db.insert_person(
        name      = req.name.strip(),
        embedding = req.embedding,
        metadata  = req.metadata,
    )

    return {"success": True, "person_id": person_id}


# ── People list ───────────────────────────────────────────────────────────────

@app.get("/people")
async def get_people():
    """
    Return all people with their embedding counts.
    Used by the dashboard viz panel and Pi status polling.

    Returns list of:
        {id, name, metadata, embedding_count, created_at}
    """
    return await db.get_all_people()


# ── Single person ─────────────────────────────────────────────────────────────

@app.get("/person/{person_id}")
async def get_person(person_id: int):
    """
    Return a single person record.

    Returns:
        {id, name, metadata, embedding_count, created_at}
    """
    person = await db.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404,
                            detail=f"Person {person_id} not found")
    return person


@app.patch("/person/{person_id}")
async def update_person(person_id: int, req: UpdatePersonRequest):
    """
    Update a person's name and/or metadata.
    Only provided fields are written — omitted fields are unchanged.

    Returns:
        {success: true}
    """
    if req.name is None and req.metadata is None:
        raise HTTPException(status_code=422,
                            detail="Provide name or metadata to update")

    exists = await db.get_person(person_id)
    if not exists:
        raise HTTPException(status_code=404,
                            detail=f"Person {person_id} not found")

    await db.update_person(
        person_id = person_id,
        name      = req.name,
        metadata  = req.metadata,
    )
    return {"success": True}


# ── Embeddings ────────────────────────────────────────────────────────────────

@app.get("/person/{person_id}/embeddings")
async def get_embeddings(person_id: int):
    """
    Return all stored embeddings for a person.
    Used by Mac client to check diversity before adding a new reference.

    Returns:
        {person_id, total_embeddings, embeddings: [[float, ...]]}
    """
    exists = await db.get_person(person_id)
    if not exists:
        raise HTTPException(status_code=404,
                            detail=f"Person {person_id} not found")

    embeddings = await db.get_embeddings_for_person(person_id)
    return {
        "person_id":        person_id,
        "total_embeddings": len(embeddings),
        "embeddings":       embeddings,
    }


@app.post("/person/{person_id}/embeddings", status_code=201)
async def add_embedding(person_id: int, req: AddEmbeddingRequest):
    """
    Add a new reference embedding to an existing person.
    Called by the Mac after a diversity check confirms the angle is novel.

    Returns:
        {success: true, person_id: int, total_embeddings: int}
    """
    if len(req.embedding) != 512:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 512-dim embedding, got {len(req.embedding)}"
        )

    exists = await db.get_person(person_id)
    if not exists:
        raise HTTPException(status_code=404,
                            detail=f"Person {person_id} not found")

    total = await db.add_embedding_for_person(person_id, req.embedding)
    return {
        "success":          True,
        "person_id":        person_id,
        "total_embeddings": total,
    }


# ── System status ─────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    """
    System health and statistics.
    Polled by the Mac dashboard server every PI_POLL_INTERVAL seconds.

    Returns:
        {
            status:            "online",
            total_people:      int,
            total_embeddings:  int,
            needs_review:      int,
            uptime:            str,
            db_size_mb:        float,
            version:           str,
            ts:                str (ISO 8601),
        }
    """
    stats   = await db.get_stats()
    uptime  = time.monotonic() - _started_at

    hours   = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    seconds = int(uptime % 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    return {
        "status":           "online",
        "total_people":     stats.get("total_people",     0),
        "total_embeddings": stats.get("total_embeddings", 0),
        "needs_review":     stats.get("needs_review",     0),
        "uptime":           uptime_str,
        "db_size_mb":       stats.get("db_size_mb",       0.0),
        "version":          "1.0.0",
        "ts":               datetime.now().isoformat(),
    }


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Minimal liveness probe. Returns 200 if the server is running."""
    return {"ok": True}