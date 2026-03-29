"""
integration/ig_server.py
─────────────────────────
Dashboard server for reviewing Instagram images and adding faces to TrueDataBase.
Runs on localhost:8767. Talks to Pi on port 8000.

Usage:
    python ig_server.py [--root ./ig_output] [--pi http://192.168.2.2:8000] [--port 8767]

Sessions are subfolders of --root that contain a catalog.json.
The most recent session is auto-selected on startup.
New scrapes can be triggered from the browser dashboard.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "Mac"))
_saved = sys.modules.pop("config", None)
import detector
if _saved is not None:
    sys.modules["config"] = _saved

log = logging.getLogger("ig_server")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

_THIS_DIR = Path(__file__).parent

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.getenv("IG_ROOT", "./ig_output"),
                   help="Base directory containing session folders (default ./ig_output)")
    p.add_argument("--pi", default=os.getenv("PI_URL", "http://192.168.2.2:8000"),
                   help="Pi API base URL")
    p.add_argument("--port", type=int, default=8767)
    return p.parse_args()

_args = parse_args()
ROOT_DIR = Path(_args.root).resolve()
PI_URL = _args.pi

# ── Session state ─────────────────────────────────────────────────────────────

_session: dict = {"dir": None, "name": None}  # currently active session

def _catalog_path() -> Optional[Path]:
    return (_session["dir"] / "catalog.json") if _session["dir"] else None

def _faces_cache_path() -> Optional[Path]:
    return (_session["dir"] / "faces_cache.json") if _session["dir"] else None

def _list_sessions() -> list[dict]:
    if not ROOT_DIR.exists():
        return []
    result = []
    for d in sorted(ROOT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "catalog.json").exists():
            result.append({"name": d.name, "path": str(d)})
    return result

# ── Background face scan ──────────────────────────────────────────────────────

_scan = {"done": 0, "total": 0, "running": False}
_scan_gen = 0
_cache_lock = threading.Lock()

def _scan_all_images(session_dir: Path, gen: int):
    catalog_file = session_dir / "catalog.json"
    faces_cache  = session_dir / "faces_cache.json"
    if not catalog_file.exists():
        return
    catalog   = json.loads(catalog_file.read_text())
    all_paths = [p for post in catalog for p in post.get("image_paths", [])]

    _scan["total"]   = len(all_paths)
    _scan["done"]    = 0
    _scan["running"] = True

    for img_path in all_paths:
        if _scan_gen != gen:          # session switched — abort
            _scan["running"] = False
            return
        key = Path(img_path).name
        with _cache_lock:
            try:
                cache = json.loads(faces_cache.read_text()) if faces_cache.exists() else {}
            except (json.JSONDecodeError, OSError):
                cache = {}
            if key not in cache:
                cache[key] = detect_faces_for_image(img_path)
                faces_cache.write_text(json.dumps(cache, indent=2))
        _scan["done"] += 1

    _scan["running"] = False
    log.info("Scan complete: %d images in %s", _scan["total"], session_dir.name)

def _activate_session(path: Path):
    global _scan_gen
    _session["dir"]  = path
    _session["name"] = path.name
    _scan.update({"done": 0, "total": 0, "running": False})
    _scan_gen += 1
    gen = _scan_gen
    threading.Thread(target=_scan_all_images, args=(path, gen), daemon=True).start()
    log.info("Active session: %s", path.name)

# ── Scrape subprocess ─────────────────────────────────────────────────────────

_scrape_job: dict = {"running": False, "username": "", "lines": [], "returncode": None}

def _monitor_scrape(proc):
    for line in proc.stdout:
        line = line.rstrip()
        log.info("[scrape] %s", line)
        _scrape_job["lines"].append(line)
        if len(_scrape_job["lines"]) > 100:
            _scrape_job["lines"].pop(0)
    proc.wait()
    _scrape_job["returncode"] = proc.returncode
    _scrape_job["running"]    = False
    if proc.returncode == 0:
        sessions = _list_sessions()
        if sessions:
            _activate_session(Path(sessions[0]["path"]))
    log.info("Scrape finished (exit %d)", proc.returncode)

# ── Face detection cache helpers ──────────────────────────────────────────────

def load_cache() -> dict:
    p = _faces_cache_path()
    if p and p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def save_cache(cache: dict):
    p = _faces_cache_path()
    if p:
        p.write_text(json.dumps(cache, indent=2))

def detect_faces_for_image(image_path: str) -> list[dict]:
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        return []
    faces = detector.detect(frame)
    return [
        {
            "index": i,
            "bbox":       f["bbox"],
            "det_score":  f["det_score"],
            "embedding":  f["embedding"],
            "registered_to": None,
        }
        for i, f in enumerate(faces)
    ]

def get_faces(image_key: str, image_path: str) -> list[dict]:
    cache = load_cache()
    if image_key not in cache:
        log.info("Detecting faces in %s", image_path)
        cache[image_key] = detect_faces_for_image(image_path)
        save_cache(cache)
    return cache[image_key]

def update_face_registration(image_key: str, face_index: int, person_id: int, person_name: str):
    cache = load_cache()
    if image_key in cache:
        for face in cache[image_key]:
            if face["index"] == face_index:
                face["registered_to"]   = person_id
                face["registered_name"] = person_name
    save_cache(cache)

# ── Pi API helpers ────────────────────────────────────────────────────────────

async def pi_get(path: str, params: dict = None) -> dict | list:
    async with httpx.AsyncClient(base_url=PI_URL, timeout=10.0) as c:
        r = await c.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

async def pi_post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(base_url=PI_URL, timeout=10.0) as c:
        r = await c.post(path, json=payload)
        r.raise_for_status()
        return r.json()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Instagram → TrueDataBase")

@app.on_event("startup")
def startup():
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading InsightFace model…")
    detector.init_worker_model()
    sessions = _list_sessions()
    if sessions:
        _activate_session(Path(sessions[0]["path"]))
        log.info("Auto-selected session: %s", sessions[0]["name"])
    else:
        log.info("No sessions found in %s — use the dashboard to start a scrape", ROOT_DIR)

# ── HTML ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTMLResponse((_THIS_DIR / "ig_dashboard.html").read_text())

# ── Images ────────────────────────────────────────────────────────────────────

@app.get("/img")
def serve_image(path: str = Query(...)):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Image not found")
    ext = p.suffix.lower()
    mt  = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
           ".png": "image/png",  ".webp": "image/webp"}.get(ext, "image/jpeg")
    return FileResponse(str(p), media_type=mt)

# ── Sessions API ──────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def get_sessions():
    active = str(_session["dir"]) if _session["dir"] else None
    sessions = _list_sessions()
    for s in sessions:
        s["active"] = (s["path"] == active)
    return sessions

@app.get("/api/session")
def get_current_session():
    if not _session["dir"]:
        return {"active": False}
    return {"active": True, "name": _session["name"], "path": str(_session["dir"])}

class SessionRequest(BaseModel):
    path: str

@app.post("/api/session")
def set_session(req: SessionRequest):
    p = Path(req.path)
    if not p.is_dir() or not (p / "catalog.json").exists():
        raise HTTPException(400, "Not a valid session directory")
    _activate_session(p)
    return {"ok": True, "name": p.name}

# ── Scrape API ────────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    username:     str
    count:        int  = 100
    mentions_only: bool = False

@app.post("/api/scrape")
def start_scrape(req: ScrapeRequest):
    if _scrape_job["running"]:
        raise HTTPException(409, "A scrape is already running")
    script = _THIS_DIR / "ig_scraper.py"
    cmd = [
        sys.executable, str(script),
        req.username.lstrip("@"),
        "--count",  str(req.count),
        "--output", str(ROOT_DIR),
    ]
    if req.mentions_only:
        cmd.append("--mentions-only")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _scrape_job.update({"running": True, "username": req.username,
                         "lines": [], "returncode": None})
    threading.Thread(target=_monitor_scrape, args=(proc,), daemon=True).start()
    log.info("Started scrape for @%s (count=%d)", req.username, req.count)
    return {"ok": True}

@app.get("/api/scrape/status")
def scrape_status():
    return _scrape_job

# ── Scan status ───────────────────────────────────────────────────────────────

@app.get("/api/scan_status")
def scan_status():
    return _scan

# ── Posts ─────────────────────────────────────────────────────────────────────

@app.get("/api/posts")
def list_posts():
    if _session["dir"] is None:
        return []
    cp = _session["dir"] / "catalog.json"
    if not cp.exists():
        return []
    catalog = json.loads(cp.read_text())
    cache   = load_cache()
    result  = []
    for post in catalog:
        faces_info         = []
        images_with_counts = []
        for img_path in post.get("image_paths", []):
            key    = Path(img_path).name
            cached = cache.get(key)
            face_count = len(cached) if cached is not None else None
            images_with_counts.append({"path": img_path, "face_count": face_count})
            if cached:
                for f in cached:
                    faces_info.append({
                        "image_path":      img_path,
                        "index":           f["index"],
                        "registered_to":   f.get("registered_to"),
                        "registered_name": f.get("registered_name"),
                    })
        result.append({
            "shortcode":    post["shortcode"],
            "url":          post["url"],
            "caption":      post["caption"],
            "mentions":     post["mentions"],
            "timestamp":    post["timestamp"],
            "image_paths":  post["image_paths"],
            "images":       images_with_counts,
            "face_summary": faces_info,
        })
    return result

@app.get("/api/posts/{shortcode}/faces")
def get_post_faces(shortcode: str, image_path: str = Query(...)):
    p = Path(image_path)
    if not p.exists():
        raise HTTPException(404, f"Image not found: {image_path}")
    faces = get_faces(p.name, image_path)
    return [{k: v for k, v in f.items() if k != "embedding"} for f in faces]

# ── Auto-identify faces via Pi ────────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    image_path: str

@app.post("/api/identify_faces")
async def identify_faces(req: IdentifyRequest):
    """
    Query Pi /identify for every cached face in the image.
    Results (including None for no-match) are written back to the cache
    so repeated calls are free. Returns list of {face_index, pi_match}.
    """
    key   = Path(req.image_path).name
    cache = load_cache()
    faces = cache.get(key, [])
    if not faces:
        return []

    results      = []
    cache_dirty  = False

    for face in faces:
        # Already queried — return cached result (None means "queried, no match")
        if "pi_match" in face:
            results.append({"face_index": face["index"], "pi_match": face["pi_match"]})
            continue

        embedding = face.get("embedding")
        if not embedding:
            face["pi_match"] = None
            cache_dirty = True
            results.append({"face_index": face["index"], "pi_match": None})
            continue

        try:
            match = await pi_post("/identify", {"embedding": embedding})
            pi_match = (
                None if (not match.get("name") or match["name"] == "Unknown")
                else {
                    "name":       match["name"],
                    "person_id":  match.get("person_id"),
                    "confidence": match.get("confidence"),
                }
            )
        except Exception as e:
            log.warning("Pi identify failed for face %d: %s", face["index"], e)
            pi_match = None     # don't cache on network error (allow retry)
            results.append({"face_index": face["index"], "pi_match": None})
            continue

        face["pi_match"] = pi_match
        cache_dirty = True
        results.append({"face_index": face["index"], "pi_match": pi_match})

    if cache_dirty:
        save_cache(cache)

    return results

# ── Register face with Pi ─────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    image_path: str
    face_index: int
    name:       str
    person_id:  Optional[int] = None

@app.post("/api/register")
async def register_face(req: RegisterRequest):
    key   = Path(req.image_path).name
    cache = load_cache()
    faces = cache.get(key)
    if not faces:
        raise HTTPException(400, "Faces not detected yet — load the image first")
    face = next((f for f in faces if f["index"] == req.face_index), None)
    if face is None:
        raise HTTPException(404, f"Face index {req.face_index} not found")

    if req.person_id is None:
        result = await pi_post("/register", {
            "name":      req.name,
            "embedding": face["embedding"],
            "metadata":  {"source": "instagram", "auto_detected": False, "needs_review": False},
        })
        person_id = result.get("person_id")
    else:
        result    = await pi_post(f"/person/{req.person_id}/embeddings",
                                  {"embedding": face["embedding"]})
        person_id = req.person_id

    if not result.get("error"):
        update_face_registration(key, req.face_index, person_id, req.name)

    return {**result, "person_id": person_id, "name": req.name}

# ── Pi proxy ──────────────────────────────────────────────────────────────────

@app.get("/api/pi/people")
async def pi_people():
    return await pi_get("/people")

@app.get("/api/pi/status")
async def pi_status():
    return await pi_get("/status")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info("Dashboard → http://localhost:%d", _args.port)
    log.info("Pi API   → %s", PI_URL)
    log.info("Sessions → %s", ROOT_DIR)
    uvicorn.run(app, host="localhost", port=_args.port, log_level="warning")
