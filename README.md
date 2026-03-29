# TrueDataBase

Distributed face recognition system. A Raspberry Pi holds the database and runs the identification API. A Mac does the heavy processing (InsightFace/ArcFace embeddings) and hosts the browser dashboard. An Instagram integration lets you scrape public profiles and add identified faces directly into the database from a review UI.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│  Raspberry Pi                                           │
│  PostgreSQL + pgvector  ←→  FastAPI on :8000            │
└──────────────────────────┬──────────────────────────────┘
                           │  HTTP (LAN / Tailscale)
┌──────────────────────────▼──────────────────────────────┐
│  Mac                                                    │
│  InsightFace workers  →  dashboard  ws://localhost:8765 │
│  Camera scan page        HTTPS :8766                    │
│  Instagram review        HTTP  :8767                    │
│  Dashboard HTML server   HTTP  :8769                    │
└─────────────────────────────────────────────────────────┘
```

**Pi** — stores embeddings, answers `/identify`, `/register`, `/people`.
**Mac** — detects faces in images/video, identifies via Pi, hosts all UIs.
**Instagram integration** — scrapes public profiles, lets you review detected faces and register them into the Pi database.

---

## Directory Structure

```
TrueDataBase/
├── Pi/
│   ├── main.py          # FastAPI app (all HTTP endpoints)
│   ├── database.py      # asyncpg pool, pgvector search, CRUD
│   └── requirements.txt
├── Mac/
│   ├── main.py          # Entry point — spawns all processes
│   ├── config.py        # All tunable settings (edit this)
│   ├── dashboard.html   # Browser UI
│   ├── dashboard_server.py  # WebSocket bridge
│   ├── scan_server.py   # HTTPS camera scan page
│   ├── worker.py        # Face detection worker process
│   ├── worker_pool.py   # Worker lifecycle management
│   ├── detector.py      # InsightFace wrapper
│   ├── client.py        # Pi HTTP client
│   ├── tracker.py       # Per-worker session cache
│   ├── queue_manager.py # Job queue
│   └── upload.py        # File type detection
├── integration/
│   ├── ig_scraper.py    # Instagram scraper CLI
│   ├── ig_server.py     # Instagram review server
│   ├── ig_dashboard.html
│   └── requirements.txt
├── schema.sql           # PostgreSQL schema (run once on Pi)
└── requirements.txt     # Mac dependencies
```

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| Pi        | Raspberry Pi 4 or 5, Raspberry Pi OS (64-bit) |
| Pi        | PostgreSQL 14+ with pgvector extension |
| Mac       | macOS, Python 3.10+ |
| Network   | Pi and Mac on same LAN, or connected via Tailscale |

---

## Pi Setup

### 1. Install PostgreSQL and pgvector

```bash
# PostgreSQL
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# pgvector (build from source)
sudo apt install -y postgresql-server-dev-all git build-essential
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
cd ..
```

### 2. Create the database

```bash
sudo -u postgres psql -c "CREATE DATABASE facedb;"
sudo -u postgres psql -c "CREATE USER pi WITH PASSWORD 'password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE facedb TO pi;"
```

### 3. Apply the schema

```bash
cd /path/to/TrueDataBase
sudo -u postgres psql -d facedb -f schema.sql
```

### 4. Install Pi dependencies

```bash
cd Pi
pip3 install fastapi uvicorn asyncpg
```

### 5. Set the database URL

```bash
export DATABASE_URL="postgresql://pi:password@localhost/facedb"
```

Add to `~/.bashrc` or `/etc/environment` to persist across reboots.

### 6. Start the Pi API server

```bash
cd Pi
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Run as a service (recommended):**

```ini
# /etc/systemd/system/truedatabase.service
[Unit]
Description=TrueDataBase Pi API
After=postgresql.service

[Service]
User=pi
WorkingDirectory=/home/pi/TrueDataBase/Pi
Environment=DATABASE_URL=postgresql://pi:password@localhost/facedb
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable truedatabase
sudo systemctl start truedatabase
```

### 7. Note the Pi's IP address

```bash
hostname -I   # e.g. 192.168.2.2
```

---

## Mac Setup

### 1. Install dependencies

```bash
cd TrueDataBase
pip3 install -r requirements.txt
pip3 install -r integration/requirements.txt
```

### 2. Set the Pi URL

Edit `Mac/config.py`:

```python
PI_URL = "http://192.168.2.2:8000"   # ← your Pi's IP
```

All other settings (thresholds, ports, worker count) are in `Mac/config.py` and documented inline.

### 3. Start the Mac dashboard

```bash
cd Mac
python3 main.py
```

You will be prompted for how many CPU cores to allocate (1 dashboard + N workers). Press Enter to use the recommended value.

The browser opens automatically at `http://localhost:8769/dashboard.html`.

**What starts:**

| Process | URL | Purpose |
|---------|-----|---------|
| Dashboard WebSocket | `ws://localhost:8765` | Real-time UI bridge |
| Camera scan page | `https://<LAN-IP>:8766` | Identify faces from phone camera |
| Instagram review | `http://localhost:8767` | Scrape & register faces from Instagram |
| Dashboard HTTP server | `http://localhost:8769` | Serves dashboard.html |
| Worker processes | — | Face detection (one per allocated core) |

---

## Using the Mac Dashboard

### Processing images and video

1. Open `http://localhost:8769/dashboard.html` (opens automatically on start)
2. Paste file or folder paths into the **Queue** tab input and click **Add Files**
3. Click **Start** — workers process jobs in parallel
4. The **Workers** panel shows live progress per core
5. Identified faces are written to the Pi database automatically

### Reviewing the database

- Click the **Database** tab to browse all people in the Pi database
- Search by name or filter to `needs_review` entries (auto-registered unknowns)
- Click a name to rename it inline

### Camera scan (mobile)

- Open `https://<LAN-IP>:8766` on any phone on the same network
- Accept the self-signed certificate warning
- Point camera at a face — identification results appear in real time

---

## Instagram Integration

### Step 1 — Scrape a profile

```bash
cd integration
python3 ig_scraper.py <username> --count 100
```

Downloads images from the last 100 posts to:
```
integration/ig_output/YYYY-MM-DD_<username>_<N>posts/
├── catalog.json      # post metadata (captions, @mentions, timestamps)
└── images/           # downloaded images
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--count N` | 100 | Maximum posts to fetch |
| `--output DIR` | `./ig_output` | Base output directory |
| `--mentions-only` | off | Only download posts that @mention someone |

Running the same account again creates a new dated folder — previous downloads are never overwritten.

### Step 2 — Review faces in the dashboard

The Instagram review UI launches automatically when you run `Mac/main.py`. Access it two ways:

- **From the Mac dashboard:** click the **Instagram** button in the topbar
- **Directly:** open `http://localhost:8767` in any browser

### Step 3 — Assign faces to people

1. **Select a session** from the dropdown (auto-selects the most recent scrape)
2. **Browse posts** in the left panel — face count badges show how many faces InsightFace detected in each image
3. **Click a post** to open the image. Face bboxes are drawn automatically:
   - 🟢 Green = already registered
   - 🔵 Blue = Pi recognises this face (≥70% confidence)
   - 🟣 Purple = weak Pi match (50–70%)
   - Plain = not yet identified
4. **Click a face** to open the assignment panel:
   - If Pi already recognises the face, a **Confirm** banner appears — one click registers the embedding
   - Caption @mentions appear as chips — click one to auto-fill the name and pre-select from the database
   - Smart auto-fill: if there's one unregistered face and one caption tag, the name is pre-filled automatically
5. **Register new person** — creates a new entry in the Pi database
6. **Add to existing person** — select from dropdown and add this face as an additional reference embedding

### Scraping from inside the dashboard

Click **+ New Scrape** in the Instagram panel topbar, enter a username and post count, and click **Start**. Progress streams live in the status bar. The new session is auto-selected when the scrape finishes.

### Filtering

Check **1–8 faces only** in the post list to hide images with no faces or crowd shots. The filter applies to both the post list and the image carousel within a post.

---

## Configuration Reference (`Mac/config.py`)

```python
PI_URL                    = "http://192.168.2.2:8000"  # Pi address
PI_IDENTIFY_THRESHOLD     = 0.4    # cosine distance cutoff (lower = stricter)

DASHBOARD_PORT            = 8765   # WebSocket
SCAN_SERVER_PORT          = 8766   # Camera HTTPS page
IG_SERVER_PORT            = 8767   # Instagram review
DASHBOARD_HTTP_PORT       = 8769   # Dashboard HTML server

POLL_RATE_SECONDS         = 4.0    # Video frame sample rate
BATCH_SIZE                = 4      # Parallel frames per worker
EMBEDDING_DIVERSITY_THRESHOLD = 0.15  # Min distance for new reference embedding
MAX_EMBEDDINGS_PER_PERSON = 20     # Cap per person

AUTO_REGISTER_NEW         = True   # Auto-register unknown faces as Person_XXXXXX
DELETE_AFTER_SCAN         = False  # Delete source files after processing
INSIGHTFACE_DET_SIZE      = (640, 640)  # Detection resolution
```

---

## Pi API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Pi health, DB stats, uptime |
| `POST` | `/identify` | `{embedding}` → closest person match |
| `POST` | `/register` | `{name, embedding, metadata}` → new person |
| `GET` | `/people` | All people with embedding counts |
| `GET` | `/person/{id}` | Single person record |
| `POST` | `/person/{id}/embeddings` | Add reference embedding to existing person |
| `PATCH` | `/person/{id}` | Update name / metadata |

---

## Tailscale (optional)

If the Pi and Mac are not on the same local network:

1. Install Tailscale on both devices: `https://tailscale.com/download`
2. Authenticate both: `sudo tailscale up`
3. Find the Pi's Tailscale IP: `tailscale ip -4`
4. Update `Mac/config.py`: `PI_URL = "http://100.x.x.x:8000"`

---

## Troubleshooting

**Pi badge shows offline in dashboard**
→ Check Pi IP in `Mac/config.py`. Verify `curl http://<PI_IP>:8000/status` returns JSON.

**InsightFace model download on first run**
→ `main.py` downloads `buffalo_l` (~300 MB) once before spawning workers. Requires internet access on first launch.

**Port already in use**
→ A previous server is still running. Find and kill it: `lsof -ti :<port> | xargs kill`

**Camera scan page shows certificate warning**
→ Expected — the cert is self-signed. Accept the exception in your browser/phone once.

**Instagram scraper fails on private profiles**
→ Only public profiles are supported. Private accounts return a `ProfileNotExistsException`.

**`Address already in use` for ig_server on launch**
→ A standalone `ig_server.py` process is still running from a previous session. Kill it: `lsof -ti :8767 | xargs kill`
