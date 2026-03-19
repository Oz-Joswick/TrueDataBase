"""
face_system/mac/dashboard_server.py
─────────────────────────────────────
Runs on the dashboard core (core 0).
Responsibilities:
  - Serve a websocket on ws://localhost:8765
  - Drain state updates from workers via state_queue → broadcast to browser
  - Receive commands from browser → act on them (start, stop, add files, etc.)
  - Poll Pi periodically for status → broadcast to browser
  - Maintain a live snapshot of system state for late-connecting browsers
"""

import asyncio
import json
import multiprocessing
import os
import time
from datetime import datetime
from typing import Optional

import httpx
import websockets
from websockets.server import WebSocketServerProtocol

from config import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    PI_URL,
    PI_POLL_INTERVAL,
    PI_REQUEST_TIMEOUT,
    SHOW_PREVIEWS_DEFAULT,
    SHOW_VIZ_DEFAULT,
    DIVERSITY_CHECK_DEFAULT,
    AUTO_REGISTER_NEW,
    DELETE_AFTER_SCAN,
)

_PI_URL = PI_URL   # local alias so it's accessible inside closures

# Build the scan server URL once at import time
def _scan_url() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "localhost"
    from config import SCAN_SERVER_PORT
    return f"https://{ip}:{SCAN_SERVER_PORT}"

_SCAN_URL = _scan_url()
from queue_manager import QueueManager, JobStatus


# ── Live state snapshot ───────────────────────────────────────────────────────
# Kept in memory so any browser that connects mid-session gets the full picture.

class SystemState:
    def __init__(self):
        self.workers:  dict[int, dict] = {}   # worker_id → latest state
        self.people:   list[dict]      = []
        self.pi:       dict            = {"status": "unknown"}
        self.settings: dict            = {
            "show_previews":   SHOW_PREVIEWS_DEFAULT,
            "show_viz":        SHOW_VIZ_DEFAULT,
            "diversity_check": DIVERSITY_CHECK_DEFAULT,
            "auto_register":   AUTO_REGISTER_NEW,
            "delete_after":    DELETE_AFTER_SCAN,
        }
        self.running: bool = False

    def full_snapshot(self, queue_manager: QueueManager) -> dict:
        """Complete state dump sent to newly connected browsers."""
        return {
            "type":     "snapshot",
            "workers":  list(self.workers.values()),
            "jobs":     queue_manager.to_dict_list(),
            "queue":    queue_manager.summary(),
            "people":   self.people,
            "pi":       self.pi,
            "pi_url":   _PI_URL,
            "scan_url": _SCAN_URL,
            "settings": self.settings,
            "running":  self.running,
            "ts":       datetime.now().isoformat(),
        }


# ── Pi polling ────────────────────────────────────────────────────────────────

async def _poll_pi() -> dict:
    """Fetch /status from Pi. Returns a status dict regardless of outcome."""
    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=PI_REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{PI_URL}/status")
            ping_ms = round((time.monotonic() - t0) * 1000, 1)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status":   "online",
                    "ping_ms":  ping_ms,
                    "api":      "online",
                    "db_rows":  data.get("total_people", "—"),
                    "uptime":   data.get("uptime", "—"),
                    "version":  data.get("version", "—"),
                    "ts":       datetime.now().isoformat(),
                }
            return {
                "status":  "error",
                "ping_ms": ping_ms,
                "api":     f"HTTP {resp.status_code}",
                "ts":      datetime.now().isoformat(),
            }
    except httpx.ConnectError:
        return {"status": "offline", "api": "unreachable",
                "ts": datetime.now().isoformat()}
    except httpx.TimeoutException:
        return {"status": "timeout", "api": "timeout",
                "ts": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "api": str(e),
                "ts": datetime.now().isoformat()}


async def _fetch_people() -> list[dict]:
    """Pull full people list from Pi for the viz panel."""
    try:
        async with httpx.AsyncClient(timeout=PI_REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{PI_URL}/people")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return []


# ── Broadcast helpers ─────────────────────────────────────────────────────────

async def _broadcast(clients: set, message: dict):
    """Send a JSON message to all connected browser clients."""
    if not clients:
        return
    payload = json.dumps(message)
    await asyncio.gather(
        *[_safe_send(ws, payload) for ws in clients],
        return_exceptions=True,
    )


async def _safe_send(ws: WebSocketServerProtocol, payload: str):
    try:
        await ws.send(payload)
    except Exception:
        pass    # client disconnected — handled in connection handler


# ── Command handlers ──────────────────────────────────────────────────────────

async def _handle_command(msg:           dict,
                           state:         SystemState,
                           queue_manager: QueueManager,
                           job_queue:     multiprocessing.Queue,
                           clients:       set):
    """
    Process a command sent from the browser dashboard.
    All commands arrive as JSON objects with a 'type' field.
    """
    cmd = msg.get("type")

    # ── start / stop ──────────────────────────────────────────────────────────
    if cmd == "start":
        state.running = True
        await _broadcast(clients, {"type": "running_state", "running": True})

    elif cmd == "stop":
        state.running = False
        await _broadcast(clients, {"type": "running_state", "running": False})

    # ── file / folder ingestion ───────────────────────────────────────────────
    elif cmd == "add_files":
        paths = msg.get("paths", [])
        jobs  = queue_manager.add_paths(paths)
        await _broadcast(clients, {
            "type":  "queue_update",
            "jobs":  queue_manager.to_dict_list(),
            "queue": queue_manager.summary(),
        })
        # If running, push jobs straight to the job_queue for workers
        if state.running:
            for job in jobs:
                job_queue.put(job)

    # ── remove a pending job ──────────────────────────────────────────────────
    elif cmd == "remove_job":
        job_id = msg.get("job_id")
        if job_id is not None:
            queue_manager.remove(int(job_id))
            await _broadcast(clients, {
                "type":  "queue_update",
                "jobs":  queue_manager.to_dict_list(),
                "queue": queue_manager.summary(),
            })

    # ── clear all pending jobs ────────────────────────────────────────────────
    elif cmd == "clear_queue":
        removed = queue_manager.clear_pending()
        await _broadcast(clients, {
            "type":    "queue_update",
            "jobs":    queue_manager.to_dict_list(),
            "queue":   queue_manager.summary(),
            "cleared": removed,
        })

    # ── settings toggles ──────────────────────────────────────────────────────
    elif cmd == "settings":
        key   = msg.get("key")
        value = msg.get("value")
        if key and key in state.settings:
            state.settings[key] = value
            await _broadcast(clients, {
                "type":     "settings_update",
                "settings": state.settings,
            })

    # ── manual Pi ping ────────────────────────────────────────────────────────
    elif cmd == "ping_pi":
        pi_status = await _poll_pi()
        state.pi  = pi_status
        await _broadcast(clients, {"type": "pi_status", "pi": pi_status})

    # ── rename a person on the Pi ─────────────────────────────────────────────
    elif cmd == "rename_person":
        person_id = msg.get("person_id")
        new_name  = msg.get("name", "").strip()
        if person_id and new_name:
            try:
                async with httpx.AsyncClient(timeout=PI_REQUEST_TIMEOUT) as client:
                    resp = await client.patch(
                        f"{_PI_URL}/person/{person_id}",
                        json={"name": new_name},
                    )
                if resp.status_code == 200:
                    # Refresh people list and broadcast
                    people       = await _fetch_people()
                    state.people = people
                    await _broadcast(clients, {
                        "type":   "pi_status",
                        "pi":     state.pi,
                        "people": people,
                    })
                    await _broadcast(clients, {
                        "type":    "rename_result",
                        "ok":      True,
                        "person_id": person_id,
                        "name":    new_name,
                    })
                else:
                    await _broadcast(clients, {
                        "type":  "rename_result",
                        "ok":    False,
                        "error": f"Pi returned {resp.status_code}",
                    })
            except Exception as e:
                await _broadcast(clients, {
                    "type":  "rename_result",
                    "ok":    False,
                    "error": str(e),
                })

    # ── request full snapshot (e.g. browser reload) ───────────────────────────
    elif cmd == "get_snapshot":
        pass    # handled in connection handler — snapshot sent on connect

    else:
        print(f"  [dashboard] unknown command: {cmd}")


# ── State queue drainer ───────────────────────────────────────────────────────

async def _drain_state_queue(state_queue:   multiprocessing.Queue,
                              state:         SystemState,
                              queue_manager: QueueManager,
                              clients:       set):
    """
    Runs as a background task.
    Continuously drains worker state updates and broadcasts them.
    """
    loop = asyncio.get_event_loop()

    while True:
        # Non-blocking drain — process up to 50 messages per tick
        messages_this_tick = 0
        while messages_this_tick < 50:
            try:
                update = state_queue.get_nowait()
            except Exception:
                break   # queue empty

            messages_this_tick += 1
            worker_id = update.get("worker_id")
            msg_type  = update.get("type")

            # ── Update worker state ───────────────────────────────────────────
            if worker_id is not None:
                if worker_id not in state.workers:
                    state.workers[worker_id] = {}
                state.workers[worker_id].update(update)

            # ── Update job registry ───────────────────────────────────────────
            job_id = update.get("job_id")
            if job_id is not None:
                if msg_type == "job_start":
                    queue_manager.update(
                        job_id,
                        status=JobStatus.PROCESSING,
                        worker_id=worker_id,
                        started_at=datetime.fromisoformat(
                            update["started_at"]
                        ) if update.get("started_at") else datetime.now(),
                    )
                elif msg_type == "job_done":
                    queue_manager.update(
                        job_id,
                        status=JobStatus.DONE,
                        finished_at=datetime.fromisoformat(
                            update["finished_at"]
                        ) if update.get("finished_at") else datetime.now(),
                        result=update.get("result", {}),
                    )
                elif msg_type == "job_failed":
                    queue_manager.update(
                        job_id,
                        status=JobStatus.FAILED,
                        finished_at=datetime.fromisoformat(
                            update["finished_at"]
                        ) if update.get("finished_at") else datetime.now(),
                        error=update.get("error", "unknown error"),
                    )

            # ── Broadcast to browser ──────────────────────────────────────────
            await _broadcast(clients, {
                "type":   "worker_state",
                "update": update,
                "queue":  queue_manager.summary(),
                "jobs":   queue_manager.to_dict_list(),
            })

        # Yield to event loop between draining ticks
        await asyncio.sleep(0.05)   # 20 ticks/sec


# ── Pi polling loop ───────────────────────────────────────────────────────────

async def _pi_poll_loop(state: SystemState, clients: set):
    """Poll Pi on a fixed interval and broadcast results."""
    while True:
        pi_status    = await _poll_pi()
        state.pi     = pi_status

        # Also refresh people list so viz panel stays current
        people       = await _fetch_people()
        state.people = people

        await _broadcast(clients, {
            "type":   "pi_status",
            "pi":     pi_status,
            "people": people,
        })

        await asyncio.sleep(PI_POLL_INTERVAL)


# ── Connection handler ────────────────────────────────────────────────────────

async def _connection_handler(websocket:     WebSocketServerProtocol,
                               state:         SystemState,
                               queue_manager: QueueManager,
                               job_queue:     multiprocessing.Queue,
                               clients:       set):
    """Handles one browser connection for its lifetime."""
    clients.add(websocket)
    print(f"  [dashboard] browser connected  "
          f"({len(clients)} client(s))")

    try:
        # Send full snapshot so the browser renders immediately
        await websocket.send(
            json.dumps(state.full_snapshot(queue_manager))
        )

        # Listen for commands
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await _handle_command(
                msg, state, queue_manager, job_queue, clients
            )

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        print(f"  [dashboard] browser disconnected  "
              f"({len(clients)} client(s))")


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_dashboard_server(state_queue:   multiprocessing.Queue,
                                command_queue: multiprocessing.Queue,
                                job_queue:     multiprocessing.Queue):
    """
    Called by main.py as the asyncio entry point for the dashboard process.
    Starts the websocket server and all background tasks.
    """
    state         = SystemState()
    queue_manager = QueueManager(job_queue)
    clients:  set = set()

    print(f"  [dashboard] websocket server starting  "
          f"ws://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    # Background tasks
    drain_task = asyncio.create_task(
        _drain_state_queue(state_queue, state, queue_manager, clients)
    )
    pi_task = asyncio.create_task(
        _pi_poll_loop(state, clients)
    )

    # Websocket server
    async with websockets.serve(
        lambda ws: _connection_handler(
            ws, state, queue_manager, job_queue, clients
        ),
        DASHBOARD_HOST,
        DASHBOARD_PORT,
    ):
        print(f"  [dashboard] ready  "
              f"ws://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        try:
            await asyncio.gather(drain_task, pi_task)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass