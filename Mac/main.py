"""
face_system/mac/main.py
───────────────────────
Entry point. Responsibilities:
  1. Detect logical core count
  2. Prompt user for how many cores to use
  3. Spawn dashboard_server process (core 0 / dashboard manager)
  4. Spawn worker processes (cores 1..N)
  5. Open dashboard.html in the default browser
  6. Stay alive, forwarding shutdown signals to children
"""

import functools
import http.server
import multiprocessing
import os
import subprocess
import sys
import signal
import threading
import time
import webbrowser
from pathlib import Path

# ── local imports ─────────────────────────────────────────────────────────────
from config import (
    DASHBOARD_PORT,
    DASHBOARD_HTTP_PORT,
    SCAN_SERVER_PORT,
    IG_SERVER_PORT,
    PI_URL,
    WORKER_CORES_OVERRIDE,   # set to int in config.py to skip the prompt
)
from worker_pool import WorkerPool
from dashboard_server import run_dashboard_server
from detector import init_worker_model
from scan_server import run_scan_server, _local_ip

_IG_SERVER_SCRIPT = Path(__file__).parent.parent / "integration" / "ig_server.py"
_IG_ROOT          = Path(__file__).parent.parent / "integration" / "ig_output"


# ── helpers ───────────────────────────────────────────────────────────────────

def detect_cores() -> int:
    """Return logical CPU count, with a sensible fallback."""
    return multiprocessing.cpu_count() or 4


def prompt_core_count(detected: int) -> int:
    """
    Ask the user how many cores to allocate.
    Returns the chosen total (minimum 2: 1 dashboard + 1 worker).
    """
    print()
    print("┌─────────────────────────────────────────┐")
    print("│          face_system  —  setup          │")
    print("└─────────────────────────────────────────┘")
    print(f"  Detected cores : {detected}")
    print(f"  Recommended    : {min(detected, 8)}  "
          f"(1 dashboard + {min(detected, 8) - 1} workers)")
    print()

    while True:
        raw = input(f"  How many cores to use? [2–{detected}]  ").strip()
        if not raw:
            # default: all detected, capped at 8
            chosen = min(detected, 8)
            break
        try:
            chosen = int(raw)
            if 2 <= chosen <= detected:
                break
            print(f"  Please enter a number between 2 and {detected}.")
        except ValueError:
            print("  Please enter a whole number.")

    workers = chosen - 1
    print()
    print(f"  Allocating  1 dashboard core  +  {workers} worker core(s)")
    print()
    return chosen


def _run_html_server(port: int, directory: Path):
    """Serve the Mac/ directory over plain HTTP so iframes to localhost work."""
    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass  # suppress per-request logs

    handler = functools.partial(_QuietHandler, directory=str(directory))
    with http.server.HTTPServer(("localhost", port), handler) as httpd:
        httpd.serve_forever()


def open_dashboard():
    """Start a tiny HTTP file-server, then open dashboard.html via HTTP."""
    mac_dir = Path(__file__).parent
    threading.Thread(
        target=_run_html_server,
        args=(DASHBOARD_HTTP_PORT, mac_dir),
        daemon=True,
    ).start()
    time.sleep(0.3)   # let the server bind
    url = f"http://localhost:{DASHBOARD_HTTP_PORT}/dashboard.html"
    webbrowser.open(url)
    print(f"  [html]       serving Mac/ at http://localhost:{DASHBOARD_HTTP_PORT}/")


# ── process targets ───────────────────────────────────────────────────────────

def _dashboard_process_target(state_queue: multiprocessing.Queue,
                               command_queue: multiprocessing.Queue,
                               job_queue: multiprocessing.Queue):
    """
    Runs on the dashboard core.
    Starts the asyncio websocket server that bridges workers ↔ browser UI.
    """
    import asyncio
    try:
        asyncio.run(
            run_dashboard_server(state_queue, command_queue, job_queue)
        )
    except (KeyboardInterrupt, SystemExit):
        pass


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Core count ────────────────────────────────────────────────────────
    detected = detect_cores()

    if WORKER_CORES_OVERRIDE is not None:
        total_cores = max(2, min(WORKER_CORES_OVERRIDE + 1, detected))
        print(f"[config] Using {total_cores} cores "
              f"(override: {WORKER_CORES_OVERRIDE} workers)")
    else:
        total_cores = prompt_core_count(detected)

    num_workers = total_cores - 1   # one core reserved for dashboard

    # ── 2. Shared inter-process queues ───────────────────────────────────────
    #
    #   job_queue     workers pull jobs from here
    #   state_queue   workers push state updates here → dashboard_server reads
    #   command_queue dashboard UI sends commands here → dashboard_server reads
    #
    job_queue     = multiprocessing.Queue()
    state_queue   = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()

    # ── 3. Spawn dashboard server process ────────────────────────────────────
    dashboard_proc = multiprocessing.Process(
        target=_dashboard_process_target,
        args=(state_queue, command_queue, job_queue),
        name="dashboard-core",
        daemon=True,
    )
    dashboard_proc.start()
    print(f"  [dashboard]  PID {dashboard_proc.pid} "
          f"— ws://localhost:{DASHBOARD_PORT}")

    # ── 3b. Spawn scan server process ────────────────────────────────────────
    scan_proc = multiprocessing.Process(
        target=run_scan_server,
        name="scan-server",
        daemon=True,
    )
    scan_proc.start()
    print(f"  [scan]       PID {scan_proc.pid} "
          f"— https://{_local_ip()}:{SCAN_SERVER_PORT}")

    # ── 3c. Spawn Instagram review server ────────────────────────────────────
    _IG_ROOT.mkdir(parents=True, exist_ok=True)
    ig_proc = subprocess.Popen(
        [
            sys.executable, str(_IG_SERVER_SCRIPT),
            "--root",  str(_IG_ROOT),
            "--pi",    PI_URL,
            "--port",  str(IG_SERVER_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  [instagram]  PID {ig_proc.pid} "
          f"— http://localhost:{IG_SERVER_PORT}")

    # Give the websocket server a moment to bind before the browser opens
    time.sleep(0.8)

    # ── 4. Pre-download model before spawning workers ─────────────────────────
    # Ensures the InsightFace model is on disk exactly once. Without this,
    # all worker processes race to download buffalo_l.zip simultaneously.
    print("  [main] pre-loading InsightFace model (one-time download if needed)...")
    init_worker_model()

    # ── 5. Spawn worker pool ──────────────────────────────────────────────────
    pool = WorkerPool(
        num_workers=num_workers,
        job_queue=job_queue,
        state_queue=state_queue,
    )
    pool.start()
    print(f"  [workers]    {num_workers} worker process(es) started")

    # ── 6. Open browser ───────────────────────────────────────────────────────
    open_dashboard()
    print(f"  [browser]    opening dashboard.html")
    print()
    print("  face_system running — press Ctrl+C to quit")
    print()

    # ── 7. Stay alive ────────────────────────────────────────────────────────
    def _shutdown(sig, frame):
        print("\n  Shutting down...")
        pool.shutdown()
        dashboard_proc.terminate()
        dashboard_proc.join(timeout=3)
        ig_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep the main process alive while children run.
    # Restart any worker that dies unexpectedly.
    while True:
        time.sleep(5)

        if not dashboard_proc.is_alive():
            print("  [warn] dashboard process died — restarting...")
            dashboard_proc = multiprocessing.Process(
                target=_dashboard_process_target,
                args=(state_queue, command_queue, job_queue),
                name="dashboard-core",
                daemon=True,
            )
            dashboard_proc.start()

        if not scan_proc.is_alive():
            print("  [warn] scan server died — restarting...")
            scan_proc = multiprocessing.Process(
                target=run_scan_server,
                name="scan-server",
                daemon=True,
            )
            scan_proc.start()

        if ig_proc.poll() is not None:
            print("  [warn] Instagram server died — restarting...")
            ig_proc = subprocess.Popen(
                [
                    sys.executable, str(_IG_SERVER_SCRIPT),
                    "--root",  str(_IG_ROOT),
                    "--pi",    PI_URL,
                    "--port",  str(IG_SERVER_PORT),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        pool.health_check()


if __name__ == "__main__":
    # Required on macOS / Windows for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()