"""
face_system/mac/worker_pool.py
───────────────────────────────
Spawns and manages the worker processes.
Responsibilities:
  - Start one Process per worker core
  - Monitor process health and restart dead workers
  - Submit jobs to the shared job_queue
  - Forward state updates from workers to dashboard_server
  - Cleanly shut everything down on exit
"""

import multiprocessing
import os
import time
from datetime import datetime
from typing import Optional

from config import POLL_RATE_SECONDS
from worker import run_worker
from queue_manager import Job


# Module-level sentinel — must be at top level so multiprocessing can pickle it
class _Sentinel:
    """Poison pill placed on the job queue to signal a worker to exit cleanly."""
    id     = -1
    status = "sentinel"


# ── Worker process descriptor ─────────────────────────────────────────────────

class WorkerProcess:
    """Tracks a single worker process and its metadata."""

    def __init__(self, worker_id: int,
                 job_queue:   multiprocessing.Queue,
                 state_queue: multiprocessing.Queue):
        self.worker_id   = worker_id
        self.job_queue   = job_queue
        self.state_queue = state_queue
        self.process: Optional[multiprocessing.Process] = None
        self.started_at: Optional[datetime] = None
        self.restart_count = 0

    def start(self):
        self.process = multiprocessing.Process(
            target=run_worker,
            args=(
                self.worker_id,
                self.job_queue,
                self.state_queue,
                POLL_RATE_SECONDS,
            ),
            name=f"worker-{self.worker_id}",
            daemon=True,
        )
        self.process.start()
        self.started_at = datetime.now()
        print(f"  [pool] worker-{self.worker_id} started  "
              f"PID {self.process.pid}")

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def restart(self):
        """Terminate (if running) and start fresh."""
        self.terminate()
        time.sleep(0.5)
        self.restart_count += 1
        print(f"  [pool] restarting worker-{self.worker_id} "
              f"(restart #{self.restart_count})")
        self.start()

    def terminate(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)

    def send_poison_pill(self):
        """
        Put a sentinel job onto the queue so this worker exits its loop
        cleanly rather than being force-terminated mid-job.
        """
        self.job_queue.put(_Sentinel())

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    @property
    def uptime_seconds(self) -> Optional[float]:
        if self.started_at:
            return round((datetime.now() - self.started_at).total_seconds(), 1)
        return None

    def to_dict(self) -> dict:
        return {
            "worker_id":     self.worker_id,
            "pid":           self.pid,
            "alive":         self.is_alive(),
            "restart_count": self.restart_count,
            "uptime":        self.uptime_seconds,
        }


# ── Worker pool ───────────────────────────────────────────────────────────────

class WorkerPool:
    """
    Owns all worker processes.
    Called by main.py after the queues are created.

    Usage:
        pool = WorkerPool(num_workers=3, job_queue=jq, state_queue=sq)
        pool.start()

        pool.submit(job)            # put a job on the queue
        pool.health_check()         # call periodically to restart dead workers
        pool.shutdown()             # graceful shutdown
    """

    MAX_RESTARTS = 5    # stop restarting a worker that keeps crashing

    def __init__(self,
                 num_workers: int,
                 job_queue:   multiprocessing.Queue,
                 state_queue: multiprocessing.Queue):
        self.num_workers = num_workers
        self.job_queue   = job_queue
        self.state_queue = state_queue
        self._workers: list[WorkerProcess] = []
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Spawn all worker processes."""
        if self._running:
            print("  [pool] already running")
            return

        self._workers = [
            WorkerProcess(
                worker_id   = i + 1,     # 1-indexed; 0 is dashboard
                job_queue   = self.job_queue,
                state_queue = self.state_queue,
            )
            for i in range(self.num_workers)
        ]

        for wp in self._workers:
            wp.start()

        self._running = True
        print(f"  [pool] {self.num_workers} worker(s) running")

    def shutdown(self, graceful: bool = True):
        """
        Stop all workers.
        graceful=True  → send poison pills and wait for clean exit
        graceful=False → force terminate immediately
        """
        if not self._running:
            return

        print(f"  [pool] shutting down "
              f"({'graceful' if graceful else 'forced'})...")

        if graceful:
            # One poison pill per worker
            for wp in self._workers:
                if wp.is_alive():
                    wp.send_poison_pill()

            # Wait up to 10s for clean exits
            deadline = time.time() + 10
            while time.time() < deadline:
                if not any(wp.is_alive() for wp in self._workers):
                    break
                time.sleep(0.2)

        # Force-kill anything still alive
        for wp in self._workers:
            if wp.is_alive():
                print(f"  [pool] force-killing worker-{wp.worker_id}")
                wp.terminate()

        self._workers  = []
        self._running  = False
        print("  [pool] all workers stopped")

    # ── Job submission ────────────────────────────────────────────────────────

    def submit(self, job: Job):
        """Put a job on the shared queue. Any idle worker will pick it up."""
        self.job_queue.put(job)

    def submit_many(self, jobs: list[Job]):
        for job in jobs:
            self.submit(job)

    # ── Health monitoring ─────────────────────────────────────────────────────

    def health_check(self):
        """
        Called by main.py every few seconds.
        Restarts any worker that has died unexpectedly,
        up to MAX_RESTARTS per worker.
        """
        for wp in self._workers:
            if not wp.is_alive():
                if wp.restart_count >= self.MAX_RESTARTS:
                    print(f"  [pool] worker-{wp.worker_id} exceeded "
                          f"max restarts ({self.MAX_RESTARTS}) — giving up")
                    # Push a permanent-failure state so the dashboard knows
                    self.state_queue.put_nowait({
                        "worker_id": wp.worker_id,
                        "type":      "worker_dead",
                        "ts":        datetime.now().isoformat(),
                        "restarts":  wp.restart_count,
                    })
                else:
                    print(f"  [pool] worker-{wp.worker_id} died — restarting")
                    wp.restart()
                    self.state_queue.put_nowait({
                        "worker_id": wp.worker_id,
                        "type":      "worker_restarted",
                        "ts":        datetime.now().isoformat(),
                        "restarts":  wp.restart_count,
                    })

    # ── Status ────────────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """True if at least one worker is still running."""
        return any(wp.is_alive() for wp in self._workers)

    def all_alive(self) -> bool:
        return all(wp.is_alive() for wp in self._workers)

    def alive_count(self) -> int:
        return sum(1 for wp in self._workers if wp.is_alive())

    def status(self) -> dict:
        return {
            "num_workers": self.num_workers,
            "alive":       self.alive_count(),
            "running":     self._running,
            "workers":     [wp.to_dict() for wp in self._workers],
        }

    def print_status(self):
        s = self.status()
        print(f"  [pool] {s['alive']}/{s['num_workers']} workers alive")
        for w in s["workers"]:
            alive = "✓" if w["alive"] else "✗"
            print(f"    {alive} worker-{w['worker_id']}  "
                  f"PID {w['pid']}  "
                  f"uptime {w['uptime']}s  "
                  f"restarts {w['restart_count']}")