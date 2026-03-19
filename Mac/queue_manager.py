"""
face_system/mac/queue_manager.py
─────────────────────────────────
Manages the job queue. Handles:
  - Adding individual files, folders, or mixed lists
  - Tracking job status through its lifecycle
  - Providing filtered views and summary stats
  - Thread/process-safe via multiprocessing.Queue
"""

import os
import multiprocessing
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from upload import get_file_type


# ── Job status ────────────────────────────────────────────────────────────────

class JobStatus(Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"
    SKIPPED    = "skipped"


# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    id:           int
    path:         str
    file_type:    str                   # 'image' or 'video'
    status:       JobStatus = JobStatus.PENDING
    added_at:     datetime  = field(default_factory=datetime.now)
    started_at:   Optional[datetime] = None
    finished_at:  Optional[datetime] = None
    worker_id:    Optional[int]      = None
    result:       dict = field(default_factory=dict)
    error:        Optional[str]      = None

    # ── computed ──────────────────────────────────────────────────────────────

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def duration(self) -> Optional[float]:
        """Wall-clock seconds from start to finish. None if not complete."""
        if self.started_at and self.finished_at:
            return round(
                (self.finished_at - self.started_at).total_seconds(), 2
            )
        return None

    @property
    def is_terminal(self) -> bool:
        """True if the job will not be processed again."""
        return self.status in (
            JobStatus.DONE,
            JobStatus.FAILED,
            JobStatus.SKIPPED,
        )

    def to_dict(self) -> dict:
        """Serialisable snapshot for websocket broadcasts."""
        return {
            "id":          self.id,
            "filename":    self.filename,
            "path":        self.path,
            "file_type":   self.file_type,
            "status":      self.status.value,
            "worker_id":   self.worker_id,
            "duration":    self.duration,
            "error":       self.error,
            "faces_found": self.result.get("faces_found", 0),
            "added_at":    self.added_at.isoformat(),
        }


# ── Queue manager ─────────────────────────────────────────────────────────────

class QueueManager:
    """
    Wraps a multiprocessing.Queue so it can be shared across processes.
    The internal _jobs registry lives in the main/dashboard process only —
    workers communicate status back via the state_queue, and the dashboard
    server calls update() to keep the registry current.
    """

    def __init__(self, job_queue: multiprocessing.Queue):
        self._job_queue: multiprocessing.Queue = job_queue
        self._jobs:      dict[int, Job]        = {}
        self._counter:   int                   = 0
        self._lock = multiprocessing.Lock()

    # ── Adding jobs ───────────────────────────────────────────────────────────

    def add_file(self, path: str) -> Optional[Job]:
        """
        Validate file type, create a Job, push it onto the shared queue.
        Returns None if the file type is unsupported.
        """
        path = os.path.abspath(path)

        if not os.path.isfile(path):
            print(f"  [queue] not a file: {path}")
            return None

        try:
            file_type = get_file_type(path)
        except ValueError as e:
            print(f"  [queue] skipping — {e}")
            return None

        with self._lock:
            self._counter += 1
            job = Job(id=self._counter, path=path, file_type=file_type)
            self._jobs[job.id] = job

        self._job_queue.put(job)
        print(f"  [queue] +{job.id:04}  {job.filename}  ({file_type})")
        return job

    def add_folder(self, folder_path: str) -> list[Job]:
        """
        Recursively add all supported files under folder_path.
        Images are enqueued before videos so fast jobs run first.
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(folder_path):
            print(f"  [queue] not a directory: {folder_path}")
            return []

        images, videos = [], []

        for root, _, files in os.walk(folder_path):
            for fname in sorted(files):
                full = os.path.join(root, fname)
                try:
                    ft = get_file_type(full)
                    (images if ft == "image" else videos).append(full)
                except ValueError:
                    pass

        jobs = []
        for path in images + videos:
            job = self.add_file(path)
            if job:
                jobs.append(job)

        print(f"  [queue] folder scan: "
              f"{len(images)} image(s), {len(videos)} video(s) added")
        return jobs

    def add_paths(self, paths: list[str]) -> list[Job]:
        """
        Accept a mixed list of file paths and/or folder paths.
        Folders are expanded; unsupported files are silently skipped.
        """
        jobs = []
        for path in paths:
            path = os.path.abspath(path)
            if os.path.isdir(path):
                jobs += self.add_folder(path)
            elif os.path.isfile(path):
                job = self.add_file(path)
                if job:
                    jobs.append(job)
            else:
                print(f"  [queue] path not found: {path}")
        return jobs

    # ── Status updates ────────────────────────────────────────────────────────

    def update(self, job_id: int, **kwargs) -> Optional[Job]:
        """
        Update any field on a job by keyword argument.
        Called by dashboard_server when a worker pushes a state update.

        Example:
            manager.update(3, status=JobStatus.PROCESSING, worker_id=2,
                           started_at=datetime.now())
        """
        job = self._jobs.get(job_id)
        if not job:
            return None
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
            else:
                print(f"  [queue] unknown field '{key}' on Job")
        return job

    def remove(self, job_id: int) -> bool:
        """
        Remove a PENDING job from the registry.
        Cannot remove jobs that are already processing or complete.
        Note: the job may already be in the multiprocessing.Queue —
        workers check job status before processing and skip non-PENDING jobs.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status != JobStatus.PENDING:
            print(f"  [queue] cannot remove job {job_id} "
                  f"— status is {job.status.value}")
            return False
        with self._lock:
            del self._jobs[job_id]
        return True

    def clear_pending(self) -> int:
        """Remove all pending jobs. Returns count removed."""
        pending_ids = [j.id for j in self.all_jobs
                       if j.status == JobStatus.PENDING]
        for jid in pending_ids:
            self.remove(jid)
        return len(pending_ids)

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def all_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    @property
    def pending(self) -> list[Job]:
        return [j for j in self.all_jobs if j.status == JobStatus.PENDING]

    @property
    def processing(self) -> list[Job]:
        return [j for j in self.all_jobs if j.status == JobStatus.PROCESSING]

    @property
    def done_jobs(self) -> list[Job]:
        return [j for j in self.all_jobs if j.status == JobStatus.DONE]

    @property
    def failed_jobs(self) -> list[Job]:
        return [j for j in self.all_jobs if j.status == JobStatus.FAILED]

    def get(self, job_id: int) -> Optional[Job]:
        return self._jobs.get(job_id)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        jobs = self.all_jobs
        return {
            "total":      len(jobs),
            "pending":    sum(1 for j in jobs if j.status == JobStatus.PENDING),
            "processing": sum(1 for j in jobs if j.status == JobStatus.PROCESSING),
            "done":       sum(1 for j in jobs if j.status == JobStatus.DONE),
            "failed":     sum(1 for j in jobs if j.status == JobStatus.FAILED),
            "skipped":    sum(1 for j in jobs if j.status == JobStatus.SKIPPED),
        }

    def to_dict_list(self) -> list[dict]:
        """Serialisable list of all jobs for websocket broadcast."""
        return [j.to_dict() for j in self.all_jobs]

    def print_summary(self):
        s = self.summary()
        print()
        print(f"  {'─'*44}")
        print(f"  Queue summary")
        print(f"  {'─'*44}")
        print(f"  Total      {s['total']}")
        print(f"  Done    ✓  {s['done']}")
        print(f"  Pending ·  {s['pending']}")
        print(f"  Failed  ✗  {s['failed']}")
        print(f"  Skipped ⊘  {s['skipped']}")
        print(f"  {'─'*44}")
        for job in self.all_jobs:
            icon = {
                JobStatus.DONE:       "✓",
                JobStatus.FAILED:     "✗",
                JobStatus.PROCESSING: "⚙",
                JobStatus.PENDING:    "·",
                JobStatus.SKIPPED:    "⊘",
            }.get(job.status, "?")
            dur = f"  {job.duration}s" if job.duration else ""
            err = f"  — {job.error}"   if job.error    else ""
            print(f"  {icon} [{job.id:04}]  {job.filename}{dur}{err}")
        print()