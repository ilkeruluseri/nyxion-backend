# app/state/job_store.py
from __future__ import annotations
from typing import Dict, Any, Iterable
from threading import RLock

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = RLock()

def init_job(job_id: str, **fields: Any) -> None:
    with _lock:
        _jobs[job_id] = {"job_id": job_id, **fields}

def update_job(job_id: str, **fields: Any) -> None:
    with _lock:
        if job_id not in _jobs:
            _jobs[job_id] = {"job_id": job_id}
        _jobs[job_id].update(fields)

def append_logs(job_id: str, lines: Iterable[str]) -> None:
    with _lock:
        job = _jobs.setdefault(job_id, {"job_id": job_id})
        job.setdefault("logs", [])
        for ln in lines:
            if ln:
                job["logs"].append(ln)

def get_job(job_id: str) -> Dict[str, Any] | None:
    with _lock:
        return _jobs.get(job_id)
