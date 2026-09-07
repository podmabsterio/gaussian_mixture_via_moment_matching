from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .result_files import read_result_directory


class RunNotFound(KeyError):
    pass


class RunManager:
    def __init__(self, project_root: str | Path, runtime_dir: str | Path):
        self.project_root = Path(project_root).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def start(
        self,
        run_id: str,
        config: DictConfig,
        metadata: dict[str, Any],
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        job_dir = self.runtime_dir / "jobs" / run_id
        job_dir.mkdir(parents=True, exist_ok=False)
        config_path = job_dir / "config.yaml"
        request_path = job_dir / "request.json"
        log_path = job_dir / "run.log"
        OmegaConf.save(config, config_path)
        request_path.write_text(json.dumps(request_body, ensure_ascii=False, indent=2), encoding="utf-8")

        now = time.time()
        record: dict[str, Any] = {
            "id": run_id,
            "status": "starting",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "error": None,
            "metadata": metadata,
            "config": OmegaConf.to_container(config, resolve=True),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "process": None,
        }
        with self._lock:
            self._runs[run_id] = record

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "ui.backend.worker", str(config_path)],
                cwd=self.project_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
            )
        except Exception:
            log_handle.close()
            with self._lock:
                record["status"] = "error"
                record["finished_at"] = time.time()
            raise

        with self._lock:
            record["process"] = process
            record["status"] = "running"
            record["started_at"] = time.time()

        watcher = threading.Thread(
            target=self._watch,
            args=(run_id, process, log_handle),
            daemon=True,
            name=f"ui-run-{run_id}",
        )
        watcher.start()
        return self.get(run_id)

    def _watch(self, run_id: str, process: subprocess.Popen[str], log_handle: Any) -> None:
        return_code = process.wait()
        log_handle.close()
        with self._lock:
            record = self._runs[run_id]
            record["return_code"] = return_code
            record["finished_at"] = time.time()
            if record["status"] == "cancelling":
                record["status"] = "cancelled"
            elif return_code == 0:
                record["status"] = "completed"
            else:
                record["status"] = "error"
                record["error"] = self._last_nonempty_line(Path(record["log_path"]))

    @staticmethod
    def _last_nonempty_line(path: Path) -> str:
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        except OSError:
            return "Experiment worker failed"
        return lines[-1] if lines else "Experiment worker failed"

    def _record(self, run_id: str) -> dict[str, Any]:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RunNotFound(run_id) from exc

    def get(self, run_id: str, include_config: bool = True) -> dict[str, Any]:
        with self._lock:
            record = self._record(run_id)
            public = {key: value for key, value in record.items() if key != "process"}
            if not include_config:
                public.pop("config", None)
            public["log_tail"] = self._log_tail(Path(record["log_path"]))
            return public

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = sorted(self._runs, key=lambda key: self._runs[key]["created_at"], reverse=True)
        return [self.get(run_id, include_config=False) for run_id in ids]

    @staticmethod
    def _log_tail(path: Path, limit: int = 80) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-limit:])

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._record(run_id)
            if record["status"] not in {"starting", "running"}:
                return self.get(run_id)
            record["status"] = "cancelling"
            process = record["process"]
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        return self.get(run_id)

    def results(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._record(run_id)
            result_path = Path(record["metadata"]["result_path"])
            status = record["status"]
        return {
            "run_id": run_id,
            "status": status,
            "result_path": str(result_path),
            "datasets": read_result_directory(result_path),
        }
