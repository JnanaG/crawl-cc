import json
import os
from datetime import datetime


class JobStateStore:
    def __init__(self, state_path: str):
        self.state_path = state_path
        self.state = {"updated_at": None, "jobs": {}}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            self._flush()
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
            if "jobs" not in self.state:
                self.state["jobs"] = {}
        except Exception:
            # 状态文件损坏时保底重建，避免阻塞任务
            self.state = {"updated_at": None, "jobs": {}}
            self._flush()

    def _flush(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def get_job(self, series_id: str) -> dict:
        return self.state["jobs"].get(str(series_id), {})

    def can_process(self, series_id: str, max_retry: int) -> tuple[bool, str]:
        job = self.get_job(series_id)
        status = job.get("status", "pending")
        retry_count = int(job.get("retry_count", 0))

        if status == "success":
            return False, "already_success"
        if status == "failed" and retry_count >= max_retry:
            return False, "retry_exhausted"
        return True, "processable"

    def mark_running(self, series_id: str) -> None:
        sid = str(series_id)
        old = self.state["jobs"].get(sid, {})
        self.state["jobs"][sid] = {
            **old,
            "status": "running",
            "last_error": "",
            "last_start_at": datetime.now().isoformat(),
        }
        self._flush()

    def mark_success(self, series_id: str, meta: dict | None = None) -> None:
        sid = str(series_id)
        old = self.state["jobs"].get(sid, {})
        self.state["jobs"][sid] = {
            **old,
            "status": "success",
            "last_error": "",
            "last_finish_at": datetime.now().isoformat(),
            "meta": meta or {},
        }
        self._flush()

    def mark_failed(self, series_id: str, error: str, meta: dict | None = None) -> None:
        sid = str(series_id)
        old = self.state["jobs"].get(sid, {})
        retry_count = int(old.get("retry_count", 0)) + 1
        self.state["jobs"][sid] = {
            **old,
            "status": "failed",
            "retry_count": retry_count,
            "last_error": error,
            "last_finish_at": datetime.now().isoformat(),
            "meta": meta or {},
        }
        self._flush()

    def mark_skipped(self, series_id: str, reason: str) -> None:
        sid = str(series_id)
        old = self.state["jobs"].get(sid, {})
        self.state["jobs"][sid] = {
            **old,
            "status": "skipped",
            "skip_reason": reason,
            "last_finish_at": datetime.now().isoformat(),
        }
        self._flush()
