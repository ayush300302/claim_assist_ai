"""Audit and compliance logging for ClaimAssist AI conversations."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import settings


class AuditLogger:
    """Logs conversation history and decision trails for compliance review."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or settings.resolved_log_file_path
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Create parent log directory if absent."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        citations: List[str],
        tool_calls: List[Dict[str, Any]],
        model_used: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Appends a structured interaction record to the JSONL log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "citations": citations,
            "tool_calls": tool_calls,
            "model_used": model_used,
            "latency_ms": round(latency_ms, 2),
            "metadata": metadata or {}
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # Audit logging failure should not crash user-facing response
            print(f"[AuditLogger Warning] Could not write log record: {e}")

        return record

    def read_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Reads the most recent log entries."""
        if not self.log_path.exists():
            return []

        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in reversed(lines[-limit:]):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
