"""Persistent SQLite storage layer for claims, accumulators, and escalations."""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.config import settings


def get_db_path() -> Path:
    data_dir = settings.resolved_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return settings.resolved_db_path


from contextlib import contextmanager


class Database:
    """Thread-safe SQLite manager for persistent application state."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
        self._init_tables()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Create necessary tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Claims Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    policyholder_id TEXT NOT NULL,
                    policyholder_name TEXT NOT NULL,
                    claim_type TEXT NOT NULL,
                    claim_amount REAL NOT NULL,
                    approved_amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    submission_date TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    next_action TEXT NOT NULL,
                    notes TEXT NOT NULL
                )
            """)

            # 2. Benefit Accumulators Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accumulators (
                    policyholder_id TEXT PRIMARY KEY,
                    is_family_plan INTEGER NOT NULL DEFAULT 0,
                    deductible_limit REAL NOT NULL,
                    oop_max_limit REAL NOT NULL,
                    deductible_satisfied REAL NOT NULL DEFAULT 0.0,
                    oop_accumulated REAL NOT NULL DEFAULT 0.0,
                    last_updated TEXT NOT NULL
                )
            """)

            # 3. Escalation Tickets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    ticket_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    policyholder_id TEXT,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    assigned_queue TEXT NOT NULL,
                    specialist_hotline TEXT NOT NULL,
                    specialist_email TEXT NOT NULL,
                    turnaround_notice TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            conn.commit()

    def seed_claims_if_empty(self, json_path: Path) -> int:
        """Seeds claims from initial sample JSON if database table is currently empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM claims")
            count = cursor.fetchone()[0]
            if count > 0:
                return count

            if not json_path.exists():
                return 0

            claims_data = json.loads(json_path.read_text(encoding="utf-8"))
            for item in claims_data:
                cursor.execute("""
                    INSERT OR REPLACE INTO claims (
                        claim_id, policyholder_id, policyholder_name, claim_type,
                        claim_amount, approved_amount, status, submission_date,
                        last_updated, next_action, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["claim_id"].upper(),
                    item["policyholder_id"],
                    item["policyholder_name"],
                    item["claim_type"],
                    float(item["claim_amount"]),
                    float(item["approved_amount"]),
                    item["status"],
                    item["submission_date"],
                    item["last_updated"],
                    item["next_action"],
                    item["notes"]
                ))
            conn.commit()
            return len(claims_data)

    # Claims CRUD
    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM claims WHERE UPPER(claim_id) = ?", (claim_id.strip().upper(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_all_claims(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM claims ORDER BY submission_date DESC")
            return [dict(row) for row in cursor.fetchall()]

    def search_claims(self, query: str) -> List[Dict[str, Any]]:
        query_pattern = f"%{query.strip().lower()}%"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM claims
                WHERE LOWER(claim_id) LIKE ?
                   OR LOWER(policyholder_id) LIKE ?
                   OR LOWER(policyholder_name) LIKE ?
                   OR LOWER(claim_type) LIKE ?
                   OR LOWER(status) LIKE ?
                ORDER BY submission_date DESC
            """, (query_pattern, query_pattern, query_pattern, query_pattern, query_pattern))
            return [dict(row) for row in cursor.fetchall()]

    def save_claim(self, claim: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO claims (
                    claim_id, policyholder_id, policyholder_name, claim_type,
                    claim_amount, approved_amount, status, submission_date,
                    last_updated, next_action, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                claim["claim_id"].upper(),
                claim["policyholder_id"],
                claim["policyholder_name"],
                claim["claim_type"],
                float(claim["claim_amount"]),
                float(claim["approved_amount"]),
                claim["status"],
                claim["submission_date"],
                claim["last_updated"],
                claim["next_action"],
                claim["notes"]
            ))
            conn.commit()

    # Accumulator Storage
    def get_accumulator(self, policyholder_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM accumulators WHERE policyholder_id = ?", (policyholder_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["is_family_plan"] = bool(d["is_family_plan"])
                return d
            return None

    def save_accumulator(self, acc: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO accumulators (
                    policyholder_id, is_family_plan, deductible_limit,
                    oop_max_limit, deductible_satisfied, oop_accumulated, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                acc["policyholder_id"],
                1 if acc.get("is_family_plan", False) else 0,
                float(acc["deductible_limit"]),
                float(acc["oop_max_limit"]),
                float(acc["deductible_satisfied"]),
                float(acc["oop_accumulated"]),
                acc.get("last_updated", "")
            ))
            conn.commit()

    # Escalations
    def save_escalation(self, esc: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO escalations (
                    ticket_id, claim_id, policyholder_id, reason, status,
                    assigned_queue, specialist_hotline, specialist_email,
                    turnaround_notice, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                esc["ticket_id"],
                esc["claim_id"],
                esc.get("policyholder_id"),
                esc["reason"],
                esc.get("status", "Escalated"),
                esc.get("assigned_queue", "Human Adjudication Unit"),
                esc.get("specialist_hotline", ""),
                esc.get("specialist_email", ""),
                esc.get("turnaround_notice", ""),
                esc.get("created_at", "")
            ))
            conn.commit()

    def list_escalations(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM escalations ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]


# Global database singleton
db = Database()
