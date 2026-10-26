"""SQLite persistence layer for PlantPilot.

Uses Python's built-in sqlite3 module with parameterised queries. The
database schema supports incidents, investigations, evidence, tool
executions, and audit events.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from .config import get_config
from .schemas import (
    Incident,
    IncidentCreate,
    Investigation,
    InvestigationCreate,
    Status,
    EvidenceItem,
    Hypothesis,
    Recommendation,
    ToolExecution,
)


class DatabaseError(RuntimeError):
    """Raised on database operation failures."""


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open connection and initialize schema if needed."""
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                status TEXT NOT NULL,
                findings_json TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (incident_id) REFERENCES incidents(id)
            );

            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                value_json TEXT,
                timestamp TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                explanation TEXT NOT NULL,
                confidence TEXT,
                supporting_evidence_json TEXT,
                contradicting_evidence_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                requires_approval INTEGER NOT NULL DEFAULT 1,
                approval_status TEXT NOT NULL,
                priority TEXT NOT NULL,
                created_at TEXT NOT NULL,
                executed_at TEXT,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            );

            CREATE TABLE IF NOT EXISTS tool_executions (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                result TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                executed_at TEXT NOT NULL,
                input_size_bytes INTEGER NOT NULL DEFAULT 0,
                output_size_bytes INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                investigation_id TEXT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
            CREATE INDEX IF NOT EXISTS idx_investigations_incident ON investigations(incident_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_investigation ON evidence(investigation_id);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_investigation ON hypotheses(investigation_id);
            CREATE INDEX IF NOT EXISTS idx_recommendations_investigation ON recommendations(investigation_id);
            CREATE INDEX IF NOT EXISTS idx_tool_executions_investigation ON tool_executions(investigation_id);

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread
                ON conversation_messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_investigation_created
                ON audit_events(investigation_id, created_at);
        """)
        self._conn.commit()

    # --- Incidents ---

    def create_incident(self, create: IncidentCreate) -> Incident:
        if self._conn is None:
            raise DatabaseError("Not connected")

        incident_id = f"inc_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO incidents
               (id, description, severity, category, status, created_at, updated_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                incident_id,
                create.description,
                create.severity,
                create.category,
                Status.PENDING.value,
                now,
                now,
                json.dumps(create.metadata),
            ),
        )
        self._conn.commit()

        return Incident(
            id=incident_id,
            description=create.description,
            severity=create.severity,
            category=create.category,
            status=Status.PENDING,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            metadata=create.metadata,
        )

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        row = self._conn.execute(
            "SELECT * FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()

        if row is None:
            return None

        return Incident(
            id=row["id"],
            description=row["description"],
            severity=row["severity"],
            category=row["category"],
            status=Status(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def list_incidents(self, limit: int = 50, offset: int = 0) -> List[Incident]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        rows = self._conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        return [
            Incident(
                id=row["id"],
                description=row["description"],
                severity=row["severity"],
                category=row["category"],
                status=Status(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    # --- Investigations ---

    def create_investigation(
        self, create: InvestigationCreate, incident: Incident
    ) -> Investigation:
        if self._conn is None:
            raise DatabaseError("Not connected")

        investigation_id = f"inv_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO investigations
               (id, incident_id, status, findings_json, started_at, created_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                investigation_id,
                create.incident_id,
                Status.IN_PROGRESS.value,
                json.dumps([]),
                now,
                now,
                create.notes,
            ),
        )
        self._conn.commit()

        return Investigation(
            id=investigation_id,
            incident_id=create.incident_id,
            status=Status.IN_PROGRESS,
            findings=[],
            evidence=[],
            hypotheses=[],
            tool_executions=[],
            recommendations=[],
            errors=[],
            started_at=datetime.fromisoformat(now),
            completed_at=None,
            created_at=datetime.fromisoformat(now),
            notes=create.notes,
        )

    def get_investigation(self, investigation_id: str) -> Optional[Investigation]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        row = self._conn.execute(
            "SELECT * FROM investigations WHERE id = ?", (investigation_id,)
        ).fetchone()

        if row is None:
            return None

        # Load related records
        evidence = self._load_evidence(investigation_id)
        hypotheses = self._load_hypotheses(investigation_id)
        recommendations = self._load_recommendations(investigation_id)
        tool_executions = self._load_tool_executions(investigation_id)

        return Investigation(
            id=row["id"],
            incident_id=row["incident_id"],
            status=Status(row["status"]),
            findings=json.loads(row["findings_json"] or "[]"),
            evidence=evidence,
            hypotheses=hypotheses,
            tool_executions=tool_executions,
            recommendations=recommendations,
            errors=[],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            notes=row["notes"],
        )

    def update_investigation(
        self,
        investigation_id: str,
        status: Optional[Status] = None,
        findings: Optional[List[str]] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status.value)
        if findings is not None:
            updates.append("findings_json = ?")
            params.append(json.dumps(findings))
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at.isoformat())

        if not updates:
            return

        params.append(investigation_id)
        self._conn.execute(
            f"UPDATE investigations SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        self._conn.commit()

    # --- Evidence ---

    def save_evidence(self, investigation_id: str, evidence: EvidenceItem) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT INTO evidence
               (id, investigation_id, evidence_type, description, source,
                value_json, timestamp, verified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence.id,
                investigation_id,
                evidence.evidence_type.value,
                evidence.description,
                evidence.source,
                json.dumps(evidence.value) if evidence.value is not None else None,
                evidence.timestamp.isoformat(),
                1 if evidence.verified else 0,
            ),
        )
        self._conn.commit()

    def _load_evidence(self, investigation_id: str) -> List[EvidenceItem]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE investigation_id = ? ORDER BY timestamp",
            (investigation_id,),
        ).fetchall()

        return [
            EvidenceItem(
                id=row["id"],
                evidence_type=row["evidence_type"],
                description=row["description"],
                source=row["source"],
                value=json.loads(row["value_json"]) if row["value_json"] else None,
                timestamp=datetime.fromisoformat(row["timestamp"]),
                verified=bool(row["verified"]),
            )
            for row in rows
        ]

    # --- Hypotheses ---

    def save_hypothesis(self, investigation_id: str, hypothesis: Hypothesis) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT INTO hypotheses
               (id, investigation_id, title, explanation, confidence,
                supporting_evidence_json, contradicting_evidence_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hypothesis.id,
                investigation_id,
                hypothesis.title,
                hypothesis.explanation,
                hypothesis.confidence,
                json.dumps(hypothesis.supporting_evidence_ids),
                json.dumps(hypothesis.contradicting_evidence_ids),
                hypothesis.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def _load_hypotheses(self, investigation_id: str) -> List[Hypothesis]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM hypotheses WHERE investigation_id = ? ORDER BY created_at",
            (investigation_id,),
        ).fetchall()

        return [
            Hypothesis(
                id=row["id"],
                title=row["title"],
                explanation=row["explanation"],
                confidence=row["confidence"],
                supporting_evidence_ids=json.loads(row["supporting_evidence_json"] or "[]"),
                contradicting_evidence_ids=json.loads(row["contradicting_evidence_json"] or "[]"),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    # --- Recommendations ---

    def save_recommendation(
        self, investigation_id: str, recommendation: Recommendation
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT INTO recommendations
               (id, investigation_id, title, description, requires_approval,
                approval_status, priority, created_at, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                recommendation.id,
                investigation_id,
                recommendation.title,
                recommendation.description,
                1 if recommendation.requires_approval else 0,
                recommendation.approval_status.value,
                recommendation.priority,
                recommendation.created_at.isoformat(),
                recommendation.executed_at.isoformat() if recommendation.executed_at else None,
            ),
        )
        self._conn.commit()

    def _load_recommendations(self, investigation_id: str) -> List[Recommendation]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM recommendations WHERE investigation_id = ? ORDER BY created_at",
            (investigation_id,),
        ).fetchall()

        return [
            Recommendation(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                requires_approval=bool(row["requires_approval"]),
                approval_status=row["approval_status"],
                priority=row["priority"],
                created_at=datetime.fromisoformat(row["created_at"]),
                executed_at=(
                    datetime.fromisoformat(row["executed_at"])
                    if row["executed_at"]
                    else None
                ),
            )
            for row in rows
        ]

    # --- Tool Executions ---

    def save_tool_execution(
        self, investigation_id: str, execution: ToolExecution
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT INTO tool_executions
               (id, investigation_id, tool_name, arguments_json, result, status,
                error_message, executed_at, input_size_bytes, output_size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                execution.id,
                investigation_id,
                execution.tool_name,
                json.dumps(execution.arguments),
                execution.result,
                execution.status.value,
                execution.error_message,
                execution.executed_at.isoformat(),
                execution.input_size_bytes,
                execution.output_size_bytes,
            ),
        )
        self._conn.commit()

    def _load_tool_executions(self, investigation_id: str) -> List[ToolExecution]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM tool_executions WHERE investigation_id = ? ORDER BY executed_at",
            (investigation_id,),
        ).fetchall()

        return [
            ToolExecution(
                id=row["id"],
                tool_name=row["tool_name"],
                arguments=json.loads(row["arguments_json"]),
                result=row["result"],
                status=Status(row["status"]),
                error_message=row["error_message"],
                executed_at=datetime.fromisoformat(row["executed_at"]),
                input_size_bytes=row["input_size_bytes"],
                output_size_bytes=row["output_size_bytes"],
            )
            for row in rows
        ]

    # --- Audit Events ---

    def save_audit_event(
        self,
        event_type: str,
        message: str,
        details: Dict[str, Any],
        investigation_id: Optional[str] = None,
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO audit_events
               (id, investigation_id, event_type, message, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                investigation_id,
                event_type,
                message,
                json.dumps(details),
                now,
            ),
        )
        self._conn.commit()


    # --- Conversational follow-ups ---

    def create_conversation(self, investigation_id: str) -> Dict[str, Any]:
        if self._conn is None:
            raise DatabaseError("Not connected")
        if self.get_investigation(investigation_id) is None:
            raise KeyError(f"Investigation {investigation_id!r} not found")
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO conversations (id, investigation_id, created_at) VALUES (?, ?, ?)",
            (conversation_id, investigation_id, now),
        )
        self._conn.commit()
        return {
            "id": conversation_id,
            "investigation_id": investigation_id,
            "created_at": now,
            "messages": [],
        }

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        if self._conn is None:
            raise DatabaseError("Not connected")
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        messages = self._conn.execute(
            """SELECT id, conversation_id, role, content, created_at
               FROM conversation_messages WHERE conversation_id = ?
               ORDER BY created_at, rowid""",
            (conversation_id,),
        ).fetchall()
        return {
            "id": row["id"],
            "investigation_id": row["investigation_id"],
            "created_at": row["created_at"],
            "messages": [dict(m) for m in messages],
        }

    def add_conversation_message(
        self, conversation_id: str, role: str, content: str
    ) -> Dict[str, Any]:
        if self._conn is None:
            raise DatabaseError("Not connected")
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        if self.get_conversation(conversation_id) is None:
            raise KeyError(f"Conversation {conversation_id!r} not found")
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO conversation_messages
               (id, conversation_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (message_id, conversation_id, role, content, now),
        )
        self._conn.commit()
        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

    def list_audit_events(
        self, investigation_id: str, since: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        if self._conn is None:
            raise DatabaseError("Not connected")
        params: List[Any] = [investigation_id]
        query = """SELECT id, investigation_id, event_type, message,
                          details_json, created_at
                   FROM audit_events WHERE investigation_id = ?"""
        if since:
            query += " AND created_at > ?"
            params.append(since)
        query += " ORDER BY created_at, rowid LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "id": row["id"],
                "investigation_id": row["investigation_id"],
                "type": row["event_type"],
                "message": row["message"],
                "details": json.loads(row["details_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


_db: Optional[Database] = None
_event_streams: dict = {}


def get_database() -> Database:
    """Return the cached database connection, opening it on first access."""
    global _db
    if _db is None:
        config = get_config()
        _db = Database(config.database_path)
        _db.connect()
    return _db


def reset_database() -> None:
    """Close and clear the cached database. Useful for tests."""
    global _db
    if _db is not None:
        _db.close()
        _db = None

# --- Live SSE event support ---

from typing import Optional

class EventStream:
    """In-memory event stream for a running investigation."""
    def __init__(self, investigation_id: str):
        import asyncio
        self.investigation_id = investigation_id
        self.queue: Optional[asyncio.Queue] = None
        self._last_event_id: Optional[str] = None

    def init_queue(self) -> None:
        import asyncio
        if self.queue is None:
            self.queue = asyncio.Queue()

    def put(self, event_id: str, event_type: str, message: str, details: dict) -> None:
        import asyncio
        if self.queue is None:
            self.init_queue()
        self.queue.put_nowait({
            "id": event_id,
            "event": event_type,
            "data": {"message": message, "details": details},
        })
        self._last_event_id = event_id
