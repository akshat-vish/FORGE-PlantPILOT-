"""Persistent software digital-twin records and isolated what-if simulation."""
from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime, timezone
from typing import Any
from .config import get_config


def _now(): return datetime.now(timezone.utc).isoformat()

class TwinStore:
    def __init__(self):
        self.path = get_config().database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS digital_twins (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, machine_type TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS twin_readings (
              id TEXT PRIMARY KEY, twin_id TEXT NOT NULL, reading_json TEXT NOT NULL,
              assessment_json TEXT NOT NULL, evidence_type TEXT NOT NULL DEFAULT 'observed', created_at TEXT NOT NULL,
              FOREIGN KEY(twin_id) REFERENCES digital_twins(id));
            CREATE INDEX IF NOT EXISTS idx_twin_readings ON twin_readings(twin_id, created_at);
            ''')
    def _connect(self):
        c=sqlite3.connect(str(self.path), timeout=10); c.row_factory=sqlite3.Row; c.execute('PRAGMA foreign_keys=ON'); return c
    def create(self, name: str, machine_type: str, metadata: dict[str,Any]):
        twin_id='twin_'+uuid.uuid4().hex[:12]; now=_now()
        with self._connect() as c:
            c.execute('INSERT INTO digital_twins VALUES (?,?,?,?,?,?)',(twin_id,name,machine_type,json.dumps(metadata),now,now))
        return self.get(twin_id)
    def get(self, twin_id: str):
        with self._connect() as c:
            r=c.execute('SELECT * FROM digital_twins WHERE id=?',(twin_id,)).fetchone()
            if not r: return None
            out=dict(r); out['metadata']=json.loads(out.pop('metadata_json'))
            latest=c.execute("SELECT * FROM twin_readings WHERE twin_id=? AND evidence_type='observed' ORDER BY created_at DESC LIMIT 1",(twin_id,)).fetchone()
            out['latest_observed_reading']=self._reading(latest) if latest else None
            count=c.execute("SELECT COUNT(*) n FROM twin_readings WHERE twin_id=? AND evidence_type='observed'",(twin_id,)).fetchone()['n']
            out['observed_reading_count']=count
            return out
    def list(self, limit=50):
        with self._connect() as c: ids=[r['id'] for r in c.execute('SELECT id FROM digital_twins ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()]
        return [self.get(i) for i in ids]
    def _reading(self,r):
        if not r: return None
        d=dict(r); d['reading']=json.loads(d.pop('reading_json')); d['assessment']=json.loads(d.pop('assessment_json')); return d
    def history(self,twin_id: str, limit=100):
        with self._connect() as c:
            if not c.execute('SELECT 1 FROM digital_twins WHERE id=?',(twin_id,)).fetchone(): return None
            rows=c.execute('SELECT * FROM twin_readings WHERE twin_id=? ORDER BY created_at DESC LIMIT ?', (twin_id,limit)).fetchall()
            return [self._reading(r) for r in rows]
    def add_reading(self,twin_id: str, reading: dict, assessment: dict, evidence_type='observed'):
        rid='rdg_'+uuid.uuid4().hex[:12]; now=_now()
        with self._connect() as c:
            if not c.execute('SELECT 1 FROM digital_twins WHERE id=?',(twin_id,)).fetchone(): return None
            c.execute('INSERT INTO twin_readings VALUES (?,?,?,?,?,?)',(rid,twin_id,json.dumps(reading),json.dumps(assessment),evidence_type,now))
            if evidence_type=='observed': c.execute('UPDATE digital_twins SET updated_at=? WHERE id=?',(now,twin_id))
        return {'id':rid,'twin_id':twin_id,'reading':reading,'assessment':assessment,'evidence_type':evidence_type,'created_at':now}

_store: TwinStore | None = None
def get_twin_store():
    global _store
    if _store is None: _store=TwinStore()
    return _store
def reset_twin_store():
    global _store; _store=None
