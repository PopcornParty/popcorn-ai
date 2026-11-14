"""Explicitly approved SQLite memory prototype; not connected to the live chat."""
import json,re,sqlite3
from pathlib import Path
from typing import Any
class MemoryConsentError(ValueError): pass
class MemoryStore:
 def __init__(self,database_path: str|Path):
  self.database_path=str(database_path)
  with self._connect() as db: db.execute("""CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,content TEXT NOT NULL,source TEXT NOT NULL,confidence REAL NOT NULL,user_confirmed INTEGER NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
 def _connect(self):
  db=sqlite3.connect(self.database_path); db.row_factory=sqlite3.Row; return db
 @staticmethod
 def _looks_sensitive(content: str)->bool:
  patterns=[r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]",r"(?i)-----BEGIN .*PRIVATE KEY-----",r"(?i)password\s*[:=]"]
  return any(re.search(p,content) for p in patterns)
 def remember(self,content: str,*,kind: str,source: str,user_confirmed: bool,confidence: float=1.0,metadata: dict[str,Any]|None=None)->int:
  content=content.strip()
  if not user_confirmed: raise MemoryConsentError("A memory must be explicitly confirmed before it can be saved")
  if not content or len(content)>4000: raise ValueError("Memory content must contain 1–4000 characters")
  if kind not in {"preference","project","fact","instruction"}: raise ValueError("Unsupported memory kind")
  if not 0<=confidence<=1: raise ValueError("confidence must be between 0 and 1")
  if self._looks_sensitive(content): raise MemoryConsentError("This looks like a secret or credential and cannot be stored")
  with self._connect() as db:
   cur=db.execute("INSERT INTO memories(kind,content,source,confidence,user_confirmed,metadata_json) VALUES(?,?,?,?,1,?)",(kind,content,source[:200],confidence,json.dumps(metadata or {},ensure_ascii=False)))
   return int(cur.lastrowid)
 def search(self,query: str,limit: int=10)->list[dict]:
  if not 1<=limit<=50: raise ValueError("limit must be between 1 and 50")
  terms=query.strip().split()
  if not terms: return []
  where=" AND ".join(["content LIKE ?" for _ in terms]); params=[f"%{term}%" for term in terms]
  with self._connect() as db: rows=db.execute(f"SELECT id,kind,content,source,confidence,created_at FROM memories WHERE user_confirmed=1 AND {where} ORDER BY id DESC LIMIT ?",(*params,limit)).fetchall()
  return [dict(row) for row in rows]
 def delete(self,memory_id: int)->bool:
  with self._connect() as db:
   cur=db.execute("DELETE FROM memories WHERE id=?",(memory_id,)); return cur.rowcount>0
