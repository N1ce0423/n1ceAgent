"""只持久化可选择的 session_id，不保存对话正文或摘要。"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4


@dataclass(frozen=True)
class SessionInfo:
    session_id: str


class SessionStore:
    def __init__(self, database_path: str):
        self.database_path = str(Path(database_path).resolve())
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            columns = [row["name"] for row in connection.execute("PRAGMA table_info(chat_sessions)")]
            # 用户要求只保留 ID：事务内迁移旧记录，同时清除历史正文及摘要。
            connection.execute("DROP TABLE IF EXISTS chat_messages")
            if columns and columns != ["session_id"]:
                connection.execute("CREATE TABLE chat_session_ids (session_id TEXT PRIMARY KEY)")
                connection.execute("INSERT INTO chat_session_ids SELECT session_id FROM chat_sessions")
                connection.execute("DROP TABLE chat_sessions")
                connection.execute("ALTER TABLE chat_session_ids RENAME TO chat_sessions")
            else:
                connection.execute("CREATE TABLE IF NOT EXISTS chat_sessions (session_id TEXT PRIMARY KEY)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # 删除旧历史时清除对应页的内容，避免正文留在 SQLite 的空闲页中。
        connection.execute("PRAGMA secure_delete = ON")
        try:
            # DDL 和数据迁移在同一事务中提交；失败时保留原数据库状态。
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                yield connection
        finally:
            connection.close()

    def create_session(self) -> str:
        session_id = f"session_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute("INSERT INTO chat_sessions(session_id) VALUES (?)", (session_id,))
        return session_id

    def require_session(self, session_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id FROM chat_sessions WHERE session_id = ?", (session_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Session not found: {session_id}")

    def list_sessions(self) -> list[SessionInfo]:
        with self._connect() as connection:
            rows = connection.execute("SELECT session_id FROM chat_sessions ORDER BY session_id").fetchall()
        return [SessionInfo(row["session_id"]) for row in rows]

    def delete_session(self, session_id: str) -> None:
        """删除会话 ID；调用方必须先完成该会话的记忆清理。"""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            if cursor.rowcount == 0:
                raise ValueError(f"Session not found: {session_id}")
