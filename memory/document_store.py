import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import Any, Optional

from memory.memory_item import MemoryItem


_DEFAULT_TTL = object()


class SQLiteDocumentStore:
    """使用 SQLite 保存记忆正文、元数据和结构化字段。"""

    def __init__(self, database_path: str = "memory.db", capacity: int = 1000,
                 ttl_days: Optional[int] = 30):
        self.database_path = database_path
        self.capacity = capacity
        self.ttl_days = ttl_days
        parent = os.path.dirname(os.path.abspath(database_path))
        os.makedirs(parent, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at "
                "ON memories(created_at)"
            )

    @staticmethod
    def _to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            type=row["type"],
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _to_utc(value: Any, field_name: str = "created_at") -> datetime:
        """保留原始时间戳；比较时将无时区时间按本地时间解释，再统一到 UTC。"""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be a valid ISO 8601 datetime string") from exc
        elif not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime or an ISO 8601 datetime string")
        return value.astimezone(timezone.utc)

    def add(self, memory_item: MemoryItem, *, cleanup: bool = True) -> str:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, content, type, importance, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_item.id,
                    memory_item.content,
                    memory_item.type,
                    memory_item.importance,
                    memory_item.created_at.isoformat(),
                    json.dumps(memory_item.metadata, ensure_ascii=False),
                ),
            )
        return memory_item.id

    def _get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._to_item(row) if row is not None else None

    def _get_by_ids(self, memory_ids: list[str]) -> list[MemoryItem]:
        """按向量检索的相关性顺序读取正文，跳过不存在的 ID。"""
        items: dict[str, MemoryItem] = {}
        with self._connect() as connection:
            # 分批查询，避免超过 SQLite 的绑定参数数量限制。
            for offset in range(0, len(memory_ids), 500):
                batch = memory_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})", batch
                ).fetchall()
                items.update((row["id"], self._to_item(row)) for row in rows)
        return [items[memory_id] for memory_id in memory_ids if memory_id in items]

    def _get_before_time(self, end_time: datetime, start_time: Optional[datetime] = None) -> list[MemoryItem]:
        """按实际时间降序读取正文，时间范围为左闭右开，兼容旧记录的时区。"""
        end_time_utc = self._to_utc(end_time, "end_time")
        start_time_utc = self._to_utc(start_time, "start_time") if start_time is not None else None
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memories").fetchall()
        results = []
        for row in rows:
            created_at = self._to_utc(row["created_at"])
            if created_at >= end_time_utc:
                continue
            if start_time_utc is not None and created_at < start_time_utc:
                continue
            results.append((created_at, row["id"], self._to_item(row)))
        results.sort(key=lambda result: result[:2], reverse=True)
        return [item for _, _, item in results]

    def retrieve(self, query: str = "", limit: int = 5, **kwargs: Any) -> list[MemoryItem]:
        sql = "SELECT * FROM memories"
        parameters: list[Any] = []
        conditions: list[str] = []
        if query:
            conditions.append("content LIKE ?")
            parameters.append(f"%{query}%")
        if kwargs.get("session_id") is not None:
            conditions.append("json_extract(metadata, '$.session_id') = ?")
            parameters.append(kwargs["session_id"])
        if kwargs.get("min_importance") is not None:
            conditions.append("importance >= ?")
            parameters.append(kwargs["min_importance"])
        boundaries = {
            key: self._to_utc(kwargs[key], key)
            for key in ("start_time", "end_time") if kwargs.get(key) is not None
        }
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        # 旧记录可能混用本地时间和不同偏移量，不能按 ISO 字符串过滤或排序。
        results = []
        for row in rows:
            created_at = self._to_utc(row["created_at"])
            if "start_time" in boundaries and created_at < boundaries["start_time"]:
                continue
            if "end_time" in boundaries and created_at > boundaries["end_time"]:
                continue
            results.append((created_at, row["id"], self._to_item(row)))
        results.sort(key=lambda result: result[:2], reverse=True)
        return [item for _, _, item in results[:max(0, limit)]]

    def get_all(self) -> list[MemoryItem]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories"
            ).fetchall()
        rows = sorted(rows, key=lambda row: (self._to_utc(row["created_at"]), row["id"]), reverse=True)
        return [self._to_item(row) for row in rows]

    def overflow_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at FROM memories"
            ).fetchall()
        rows = sorted(rows, key=lambda row: (self._to_utc(row["created_at"]), row["id"]), reverse=True)
        return [row["id"] for row in rows[self.capacity:]]

    def remove(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def contains(self, memory_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return row is not None

    def _enforce_capacity(self) -> int:
        return sum(self.remove(memory_id) for memory_id in self.overflow_ids())

    def _remove_low_importance_memory(self, threshold: float = 0.1) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE importance < ?", (threshold,)
            )
        return cursor.rowcount
