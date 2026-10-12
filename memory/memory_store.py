from typing import Protocol

from memory.memory_item import MemoryItem
from memory.retrieval_hit import MemoryRetrievalHit


class MemoryStore(Protocol):
    """各类记忆存储需要实现的公共操作。"""

    def add(self, memory_item: MemoryItem) -> str:
        """保存记忆并返回其 ID。"""
        ...

    def update(self, memory_item: MemoryItem) -> bool:
        """更新指定 ID 的记忆，返回是否找到并更新。"""
        ...

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> list[MemoryRetrievalHit]:
        """返回最多 limit 条记忆及本次检索分数，按 score 降序排列。"""
        ...

    def get_all(self) -> list[MemoryItem]:
        """返回存储中的记忆快照。"""
        ...

    def remove(self, memory_id: str) -> bool:
        """删除指定 ID 的记忆，返回是否找到并删除。"""
        ...

    def remove_memories_by_session(self, session_id: str) -> int:
        """删除指定 session_id 的记忆，返回删除的数量。"""
        ...

    def contains(self, memory_id: str) -> bool:
        """判断指定 ID 是否存在。"""
        ...

    def _expire_old_memories(self) -> int:
        """清理过期的记忆，返回清理的数量。"""
        ...

    def _remove_low_importance_memory(self, treshold: float = 0.1) -> int:
        """移除重要性低于阈值的记忆，返回移除的数量。"""
        ...
