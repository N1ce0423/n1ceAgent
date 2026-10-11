from typing import TypedDict

from memory.memory_item import MemoryItem


class MemoryRetrievalHit(TypedDict):
    """本次检索的记忆和排序分数；分数不写入持久化记忆。"""

    memory: MemoryItem
    score: float
