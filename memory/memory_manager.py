from typing import Any, Optional, List
from uuid import uuid4

from memory_config import MemoryConfig
from memory.memory_item import MemoryItem
from memory.retrieval_hit import MemoryRetrievalHit
from memory_store import MemoryStore
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory

class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = True
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        self.memory_types: dict[str, MemoryStore] = {}

        if enable_working:
            self.memory_types['working'] = WorkingMemory(self.config)

        if enable_episodic:
            self.memory_types['episodic'] = EpisodicMemory(self.config)

        if enable_semantic:
            self.memory_types['semantic'] = SemanticMemory(self.config)

    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        auto_classify: bool,
        metadata: Optional[dict] = None
    ) -> str:
        if memory_type not in self.memory_types:
            raise ValueError(f"Memory type '{memory_type}' is not enabled.")

        memory = self.memory_types[memory_type]
        memory_item = MemoryItem(
            id=f"{memory_type}_{uuid4().hex}",
            content=content,
            type=memory_type,
            importance=importance,
            metadata=metadata or {}
        )
        return memory.add(memory_item)

    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5,
        min_importance: float = 0.1,
        threshold: float = 0.7,
        session_id: Optional[str] = None,
    ) -> List[MemoryRetrievalHit]:
        """合并各类检索结果，按本次检索 score 降序截取。"""
        if limit <= 0:
            return []
        result: List[MemoryRetrievalHit] = []

        if memory_types is None:
            memory_types = list(self.memory_types.keys())

        for memory_type in memory_types:
            if memory_type not in self.memory_types:
                raise ValueError(f"Memory type '{memory_type}' is not enabled.")
            memory = self.memory_types[memory_type]
            filters: dict[str, Any] = {"min_importance": min_importance}
            if session_id is not None:
                filters["session_id"] = session_id
            retrieved_memories = memory.retrieve(query, limit=limit, threshold=threshold, **filters)
            for hit in retrieved_memories:
                retrieved_memory = hit["memory"]
                if (retrieved_memory.importance >= min_importance
                        and (session_id is None or retrieved_memory.metadata.get("session_id") == session_id)):
                    result.append(hit)

        result.sort(key=lambda hit: hit["score"], reverse=True)
        return result[:limit]

    def forget_memories(
        self,
        strategy: str,
        treshold: Optional[float] = None
    ) -> int:
        count = 0
        if strategy not in ['importance_based', 'time_based']:
            raise ValueError(f"Invalid forgetting strategy: {strategy}")
        elif strategy == 'importance_based':
            if treshold is None:
                raise ValueError("Threshold must be provided for importance-based forgetting.")
            else:
                for memory_type in self.memory_types:
                    count += getattr(
                        self.memory_types[memory_type],
                        '_remove_low_importance_memory'
                    )(treshold)
        elif strategy == 'time_based':
            for memory_type in self.memory_types:
                count += getattr(
                    self.memory_types[memory_type],
                    '_expire_old_memories'
                )()
        return count

    def get_all_working_memory(self) -> List[MemoryItem]:
        """获取所有工作记忆"""
        return self.memory_types['working'].get_all()


    # def consolidate_memories(
    #     self,
    #     from_type: str,
    #     to_type: str,
    #     importance_threshold: float = 0.5
    # ) -> int:
    #     """整合记忆（将重要的短期记忆提升为长期记忆）"""
        

            
        

        


