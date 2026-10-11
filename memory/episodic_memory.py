import math
import os
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional

from memory_config import MemoryConfig
from memory.memory_item import MemoryItem
from memory.retrieval_hit import MemoryRetrievalHit
from memory_store import MemoryStore
from document_store import SQLiteDocumentStore
from vector_store import QdrantVectorStore

def create_embedder(model_name: str):
    """创建 OpenAI 兼容的文本嵌入函数。"""
    from openai import OpenAI

    api_key = os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set EMBED_API_KEY in .env")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("EMBED_BASE_URL"),
        timeout=30,
        max_retries=1,
    )

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(model=model_name, input=text)
        return response.data[0].embedding

    return embed


class EpisodicMemory(MemoryStore):
    """情景记忆实现
    特点：
    - 容量有限（默认1000条）+ TTL自动清理
    - SQLite+Qdrant混合存储架构
    - 支持时间序列和会话级检索
    - 结构化过滤 + 语义向量检索
    """

    def __init__(
            self, 
            config: MemoryConfig,
            embedder: Optional[Callable[[str], list[float]]] = None
        ):
        self.config = config
        self.sessions = {}
        self.doc_store = SQLiteDocumentStore(
            config.database_path,
            config.eposidic_memory_capacity,
            config.episodic_memory_ttl_days,
        )
        self.embedder = (embedder or create_embedder(
            os.getenv("EMBED_MODEL_NAME", "text-embedding-v3")
        ))
        self.vector_store = QdrantVectorStore(
            qdrant_collection_name = "episodic_memories",
            embedder = self.embedder
        )
        self._refresh_sessions()
        self.ttl_days = config.episodic_memory_ttl_days

    def add(self, memory_item: MemoryItem) -> str:
        """
        添加记忆项到情景记忆中。
        """
        try:
            return self._persist_episode(memory_item)
        finally:
            self._refresh_sessions()

    def update(self, memory_item: MemoryItem) -> bool:
        """更新已有记忆的正文、重要性和 metadata；metadata 整体替换。

        保留原记录的 ID、记忆类型及创建时间，不延长 TTL。
        不存在或已过期时返回 False；成功返回 True，存储失败时抛出异常。
        """
        self._expire_old_memories()
        previous = self.doc_store._get_by_id(memory_item.id)
        if previous is None:
            return False
        updated = replace(
            memory_item,
            id=previous.id,
            type=previous.type,
            created_at=previous.created_at,
        )
        try:
            self._persist_episode(updated)
        finally:
            self._refresh_sessions()
        return True

    def _refresh_sessions(self) -> None:
        """根据 SQLite 中实际存在的记忆重建会话索引。"""
        sessions: dict[Any, list[str]] = {}
        for item in self.doc_store.get_all():
            session_id = item.metadata.get("session_id")
            sessions.setdefault(session_id, []).append(item.id)
        self.sessions = sessions

    def _persist_episode(self, memory_item: MemoryItem) -> str:
        """将记忆项持久化到 SQLite 和 Qdrant 中，返回记忆 ID。"""
        self._expire_old_memories()
        previous = self.doc_store._get_by_id(memory_item.id)
        if self.config.qdrant_url:
            self.vector_store.add(memory_item)
        try:
            self.doc_store.add(memory_item, cleanup=False)
        except Exception:
            # SQL 写入失败时尽量恢复原来的向量；两种数据库不共享事务。
            if self.config.qdrant_url:
                if previous is None:
                    self.vector_store.remove(memory_item.id)
                else:
                    self.vector_store.add(previous)
            raise
        self._expire_old_memories()
        self._enforce_capacity()
        return memory_item.id

    def retrieve(self, query: str = "", limit: int = 5, threshold: float = 0.7, **kwargs: Any) -> list[MemoryRetrievalHit]:
        """返回记忆及本次排序分数；向量检索保留原有综合评分。

        SQLite 回退使用正文子串匹配，命中记为 1.0；空查询列举记为 0.0。
        """
        self._expire_old_memories()
        if limit <= 0:
            return []
        if not query.strip() or not self.config.qdrant_url:
            return [{"memory": item, "score": 1.0 if query.strip() else 0.0}
                    for item in self.doc_store.retrieve(query, limit=limit, **kwargs)]
        # 先筛选有效正文 ID，再执行向量检索，过滤掉过期或孤立向量。
        # 时间范围在 _should_include 中统一按 UTC 比较，避免 ISO 字符串字典序误判。
        candidate_filters = {
            key: kwargs[key] for key in ("session_id", "min_importance") if key in kwargs
        }
        candidates = self.doc_store.retrieve(
            "", limit=self.doc_store.capacity, **candidate_filters
        )
        # 先筛选正文，再限制向量查询，避免不符合条件的记忆占用结果名额。
        candidates = [item for item in candidates if self._should_include(
            {"id": item.id}, [item],
            {key: value for key, value in kwargs.items() if key != "score_threshold"},
        )]
        candidates_by_id = {item.id: item for item in candidates}
        hits = self.vector_store.retrieve(
            query, limit=len(candidates),
            allowed_ids=[item.id for item in candidates],
            threshold=threshold,
        )
        results = []
        for vector_hit in hits:
            item = candidates_by_id.get(vector_hit["memory_id"])
            if item is None:
                continue
            # 正文和结构化字段以 SQLite 为准，Qdrant 仅提供相关性分数。
            hit = {
                "id": item.id,
                "content": item.content,
                "memory_type": item.type,
                "importance": item.importance,
                "created_at": item.created_at,
                "metadata": item.metadata,
                "score": vector_hit["score"],
            }
            if self._should_include(hit, candidates, kwargs):
                score = self._calculate_episode_score(hit)
                memory_item = self._create_memory_item_from_hit(hit)
                results.append({"memory": memory_item, "score": float(score)})
        results.sort(key=lambda hit: hit["score"], reverse=True)
        return results[:limit]

    def _create_memory_item_from_hit(self, hit: dict[str, Any]) -> MemoryItem:
        """将检索结果转换为记忆项，恢复并校验原始创建时间。"""
        created_at = hit.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError as exc:
                raise ValueError(
                    "hit['created_at'] must be a valid ISO 8601 datetime string"
                ) from exc
        elif not isinstance(created_at, datetime):
            raise TypeError(
                "hit['created_at'] must be a datetime or an ISO 8601 datetime string"
            )

        return MemoryItem(
            id = hit["id"],
            content = hit["content"],
            type = hit["memory_type"],
            importance = hit["importance"],
            metadata = hit.get("metadata", {}),
            created_at = created_at,
        )

    def _should_include(self, hit: dict[str, Any], candidates: list[MemoryItem], kwargs: dict[str, Any]) -> bool:
        """根据候选记忆和过滤条件判断是否包含该检索结果。"""
        memory_id = hit.get("id") or hit.get("memory_id")
        candidate = next((item for item in candidates if item.id == memory_id), None)
        if candidate is None:
            return False

        # 使用正文存储的字段，避免向量 payload 的旧数据影响过滤结果。
        session_id = kwargs.get("session_id")
        if session_id is not None and candidate.metadata.get("session_id") != session_id:
            return False
        min_importance = kwargs.get("min_importance")
        if min_importance is not None and candidate.importance < float(min_importance):
            return False

        created_at = candidate.created_at.astimezone(timezone.utc)
        for key in ("start_time", "end_time"):
            boundary = kwargs.get(key)
            if boundary is None:
                continue
            if isinstance(boundary, str):
                try:
                    boundary = datetime.fromisoformat(boundary)
                except ValueError as exc:
                    raise ValueError(f"{key} must be a valid ISO 8601 datetime string") from exc
            elif not isinstance(boundary, datetime):
                raise TypeError(f"{key} must be a datetime or an ISO 8601 datetime string")
            boundary = boundary.astimezone(timezone.utc)
            if key == "start_time" and created_at < boundary:
                return False
            if key == "end_time" and created_at > boundary:
                return False

        try:
            score = float(hit.get("score", 0.0))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(score):
            return False
        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None and score < float(score_threshold):
            return False
        return True

    def get_all(self) -> list[MemoryItem]:
        self._expire_old_memories()
        return self.doc_store.get_all()

    def contains(self, memory_id: str) -> bool:
        self._expire_old_memories()
        return self.doc_store.contains(memory_id)

    def remove(self, memory_id: str) -> bool:
        # 先删除向量。远端失败时保留 SQL 记录，便于下次重试。
        try:
            if self.config.qdrant_url:
                self.vector_store.remove(memory_id)
            return self.doc_store.remove(memory_id)
        finally:
            self._refresh_sessions()

    def _expire_old_memories(self) -> int:
        """删除过期记忆，返回删除数量。"""
        age_limit = self._ttl_delta(self.ttl_days)
        if age_limit is None:
            return 0
        expired_datetime = datetime.now(timezone.utc) - age_limit
        expired_memories = self.doc_store._get_before_time(expired_datetime)
        return sum(self.remove(memory.id) for memory in expired_memories)
        
    def _enforce_capacity(self) -> int:
        return sum(self.remove(memory_id) for memory_id in self.doc_store.overflow_ids())

    def _remove_low_importance_memory(self, threshold: float = 0.1) -> int:
        return sum(
            self.remove(item.id) for item in self.doc_store.get_all()
            if item.importance < threshold
        )

    def _calculate_episode_score(self, hit) -> float:
        """计算情景记忆的综合评分，结合向量相似度和重要性。"""
        vec_score = float(hit.get("score", 0.0))
        recency_score = self._calculate_recency_score(hit["created_at"])
        importance = hit["importance"]

        base_relevance = vec_score * 0.8 + recency_score * 0.2
        importance_weight = 0.8 + (importance * 0.4)

        return base_relevance * importance_weight

    def _calculate_recency_score(self, created_at: datetime | str) -> float:
        """按记忆年龄计算指数衰减分数，范围为 0 到 1。

        支持 datetime 和 ISO 时间字符串；无时区时间按本地时间解释。
        使用情景记忆 TTL 天数作为衰减尺度；未设置 TTL 时使用 30 天，
        尺度最小为 1 天，避免除以零。未来时间按年龄为零处理。
        """
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError as exc:
                raise ValueError(
                    "created_at must be a valid ISO 8601 datetime string"
                ) from exc
        elif not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime or an ISO 8601 datetime string")

        # 统一到 UTC 再相减，兼容带时区时间，并避免夏令时切换影响。
        created_at_utc = created_at.astimezone(timezone.utc)
        age_days = max(
            0.0,
            (datetime.now(timezone.utc) - created_at_utc).total_seconds() / 86400,
        )
        ttl_days = self.config.episodic_memory_ttl_days
        decay_days = max(1.0, float(ttl_days) if ttl_days is not None else 30.0)
        return math.exp(-age_days / decay_days)

    @staticmethod
    def _ttl_delta(threshold: Optional[float]) -> Optional[timedelta]:
        if threshold is None:
            return None
        try:
            days = float(threshold)
            if not math.isfinite(days) or days < 0:
                raise ValueError
            return timedelta(days=days)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("threshold must be a finite, non-negative number of days or None") from exc

