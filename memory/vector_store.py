import uuid
import os
import math
from copy import deepcopy
from datetime import datetime
from importlib import import_module
from typing import Any, Callable, Optional
from memory.memory_item import MemoryItem
from qdrant_client.models import UpdateStatus

class QdrantVectorStore:
    """将 MemoryItem 的向量保存到 Qdrant。"""

    def __init__(
            self, 
            qdrant_collection_name: str,
            url: Optional[str] = None, 
            api_key: Optional[str] = None, 
            embedder: Optional[Callable[[str], list[float]]] = None,
            vector_size: Optional[int] = None,
            memory_loader: Optional[Callable[[list[str]], list[MemoryItem]]] = None,
            ttl_days: Optional[int] = 30,
        ):
        self.url = os.getenv("QDRANT_URL") if url is None else url
        self.api_key = os.getenv("QDRANT_API_KEY") if api_key is None else api_key
        self.collection_name = qdrant_collection_name
        self.embedder = embedder
        self.memory_loader = memory_loader
        self._client = None
        self._vector_size = vector_size
        self._collection_ready = False
        if self.url:
            try:
                QdrantClient = getattr(import_module("qdrant_client"), "QdrantClient")
            except ImportError as exc:
                raise RuntimeError(
                    "Install qdrant-client before using QdrantVectorStore: "
                    "pip install qdrant-client"
                ) from exc
            self._client = (
                QdrantClient(location=":memory:") if self.url == ":memory:"
                else QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
            )

    def _require_ready(self) -> None:
        if self._client is None:
            raise RuntimeError("QDRANT_URL is not configured; QdrantVectorStore is unavailable")
        if self.embedder is None:
            raise RuntimeError("Provide an embedder or configure an embedding model for QdrantVectorStore")

    def _ensure_collection(self, vector_size: int, *, create: bool = True) -> bool:
        from qdrant_client.models import Distance, VectorParams

        if vector_size <= 0:
            raise ValueError("The embedding model returned an empty vector")
        if self._vector_size is not None and self._vector_size != vector_size:
            raise ValueError("Vector dimensions do not match the Qdrant collection")
        if self._collection_ready:
            return True
        if not self._client.collection_exists(self.collection_name):
            if not create:
                return False
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            params = self._client.get_collection(self.collection_name).config.params.vectors
            if not isinstance(params, VectorParams):
                raise ValueError("This implementation requires a Qdrant collection with a single unnamed vector")
            if params.size != vector_size or params.distance != Distance.COSINE:
                raise ValueError(
                    f"The collection uses {params.size} dimensions/{params.distance}, "
                    f"but the embedding uses {vector_size} dimensions/Cosine; use a new collection"
                )
        self._vector_size = vector_size
        self._collection_ready = True
        return True

    @staticmethod
    def _point_id(memory_id: str) -> str:
        # MemoryManager 的 episodic_xxx 不是 Qdrant 支持的 UUID，使用稳定映射。
        return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))

    def add(self, memory_item: MemoryItem) -> bool:
        self._require_ready()
        return self._upsert_precomputed(memory_item, self.embedder(memory_item.content))

    def update(self, memory_item: MemoryItem) -> bool:
        self._require_ready()
        return self._upsert_precomputed(memory_item, self.embedder(memory_item.content))

    def _upsert_precomputed(self, memory_item: MemoryItem, vector: list[float]) -> bool:
        """写入已生成的向量，供上层在开始修改数据库前完成抽取及 embedding。"""
        self._require_ready()
        from qdrant_client.models import PointStruct

        if not vector or any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("Embedding vectors must contain finite numbers")
        self._ensure_collection(len(vector))
        point_id = self._point_id(memory_item.id)
        result = self._client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "memory_id": memory_item.id,
                    "content": memory_item.content,
                    "type": memory_item.type,
                    "importance": memory_item.importance,
                    "created_at": memory_item.created_at.isoformat(),
                    "metadata": memory_item.metadata,
                },
            )],
            wait=True,
        )
        return result.status == UpdateStatus.COMPLETED

    def _snapshot(self, memory_id: str) -> Optional[dict[str, Any]]:
        """读取原始向量和 payload，用于精确恢复，不重新调用 embedding。"""
        self._require_ready()
        if not self._client.collection_exists(self.collection_name):
            return None
        points = self._client.retrieve(
            self.collection_name, ids=[self._point_id(memory_id)], with_payload=True, with_vectors=True,
        )
        if not points:
            return None
        point = points[0]
        return {"id": str(point.id), "vector": deepcopy(point.vector), "payload": deepcopy(point.payload)}

    def _restore_snapshot(self, memory_id: str, snapshot: Optional[dict[str, Any]]) -> None:
        """从快照恢复向量和 payload。"""
        self._require_ready()
        if snapshot is None:
            self.remove(memory_id)
            return
        from qdrant_client.models import PointStruct

        if snapshot["id"] != self._point_id(memory_id):
            raise ValueError("Snapshot does not belong to the requested memory")
        result = self._client.upsert(
            collection_name=self.collection_name, points=[PointStruct(**snapshot)], wait=True,
        )
        if result.status != UpdateStatus.COMPLETED:
            raise RuntimeError("Failed to restore the original vector")

    def retrieve(self, query: str, limit: int = 5, threshold: float = 0.7, **kwargs: Any) -> list[dict[str, Any]]:
        """
        检索相似记忆，返回包含 memory、memory_id 和 score 的字典列表。

        配置 memory_loader 时批量从正文存储读取完整记忆，跳过孤立向量；
        未配置时从 payload 还原 MemoryItem，缺少正文的旧记录 memory 为 None。
        原有 payload 字段继续保留，兼容只读取 memory_id 的调用方。
        支持 allowed_ids 和 score_threshold，供上层筛选和重排。
        仅需要记忆 ID 时，可提取每个结果的 memory_id 字段。
        """
        self._require_ready()
        if limit <= 0 or not query.strip():
            return []
        from qdrant_client.models import Filter, HasIdCondition

        allowed_ids = kwargs.get("allowed_ids")
        if allowed_ids is not None and not allowed_ids:
            return []
        vector = self.embedder(query)
        if not self._ensure_collection(len(vector), create=False):
            return []
        query_filter = None
        if allowed_ids is not None:
            query_filter = Filter(must=[HasIdCondition(
                has_id=[self._point_id(memory_id) for memory_id in allowed_ids]
            )])
        result = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=kwargs.get("score_threshold", threshold),
            with_payload=True,
            with_vectors=False,
        )
        hits = [
            {**point.payload, "memory_id": str(point.payload["memory_id"]),
             "score": float(point.score)} for point in result.points
            if point.payload and "memory_id" in point.payload
        ]
        if not hits:
            return []
        if self.memory_loader is not None:
            ids = list(dict.fromkeys(hit["memory_id"] for hit in hits))
            memories = {item.id: item for item in self.memory_loader(ids)}
            return [{**hit, "memory": memories[hit["memory_id"]]} for hit in hits
                    if hit["memory_id"] in memories]
        return [{**hit, "memory": self._memory_from_payload(hit)} for hit in hits]

    @staticmethod
    def _memory_from_payload(payload: dict[str, Any]) -> Optional[MemoryItem]:
        """还原完整记忆；旧 payload 缺字段时不编造原文或创建时间。"""
        required = ("memory_id", "content", "type", "importance", "created_at")
        if any(key not in payload for key in required):
            return None
        return MemoryItem(
            id=str(payload["memory_id"]), content=payload["content"], type=payload["type"],
            importance=payload["importance"], created_at=datetime.fromisoformat(payload["created_at"]),
            metadata=deepcopy(payload.get("metadata") or {}),
        )

    def remove(self, memory_id: str) -> bool:
        if self._client is not None:
            if not self._collection_ready and not self._client.collection_exists(self.collection_name):
                return False
            from qdrant_client.models import PointIdsList
            result = self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=[self._point_id(memory_id)]
                ),
                wait=True,
            )
            if result.status != UpdateStatus.COMPLETED:
                raise RuntimeError("Failed to delete the memory vector")
            return True
        return False
