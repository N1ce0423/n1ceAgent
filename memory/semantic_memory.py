import json
import math
import os
import hashlib
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional, Dict, List
from uuid import NAMESPACE_URL, uuid4, uuid5

from memory.graph_store import Neo4jGraphStore
from memory_store import MemoryStore
from memory.retrieval_hit import MemoryRetrievalHit
from memory_config import MemoryConfig
from memory.memory_item import MemoryItem
from vector_store import QdrantVectorStore
from entity import Entity
from relation import Relation
from memory.gliner_relation_extractor import GLiNER2RelationExtractor

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

class SemanticMemory(MemoryStore):
    """
    语义记忆实现
    
    特点：
    - 使用阿里云text-embedding-v3预训练模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量+图+语义推理
    """

    def __init__(
            self, 
            config: MemoryConfig, 
            storage_backend = None,
            embedder: Optional[Callable[[str], list[float]]] = None,
            relation_extractor: Optional[GLiNER2RelationExtractor] = None
        ):
        super().__init__()

        self.embedder = (embedder or create_embedder(
            os.getenv("EMBED_MODEL_NAME", "text-embedding-v3")
        ))

        self.graph_store = Neo4jGraphStore()
        self.vector_store = QdrantVectorStore(
            qdrant_collection_name = "semantic_memories",
            embedder = self.embedder,
            memory_loader = self.graph_store.get_many,
        )

        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.ttl_days = config.semantic_memory_ttl_days
        self._write_lock = RLock()
        self.repair_directory = Path(os.getenv(
            "MEMORY_REPAIR_DIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "memory-repairs")
        ))
        
        self.nlp = self._init_nlp()
        self.relation_extractor = relation_extractor or GLiNER2RelationExtractor()

    def add(self, memory_item: MemoryItem) -> str:
        """新增或替换记忆，正文、向量和本条记忆的图证据使用同一份抽取结果。"""
        with self._write_guard():
            previous = self.graph_store._get_by_id(memory_item.id)
            stored, entities, relations, vector = self._prepare_update(memory_item, previous)
            if not self._persist_semantic(stored, entities, relations, vector, previous):
                raise RuntimeError("Memory disappeared during insertion; retry the operation")
            return memory_item.id

    def update(self, memory_item: MemoryItem) -> bool:
        """替换正文及其派生图，保留类型和创建时间；不存在时返回 False。"""
        with self._write_guard():
            previous = self.graph_store._get_by_id(memory_item.id)
            if previous is None:
                return False
            stored, entities, relations, vector = self._prepare_update(memory_item, previous)
            return self._persist_semantic(stored, entities, relations, vector, previous)

    def remove(self, memory_id: str) -> bool:
        """删除正文、向量及关系来源；清理孤立向量也返回 True，均不存在返回 False。"""
        with self._write_guard():
            previous = self.graph_store._get_by_id(memory_id)
            if previous is None:
                return self._persist_semantic(None, [], [], None, None, deleting=True, memory_id=memory_id)
            return self._persist_semantic(previous, [], [], None, previous, deleting=True)

    def remove_memories_by_session(self, session_id: str) -> int:
        """删除指定 session_id 的记忆，返回删除的数量。"""
        with self._write_guard():
            memory_ids = [item.id for item in self.get_all()
                          if item.metadata.get("session_id") == session_id]
            return sum(self.remove(memory_id) for memory_id in memory_ids)

    def _write_guard(self):
        return self.__dict__.setdefault("_write_lock", RLock())

    def _prepare_update(self, memory_item: MemoryItem, previous: Optional[MemoryItem]):
        entities = self._extract_entities(memory_item.content)
        relations = self._extract_relations(memory_item.content, entities)
        stored = replace(memory_item, metadata={
            **memory_item.metadata,
            "entities": [entity.id for entity in entities],
            "entity_count": len(entities), "relation_count": len(relations),
        })
        if previous is not None:
            stored = replace(stored, type=previous.type, created_at=previous.created_at)
        vector = self.embedder(stored.content)
        if not vector or any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("Embedding vectors must contain finite numbers")
        # 在写入前检查 JSON，模型或序列化失败不会修改任何数据库。
        json.dumps(stored.metadata)
        entity_ids = {entity.id for entity in entities}
        for entity in entities:
            json.dumps(entity.properties)
        for relation in relations:
            if relation.source_id not in entity_ids or relation.target_id not in entity_ids:
                raise ValueError("Relation endpoints must belong to the memory's extracted entities")
            Neo4jGraphStore.relationship_type(relation.relationship_type)
            if relation.relationship_type == "MENTIONS":
                raise ValueError("MENTIONS is reserved for memory-to-entity links")
            if not isinstance(relation.properties.get("evidence", []), list):
                raise ValueError("Relation evidence must be a list")
            json.dumps(relation.properties)
        return stored, entities, relations, vector

    def _repair_path(self, memory_id: str) -> Path:
        directory = getattr(self, "repair_directory", Path(__file__).resolve().parents[1] / ".runtime" / "memory-repairs")
        return Path(directory) / (hashlib.sha256(memory_id.encode("utf-8")).hexdigest() + ".json")

    @staticmethod
    def _save_repair(path: Path, record: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _persist_semantic(
        self, stored, entities, relations, vector, previous, *,
        deleting=False, memory_id: Optional[str] = None,
    ) -> bool:
        memory_id = stored.id if stored is not None else memory_id
        if memory_id is None or stored is None and not deleting:
            raise ValueError("A memory item is required unless deleting by ID")
        path = self._repair_path(memory_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"memory_id": memory_id, "token": uuid4().hex,
                  "deleting": deleting, "phase": "preparing"}
        # 同一 ID 的其他写入必须等待，留下的日志必须先恢复，避免覆盖补偿状态。
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream)
        except FileExistsError as exc:
            raise RuntimeError("Memory has an active or pending write; recover pending writes before retrying") from exc
        try:
            record["snapshot"] = self.vector_store._snapshot(memory_id)
            if deleting and previous is None and record["snapshot"] is None:
                path.unlink()
                return False
            record["phase"] = "vector"
            self._save_repair(path, record)
            self._persist_vector(stored, vector, deleting=deleting, memory_id=memory_id)
            record["phase"] = "graph"
            self._save_repair(path, record)
            success = self._persist_graph(
                stored, entities, relations, previous, write_token=record["token"],
                deleting=deleting, memory_id=memory_id,
            )
            if success:
                path.unlink()
                return True
            return self._resolve_pending_write(path, record)
        except Exception:
            try:
                committed = self._resolve_pending_write(path, record)
            except Exception as recovery_error:
                raise RuntimeError(f"Memory write needs recovery; journal saved at {path}") from recovery_error
            if committed:
                return True
            raise

    def _persist_vector(
        self, stored, vector, *, deleting=False, memory_id: Optional[str] = None,
    ) -> None:
        """写入已生成的向量，或删除指定记忆的向量。"""
        if deleting:
            self.vector_store.remove(stored.id if stored is not None else memory_id)
        elif not self.vector_store._upsert_precomputed(stored, vector):
            raise RuntimeError("Vector write did not complete")

    def _persist_graph(
        self, stored, entities, relations, previous, *, write_token: str,
        deleting=False, memory_id: Optional[str] = None,
    ) -> bool:
        """替换记忆正文及其实体关系，或删除该记忆的图数据。"""
        if deleting:
            return self.graph_store.remove_memory_graph(stored.id if stored is not None else memory_id)
        return self.graph_store.replace_memory_graph(
            stored, entities, relations, previous=previous, write_token=write_token,
        )

    def _resolve_pending_write(self, path: Path, record: dict[str, Any]) -> bool:
        """检查日志并尝试补偿或确认已提交的写入。"""
        if record["phase"] == "preparing":
            path.unlink(missing_ok=True)
            return False
        committed = False
        if record["phase"] == "graph":
            if record["deleting"]:
                committed = self.graph_store._get_by_id(record["memory_id"]) is None
            else:
                committed = self.graph_store._get_write_token(record["memory_id"]) == record["token"]
        if not committed:
            self.vector_store._restore_snapshot(record["memory_id"], record["snapshot"])
        path.unlink(missing_ok=True)
        return committed

    def recover_pending_writes(self) -> int:
        """确认没有其他实例正在写入后，重试日志中的补偿或确认已提交的写入。"""
        with self._write_guard():
            directory = self._repair_path("unused").parent
            count = 0
            for path in sorted(directory.glob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if path != self._repair_path(record["memory_id"]):
                    raise ValueError("Recovery journal ID does not match its filename")
                self._resolve_pending_write(path, record)
                count += 1
            return count

    def rebuild_graph_sources(self, memory_ids: Optional[List[str]] = None) -> int:
        """显式从现存正文重建来源，保留无法归属的历史关系，不自动运行。"""
        items = self.graph_store.get_all() if memory_ids is None else self.graph_store.get_many(memory_ids)
        return sum(self.update(item) for item in items)

    def _remove_low_importance_memory(self, threshold: float = 0.1) -> int:
        """
        移除重要性低于阈值的记忆项
        """
        return sum(self.remove(item.id) for item in self.get_all() if item.importance < threshold)

    def contains(self, memory_id: str) -> bool:
        return self.graph_store.contains(memory_id)

    def get_all(self) -> List[MemoryItem]:
        """
        获取所有记忆项
        """
        return self.graph_store.get_all()
    
    def _expire_old_memories(self) -> int:
        """
        清理过期的记忆项
        """
        age_limit = self._ttl_delta(self.ttl_days)
        if age_limit is None:
            return 0
        expired_datetime = datetime.now(timezone.utc) - age_limit
        expired_memories = self.graph_store._get_before_time(expired_datetime)
        return sum(self.remove(memory.id) for memory in expired_memories)
        
    def retrieve(self, query: str, limit: int = 5, threshold: float = 0.7, **kwargs) -> List[MemoryRetrievalHit]:
        """按两路召回的综合评分返回记忆和评分"""
        return [{"memory": hit["memory"], "score": hit["score"]}
                for hit in self.retrieve_with_scores(query, limit, **kwargs)
                if hit["score"] >= threshold]

    def retrieve_with_scores(
        self, query: str, limit: int = 5, *,
        candidate_limit: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        allowed_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
        final_score_threshold: Optional[float] = None,
        min_importance: float = 0,
        recency_half_life_days: float = 90,
        relationship_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """向量和图独立召回、按 ID 合并，再用四因素评分排序。

        默认 score = 0.55*vector + 0.25*graph + 0.10*importance + 0.10*recency。
        vector 将余弦分数从 [-1, 1] 映射到 [0, 1]；图分数已经是 [0, 1]。
        未命中的检索通道计零，不按单个候选重新分配权重。
        score_threshold 只过滤原始向量分数；final_score_threshold 过滤最终分数。
        分数单独返回，不修改 MemoryItem.importance 或原始 metadata。
        两种数据库失败会直接抛出异常，不静默伪装为无结果。
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0 or not query.strip() or allowed_ids == []:
            return []
        candidate_limit = max(20, limit * 4) if candidate_limit is None else candidate_limit
        if type(candidate_limit) is not int or candidate_limit < limit:
            raise ValueError("candidate_limit must be an integer greater than or equal to limit")
        scoring_weights = {"vector": 0.55, "graph": 0.25, "importance": 0.10, "recency": 0.10}
        if weights is not None:
            if set(weights) - set(scoring_weights):
                raise ValueError("Unknown scoring weight")
            scoring_weights.update(weights)
        scoring_weights = {key: float(value) for key, value in scoring_weights.items()}
        if any(not math.isfinite(value) or value < 0 for value in scoring_weights.values()):
            raise ValueError("Scoring weights must be finite and non-negative")
        total_weight = sum(scoring_weights.values())
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("At least one scoring weight must be positive")
        scoring_weights = {key: value / total_weight for key, value in scoring_weights.items()}
        if not math.isfinite(float(recency_half_life_days)) or recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be finite and positive")
        if not math.isfinite(float(min_importance)):
            raise ValueError("min_importance must be finite")
        for value, lower, name in [(score_threshold, -1, "score_threshold"),
                                   (final_score_threshold, 0, "final_score_threshold")]:
            if value is not None and (not math.isfinite(float(value)) or not lower <= value <= 1):
                raise ValueError(f"{name} is outside its valid range")
        # 只把当前通道支持的参数传给它，避免把融合权重等误传给数据库。
        # 先由正文存储确定所属会话，再限制两路召回，避免别的会话占用候选名额。
        if session_id is not None:
            session_ids = set(self.graph_store.get_session_ids(session_id, min_importance))
            if allowed_ids is not None:
                session_ids.intersection_update(allowed_ids)
            allowed_ids = sorted(session_ids)
            if not allowed_ids:
                return []
        vector_hits = self.vector_store.retrieve(
            query, limit=candidate_limit, allowed_ids=allowed_ids, score_threshold=score_threshold,
        )
        entity_names, keywords = self._graph_query_terms(query)
        graph_hits = self.graph_store.retrieve(
            query, limit=candidate_limit, entity_names=entity_names, keywords=keywords,
            allowed_ids=allowed_ids, min_importance=min_importance,
            relationship_types=relationship_types,
        )
        vector_scores, memories = {}, {}
        for hit in vector_hits:
            raw_score = self._finite_score(hit.get("score"))
            memory_id = hit.get("memory_id")
            if not memory_id or raw_score is None or not -1.000001 <= raw_score <= 1.000001:
                continue
            if score_threshold is not None and raw_score < score_threshold:
                continue
            vector_scores[memory_id] = max(raw_score, vector_scores.get(memory_id, -math.inf))
            item = hit.get("memory")
            if item is not None and item.id == memory_id:
                memories[memory_id] = item
        graph_scores, graph_details = {}, {}
        for hit in graph_hits:
            raw_score = self._finite_score(hit.get("score"))
            if raw_score is None or not 0 <= raw_score <= 1:
                continue
            item = hit["memory"]
            if raw_score >= graph_scores.get(item.id, -math.inf):
                memories[item.id] = item
                graph_scores[item.id] = raw_score
                graph_details[item.id] = {key: hit.get(key, 0.0) for key in ("lexical_score", "path_score")}
        missing_ids = [memory_id for memory_id in vector_scores if memory_id not in memories]
        if missing_ids:
            for item in self.graph_store.get_many(missing_ids):
                memories[item.id] = item
        allowed = set(allowed_ids) if allowed_ids is not None else None
        now = datetime.now(timezone.utc)
        results = []
        for memory_id, item in memories.items():
            if session_id is not None and item.metadata.get("session_id") != session_id:
                continue
            if (memory_id not in vector_scores and memory_id not in graph_scores
                    or allowed is not None and memory_id not in allowed):
                continue
            importance = self._finite_score(item.importance)
            if importance is None or importance < min_importance:
                continue
            created_at = item.created_at.astimezone(timezone.utc)
            age_days = max(0.0, (now - created_at).total_seconds() / 86400)
            factors = {
                "vector": max(0.0, min(1.0, (vector_scores[memory_id] + 1) / 2))
                          if memory_id in vector_scores else 0.0,
                "graph": graph_scores.get(memory_id, 0.0),
                "importance": max(0.0, min(1.0, importance)),
                "recency": math.exp(-math.log(2) * age_days / recency_half_life_days),
            }
            score = sum(scoring_weights[key] * value for key, value in factors.items())
            if final_score_threshold is not None and score < final_score_threshold:
                continue
            results.append({
                "memory": item, "memory_id": memory_id, "score": score,
                "factors": factors, "weights": scoring_weights.copy(),
                "vector_score": vector_scores.get(memory_id),
                "graph_score": graph_scores.get(memory_id),
                "graph_details": graph_details.get(memory_id, {}),
                "sources": [source for source, scores in [("vector", vector_scores), ("graph", graph_scores)]
                            if memory_id in scores],
            })
        results.sort(key=lambda hit: (-hit["score"], hit["memory_id"]))
        return results[:limit]

    @staticmethod
    def _finite_score(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    def _graph_query_terms(self, query: str) -> tuple[List[str], List[str]]:
        """使用已有 spaCy 管线解析实体及内容词，不调用 GLiNER2。"""
        doc = self.nlp(query)
        names = list(dict.fromkeys(span.text.strip() for span in doc.ents if span.text.strip()))
        keywords = list(dict.fromkeys(names + [
            token.text.strip() for token in doc
            if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop and token.text.strip()
        ]))
        return names, keywords or [query.strip()]

    def _add_entity_to_graph(self, entity: Entity, memory_item: MemoryItem):
        """
        将实体添加到知识图谱中
        """
        self.graph_store.add_entity(entity, memory_item)

    def _add_relation_to_graph(self, relation: Relation, memory_item: MemoryItem):
        """
        将关系添加到知识图谱中
        """
        self.graph_store.add_relation(relation)

    def _extract_entities(self, text: str) -> List[Entity]:
        """
        将 NLP 识别结果转换为实体，按首次出现的顺序返回。
        同类型、同名称的提及合并为一个实体，并生成跨调用稳定的 ID。
        properties["mentions"] 保存本次文本中的字符区间，结束位置不包含在内。
        此方法只提取数据，不修改 self.entities，也不写入数据库。
        名称和类型用于初步去重，不进行别名识别或同名实体消歧。
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return []

        doc = self.nlp(text)
        entities: Dict[tuple[str, str], Entity] = {}
        for span in doc.ents:
            name = span.text.strip()
            if not name:
                continue
            entity_type = span.label_ or "Entity"
            key = (entity_type, name)
            if key not in entities:
                # JSON 编码避免名称中的分隔符导致 ID 冲突；不使用随机 hash()。
                identity = json.dumps(key, ensure_ascii=False)
                entities[key] = Entity(
                    id=f"entity_{uuid5(NAMESPACE_URL, identity).hex}",
                    name=name,
                    entity_type=entity_type,
                    properties={"mentions": []},
                )
            entities[key].properties["mentions"].append({
                "start_char": span.start_char,
                "end_char": span.end_char,
            })

        return list(entities.values())

    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """使用 GLiNER2 抽取关系，映射到已有实体，不修改缓存或数据库。

        关系类型由语义描述定义，无需维护中文谓词同义词列表。
        模型对否定、计划及含糊表达可能误判，结果应视为候选关系。
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip() or len({entity.id for entity in entities}) < 2:
            return []
        if getattr(self, "relation_extractor", None) is None:
            self.relation_extractor = GLiNER2RelationExtractor()
        return self.relation_extractor.extract(text, entities)

    def _init_nlp(self):
        """
        初始化 spaCy NLP 模型，用于实体识别
        """
        import spacy

        return spacy.load("zh_core_web_sm")

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
