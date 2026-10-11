import json
import math
import os
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from entity import Entity
from memory.memory_item import MemoryItem
from relation import Relation


# 固定读取项目根目录，支持单独导入 graph_store 时加载配置。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Neo4jGraphStore:
    """使用 Neo4j 保存记忆、实体及实体之间的有向关系。

    默认使用项目 .env 的 NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD 和
    NEO4J_DATABASE，也可通过构造参数覆盖。
    加密连接未指定 CA 时为当前进程使用 certifi；保留显式 CA 配置。
    构造时不连接服务器，首次写入时创建 ID 唯一约束。
    metadata 和 properties 使用 JSON 保存，支持嵌套字典。
    实体和关系接口使用 Entity、Relation 对象。
    实体抽取及向量检索由上层实现，本类不自动调用 LLM。
    """

    def __init__(
            self, 
            uri: Optional[str] = None, 
            username: Optional[str] = None,
            password: Optional[str] = None, 
            database: Optional[str] = None,
            *, query_timeout: float = 30
        ):
        uri = uri if uri is not None else os.getenv("NEO4J_URI")
        username = username if username is not None else os.getenv("NEO4J_USERNAME", "neo4j")
        password = password if password is not None else os.getenv("NEO4J_PASSWORD")
        self.database = database if database is not None else os.getenv("NEO4J_DATABASE", "neo4j")
        if not uri or not username or not password or not self.database:
            raise ValueError(
                "Configure NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD and NEO4J_DATABASE"
            )
        if query_timeout <= 0:
            raise ValueError("query_timeout must be positive")
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install neo4j in the project conda environment") from exc
        # Conda/Windows 的默认 CA 可能不完整，TLS 失败会被包装成路由错误。
        # +s 保持完整证书和主机名校验；仅补充当前进程的 CA 文件，不修改系统配置。
        if uri.startswith(("neo4j+s://", "bolt+s://")) and not (
            os.getenv("SSL_CERT_FILE") or os.getenv("SSL_CERT_DIR")
        ):
            try:
                import certifi
            except ImportError as exc:
                raise RuntimeError("Install certifi in the project conda environment") from exc
            os.environ["SSL_CERT_FILE"] = certifi.where()
        self._driver = GraphDatabase.driver(
            uri, auth=(username, password), connection_timeout=10,
            max_transaction_retry_time=15,
        )
        self._query_timeout = query_timeout
        self._schema_ready = False
        self._closed = False

    def query(
            self, 
            cypher: str, 
            parameters: Optional[dict[str, Any]] = None,
            *, read_only: bool = False
        ) -> list[dict[str, Any]]:
        """执行参数化 Cypher 并返回字典列表；调用方负责传入可信的查询语句。

        read_only 控制驱动的读事务及路由，不是用户输入的安全过滤器。
        """
        if self._closed:
            raise RuntimeError("Neo4jGraphStore is closed")
        from neo4j import Query

        result = self._driver.execute_query(
            Query(cypher, timeout=self._query_timeout),
            parameters_=parameters or {}, database_=self.database,
            routing_="r" if read_only else "w",
        )
        return [record.data() for record in result.records]

    def verify_connectivity(self) -> None:
        """验证认证、连接及配置的数据库是否可访问；失败时抛出驱动异常。"""
        self.query("RETURN 1 AS ok", read_only=True)

    def _get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        rows = self.query(
            "MATCH (m:SemanticMemory {id: $id}) RETURN properties(m) AS memory",
            {"id": memory_id}, read_only=True,
        )
        return self.to_item(rows[0]["memory"]) if rows else None

    def _get_write_token(self, memory_id: str) -> Optional[str]:
        rows = self.query(
            "MATCH (m:SemanticMemory {id: $id}) RETURN m.write_token AS token",
            {"id": memory_id}, read_only=True,
        )
        return rows[0]["token"] if rows else None

    def replace_memory_graph(
        self, memory_item: MemoryItem, entities: list[Entity], relations: list[Relation],
        *, previous: Optional[MemoryItem], write_token: str,
    ) -> bool:
        """在单个事务中新增或替换记忆及其派生图。

        previous 为 None 时新增，否则校验旧记忆后替换；空实体和关系列表也会保存正文。
        共享实体和其他来源证据保持不变，write_token 用于上层确认提交及失败恢复。
        """
        if self._closed:
            raise RuntimeError("Neo4jGraphStore is closed")
        self.require_id(memory_item.id)
        properties = self.memory_properties(memory_item)
        entity_ids = {entity.id for entity in entities}
        for entity in entities:
            self.require_id(entity.id)
            json.dumps(entity.properties)
        for relation in relations:
            self.require_id(relation.source_id)
            self.require_id(relation.target_id)
            self.relationship_type(relation.relationship_type)
            if relation.relationship_type == "MENTIONS":
                raise ValueError("MENTIONS is reserved for memory-to-entity links")
            if relation.source_id not in entity_ids or relation.target_id not in entity_ids:
                raise ValueError("Relation endpoints must belong to the memory's extracted entities")
            if not isinstance(relation.properties.get("evidence", []), list):
                raise ValueError("Relation evidence must be a list")
            json.dumps(relation.properties)
        self.ensure_schema()

        from neo4j import unit_of_work

        @unit_of_work(timeout=self._query_timeout)
        def replace_graph(tx):
            rows = self._tx_query(tx,
                "MATCH (m:SemanticMemory {id: $id}) SET m._write_lock = true "
                "RETURN properties(m) AS memory", {"id": memory_item.id})
            if previous is None:
                if rows:
                    raise RuntimeError("Memory changed during graph preparation; retry the operation")
                rows = self._tx_query(tx,
                    "MERGE (m:SemanticMemory {id: $id}) SET m._write_lock = true "
                    "RETURN properties(m) AS memory", {"id": memory_item.id})
                if "content" in rows[0]["memory"]:
                    raise RuntimeError("Memory was created concurrently; retry the operation")
            elif not rows:
                return False
            else:
                current = self.to_item(rows[0]["memory"])
                if any(getattr(current, key) != getattr(previous, key)
                       for key in ("id", "content", "type", "importance", "created_at", "metadata")):
                    raise RuntimeError("Memory changed during graph preparation; retry the operation")

            self._quarantine_legacy_relations(tx, memory_item.id)
            self._remove_relation_source(tx, memory_item.id)
            self._tx_query(tx,
                "MATCH (m:SemanticMemory {id: $id})-[r:MENTIONS]->() DELETE r",
                {"id": memory_item.id})
            self._tx_query(tx,
                "MATCH (m:SemanticMemory {id: $id}) SET m += $properties, "
                "m.write_token = $token REMOVE m._write_lock",
                {"id": memory_item.id, "properties": properties, "token": write_token})
            for entity in sorted(entities, key=lambda entity: entity.id):
                shared_properties = {key: value for key, value in entity.properties.items() if key != "mentions"}
                self._tx_query(tx,
                    "MERGE (e:SemanticEntity {id: $entity_id}) "
                    "SET e.name = $name, e.entity_type = $entity_type, e.properties_json = $properties "
                    "WITH e MATCH (m:SemanticMemory {id: $memory_id}) "
                    "MERGE (m)-[r:MENTIONS]->(e) SET r.properties_json = $mentions",
                    {"entity_id": entity.id, "name": entity.name, "entity_type": entity.entity_type,
                     "properties": json.dumps(shared_properties, ensure_ascii=False),
                     "memory_id": memory_item.id,
                     "mentions": json.dumps({"mentions": entity.properties.get("mentions", [])}, ensure_ascii=False)})
            for relation in sorted(relations, key=lambda relation: (
                relation.source_id, relation.target_id, relation.relationship_type
            )):
                self._add_relation_source(tx, relation, memory_item.id)
            return True

        with self._driver.session(database=self.database) as session:
            return session.execute_write(replace_graph)

    def remove_memory_graph(self, memory_id: str) -> bool:
        """删除记忆及其来源证据；共享实体和其他记忆的关系仍然保留。"""
        if self._closed:
            raise RuntimeError("Neo4jGraphStore is closed")

        from neo4j import unit_of_work

        @unit_of_work(timeout=self._query_timeout)
        def remove_graph(tx):
            rows = self._tx_query(tx,
                "MATCH (m:SemanticMemory {id: $id}) SET m._write_lock = true RETURN m.id AS id",
                {"id": memory_id})
            if not rows:
                return False
            self._quarantine_legacy_relations(tx, memory_id)
            self._remove_relation_source(tx, memory_id)
            self._tx_query(tx, "MATCH (m:SemanticMemory {id: $id}) DETACH DELETE m", {"id": memory_id})
            return True

        with self._driver.session(database=self.database) as session:
            return session.execute_write(remove_graph)

    def _tx_query(self, tx, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        return [record.data() for record in tx.run(cypher, parameters)]

    def _quarantine_legacy_relations(self, tx, memory_id: str) -> None:
        # 缺少来源的旧关系不能推断归属，也不能自动删除；失效后退出路径推理。
        self._tx_query(tx,
            "MATCH (m:SemanticMemory {id: $id})-[:MENTIONS]->(e:SemanticEntity)"
            "-[r]-(other:SemanticEntity) WHERE coalesce(r.provenance_version, 0) = 0 "
            "SET r.legacy_unattributed = true", {"id": memory_id})

    def _remove_relation_source(self, tx, memory_id: str) -> None:
        rows = self._tx_query(tx,
            "MATCH (:SemanticEntity)-[r]->(:SemanticEntity) "
            "WHERE $id IN coalesce(r.source_memory_ids, []) "
            "WITH r ORDER BY elementId(r) SET r._write_lock = true "
            "RETURN elementId(r) AS relation_id, r.properties_json AS properties, "
            "coalesce(r.legacy_unattributed, false) AS legacy", {"id": memory_id})
        for row in rows:
            properties = json.loads(row["properties"] or "{}")
            properties.get("evidence_by_memory", {}).pop(memory_id, None)
            properties.get("source_properties_by_memory", {}).pop(memory_id, None)
            sources = sorted(properties.get("evidence_by_memory", {}))
            if not sources and not row["legacy"]:
                self._tx_query(tx,
                    "MATCH ()-[r]->() WHERE elementId(r) = $id DELETE r", {"id": row["relation_id"]})
            else:
                self._tx_query(tx,
                    "MATCH ()-[r]->() WHERE elementId(r) = $id "
                    "SET r.properties_json = $properties, r.source_memory_ids = $sources REMOVE r._write_lock",
                    {"id": row["relation_id"], "properties": json.dumps(properties, ensure_ascii=False),
                     "sources": sources})

    @staticmethod
    def _merge_relation_evidence(
        existing: dict[str, Any], relation: Relation, memory_id: str,
    ) -> dict[str, Any]:
        properties = deepcopy(existing)
        properties.setdefault("evidence_by_memory", {})[memory_id] = deepcopy(relation.properties.get("evidence", []))
        properties.setdefault("source_properties_by_memory", {})[memory_id] = {
            key: deepcopy(value) for key, value in relation.properties.items() if key != "evidence"
        }
        return properties

    def _add_relation_source(self, tx, relation: Relation, memory_id: str) -> None:
        relationship_type = self.relationship_type(relation.relationship_type)
        rows = self._tx_query(tx,
            "MATCH (s:SemanticEntity {id: $source}), (t:SemanticEntity {id: $target}) "
            f"MERGE (s)-[r:{relationship_type}]->(t) "
            "ON CREATE SET r.properties_json = '{}', r.source_memory_ids = [] "
            "SET r._write_lock = true RETURN elementId(r) AS relation_id, "
            "r.properties_json AS properties, r.source_memory_ids AS sources, "
            "coalesce(r.legacy_unattributed, false) AS legacy",
            {"source": relation.source_id, "target": relation.target_id})
        if not rows:
            raise RuntimeError("Relation endpoints disappeared during graph update")
        row = rows[0]
        properties = self._merge_relation_evidence(json.loads(row["properties"] or "{}"), relation, memory_id)
        self._tx_query(tx,
            "MATCH ()-[r]->() WHERE elementId(r) = $id "
            "SET r.properties_json = $properties, r.source_memory_ids = $sources, "
            "r.provenance_version = 1, r.legacy_unattributed = $legacy REMOVE r._write_lock",
            {"id": row["relation_id"], "properties": json.dumps(properties, ensure_ascii=False),
             "sources": sorted(properties["evidence_by_memory"]),
             "legacy": row["legacy"] or row["sources"] is None})

    def _get_before_time(self, end_time: datetime, start_time: Optional[datetime] = None) -> list[MemoryItem]:
        """按实际时间降序读取正文，时间范围为左闭右开，兼容旧记录的时区。"""
        end_time_utc = self.to_utc(end_time, "end_time")
        start_time_utc = self.to_utc(start_time, "start_time") if start_time is not None else None
        cypher = (
            "MATCH (m:SemanticMemory) "
            "WHERE m.created_at < $end_time "
        )
        if start_time_utc is not None:
            cypher += "AND m.created_at >= $start_time "
        cypher += "RETURN properties(m) AS memory ORDER BY m.created_at DESC, m.id"
        parameters = {"end_time": end_time_utc.isoformat()}
        if start_time_utc is not None:
            parameters["start_time"] = start_time_utc.isoformat()
        rows = self.query(cypher, parameters, read_only=True)
        return [self.to_item(row["memory"]) for row in rows]

    def get_all(self) -> list[MemoryItem]:
        rows = self.query(
            "MATCH (m:SemanticMemory) RETURN properties(m) AS memory ORDER BY m.id",
            read_only=True,
        )
        return [self.to_item(row["memory"]) for row in rows]

    def get_many(self, memory_ids: list[str]) -> list[MemoryItem]:
        """批量读取原始记忆，忽略已经删除的 ID。"""
        if not memory_ids:
            return []
        rows = self.query(
            "MATCH (m:SemanticMemory) WHERE m.id IN $memory_ids "
            "RETURN properties(m) AS memory ORDER BY m.id",
            {"memory_ids": list(dict.fromkeys(memory_ids))}, read_only=True,
        )
        return [self.to_item(row["memory"]) for row in rows]

    def get_session_ids(self, session_id: str, min_importance: float = 0) -> list[str]:
        """从原始 metadata 精确筛选会话；兼容旧记录的 JSON 空格和缩进。"""
        rows = self.query(
            "MATCH (m:SemanticMemory) WHERE m.importance >= $min_importance "
            "RETURN m.id AS id, m.metadata_json AS metadata_json",
            {"min_importance": float(min_importance)}, read_only=True,
        )
        return [row["id"] for row in rows
                if json.loads(row["metadata_json"] or "{}").get("session_id") == session_id]

    def contains(self, memory_id: str) -> bool:
        return bool(self.query(
            "MATCH (m:SemanticMemory {id: $id}) RETURN m.id AS id LIMIT 1",
            {"id": memory_id}, read_only=True,
        ))

    def update(self, memory_item: MemoryItem) -> bool:
        """更新正文、重要性和 metadata；保留类型和创建时间，不存在则返回 False。"""
        rows = self.query(
            "MATCH (m:SemanticMemory {id: $id}) "
            "SET m.content = $content, m.importance = $importance, "
            "m.metadata_json = $metadata RETURN m.id AS id",
            {"id": memory_item.id, "content": memory_item.content,
             "importance": memory_item.importance,
             "metadata": json.dumps(memory_item.metadata, ensure_ascii=False)},
        )
        return bool(rows)

    # def retrieve(self, query: str = "", limit: int = 5,
    #              *, min_importance: float = 0) -> list[MemoryItem]:
    #     """按正文子串筛选，再按重要性排序；这是文本检索，不是向量相似度检索。"""
    #     if limit <= 0:
    #         return []
    #     rows = self.query(
    #         "MATCH (m:SemanticMemory) "
    #         "WHERE toLower(m.content) CONTAINS toLower($query) "
    #         "AND m.importance >= $min_importance "
    #         "RETURN properties(m) AS memory "
    #         "ORDER BY m.importance DESC, m.created_at DESC, m.id LIMIT $limit",
    #         {"query": query, "min_importance": min_importance, "limit": limit},
    #         read_only=True,
    #     )
    #     return [self._to_item(row["memory"]) for row in rows]

    def retrieve(
        self, query: str, limit: int = 20, *,
        entity_names: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        allowed_ids: Optional[list[str]] = None,
        min_importance: float = 0,
        relationship_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """召回原文词项及实体两跳邻域中的记忆，返回 0 到 1 的规则相关性分数。

        直接提及查询实体为 1，一跳邻居为 1/2，两跳邻居为 1/3；
        原文词项匹配按查询词项覆盖率评分。两种分数取最大值。
        路径只经过实体，排除 MENTIONS；可限制知识关系类型，按双向邻域召回。
        这是规则评分，不是全文索引分数或事实置信度，不需要新建全文索引。
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if limit <= 0 or not query.strip() or allowed_ids == []:
            return []
        if not math.isfinite(float(min_importance)):
            raise ValueError("min_importance must be finite")
        names = sorted({name.strip().lower() for name in entity_names or [] if name.strip()})
        terms = sorted({term.strip().lower() for term in (keywords or [query]) if term.strip()})
        if relationship_types is not None:
            for relationship_type in relationship_types:
                self.relationship_type(relationship_type)
        rows = self.query(
            "CALL () { "
            "  MATCH (m:SemanticMemory) "
            "  WHERE any(term IN $terms WHERE toLower(m.content) CONTAINS term) "
            "  RETURN m, toFloat(size([term IN $terms WHERE toLower(m.content) CONTAINS term])) "
            "    / size($terms) AS lexical_score, 0.0 AS path_score "
            "  UNION ALL "
            "  MATCH (seed:SemanticEntity) WHERE toLower(seed.name) IN $entity_names "
            "  MATCH (m:SemanticMemory)-[:MENTIONS]->(seed) "
            "  RETURN m, 0.0 AS lexical_score, 1.0 AS path_score "
            "  UNION ALL "
            "  MATCH (seed:SemanticEntity) WHERE toLower(seed.name) IN $entity_names "
            "  MATCH p=(seed)-[*1..2]-(neighbor:SemanticEntity) "
            "  WHERE all(n IN nodes(p) WHERE n:SemanticEntity) "
            "    AND all(r IN relationships(p) WHERE type(r) <> 'MENTIONS' "
            "      AND (NOT coalesce(r.legacy_unattributed, false) "
            "        OR size(coalesce(r.source_memory_ids, [])) > 0) "
            "      AND ($relationship_types IS NULL OR type(r) IN $relationship_types)) "
            "  MATCH (m:SemanticMemory)-[:MENTIONS]->(neighbor) "
            "  RETURN m, 0.0 AS lexical_score, 1.0 / (1 + length(p)) AS path_score "
            "} "
            "WITH m, max(lexical_score) AS lexical_score, max(path_score) AS path_score "
            "WHERE m.importance >= $min_importance "
            "  AND ($allowed_ids IS NULL OR m.id IN $allowed_ids) "
            "WITH m, lexical_score, path_score, "
            "  CASE WHEN lexical_score > path_score THEN lexical_score ELSE path_score END AS score "
            "RETURN properties(m) AS memory, score, lexical_score, path_score "
            "ORDER BY score DESC, m.id LIMIT $limit",
            {"terms": terms, "entity_names": names, "allowed_ids": allowed_ids,
             "min_importance": float(min_importance), "relationship_types": relationship_types,
             "limit": limit}, read_only=True,
        )
        return [{"memory": self.to_item(row["memory"]), "score": float(row["score"]),
                 "lexical_score": float(row["lexical_score"]), "path_score": float(row["path_score"])}
                for row in rows]

    def remove(self, memory_id: str) -> bool:
        """删除记忆及其关联边，不删除其他实体节点。"""
        return bool(self.query(
            "MATCH (m:SemanticMemory {id: $id}) DETACH DELETE m RETURN $id AS id",
            {"id": memory_id},
        ))

    def _remove_low_importance_memory(self, threshold: float = 0.1) -> int:
        """删除重要性低于阈值的记忆及其关联边。"""
        rows = self.query(
            "MATCH (m:SemanticMemory) "
            "WHERE m.importance < $threshold "
            "DETACH DELETE m "
            "RETURN count(*) AS removed",
            {"threshold": float(threshold)},
        )
        return int(rows[0]["removed"])

    def add_entity(self, entity: Entity, memory_item: Optional[MemoryItem] = None) -> str:
        """按 ID 保存实体；提供来源记忆时同时保存原文并建立 MENTIONS 关联。"""
        self.require_id(entity.id)
        properties_json = json.dumps(entity.properties, ensure_ascii=False)
        cypher = (
            "MERGE (e:SemanticEntity {id: $id}) "
            "SET e.name = $name, e.entity_type = $entity_type, e.properties_json = $properties "
        )
        parameters = {"id": entity.id, "name": entity.name, "entity_type": entity.entity_type,
                      "properties": properties_json}
        if memory_item is not None:
            self.require_id(memory_item.id)
            parameters.update({"memory_id": memory_item.id,
                               "memory_properties": self.memory_properties(memory_item)})
            cypher += (
                "MERGE (m:SemanticMemory {id: $memory_id}) SET m += $memory_properties "
                "MERGE (m)-[:MENTIONS]->(e)"
            )
        self.ensure_schema()
        self.query(cypher, parameters)
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """按 ID 读取实体并还原为 Entity，不存在则返回 None。"""
        rows = self.query(
            "MATCH (e:SemanticEntity {id: $id}) RETURN properties(e) AS entity",
            {"id": entity_id}, read_only=True,
        )
        if not rows:
            return None
        data = rows[0]["entity"]
        return Entity(
            id=data["id"], name=data["name"], entity_type=data["entity_type"],
            properties=json.loads(data.get("properties_json") or "{}"),
        )

    def remove_entity(self, entity_id: str) -> bool:
        """删除实体及其关联边，不删除相邻节点。"""
        return bool(self.query(
            "MATCH (e:SemanticEntity {id: $id}) DETACH DELETE e RETURN $id AS id",
            {"id": entity_id},
        ))

    def add_relation(self, relation: Relation) -> bool:
        """在已有实体间新增或覆盖关系；任一实体不存在则返回 False。

        同一对实体、同一方向及类型仅表示一条关系，properties 整体替换。
        此无来源接口不能覆盖已由记忆来源管理的关系。
        """
        self.require_id(relation.source_id)
        self.require_id(relation.target_id)
        relationship_type = self.relationship_type(relation.relationship_type)
        return bool(self.query(
            "MATCH (s:SemanticEntity {id: $source}), (t:SemanticEntity {id: $target}) "
            f"MERGE (s)-[r:{relationship_type}]->(t) "
            "WITH r WHERE coalesce(r.provenance_version, 0) = 0 "
            "SET r.properties_json = $properties RETURN type(r) AS type",
            {"source": relation.source_id, "target": relation.target_id,
             "properties": json.dumps(relation.properties, ensure_ascii=False)},
        ))

    def get_relationships(self, entity_id: str, limit: int = 100) -> list[Relation]:
        """读取实体的入边和出边，还原为 Relation 并保留关系方向。"""
        if limit <= 0:
            return []
        rows = self.query(
            "MATCH (s:SemanticEntity)-[r]->(t:SemanticEntity) "
            "WHERE s.id = $id OR t.id = $id "
            "RETURN s.id AS source_id, t.id AS target_id, type(r) AS type, "
            "r.properties_json AS properties_json "
            "ORDER BY source_id, target_id, type LIMIT $limit",
            {"id": entity_id, "limit": limit}, read_only=True,
        )
        return [
            Relation(
                source_id=row["source_id"], target_id=row["target_id"],
                relationship_type=row["type"],
                properties=json.loads(row.get("properties_json") or "{}"),
            )
            for row in rows
        ]

    def remove_relationship(self, relation: Relation) -> bool:
        """按起点、终点和类型删除关系，properties 不参与匹配。"""
        self.require_id(relation.source_id)
        self.require_id(relation.target_id)
        relationship_type = self.relationship_type(relation.relationship_type)
        return bool(self.query(
            "MATCH (s:SemanticEntity {id: $source})"
            f"-[r:{relationship_type}]->(t:SemanticEntity {{id: $target}}) "
            "DELETE r RETURN $source AS id",
            {"source": relation.source_id, "target": relation.target_id},
        ))

    def close(self) -> None:
        if not self._closed:
            self._driver.close()
            self._closed = True

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def ensure_schema(self) -> None:
        if not self._schema_ready:
            for label, name in (("SemanticMemory", "semantic_memory_id_unique"),
                                ("SemanticEntity", "semantic_entity_id_unique")):
                self.query(
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )
            self._schema_ready = True

    @staticmethod
    def require_id(value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("IDs must be non-empty strings")

    @staticmethod
    def relationship_type(value: str) -> str:
        # 关系类型不能在兼容旧版本的 Cypher 中直接绑定参数，严格限制标识符。
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("Relationship type must be an ASCII identifier")
        return value

    @staticmethod
    def to_item(data: dict[str, Any]) -> MemoryItem:
        return MemoryItem(
            id=data["id"], content=data["content"], type=data["type"],
            importance=data["importance"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=json.loads(data["metadata_json"]),
        )

    @staticmethod
    def memory_properties(memory_item: MemoryItem) -> dict[str, Any]:
        return {
            "id": memory_item.id, "content": memory_item.content,
            "type": memory_item.type, "importance": memory_item.importance,
            "created_at": memory_item.created_at.isoformat(),
            "metadata_json": json.dumps(memory_item.metadata, ensure_ascii=False),
        }

    @staticmethod
    def to_utc(value: Any, field_name: str = "created_at") -> datetime:
        """保留原始时间戳；比较时将无时区时间按本地时间解释，再统一到 UTC。"""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be a valid ISO 8601 datetime string") from exc
        elif not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime or an ISO 8601 datetime string")
        return value.astimezone(timezone.utc)

    # def expire_memories(self, threshold: float = 30.0) -> int:
    #     """按创建时间删除超过 threshold 天的记忆及其关联边，返回删除数量。

    #     threshold 为非负天数，默认 30 天；恰好达到边界的记忆保留。
    #     带时区时间统一到 UTC，无时区时间按本地时间解释。
    #     缺失或无效的创建时间保留；相邻实体节点不删除。
    #     """
    #     threshold = float(threshold)
    #     if not math.isfinite(threshold) or threshold < 0:
    #         raise ValueError("threshold must be a finite, non-negative number of days")
    #     try:
    #         cutoff = datetime.now(timezone.utc) - timedelta(days=threshold)
    #     except OverflowError as exc:
    #         raise ValueError("threshold exceeds the supported datetime range") from exc

    #     rows = self.query(
    #         "MATCH (m:SemanticMemory) "
    #         "RETURN m.id AS id, m.created_at AS created_at",
    #         read_only=True,
    #     )
    #     expired_memories = []
    #     for row in rows:
    #         stored_time = row.get("created_at")
    #         if not isinstance(stored_time, str):
    #             continue
    #         try:
    #             created_at = datetime.fromisoformat(stored_time).astimezone(timezone.utc)
    #         except (ValueError, OverflowError):
    #             continue
    #         if created_at < cutoff:
    #             expired_memories.append({"id": row["id"], "created_at": stored_time})
    #     if not expired_memories:
    #         return 0

    #     # 删除前核对创建时间，避免清理期间同 ID 的新记录被误删。
    #     rows = self.query(
    #         "UNWIND $expired_memories AS expired "
    #         "MATCH (m:SemanticMemory {id: expired.id}) "
    #         "WHERE m.created_at = expired.created_at "
    #         "DETACH DELETE m RETURN count(*) AS removed",
    #         {"expired_memories": expired_memories},
    #     )
    #     return int(rows[0]["removed"])
