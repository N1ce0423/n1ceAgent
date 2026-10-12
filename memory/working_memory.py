import datetime
import math
import re
from jieba import tokenize
from collections import Counter
from typing import Dict, List
from memory.memory_item import MemoryItem
from memory.retrieval_hit import MemoryRetrievalHit
from memory_config import MemoryConfig
from memory_store import MemoryStore

class WorkingMemory(MemoryStore):
    """工作记忆实现
    特点：
    - 容量有限（默认50条）+ TTL自动清理
    - 纯内存存储，访问速度极快
    - 组合词法检索：TF-IDF 词项权重相似度 + 关键词匹配
    """

    def __init__(self, config: MemoryConfig):
        self.max_capacity = config.working_memory_capacity or 50
        self.ttl_hours = config.working_memory_ttl_hours or 3
        self.memories = []

    def add(self, memory_item: MemoryItem) -> str:
        self._expire_old_memories()

        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_importance_memory()
        self.memories.append(memory_item)
        return memory_item.id

    def update(self, memory_item: MemoryItem) -> bool:
        for i, memory in enumerate(self.memories):
            if memory.id == memory_item.id:
                self.memories[i] = memory_item
                return True
        return False

    def get_all(self) -> List[MemoryItem]:
        self._expire_old_memories()
        return list(self.memories)

    def contains(self, memory_id: str) -> bool:
        self._expire_old_memories()
        return any(memory.id == memory_id for memory in self.memories)

    def remove(self, memory_id: str) -> bool:
        for memory in self.memories:
            if memory.id == memory_id:
                self.memories.remove(memory)
                return True
        return False

    def remove_memories_by_session(self, session_id: str) -> int:
        """删除指定 session_id 的记忆，返回删除的数量。"""
        count = 0
        for memory in self.memories[:]:
            if memory.metadata.get("session_id") == session_id:
                self.memories.remove(memory)
                count += 1
        return count

    def _expire_old_memories(self) -> int:
        """清理过期的记忆"""
        current_time = datetime.datetime.now(datetime.timezone.utc)
        count = 0
        end_time = current_time - self.ttl_hours * datetime.timedelta(minutes=1)

        for memory in self.memories[:]:
            # 无时区时间按本地时间解释，带时区时间转换为 UTC 后比较。
            created_at_utc = memory.created_at.astimezone(datetime.timezone.utc)
            if created_at_utc < end_time:
                self.memories.remove(memory)
                count += 1
        return count

    def _remove_low_importance_memory(self, treshold: float = 0.1) -> int:
        """移除重要性低于阈值的记忆"""
        count = 0
        while self.memories:
            lowest_importance_memory = min(
                self.memories,
                key=lambda memory: memory.importance,
            )
            if lowest_importance_memory.importance < treshold:
                self.memories.remove(lowest_importance_memory)
                count += 1
            else:
                break
        return count

    def _remove_lowest_importance_memory(self) -> bool:
        """移除当前记忆中重要性最低的记忆"""
        if not self.memories:
            return False
        lowest_importance_memory = min(
            self.memories,
            key=lambda memory: memory.importance,
        )
        self.memories.remove(lowest_importance_memory)
        return True

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryRetrievalHit]:
        """
        检索工作记忆
        """
        self._expire_old_memories()
        if limit <= 0:
            return []
        vector_scores = self._try_tfidf_search(query)

        scored_memories = []
        for memory in self.memories:
            if kwargs.get("session_id") is not None and memory.metadata.get("session_id") != kwargs["session_id"]:
                continue
            if memory.importance < kwargs.get("min_importance", 0):
                continue
            vector_score = vector_scores.get(memory.id, 0.0)
            keyword_score = self._calculate_key_word_score(query, memory.content)
            
            base_relevance = vector_score * 0.7 + keyword_score * 0.3 if vector_score > 0 else keyword_score
            importance_weight = 0.8 + (memory.importance * 0.4)
            time_decay = self._calculate_time_decay(memory.created_at)
            final_score = base_relevance * importance_weight * time_decay
            if final_score > 0:
                scored_memories.append({"memory": memory, "score": float(final_score)})

        scored_memories.sort(key=lambda hit: hit["score"], reverse=True)
        return scored_memories[:limit]
    
    def _try_tfidf_search(self, query: str) -> Dict[str, float]:
        """
        尝试使用TF-IDF进行检索
        """
        def tokenize(text: str) -> List[str]:
            # 同时支持英文单词、数字及中文单字，避免依赖第三方分词库。
            return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())

        query_terms = tokenize(query)
        if not query_terms or not self.memories:
            return {}

        documents = [tokenize(str(memory.content)) for memory in self.memories]
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        document_count = len(documents)

        def tfidf(terms: List[str]) -> Dict[str, float]:
            term_frequency = Counter(terms)
            return {
                term: (count / len(terms)) * math.log(
                    (document_count + 1) / (document_frequency.get(term, 0) + 1)
                )
                for term, count in term_frequency.items()
                if term in document_frequency
            }

        query_vector = tfidf(query_terms)
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0:
            return {}

        scores: Dict[str, float] = {}
        for memory, terms in zip(self.memories, documents):
            document_vector = tfidf(terms)
            document_norm = math.sqrt(
                sum(value * value for value in document_vector.values())
            )
            if document_norm == 0:
                continue
            dot_product = sum(
                query_vector.get(term, 0.0) * value
                for term, value in document_vector.items()
            )
            if dot_product > 0:
                scores[memory.id] = dot_product / (query_norm * document_norm)
        return scores

    def _calculate_time_decay(self, created_at: datetime.datetime) -> float:
        """计算时间衰减因子"""
        current_time = datetime.datetime.now(datetime.timezone.utc)
        created_at_utc = created_at.astimezone(datetime.timezone.utc)
        age_minutes = (current_time - created_at_utc).total_seconds() / 60
        decay_factor = math.exp(-age_minutes / self.ttl_hours)
        return decay_factor

    def _calculate_key_word_score(self, query: str, content: str) -> float:
        """
        计算查询词在记忆内容中的匹配得分，结果范围为 0 到 1。
        """
        # jieba 返回 (词, 起始位置, 结束位置)，只取词，避免位置不同导致漏匹配。
        query_terms = {word for word, _, _ in tokenize(query.lower()) if word.strip()}
        content_terms = {word for word, _, _ in tokenize(content.lower()) if word.strip()}
        if not query_terms:
            return 0.0

        return len(query_terms & content_terms) / len(query_terms)
        
        
        
