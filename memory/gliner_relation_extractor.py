"""使用 GLiNER2 多语言模型，将关系端点映射到已有实体。"""

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from entity import Entity
from relation import Relation


# 这里定义关系的语义和方向，而不是枚举文本中必须出现的谓词。
DEFAULT_RELATION_TYPES = {
    "STUDIES_AT": "The head person studies at the tail school or university.",
    "WORKS_AT": "The head person works for the tail organization.",
    "LOCATED_IN": "The head organization or place is located in the tail location.",
    "LIVES_IN": "The head person lives in the tail location.",
    "FOUNDED": "The head person or organization founded the tail organization.",
    "ACQUIRED": "The head organization acquired or purchased the tail organization.",
    "INVESTED_IN": "The head person or organization invested money in the tail organization.",
    "OWNS": "The head person or organization owns the tail entity or asset.",
    "KNOWS": "The head person knows the tail person.",
    "LIKES": "The head person likes the tail entity.",
    "BELONGS_TO": "The head entity belongs to the tail organization or entity.",
}


class GLiNER2RelationExtractor:
    """模型按需加载并复用；只提取数据，不访问数据库。

    输出仅连接传入实体。名称或字符区间存在歧义时跳过，避免猜测身份。
    模型分数不是事实正确率；否定、计划等表达仍可能被误识别。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
        relation_types: Optional[Dict[str, str]] = None,
        model: Any = None,
    ):
        default_path = Path(__file__).resolve().parents[1] / ".models" / "gliner2.5-multi-v1"
        self.model_name = model_name or os.getenv("GLINER2_MODEL") or (
            str(default_path) if (default_path / "config.json").exists()
            else "fastino/gliner2.5-multi-v1"
        )
        self.threshold = float(threshold if threshold is not None else os.getenv("GLINER2_THRESHOLD", "0.7"))
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("GLINER2_THRESHOLD must be between 0 and 1")
        self.relation_types = dict(DEFAULT_RELATION_TYPES if relation_types is None else relation_types)
        if not self.relation_types or any(
            not isinstance(name, str) or re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None
            or not isinstance(description, str) or not description.strip()
            for name, description in self.relation_types.items()
        ):
            raise ValueError("Relation types must map uppercase identifiers to descriptions")
        self._model = model

    def _get_model(self):
        if self._model is None:
            try:
                from gliner2 import AutoExtractor
            except ImportError as exc:
                raise RuntimeError("Install memory/requirements-gliner2.txt in your conda environment") from exc
            model_path = self.model_name
            if not Path(model_path).is_dir():
                from huggingface_hub import snapshot_download

                cache_dir = Path(__file__).resolve().parents[1] / ".models" / ".cache" / "huggingface"
                model_path = snapshot_download(self.model_name, cache_dir=str(cache_dir))
            self._model = AutoExtractor.from_pretrained(
                model_path, word_splitter="char", map_location="cpu",
            )
            self._model.eval()
        return self._model

    @staticmethod
    def _endpoint_entity(text: str, endpoint: Any, entities: List[Entity]):
        """严格校验模型区间，再匹配实体的本次文本提及。"""
        if not isinstance(endpoint, dict):
            return None
        start, end, name = endpoint.get("start"), endpoint.get("end"), endpoint.get("text")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text) or not isinstance(name, str)
                or text[start:end] != name):
            return None
        matches = {}
        for entity in entities:
            if entity.name != name:
                continue
            mentions = entity.properties.get("mentions")
            if mentions is None or any(
                isinstance(mention, dict)
                and mention.get("start_char") == start and mention.get("end_char") == end
                for mention in mentions
            ):
                matches[entity.id] = entity
        return next(iter(matches.values())) if len(matches) == 1 else None

    def extract(self, text: str, entities: List[Entity]) -> List[Relation]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip() or len({entity.id for entity in entities}) < 2:
            return []
        model = self._get_model()
        schema = model.create_schema().relations(self.relation_types)
        # 字符切分配合重叠窗口，避免长文本的尾部被默默截断。
        result = model.extract_long(
            text, schema, threshold=self.threshold, chunk_size=384, chunk_overlap=64,
            batch_size=1, include_spans=True, include_confidence=True,
        )
        if not isinstance(result, dict) or not isinstance(result.get("relation_extraction"), dict):
            raise RuntimeError("GLiNER2 returned an invalid relation extraction payload")
        relations = {}
        for relationship_type, pairs in result["relation_extraction"].items():
            if relationship_type not in self.relation_types or not isinstance(pairs, list):
                continue
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                head, tail = pair.get("head"), pair.get("tail")
                source = self._endpoint_entity(text, head, entities)
                target = self._endpoint_entity(text, tail, entities)
                if source is None or target is None or source.id == target.id:
                    continue
                scores = [endpoint.get("confidence") for endpoint in (head, tail)]
                if any(
                    not isinstance(score, (int, float)) or isinstance(score, bool)
                    or not math.isfinite(score) or not self.threshold <= score <= 1
                    for score in scores
                ):
                    continue
                key = (source.id, target.id, relationship_type)
                if key not in relations:
                    relations[key] = Relation(
                        source_id=source.id, target_id=target.id,
                        relationship_type=relationship_type,
                        properties={"extractor": "GLiNER2", "model": self.model_name, "evidence": []},
                    )
                evidence = {
                    "text": text,
                    "head": {"start_char": head["start"], "end_char": head["end"], "confidence": scores[0]},
                    "tail": {"start_char": tail["start"], "end_char": tail["end"], "confidence": scores[1]},
                }
                if evidence not in relations[key].properties["evidence"]:
                    relations[key].properties["evidence"].append(evidence)
        return list(relations.values())
