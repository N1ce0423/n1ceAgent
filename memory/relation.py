from dataclasses import dataclass, field
from typing import Any

@dataclass
class Relation:
    """两个实体之间的有向关系。"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: dict[str, Any] = field(default_factory=dict)