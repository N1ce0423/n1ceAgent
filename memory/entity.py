from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """知识图谱中的实体。"""
    id: str
    name: str
    entity_type: str = "Entity"
    properties: dict[str, Any] = field(default_factory=dict)