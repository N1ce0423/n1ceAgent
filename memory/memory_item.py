from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryItem:
	"""
	用于保存一条记忆。
	"""
	id: str
	content: str
	type: str
	importance: float
	created_at: datetime = field(default_factory=datetime.now)
	metadata: dict[str, Any] = field(default_factory=dict)
