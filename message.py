from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    content: str
    role: MessageRole
    timeStamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content:str, role:MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timeStamp=kwargs.get("timeStamp", datetime.now()),
            metadata=kwargs.get("metadata", {})
        )

    def to_dictS(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"