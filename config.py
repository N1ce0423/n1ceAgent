import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    default_model: str = "gpt-5.5"
    default_provider: str = "rightcode"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    debug : bool = False
    log_level: str = "INFO"

    max_history_length: int = 100
    max_round: int = 10
    session_database_path: str = str(Path(__file__).resolve().parent / ".runtime" / "conversations.db")

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
            session_database_path=os.getenv(
                "AGENT_SESSION_DATABASE_PATH", cls.model_fields["session_database_path"].default
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
