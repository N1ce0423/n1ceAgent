import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# 固定读取项目根目录，避免从 memory/ 运行时找不到 .env。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

@dataclass
class MemoryConfig:
    working_memory_capacity: int = 50
    working_memory_ttl_hours: int = 3

    eposidic_memory_capacity: int = 1000
    episodic_memory_ttl_days: int = 30

    semantic_memory_capacity: int = 10000
    semantic_memory_ttl_days: int = 30

    database_path: str = "memory.db"

    qdrant_url = os.getenv("QDRANT_URL")
    

    
