from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


@dataclass
class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    description: str
    type: str = "string"
    required: bool = False


class Tool(ABC):
    """
    工具基类
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行工具的主要逻辑
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """
        获取工具所需的参数列表
        """
        pass