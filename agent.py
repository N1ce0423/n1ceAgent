from abc import ABC, abstractmethod
from typing import Optional, Any
from message import Message
from HelloAgentsLLM import HelloAgentsLLM
from config import Config
from session_store import SessionInfo, SessionStore

class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm_client: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        session_id: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
    ):
        self.name = name
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.session_store = session_store or SessionStore(self.config.session_database_path)
        if session_id is None:
            self.new_session()
        else:
            self.select_session(session_id)

    @abstractmethod
    def run(self, input_text: str, **kwargs:Any) -> str:
        """
        运行代理的主要逻辑。
        这个方法必须被子类实现。
        """
        pass

    def add_message(self, message: Message):
        """兼容旧调用；会话不保存消息。"""

    def clear_history(self):
        """兼容旧调用；会话没有历史可清空。"""

    def record_turn(self, input_text: str, answer: str) -> None:
        """兼容旧调用；不记录本轮问答。"""

    def new_session(self) -> str:
        """新建会话 ID，用于保存及检索该会话的记忆。"""
        self.session_id = self.session_store.create_session()
        self._on_session_changed()
        return self.session_id

    def select_session(self, session_id: str) -> str:
        """切换当前会话 ID；不恢复聊天历史。"""
        self.session_store.require_session(session_id)
        self.session_id = session_id
        self._on_session_changed()
        return self.session_id

    def list_sessions(self) -> list[SessionInfo]:
        return self.session_store.list_sessions()

    def _on_session_changed(self) -> None:
        """供子类同步当前会话关联的工具和临时执行状态。"""

    def get_history(self) -> list[Message]:
        """兼容旧调用，始终返回空列表。"""
        return []

    def __str__(self) -> str:
        return f"Agent(name={self.name}, session_id={self.session_id})"
