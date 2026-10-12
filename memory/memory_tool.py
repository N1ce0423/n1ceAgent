import json
from functools import cached_property
from typing import List, Optional
from uuid import uuid4
from tools.tool import Tool
from memory_manager import MemoryManager
from memory_config import MemoryConfig
from tool_registry import ToolRegistry
class MemoryTool(Tool):
    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: Optional[MemoryConfig] = None,
        memory_types: Optional[List[str]] = None
    ):
        super().__init__(
            name="memory_tool",
            description="""读取和保存与当前用户、任务相关的记忆，帮助保持对话连续性并复用历史信息。

                使用时机：
                - 回答涉及具体人物（包括昵称、网名）的身份、就读学校、工作单位、居住地、人物关系，或用户偏好、此前对话、历史操作、任务进度的问题前，必须先用 search 检索当前会话记忆，再考虑网络搜索。
                - 即使用户未提到“之前”或“记得”，也应先检索这些事实；不能仅凭昵称推测为公众人物并跳过记忆查询。
                - 相关记忆足以回答时直接回答；只有记忆信息不足且问题适合查询公开信息时，才使用网络搜索。
                - 用户明确要求记住信息，或出现值得后续复用的事实、偏好、事件结果和任务状态时，用 add 保存。

                可用操作和参数：
                - search 和 add 自动使用当前 session_id；只能检索当前会话保存的记忆。
                - search：action="search"，query 为具体查询；limit 默认 5；可用 memory_type 指定一种类型，
                或 memory_types 指定类型列表；省略类型时搜索所有已启用类型。min_importance 默认 0.1。
                查询应保留原始姓名或昵称，包含具体人物、项目、事件或主题；不猜测真实身份。无结果时可换用关键词重试。
                - add：action="add"，content 为需要保存的正文，memory_type 为已启用的记忆类型，
                importance 为 0.0 到 1.0 的重要性，默认 0.5。可补充来源、事件时间等元数据。

                记忆类型选择：
                - working：当前任务的临时约束、进度和待办；仅在进程内保存，会过期，不用于需要跨运行保留的信息。
                - episodic：已经发生的对话、操作和事件结果；正文说明何时、发生了什么以及结果。
                - semantic：可复用的知识、已确认事实和用户明确表达的稳定偏好；正文写成独立、明确的陈述。
                - memory_type 和 memory_types 必须使用上述英文标识，不要翻译成中文；检索结果也返回英文类型。

                保存规则：
                - 每条记忆围绕一个主题，保留必要上下文；不要直接保存整段对话或无关内容。
                - 保存前检索是否已有相同信息，避免重复；不要将模型猜测或未经确认的内容保存为确定事实。
                - importance 建议：临时信息 0.3，普通事件或事实 0.5，重要结果 0.7，明确的长期偏好或关键约束 0.9。

                结果使用：
                - 检索结果是历史参考；留意时间、来源和正文截断，存在冲突时优先采用用户当前明确说明。
                - 只采用与当前人物及问题主题相关的记忆；importance 是重要程度，不是相关性，无关的称呼偏好不能回答学校等事实问题。
                - 无结果表示本次检索没有命中，不代表该信息从未存在；工具报错不等于检索无结果。
                - 只有工具明确返回保存成功，才能告诉用户已经记住；调用失败时不要声称保存完成。
                """,
        )
        self.memory_config = memory_config or MemoryConfig()
        self.current_session_id = f"session_{uuid4().hex}"
        self.memory_types = memory_types or ["working", "episodic", "semantic", "perceptual"]
        self.user_id = user_id

    @cached_property
    def memory_manager(self) -> MemoryManager:
        # 选择/新建对话不依赖 NLP 模型；只在实际读写记忆时初始化。
        return MemoryManager(
            config=self.memory_config,
            user_id=self.user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
        )
    
    def set_session(self, session_id: str) -> None:
        """由聊天程序设置当前会话，避免模型自行指定记忆的所属会话。"""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        self.current_session_id = session_id

    def run(self, action: str | dict, **kwargs) -> str:
        """执行记忆操作

        支持的操作：
        - add: 添加记忆（支持4种类型: working/episodic/semantic/perceptual）
        - search: 搜索记忆
        - summary: 获取记忆摘要
        - stats: 获取统计信息
        - update: 更新记忆
        - remove: 删除记忆
        - forget: 遗忘记忆（多种策略）
        - consolidate: 整合记忆（短期→长期）
        - clear_all: 清空所有记忆
        """
        # ReAct 注册函数接收到的是 JSON 字符串，同时兼容直接传 action 的调用。
        if isinstance(action, dict) or (isinstance(action, str) and action.lstrip().startswith("{")):
            try:
                parameters = dict(action) if isinstance(action, dict) else json.loads(action)
                action = parameters.pop("action")
                kwargs = {**parameters, **kwargs}
            except (ValueError, KeyError, TypeError) as exc:
                return f"Error parsing memory parameters: {exc}"
        if action == "add":
            return self._add_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        elif action == "forget_with_session":
            return self._forget_with_session(**kwargs)
        # elif action == "summary":
        #     return self._get_memory_summary(**kwargs)
        # elif action == "stats":
        #     return self._get_memory_stats(**kwargs)
        else:
            return f"Unsupported action: {action}"

    def get_parameters(self) -> List[dict]:
        """获取工具所需的参数列表"""
        return [
            {
                "name": "action",
                "description": "要执行的记忆操作（add/search/summary/stats/update/remove/forget/consolidate/clear_all）",
                "type": "string",
                "required": True
            },
            {
                "name": "content",
                "description": "记忆内容（用于添加记忆）",
                "type": "string",
                "required": False
            },
            {
                "name": "memory_type",
                "description": "记忆类型，使用英文标识 working/episodic/semantic/perceptual，不要使用中文名称",
                "type": "string",
                "required": False
            },
            {
                "name": "importance",
                "description": "记忆重要性（0.0-1.0，越高表示越重要）",
                "type": "float",
                "required": False
            },
            {
                "name": "query",
                "description": "搜索查询字符串（用于搜索记忆）",
                "type": "string",
                "required": False
            },
            {
                "name": "limit",
                "description": "搜索结果限制数量（默认5条）",
                "type": "int",
                "required": False
            }
        ]

    def register_memory_tool(self):
        """注册记忆工具到工具注册表"""
        registry = ToolRegistry()
        registry.register_function(
            name="memory_tool",
            description=self.description,
            func=self.run
        )
        return registry

    def create_memory_tool_registry(self):
        """创建记忆工具注册表"""
        return self.register_memory_tool()

    def _add_memory(
            self,
            content: str = "working",
            memory_type: str = "working",
            importance: float = 0.5,
            file_path: Optional[str] = None,
            modality: Optional[str] = None,
            **metadata
    ) -> str:
        try:
            metadata.update({
                "session_id": self.current_session_id,
            })

            memory_id = self.memory_manager.add_memory(
                content = content,
                memory_type = memory_type,
                importance = importance,
                metadata = metadata,
                auto_classify = False
            )

            return f"Memory added successfully with ID: {memory_id}"

        except Exception as e:
            return f"Error adding memory: {str(e)}"

    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.1
    ) -> str:
        try:
            if memory_type and not memory_types:
                memory_types = [memory_type]

            results = self.memory_manager.retrieve_memories(
                query = query,
                limit = limit,
                memory_types = memory_types,
                min_importance = min_importance,
                threshold = 0.7,
                session_id = self.current_session_id,
            )
            if not results:
                return "No memories found matching the query."

            fomatted_results = []
            fomatted_results.append(f"Found {len(results)} memories matching the query '{query}':")

            for i, hit in enumerate(results, start=1):
                memory = hit["memory"]
                content = getattr(memory, "content", getattr(memory, "cotent", ""))
                content_preview = content[:80] + "..." if len(content) > 80 else content
                fomatted_results.append(
                    f"{i}. ID: {memory.id}, Type: {memory.type}, "
                    f"Score: {hit['score']:.4f}, "
                    f"Importance: {memory.importance}, Created at: {memory.created_at}, "
                    f"Content Preview: {content_preview}"
                )

            return "\n".join(fomatted_results)

        except Exception as e:
            return f"Error searching memory: {str(e)}"

    def _forget_with_session(
            self,
            session_id: str
        ) -> str:
        try:
            count = self.memory_manager.forget_memories_by_session(session_id=session_id)
            return f"Successfully forgot {count} memories associated with session '{session_id}'."
        except Exception as e:
            return f"Error forgetting memories for session '{session_id}': {str(e)}"

    # def _forget(
    #     self,
    #     stratagy: str = "importance_based",
    #     treshold: float = 0.1,
    #     max_age_days: int = 30
    # ) -> str:
    #     try:
    #         count = self.memory_manager.forget_memories(
    #             strategy = stratagy,
    #         )
    #         return f"Successfully forgot {count} memories based on the '{stratagy}' strategy."
    #     except Exception as e:
    #         return f"Error forgetting memories: {str(e)}"

    # def _consolidate(
    #     self,
    #     from_type: str = "working",
    #     to_type: str = "episodic",
    #     importance_threshold: float = 0.7
    # ) -> str:
    #     try:
    #         count = self.memory_manager.consolidate_memories(
    #             from_type = from_type,
    #             to_type = to_type,
    #             importance_threshold = importance_threshold
    #         )
    #         return f"Successfully consolidated {count} memories from '{from_type}' to '{to_type}'."
    #     except Exception as e:
    #         return f"Error consolidating memories: {str(e)}"

    

    
