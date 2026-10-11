from typing import Any, Callable, Protocol


class Tool(Protocol):
    name: str
    description: str

    def run(self, parameters: dict[str, Any]) -> str:
        ...


class ToolRegistry:
    """Local tool registry used by MyReActAgent."""
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}
        self.current_session_id: str | None = None

    def set_session(self, session_id: str) -> None:
        """将当前对话 ID 同步给支持会话切换的工具。"""
        self.current_session_id = session_id
        tools = list(self._tools.values()) + [
            getattr(info["func"], "__self__", None) for info in self._functions.values()
        ]
        for tool in tools:
            setter = getattr(tool, "set_session", None)
            if callable(setter):
                setter(session_id)

    def register_tool(self, tool: Tool):
        """Register a Tool object."""
        if tool.name in self._tools:
            print(f"Warning: tool '{tool.name}' already exists and will be overwritten.")

        self._tools[tool.name] = tool
        setter = getattr(tool, "set_session", None)
        if self.current_session_id is not None and callable(setter):
            setter(self.current_session_id)
        print(f"Tool '{tool.name}' registered.")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """Register a plain function as a tool."""
        if name in self._functions:
            print(f"Warning: tool '{name}' already exists and will be overwritten.")

        self._functions[name] = {
            "description": description,
            "func": func,
        }
        setter = getattr(getattr(func, "__self__", None), "set_session", None)
        if self.current_session_id is not None and callable(setter):
            setter(self.current_session_id)
        print(f"Tool '{name}' registered.")

    def get_available_tools(self) -> str:
        """Return formatted descriptions for all registered tools."""
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "No tools available."

    def execute_tool(self, name: str, input_text: str) -> str:
        """Execute a registered tool by name."""
        if name in self._tools:
            return self._tools[name].run({"input": input_text})

        if name in self._functions:
            return self._functions[name]["func"](input_text)

        available = ", ".join(self.list_tools()) or "none"
        return f"Error: tool '{name}' does not exist. Available tools: {available}"

    def list_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_tool(self, name: str) -> callable:
        """
        Get the execution function of a tool by name.
        """
        return self._functions.get(name, {}).get("func")
