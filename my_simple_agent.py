import re
from typing import Optional
from agent import Agent
from message import Message
from HelloAgentsLLM import HelloAgentsLLM
from config import Config
from tool_registry import ToolRegistry

class MySimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"MySimpleAgent initialized with tool calling {'enabled' if self.enable_tool_calling else 'disabled'}.")

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        print(f"{self.name} executing: {input_text}")

        messages = []

        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        for msg in self.get_history():
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        if not self.enable_tool_calling:
            response = self.llm.think(messages=messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"{self.name} response finish")
            return response

        return self._run_with_tool_calling(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        base_prompt = self.system_prompt or "你是一个优秀的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_available_tools()
        if not tools_description or tools_description == "No tools available.":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

    def _run_with_tool_calling(self, messages: list, input_text: str, max_tool_iterations: int, **kwargs) -> str:
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            response = self.llm.think(messages=messages, **kwargs)

            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"{self.name} detected tool calls: {tool_calls}")
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    clean_response = clean_response.replace(call["original"], "")

                messages.append({"role": "assistant", "content": clean_response})

                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": tool_results_text})

                current_iteration += 1
                continue

            final_response = response
            break

        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.think(messages=messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"{self.name} response finish")

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f"[TOOL_CALL:{tool_name}:{parameters}]"
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        if self.tool_registry is None:
            return "fail, tool registry is not available."

        try:
            result = self.tool_registry.execute_tool(
                tool_name,
                parameters,
            )

            return (
                f"Tool '{tool_name}' executed successfully. "
                f"Result:\n{result}"
            )
        except Exception as e:
            return f"fail, error executing tool '{tool_name}': {e}"

    

    
