from typing import List, Dict, Any, Optional
from tool_registry import ToolRegistry
import asyncio
import concurrent.futures

class ToolChain:
    """
    工具链
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        """
        添加工具执行步骤
        
        Arguments:
            tool_name: 工具名称
            input_template: 输入模板，支持占位符
            output_key: 输出结果的键名，如果为None，则使用默认键名"result"  
        """
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        })

    def execute(self, tool_registry: ToolRegistry, initial_input: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """
        执行工具链
        """
        context = context or {}
        context["input"] = initial_input

        print(f"Executing tool chain: {self.name}")

        for i, step in enumerate(self.steps, 1):
            tool_name = step["tool_name"]
            input_template = step["input_template"] 
            output_key = step["output_key"]

            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                return f"Error: Missing key {e} in context for step {i}."

            print(f"Step {i}: Executing tool '{tool_name}' with input: {tool_input}")

            result = tool_registry.execute_tool(tool_name, tool_input)
            context[output_key] = result

            print(f"Step {i}: Tool '{tool_name}' output stored in context key '{output_key}': {result}")

        final_result = context[self.steps[-1]["output_key"]]

class ToolChainManager:
    """
    工具链管理器
    """
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.tool_chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """
        注册工具链
        """
        self.tool_chains[chain.name] = chain
        print(f"Registered tool chain: {chain.name}")

    def execute_chain(self, chain_name: str, initial_input: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """
        执行指定的工具链
        """
        if chain_name not in self.tool_chains:
            return f"Error: Tool chain '{chain_name}' not found."

        chain = self.tool_chains[chain_name]
        return chain.execute(self.registry, initial_input, context)

    def list_chains(self) -> List[str]:
        """
        列出所有注册的工具链
        """
        return list(self.tool_chains.keys())

class AsyncToolExecutor:
    """
    异步工具执行器
    """
    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers = max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """
        异步执行工具
        """
        loop = asyncio.get_event_loop()

        def _execute():
            return self.registry.execute_tool(tool_name, input_data)

        result = await loop.run_in_executor(self.executor, _execute)
        return result

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[str]:
        """
        并行执行多个工具
        """
        print(f"Executing {len(tasks)} tools in parallel...")

        async_tasks = []
        for task in tasks:
            tool_name = task["tool_name"]
            input_data = task["input_data"]
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append(async_task)

        results = await asyncio.gather(*async_tasks)

        print(f"Completed execution of {len(tasks)} tools.")
        return results

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)