import ast
import re
from typing import Optional
from agent import Agent
from HelloAgentsLLM import HelloAgentsLLM
from tool_registry import ToolRegistry
from config import Config

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列，并且计划不需要在这一阶段执行或得到结果。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”。如果当前步骤涉及数学计算，必须调用可用的计算工具，不能直接心算或编造计算结果。

# 可用工具
{tools}

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请严格遵循以下规则：
1. 需要计算、查询或外部信息时，必须调用合适的工具。
2. 每次只能调用一个工具，工具名必须来自“可用工具”，且一旦调用工具，请仅输出工具调用格式里包含的内容，不要添加任何其他文字或描述。
3. 工具调用必须严格使用以下格式，不要加反引号、短横线或其他文字：
Action: {{tool_name}}[{{tool_input}}]
4. 工具结果会出现在下一轮历史中。
5. 不需要工具时，直接输出当前步骤的执行结果，不要输出 Action。
"""

class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question:str) -> list[str]:
        """
        使用LLM生成一个行动计划，将复杂问题分解为多个简单步骤。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question = question)

        messages = [{"role": "user", "content": prompt}]

        print("Generating plan:")
        response_text = self.llm_client.think(messages = messages) or ""

        print(f"Plan response: {response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except Exception as e:
            print(f"Error parsing plan: {e}")
            return []

class Executor:
    def __init__(
            self, 
            llm_client,
            tool_registry: Optional[ToolRegistry] = None,
            max_tool_calls: int = 3,
        ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_tool_calls = max_tool_calls

    def parse_action(self, action: str) -> tuple[str, str]:
        """
        解析工具调用的格式，提取工具名称和输入参数
        """
        match = re.search(
            r"Action:\s*([A-Za-z_]\w*)\[(.*)\]\s*$",
            action.strip(),
            re.DOTALL,
        )
        if not match:
            return "", ""
        return match.group(1), match.group(2).strip()

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划逐步执行并解决问题
        """
        history = ""

        tools_desc = (
            self.tool_registry.get_available_tools()
            if self.tool_registry
            else "无可用工具。"
        )

        for i, strp in enumerate(plan):
            print(f"Executing step {i+1}/{len(plan)}: {strp}")
            step_history = history
            response_text = ""

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                tools=tools_desc,
                plan=plan,
                history=step_history if step_history else "无",
                current_step=strp,
            )

            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.think(messages=messages) or ""

            if not response_text:
                print(f"Error: No response for step {i+1}")
                return ""

            tool_name, tool_input = self.parse_action(response_text)

            if tool_name and tool_input:
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                response_text = f"Action: {tool_name}[{tool_input}]\nObservation: {observation}"

            history += f"步骤 {i+1}: {strp}\n结果: {response_text}\n"

            print(f"Step {i+1} result: {response_text}")

        final_answer = response_text
        return final_answer

class MyPlanAndSolveAgent(Agent):
    def __init__(
            self, 
            llm_client,
            tool_registry: Optional[ToolRegistry] = None
        ):
        """
        初始化MyPlanAndSolveAgent
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.tool_registry = tool_registry
        self.executor = Executor(self.llm_client, self.tool_registry)

    def run(self, quetsion:str):
        """
        执行计划和解决问题的流程
        """
        print(f"---Starting Planning Phase---\nquestion: {quetsion}")

        plan = self.planner.plan(quetsion)

        if not plan:
            print("Error: Failed to generate a valid plan.")
            return

        final_answer = self.executor.execute(quetsion, plan)

        print(f"\n---Mission Success---\nFinal Answer: {final_answer}")
        return final_answer

    