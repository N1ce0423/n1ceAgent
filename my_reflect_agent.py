
from typing import List, Optional
from agent import Agent
from HelloAgentsLLM import HelloAgentsLLM
from config import Config
from tool_registry import ToolRegistry
from memory import Memory
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""


REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class ReflectionAgent(Agent):
    def __init__(
                self,
                name: str,
                llm_client: HelloAgentsLLM,
                tool_registry: Optional[ToolRegistry] = None,
                system_prompt: Optional[str] = None,
                config: Optional[Config] = None,
                max_steps: int = 5,
                custom_prompt: Optional[str] = None,
        ):
            super().__init__(name, llm_client, system_prompt, config)
            self.tool_registry = tool_registry
            self.llm_client = llm_client
            self.max_steps = max_steps
            self.current_history: List[str] = []
            self.memory = Memory()

    def run(self, task: str):
        print(f"Running ReflectionAgent with task: {task}")
        print("Step 1: Generating initial code...")
        initial_propt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self.llm_client.think([{"role": "user", "content": initial_propt}])
        self.memory.add_record("execution", initial_code)

        for i in range(self.max_steps):
            print(f"Step {i + 1}: Reviewing and refining code...")

            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self.llm_client.think([{"role": "user", "content": reflect_prompt}])
            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback:
                print("No further improvements needed. Final code:")
                break

            print("Refining code based on feedback...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self.llm_client.think([{"role": "user", "content": refine_prompt}])
            self.memory.add_record("execution", refined_code)

        final_code = self.memory.get_last_execution()
        print("Final optimized code:")
        print(f"```python\n{final_code}\n```")
        return final_code

