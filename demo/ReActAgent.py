import re
from HelloAgentsLLM import HelloAgentsLLM
from tool_executor import ToolExecutor

class ReActAgent:
    # ReAct 提示词模板
    REACT_PROMPT_TEMPLATE =REACT_PROMPT_TEMPLATE = """
    你是一个 ReAct 智能体。你可以通过输出 Action 来请求程序调用外部工具。

    你不能直接调用工具，只能按照指定格式输出工具调用请求。程序会执行你的 Action，并把结果作为 Observation 放入下一轮 History。

    可用工具如下：

    {tools}

    所有工具都使用统一调用格式：

    Action: 工具名[工具输入]

    其中：
    - 工具名必须从“可用工具”列表中选择，必须完全一致。
    - 工具输入必须根据该工具的描述生成。
    - 如果工具描述要求输入关键词，就传入关键词。
    - 如果工具描述要求输入路径，就传入路径。
    - 如果工具描述要求输入表达式，就传入表达式。
    - 如果没有合适工具，就不要编造工具名。

    你每次回复必须严格使用以下格式：

    Thought: 简短说明你为什么选择这个动作
    Action: 工具名[工具输入]

    或者：

    Thought: 简短说明你已经得到答案
    Action: Finish[最终答案]

    工具选择规则：

    1. 如果用户问题需要外部信息、实时信息、最新信息、文件内容、计算、查询、执行某种操作，并且可用工具中有合适工具，必须调用工具。

    2. 如果 History 为空，并且问题无法仅靠已有信息可靠回答，应优先调用最相关的工具。

    3. 如果 History 中已经有 Observation，并且 Observation 足以回答用户问题，应使用 Finish[最终答案]。

    4. 如果缺少必要参数，并且无法从用户问题中推断，应使用 Finish[请用户补充缺少的信息]，不要猜测关键参数。

    5. 一次只能调用一个工具。

    6. 不要输出多个 Action。

    7. 不要自己编造 Observation。

    8. 不要输出 Markdown、代码块、解释性段落或额外文本。

    9. 不要使用不存在的工具。

    10. Action 后面不要再写其它内容。

    当前问题：
    {question}

    历史记录：
    {history}

    现在请严格输出：

    Thought:
    Action:
    """
    
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- Step {current_step} ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = self.REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            print("Prompt for this step:")
            print(prompt)

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("Error: the LLM did not return a valid response.")
                break
            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"Thought: {thought}")

            if not action:
                print("Warning: could not parse a valid Action; terminating.")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"Final answer: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"Action: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"Error: tool '{tool_name}' was not found."
            else:
                observation = tool_function(tool_input) # 调用真实工具
            print(f"Observation: {observation}")
            # 将本轮的Action,Observation和Thought添加到历史记录中
            if thought:
                self.history.append(f"Thought: {thought}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        # 循环结束
        print("Maximum number of steps reached; terminating.")
        return None


    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
