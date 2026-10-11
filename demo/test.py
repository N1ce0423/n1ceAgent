import os
from pathlib import Path
import serpapi
from dotenv import load_dotenv
from tool_executor import ToolExecutor
from ReActAgent import ReActAgent
from HelloAgentsLLM import HelloAgentsLLM

load_dotenv(Path(__file__).resolve().parent / ".env")

def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"Searching the web with [SerpApi]: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY is not configured in the .env file."

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = serpapi.Client(api_key=api_key)
        results = client.search(params)
        
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"Sorry, no information was found for '{query}'."

    except Exception as e:
        return f"Search error: {e}"
    
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()
    helloAgentsLLM = HelloAgentsLLM()  # 初始化LLM客户端
    reactAgent = ReActAgent(tool_executor=toolExecutor, llm_client=helloAgentsLLM) 

    # 2. 注册我们的实战搜索工具
    search_description = "Web search tool for current information, news, facts, product models, versions, prices, and dates that may not be reliable in a local knowledge base. Input should be concise search keywords."
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- Available tools ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- Running Action: Search['AMD latest CPU model'] ---")
    tool_input = "AMD最新的CPU型号是什么"

    reactAgent.run(question=tool_input)
