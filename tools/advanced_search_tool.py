import os
from typing import List, Optional, Dict, Any
from tool_registry import ToolRegistry
from tools.tool import Tool, ToolParameter

class MyAdvancedSearchTool(Tool):
    """
    自定义高级搜索工具
    """

    def __init__(self):
        self.name = "my_advanced_search"
        self.description = "智能搜索工具，支持多个搜索源，自动选择最佳结果"
        super().__init__(self.name, self.description)
        self.search_sources = []
        self._setup_search_sources()

    def _setup_search_sources(self):
        """
        设置搜索源
        """
        if os.getenv("TRAILY_API_KEY"):
            try:
                from travily import TravilyClient
                self.travil_client = TravilyClient(api_key=os.getenv("TRAILY_API_KEY"))
                self.search_sources.append("travily")
                print("Travil search source enabled.")
            except ImportError:
                print("Travil package not installed. Skipping Travily search source.")

        if os.getenv("SERPAPI_API_KEY"):
            try:
                import serpapi
                self.search_sources.append("serpapi")
                print("SerpAPI search source enabled.")
            except ImportError:
                print("SerpAPI package not installed. Skipping SerpAPI search source.")

        if self.search_sources:
            print(f"Enabled search sources: {', '.join(self.search_sources)}")
        else:
            print("No search sources enabled. Please set the appropriate API keys.")

    def run(self, query: str) -> str:
        if not query.strip():
            return "Error: Query cannot be empty."

        if not self.search_sources:
            return "Error: No search sources are enabled. Please set the appropriate API keys."

        print(f"Searching for: {query}")

        for source in self.search_sources:
            try:
                if source == "travily":
                    result = self._search_with_travily(query)
                    if result and "未找到" not in result:
                        return result
                elif source == "serpapi":
                    result = self._search_with_serpapi(query)
                    if result:
                        return result
            except Exception as e:
                print(f"Error searching with {source}: {e}")
                continue
        return "No results found from any search source."

    def _search_with_travily(self, query: str) -> str:
        """
        使用Travily进行搜索
        """
        response = self.travil_client.search(
            query = query, 
            max_results = 3
        )

        if response.get('answer'):
            result = f"Answer: {response['answer']}\n\n"
        else:
            result = ""

        result += "related results:\n"
        for i, item in enumerate(response.get('related_results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"     {item.get('content', '')[:150]}...\n\n"

        return result

    def _search_with_serpapi(self, query: str) -> str:
        """
        使用SerpAPI进行搜索
        """
        import serpapi
        response = serpapi.search({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 3
        })

        results = response.get_dict() if hasattr(response, "get_dict") else response

        result = "Google Search Results:\n"
        if "organic_results" in results:
            for i, res in enumerate(results["organic_results"][:3], 1):
                result += f"[{i}] {res.get('title', '')}\n"
                result += f"     {res.get('snippet', '')[:150]}...\n\n"

        return result

    def get_parameters(self) -> List[ToolParameter]:
        """
        获取工具所需的参数列表
        """
        return [
            ToolParameter(
                name="query",
                description="The search query string.",
                type="string",
                required=True
            )
        ]

def create_advanced_search_tool_registry():
    """
    创建高级搜索工具注册表
    """
    registry = ToolRegistry()
    search_tool = MyAdvancedSearchTool()

    registry.register_function(
        name = "my_advanced_search_tool",
        description = search_tool.description,
        func = search_tool.search
    )

    return registry



