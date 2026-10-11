# my_calculator_tool.py
import ast
import operator
import math
from typing import List

from tool_registry import ToolRegistry
from tools.tool import Tool, ToolParameter

class calculator(Tool):
    def __init__(self):
        self.name = "my_calculator"
        self.description = "Simple math calculator that supports basic operators (+,-,*,/) and sqrt."
        super().__init__(self.name, self.description)

    def run(self, expression: str) -> str:
        """Simple math calculation function."""
        if not expression.strip():
            return "The calculation expression cannot be empty."

        # Supported basic operators.
        operators = {
            ast.Add: operator.add,      # +
            ast.Sub: operator.sub,      # -
            ast.Mult: operator.mul,     # *
            ast.Div: operator.truediv,  # /
        }

        # Supported basic functions.
        functions = {
            'sqrt': math.sqrt,
            'pi': math.pi,
        }

        try:
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body, operators, functions)
            return str(result)
        except:
            return "Calculation failed. Please check the expression format."

    def get_parameters(self) -> List[ToolParameter]:
        """获得工具所需的参数列表"""
        return [
            ToolParameter(
                name="expression",
                description="The mathematical expression to calculate.",
                type="string",
                required=True
            )
        ]

    def _eval_node(self, node, operators, functions):
        """分析AST节点并计算结果"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, operators, functions)
            right = self._eval_node(node.right, operators, functions)
            op = operators.get(type(node.op))
            return op(left, right)
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in functions:
                args = [self._eval_node(arg, operators, functions) for arg in node.args]
                return functions[func_name](*args)
        elif isinstance(node, ast.Name):
            if node.id in functions:
                return functions[node.id]

    def registry_calculator_tool(self):
        """注册计算器工具到工具注册表"""
        registry = ToolRegistry()
        registry.register_function(
            name="my_calculator",
            description="Simple math calculator that supports basic operators (+,-,*,/) and sqrt.",
            func=self.run
        )
        return registry

    def create_calculator_registry(self):
        return self.registry_calculator_tool()
