"""Calculator — 轻量数学计算。"""

import ast
import operator
from typing import Any


class CalculatorTool:
    name = "calculator"

    @staticmethod
    def get_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "轻量数学计算（+ - * / **），用于简单数值运算。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式，如 '2 + 3 * 4'"},
                    },
                    "required": ["expression"],
                },
            },
        }

    # 允许的操作符
    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    async def execute(self, expression: str) -> dict[str, Any]:
        """安全 eval：只允许数值运算。"""
        try:
            result = self._safe_eval(expression)
            return {"expression": expression, "result": result, "error": None}
        except Exception as e:
            return {"expression": expression, "result": None, "error": str(e)}

    def _safe_eval(self, expr: str) -> float:
        node = ast.parse(expr.strip(), mode="eval")
        return self._eval_node(node.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = self._operators[type(node.op)]
            return op(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op = self._operators[type(node.op)]
            return op(self._eval_node(node.operand))
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")
