"""Code Sandbox — Python 代码沙箱执行。"""

from typing import Any


class CodeSandboxTool:
    name = "code_sandbox"

    @staticmethod
    def get_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "code_sandbox",
                "description": "执行 Python 代码，用于复杂计算、数据分析、统计等任务。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "要执行的 Python 代码"},
                    },
                    "required": ["code"],
                },
            },
        }

    async def execute(self, code: str) -> dict[str, Any]:
        """Mock：返回预设的执行结果。"""
        return {
            "stdout": f"[Mock] Executed code:\n{code[:100]}...",
            "stderr": "",
            "return_code": 0,
            "execution_time": 0.1,
        }
