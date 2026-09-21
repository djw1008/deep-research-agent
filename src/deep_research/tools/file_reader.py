"""File Reader — 本地文件阅读。"""

from typing import Any


class FileReaderTool:
    name = "file_reader"

    @staticmethod
    def get_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "file_reader",
                "description": "读取本地文件内容（txt/pdf/csv/json/docx）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件绝对路径或相对路径"},
                    },
                    "required": ["path"],
                },
            },
        }

    async def execute(self, path: str) -> dict[str, Any]:
        """Mock：返回预设的文件内容。"""
        return {
            "path": path,
            "content": f"[Mock] Content of file '{path}'\nLine 1: Mock data\nLine 2: Research notes",
            "format": path.split(".")[-1] if "." in path else "txt",
        }
