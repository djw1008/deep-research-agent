"""Notepad — 草稿笔记。"""

from typing import Any


class NotepadTool:
    name = "notepad"

    def __init__(self) -> None:
        self._notes: list[dict[str, Any]] = []

    @staticmethod
    def get_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "notepad",
                "description": "记录中间笔记或读取已记录的笔记，支持按标签筛选。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["write", "read", "clear"],
                            "description": "write=写入笔记, read=读取笔记, clear=清空所有笔记",
                        },
                        "content": {"type": "string", "description": "写入的笔记内容（action=write 时必填）"},
                        "tag": {"type": "string", "description": "笔记标签，用于分类和筛选"},
                    },
                    "required": ["action"],
                },
            },
        }

    async def execute(self, action: str = "write", content: str = "", tag: str = "") -> dict[str, Any]:
        """Mock：记录或读取笔记。"""
        if action == "write":
            note = {"id": len(self._notes) + 1, "tag": tag, "content": content}
            self._notes.append(note)
            return {"action": "write", "note_id": note["id"]}

        if action == "read":
            if tag:
                notes = [n for n in self._notes if n["tag"] == tag]
            else:
                notes = list(self._notes)
            return {"action": "read", "notes": notes}

        if action == "clear":
            self._notes.clear()
            return {"action": "clear"}

        return {"action": action, "error": f"Unknown action: {action}"}
