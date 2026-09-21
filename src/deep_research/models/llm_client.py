"""LLM 客户端封装 — 统一调用 OpenAI 兼容 API。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    """LLM 返回的标准化响应。"""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    reasoning_content: Optional[str] = None
    model: str = ""
    usage: Optional[dict] = None


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class LLMClient:
    """
    封装 OpenAI 兼容 API 调用。

    职责：
      - 统一发请求（chat.completions.create）
      - 消息格式清洗与合并（防 400）
      - 上下文截断（保留 system + 最近交互）
      - 工具调用解析（原生 + 正则回退）
      - 错误分类处理
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any | None = None
        self.tools: list[dict] | None = None

    def set_tools(self, tools: list[dict] | None) -> None:
        """设置工具 schema，下次 chat 调用时自动传入。"""
        self.tools = tools

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            import httpx

            http_client = httpx.Client(
                transport=httpx.HTTPTransport(verify=False),  # 绕过 SSL + 不走系统代理
                timeout=self.timeout,
            )

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=http_client,
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """发送聊天请求，返回标准化 LLMResponse。"""
        sanitized = self._sanitize_messages(messages)
        sanitized = self._truncate_messages(sanitized, max_chars=35000)

        call_kwargs = dict(
            model=self.model_name,
            messages=sanitized,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        tool_choice = kwargs.get("tool_choice")
        effective_tools = tools if tools is not None else self.tools
        if effective_tools and tool_choice != "none":
            call_kwargs["tools"] = effective_tools
            call_kwargs["tool_choice"] = tool_choice or "auto"

        try:
            resp = self._get_client().chat.completions.create(**call_kwargs)
            raw_msg = resp.choices[0].message
            content = raw_msg.content or ""

            tool_calls = self._parse_tool_calls(content, raw_msg.tool_calls)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                reasoning_content=getattr(raw_msg, "reasoning_content", None),
                model=self.model_name,
                usage=getattr(resp, "usage", None),
            )

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()

            if "maximum context length" in err_lower or "context length" in err_lower:
                n_msgs = len(messages)
                total_chars = sum(
                    len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)
                )
                raise RuntimeError(
                    f"[上下文超限] n_msgs={n_msgs}, est_chars={total_chars}: {err_str}"
                ) from e

            # API 错误（402/401/429/5xx 等）统一向上抛出，由上层 Agent/Planner 处理失败逻辑
            raise RuntimeError(f"[LLM 调用失败] {err_str}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sanitize_messages(self, messages: list[dict]) -> list[dict]:
        """清洗消息格式，合并连续同角色消息。"""
        sanitized: list[dict] = []

        for m in messages:
            role, content = "user", ""
            if isinstance(m, dict):
                role, content = m.get("role", "user"), m.get("content", "")
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                role = "user" if m[0] in ("observation", "user") else "assistant"
                content = str(m[1])

            if "task=Task(" in str(content):
                continue

            new_msg = {"role": role, "content": str(content)}
            if role == "assistant" and m.get("tool_calls"):
                new_msg["tool_calls"] = m["tool_calls"]
            if role == "assistant" and m.get("reasoning_content"):
                new_msg["reasoning_content"] = m["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = m.get("tool_call_id", "")
                new_msg["name"] = m.get("name", "")

            can_merge = (
                sanitized
                and sanitized[-1]["role"] == role
                and role in ("user", "assistant")
                and "tool_calls" not in sanitized[-1]
                and "tool_calls" not in new_msg
                and "tool_call_id" not in new_msg
            )
            if can_merge:
                sanitized[-1]["content"] += "\n" + str(content)
            else:
                sanitized.append(new_msg)

        return sanitized

    def _truncate_messages(self, messages: list[dict], max_chars: int = 35000) -> list[dict]:
        """主动截断：丢弃旧轮次，保留 system + 最近交互。"""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        def _count_chars(msgs):
            total = 0
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                total += len(str(m.get("content", "")))
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        func = tc.get("function", {})
                        total += len(str(func.get("arguments", "")))
                        total += len(str(func.get("name", "")))
                if m.get("role") == "tool":
                    total += len(str(m.get("tool_call_id", "")))
                    total += len(str(m.get("name", "")))
            return total

        before_chars = _count_chars(messages)
        if before_chars <= max_chars:
            return messages

        kept = list(other_msgs)
        while len(kept) > 3:
            removed = kept.pop(0)
            if removed.get("role") == "assistant" and removed.get("tool_calls"):
                while kept and kept[0].get("role") == "tool":
                    kept.pop(0)
            if _count_chars(system_msgs + kept) <= max_chars:
                return system_msgs + kept

        after_chars = _count_chars(system_msgs + kept)
        if after_chars > max_chars and kept:
            last = kept[-1]
            excess = after_chars - max_chars
            content = str(last.get("content", ""))
            new_len = max(len(content) - excess - 100, 500)
            last["content"] = content[:new_len] + "\n[CONTENT_TRUNCATED]"
            return system_msgs + kept

        return system_msgs + kept

    def _parse_tool_calls(self, content: str, raw_tool_calls: Any) -> list[dict]:
        """解析工具调用：原生优先，正则回退。"""
        final: list[dict] = []

        if raw_tool_calls:
            for tc in raw_tool_calls:
                final.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            return final

        if "<tool_call>" in content:
            for i, m_str in enumerate(_TOOL_CALL_PATTERN.findall(content)):
                try:
                    d = json.loads(m_str.strip())
                    final.append(
                        {
                            "id": f"manual_{i}",
                            "type": "function",
                            "function": {
                                "name": d.get("name"),
                                "arguments": json.dumps(d.get("arguments", {})),
                            },
                        }
                    )
                except Exception:
                    continue

        return final
