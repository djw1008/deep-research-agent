"""Web Search — Tavily AI Search（专为 AI Agent 设计）。"""

import asyncio
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


class WebSearchTool:
    name = "web_search"
    description = "通用网页搜索，返回标题、URL 和内容摘要。"

    @staticmethod
    def get_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "通用网页搜索，用于获取新闻、市场数据、行业报告、时事信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "num_results": {"type": "integer", "description": "返回结果数量（默认 5）"},
                    },
                    "required": ["query"],
                },
            },
        }

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode
        self._tavily_key = os.getenv("TAVILY_API_KEY")
        self._metaso_key = os.getenv("METASO_KEY")

    async def execute(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """执行网页搜索。Tavily 优先，秘塔兜底。"""
        if self.mock_mode:
            return [
                {
                    "title": f"Mock: {query} #{i+1}",
                    "url": f"https://example.com/{query.replace(' ', '-')}-{i+1}",
                    "snippet": f"Mock snippet for '{query}'...",
                }
                for i in range(min(num_results, 3))
            ]

        errors: list[str] = []

        # 1. Tavily（主力，专为 Agent 设计）
        if self._tavily_key:
            try:
                return await asyncio.to_thread(self._search_tavily, query, num_results)
            except Exception as e:
                logger.warning("Tavily 搜索失败: %s", e)
                errors.append(f"Tavily: {type(e).__name__}: {e}")

        # 2. 秘塔兜底
        if self._metaso_key:
            try:
                return await asyncio.to_thread(self._search_metaso, query, num_results)
            except Exception as e:
                logger.warning("秘塔搜索失败: %s", e)
                errors.append(f"Metaso: {type(e).__name__}: {e}")

        if errors:
            return [{
                "error": "Web search failed after internal retries: " + " | ".join(errors),
                "retry_exhausted": True,
            }]
        return [{"error": "未配置搜索 API Key (TAVILY_API_KEY / METASO_KEY)"}]

    # ------------------------------------------------------------------
    # Tavily
    # ------------------------------------------------------------------
    def _search_tavily(self, query: str, num_results: int) -> list[dict[str, Any]]:
        # 与 browser.py 保持一致的代理逻辑：读环境变量，有则用
        proxies = {
            "http": os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
            "https": os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
        }
        proxies = {k: v for k, v in proxies.items() if v}

        # 在 asyncio.to_thread 中 requests 偶发 ConnectionResetError，
        # 加两次重试 + 短暂退避
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    TAVILY_URL,
                    json={
                        "api_key": self._tavily_key,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": min(num_results, 10),
                        "include_answer": True,
                    },
                    proxies=proxies if proxies else None,
                    timeout=30,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    import time
                    time.sleep(1.0 * (attempt + 1))
        else:
            raise last_err  # type: ignore[misc]
        resp.raise_for_status()
        data = resp.json()

        results: list[dict[str, Any]] = []

        # 顶层 AI 摘要（Tavily 直接生成，省了 browser 那一步）
        answer = data.get("answer", "")
        if answer:
            results.append({
                "title": f"  AI 摘要: {query[:40]}",
                "url": "",
                "snippet": answer,
            })

        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:300],
            })

        return results[:num_results]

    # ------------------------------------------------------------------
    # 秘塔（兜底）
    # ------------------------------------------------------------------
    def _search_metaso(
        self, query: str, num_results: int, max_retries: int = 2
    ) -> list[dict[str, Any]]:
        import time

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    "https://metaso.cn/api/open/search/v2",
                    headers={
                        "Authorization": f"Bearer {self._metaso_key}",
                        "Content-Type": "application/json",
                    },
                    json={"question": query, "stream": False, "lang": "zh"},
                    timeout=30,
                )
                resp.raise_for_status()
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
        else:
            raise last_error  # type: ignore[misc]
        data = resp.json()

        err_code = data.get("errCode")
        if err_code is not None and err_code != 0:
            raise RuntimeError(f"秘塔错误: [{err_code}] {data.get('errMsg', '')}")

        inner = data.get("data", {})
        references = inner.get("references", [])

        results: list[dict[str, Any]] = []
        for ref in references[:num_results]:
            snippet = ""
            if ref.get("article_type"):
                snippet += ref["article_type"]
            if ref.get("date"):
                snippet += " | " + ref["date"]
            results.append({
                "title": ref.get("title", ""),
                "url": ref.get("link", ""),
                "snippet": snippet.strip(" | "),
            })

        answer_text = inner.get("text", "")
        if answer_text and results:
            results[0]["snippet"] = answer_text[:500] + " | " + results[0]["snippet"]

        return results
