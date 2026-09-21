"""ArXiv Reader — ArXiv 论文元数据检索（真实 API）。"""

import asyncio
import xml.etree.ElementTree as ET
from typing import Any

import requests


class ArxivReaderTool:
    name = "arxiv_reader"
    description = "学术论文检索（ArXiv）。涉及论文、publication、学术引用时使用。"

    @staticmethod
    def get_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "arxiv_reader",
                "description": "在 ArXiv 检索学术论文，返回标题、作者、摘要和 PDF 链接。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "max_results": {"type": "integer", "description": "最多返回论文数（默认 3）"},
                    },
                    "required": ["query"],
                },
            },
        }

    _API_URL = "http://export.arxiv.org/api/query"
    _NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode

    async def execute(self, query: str, max_results: int = 3) -> dict[str, Any]:
        """使用 ArXiv API 查询论文（mock_mode=True 时返回预设数据）。"""
        if self.mock_mode:
            return {
                "papers": [
                    {
                        "title": f"Mock Paper: {query} — Advances in AI",
                        "authors": ["Alice Mock", "Bob Mock"],
                        "published": "2024-01-15",
                        "summary": f"This is a mock arxiv abstract for query '{query}'...",
                        "pdf_url": f"https://arxiv.org/pdf/2401.{i:05d}.pdf",
                    }
                    for i in range(min(max_results, 2))
                ]
            }

        try:
            papers = await asyncio.to_thread(self._search, query, max_results)
            return {"papers": papers}
        except Exception as e:
            return {
                "error": f"ArXiv search failed: {type(e).__name__}: {e}",
                "retry_exhausted": True,
            }

    def _search(
        self, query: str, max_results: int, max_retries: int = 2
    ) -> list[dict[str, Any]]:
        import time

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        }
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(self._API_URL, params=params, timeout=15)
                resp.raise_for_status()
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
        else:
            raise last_error  # type: ignore[misc]

        root = ET.fromstring(resp.content)
        papers: list[dict[str, Any]] = []

        for entry in root.findall("atom:entry", self._NAMESPACE):
            title = entry.findtext("atom:title", default="", namespaces=self._NAMESPACE)
            summary = entry.findtext("atom:summary", default="", namespaces=self._NAMESPACE)
            published = entry.findtext("atom:published", default="", namespaces=self._NAMESPACE)

            authors = []
            for author in entry.findall("atom:author", self._NAMESPACE):
                name = author.findtext("atom:name", default="", namespaces=self._NAMESPACE)
                if name:
                    authors.append(name)

            pdf_url = ""
            paper_id = ""
            for link in entry.findall("atom:link", self._NAMESPACE):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                if link.get("rel") == "alternate" and link.get("type") == "text/html":
                    paper_id = link.get("href", "")

            # 清理 title 中的换行和多余空格
            title = " ".join(title.split())

            papers.append({
                "title": title,
                "authors": authors,
                "published": published[:10] if published else "",
                "summary": summary.strip(),
                "pdf_url": pdf_url or (paper_id.replace("abs", "pdf") + ".pdf" if paper_id else ""),
            })

        return papers
