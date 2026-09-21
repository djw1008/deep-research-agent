"""Browser — 网页阅读器（真实网页抓取）。"""

import asyncio
import os
import re
from typing import Any

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


class _TextExtractor:
    """把 DOM 序列化成带结构的文本行：段落成块、标题/列表/表格保留层次。"""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._buf: list[str] = []

    def flush(self) -> None:
        if self._buf:
            line = re.sub(r"\s{2,}", " ", " ".join(self._buf)).strip()
            if line:
                self.lines.append(line)
            self._buf = []

    def add_text(self, raw: str) -> None:
        text = re.sub(r"\s+", " ", str(raw))
        if text.strip():
            self._buf.append(text.strip())

    def add_line(self, line: str) -> None:
        self.flush()
        if line.strip():
            self.lines.append(line.strip())


class BrowserTool:
    name = "browser"
    description = "打开 URL 提取正文。搜索结果太短时，读取原文深度分析。"

    @staticmethod
    def get_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "browser",
                "description": "打开指定 URL 抓取网页正文内容，用于深度阅读搜索结果原文。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "要抓取的网页 URL"},
                    },
                    "required": ["url"],
                },
            },
        }

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    _SKIP_TAGS = (
        "script", "style", "nav", "header", "footer", "aside", "noscript",
        "template", "svg", "canvas", "iframe", "form", "button", "select",
        "textarea", "input", "video", "audio",
    )
    # 常见噪声区块的 class/id 关键词：广告、订阅、分享、评论区等
    _NOISE_RE = re.compile(
        r"cookie|newsletter|subscri|social|share-|sharing|related|recommend"
        r"|comment|modal|popup|breadcrumb|pagination|advert|promo|sponsor|signup",
        re.I,
    )
    _CONTENT_SELECTORS = (
        "article", "main", "[role='main']", "#main-content", "#content",
        ".post-content", ".article-content", ".entry-content", ".post",
        ".article", ".content",
    )
    _BLOCK_TAGS = {
        "p", "div", "section", "article", "main", "figure", "figcaption",
        "dl", "dt", "dd", "center", "address", "details", "summary",
    }
    _HEADING_RE = re.compile(r"h([1-6])")

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode

    async def execute(self, url: str) -> dict[str, Any]:
        """抓取 URL 并提取正文内容（mock_mode=True 时返回预设数据）。"""
        if self.mock_mode:
            return {
                "url": url,
                "title": f"Mock page title for {url}",
                "content": f"This is mock extracted content from {url}. "
                           f"It contains relevant information for the research task.",
                "status": 200,
            }

        # 检测 PDF URL，提前返回有意义的提示
        if url.lower().endswith(".pdf") or "/pdf/" in url.lower():
            return {
                "url": url,
                "title": "PDF 文件",
                "content": "[browser 无法解析 PDF 文件内容。请改用 arxiv_reader 或直接访问 HTML 版本。]",
                "status": 200,
            }

        try:
            result = await asyncio.to_thread(self._fetch, url)
            return result
        except Exception as e:
            return {
                "error": f"Browser failed: {type(e).__name__}: {e}",
                "url": url,
                "retry_exhausted": True,
            }

    def _fetch(self, url: str, max_retries: int = 2) -> dict[str, Any]:
        import time

        proxies = {
            "http": os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
            "https": os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
        }
        proxies = {k: v for k, v in proxies.items() if v}

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(
                    url, headers=self._HEADERS, timeout=20,
                    proxies=proxies if proxies else None,
                    verify=True,
                )
                resp.raise_for_status()
                break
            except requests.exceptions.SSLError as e:
                # SSL 错误时回退到不验证证书
                try:
                    resp = requests.get(
                        url, headers=self._HEADERS, timeout=20,
                        proxies=proxies if proxies else None,
                        verify=False,
                    )
                    resp.raise_for_status()
                    break
                except Exception as e2:
                    last_err = e2
                    if attempt < max_retries:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        raise last_err
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    raise last_err

        soup = BeautifulSoup(resp.text, "html.parser")
        self._strip_noise(soup)

        main = self._find_main(soup)
        text = self._extract_text(main)
        if not text:
            text = (main or soup).get_text(separator="\n", strip=True)

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        return {
            "url": url,
            "title": title,
            "content": text,
            "status": resp.status_code,
        }

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        for tag in soup.find_all(self._SKIP_TAGS):
            tag.decompose()
        for tag in list(soup.find_all(True)):
            if tag.parent is None:
                continue
            style = re.sub(r"\s+", "", str(tag.get("style", ""))).lower()
            ident = " ".join(str(c) for c in tag.get("class", [])) + " " + str(tag.get("id", ""))
            if (
                tag.has_attr("hidden")
                or str(tag.get("aria-hidden", "")).lower() == "true"
                or "display:none" in style
                or "visibility:hidden" in style
                or self._NOISE_RE.search(ident)
            ):
                tag.decompose()

    def _find_main(self, soup: BeautifulSoup) -> Any:
        for selector in self._CONTENT_SELECTORS:
            node = soup.select_one(selector)
            if node is not None and node.get_text(strip=True):
                return node
        return soup.body or soup

    def _extract_text(self, root: Any) -> str:
        extractor = _TextExtractor()
        self._walk(root, extractor)
        extractor.flush()
        # 折叠连续重复行：响应式布局常在 DOM 里放两份相同内容
        lines: list[str] = []
        for line in extractor.lines:
            if not line or (lines and lines[-1] == line):
                continue
            lines.append(line)
        return "\n".join(lines)

    def _walk(self, node: Any, ex: _TextExtractor, list_depth: int = 0) -> None:
        if isinstance(node, NavigableString):
            ex.add_text(str(node))
            return
        if not isinstance(node, Tag):
            return
        name = node.name or ""
        if name in self._SKIP_TAGS:
            return
        if name == "table":
            ex.flush()
            ex.lines.extend(self._table_lines(node))
            return
        if name == "pre":
            ex.flush()
            ex.lines.append("```")
            ex.lines.extend(node.get_text().strip("\n").splitlines())
            ex.lines.append("```")
            return
        if name == "br":
            ex.flush()
            return
        if name == "hr":
            ex.add_line("---")
            return
        if name == "a":
            text = node.get_text(" ", strip=True)
            href = str(node.get("href", ""))
            if text:
                ex.add_text(text)
            elif href.startswith("http"):
                ex.add_text(href)
            return
        heading = self._HEADING_RE.fullmatch(name)
        if heading:
            inner = _TextExtractor()
            for child in node.children:
                self._walk(child, inner, list_depth)
            inner.flush()
            title = " ".join(inner.lines).strip()
            if title:
                ex.add_line("#" * int(heading.group(1)) + " " + title)
            return
        if name in ("ul", "ol"):
            ex.flush()
            for child in node.children:
                self._walk(child, ex, list_depth + 1)
            ex.flush()
            return
        if name == "li":
            inner = _TextExtractor()
            for child in node.children:
                self._walk(child, inner, list_depth)
            inner.flush()
            if inner.lines:
                ex.add_line("  " * max(list_depth - 1, 0) + "- " + inner.lines[0])
                ex.lines.extend(inner.lines[1:])
            return
        if name == "blockquote":
            inner = _TextExtractor()
            for child in node.children:
                self._walk(child, inner, list_depth)
            inner.flush()
            for line in inner.lines:
                ex.add_line("> " + line)
            return
        if name in self._BLOCK_TAGS:
            ex.flush()
            for child in node.children:
                self._walk(child, ex, list_depth)
            ex.flush()
            return
        for child in node.children:
            self._walk(child, ex, list_depth)

    def _table_lines(self, table: Tag) -> list[str]:
        """把 table 渲染成 Markdown 表格，保留行列结构而不是拍平成字符流。"""
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue  # 跳过嵌套表格的行，外层单元格文本已包含其内容
            cells: list[str] = []
            for cell in tr.find_all(["th", "td"], recursive=False):
                text = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).replace("|", "\\|")
                try:
                    span = int(str(cell.get("colspan", "1")))
                except ValueError:
                    span = 1
                cells.extend([text] * max(1, min(span, 12)))
            if any(cells):
                rows.append(cells)
        if not rows:
            return []
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        if width > 20:
            # 超宽表格退化为逐行输出，避免 Markdown 表格撑爆
            return [" | ".join(cell for cell in row if cell) for row in rows]
        header, body = rows[0], rows[1:]
        lines = ["| " + " | ".join(header) + " |", "|" + " ---|" * width]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return lines


class BrowserBatchTool:
    """Fetch a small URL batch concurrently through the existing browser tool."""

    name = "browser_batch"
    description = "并发打开多个 URL 并提取正文，适合批量核对搜索结果。"

    def __init__(self, browser: BrowserTool, max_urls: int = 3) -> None:
        self.browser = browser
        self.max_urls = max_urls

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"并发读取多个网页正文；每次最多 {self.max_urls} 个 URL。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": self.max_urls,
                            "description": "要读取的 URL 列表",
                        }
                    },
                    "required": ["urls"],
                },
            },
        }

    async def execute(self, urls: list[str]) -> dict[str, Any]:
        unique_urls = list(dict.fromkeys(str(url).strip() for url in urls if str(url).strip()))
        selected = unique_urls[: self.max_urls]
        if not selected:
            return {"error": "browser_batch requires at least one URL"}
        results = await asyncio.gather(*(self.browser.execute(url) for url in selected))
        return {"results": results, "requested": len(unique_urls), "fetched": len(results)}
