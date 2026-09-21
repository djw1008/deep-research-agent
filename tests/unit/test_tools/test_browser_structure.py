from __future__ import annotations

import pytest

from deep_research.tools.browser import BrowserBatchTool, BrowserTool


class _Response:
    status_code = 200
    headers = {"Content-Type": "text/html; charset=utf-8"}

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_browser_does_not_truncate_before_compression(monkeypatch) -> None:
    long_tail = "relevant-tail " * 900
    html = f"""
    <html><head><title>Structured page</title></head><body><main>
      <h1>Model results</h1>
      <ul><li>First item</li><li>Second item</li></ul>
      <table><tr><th>Model</th><th>Score</th></tr>
      <tr><td>Qwen2.5</td><td>84.2</td></tr></table>
      <p>{long_tail}</p>
    </main></body></html>
    """
    monkeypatch.setattr("deep_research.tools.browser.requests.get", lambda *a, **k: _Response(html))

    result = BrowserTool()._fetch("https://example.com/report")

    assert "Model results" in result["content"]
    assert "First item" in result["content"]
    assert "Qwen2.5" in result["content"]
    assert "84.2" in result["content"]
    assert "relevant-tail" in result["content"]
    assert len(result["content"]) > 8000


@pytest.mark.asyncio
async def test_browser_batch_deduplicates_and_caps_urls() -> None:
    batch = BrowserBatchTool(BrowserTool(mock_mode=True), max_urls=3)

    result = await batch.execute([
        "https://a.example",
        "https://a.example",
        "https://b.example",
        "https://c.example",
        "https://d.example",
    ])

    assert result["requested"] == 4
    assert result["fetched"] == 3
    assert [page["url"] for page in result["results"]] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]
