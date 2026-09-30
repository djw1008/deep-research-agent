"""
合成 Agent (SummarizerAgent)

将多个 SubTask 的执行结果合成为结构化的研究报告。
区别于 ResearcherAgent 的多轮 tool-calling，Summarizer 是单轮长上下文生成任务：
  - 把所有子结果按置信度排序后拼接为上下文
  - 调用 LLM 一次性生成 Markdown 格式报告
  - 提取引用来源，计算整体置信度
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from ..core.schema import AgentResult, AgentStatus, ResearchReport, SubTask
from ..core.report_content import (
    prepare_sources,
    remove_invalid_citations,
    strip_reference_sections,
)
from .base_agent import BaseAgent


logger = logging.getLogger(__name__)

__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    单轮长上下文生成，禁用 tools 避免模型进入 tool-calling 模式。
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        session_memory=None,
    ) -> None:
        super().__init__(name, policy, tools)
        self.session_memory = session_memory

    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行合成任务。

        Args:
            task: 通常是一个特殊的 "synthesize" 类型任务。
            context: 合成上下文，必须包含 "results" 和 "query" 键。
                results: list[AgentResult]
                query: str 原始研究问题

        Returns:
            AgentResult，output 字段为 ResearchReport 实例。
        """
        query = context.get("query", "")
        results: list[AgentResult] = context.get("results", [])

        if not results:
            report = ResearchReport(
                query=query,
                content="无可用子任务结果进行合成。",
                confidence=0.0,
            )
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.FAILED,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        # 来源必须在合成前完成注册和编号，模型只能引用这些既有编号。
        sources = self._collect_sources(results)
        prompt = self._build_synthesis_prompt(query, results, sources)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            # 合成任务不需要工具调用，临时禁用 tools
            old_tools = getattr(self.policy, "tools", None)
            if hasattr(self.policy, "set_tools"):
                self.policy.set_tools(None)

            resp = await asyncio.to_thread(self.policy.chat, messages)

            if hasattr(self.policy, "set_tools") and old_tools is not None:
                self.policy.set_tools(old_tools)
        except RuntimeError as e:
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.FAILED,
                output=str(e),
                trajectory=[{"error": str(e)}],
                token_usage=0,
                confidence=0.0,
            )

        content = resp.content or ""
        token_usage = len(content) // 3  # 简化估算

        # 解析报告内容，提取来源和置信度
        report = self._parse_report(query, content, results, sources)

        result = AgentResult(
            task_id=task.id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[{"role": "assistant", "content": content}],
            token_usage=token_usage,
            confidence=report.confidence,
        )

        return result

    def _system_prompt(self) -> str:
        return (
            "你是一位 expert research synthesizer。"
            "你的任务是将多个研究发现整合成一篇连贯、结构清晰的研究报告。"
            "不要描述你将要做什么——直接输出合成后的报告。\n\n"
            "<format>\n"
            "1. 使用 Markdown 格式。来源清单由程序根据检索记录统一生成；正文不得创建"
            "『引用来源』『参考文献』『参考链接』『Sources』等章节。正文需要来源支持时，"
            "只能使用程序提供的数字引用，如 [1]；不得自行创建不存在的编号。\n"
            "2. 报告正文必须至少 3000 个中文字符（或 2000 个英文单词）。\n"
            "3. 结构：执行摘要 → 背景 → 关键发现（附细节）→ 分析 → 比较 → 影响 → 结论。\n"
            "4. 静默解决来源之间的矛盾：直接输出最终结论；数据有分歧时在行内标注"
            "（如「84.1–89.0，来源存在分歧」或「存疑」），不要描述解决过程。\n"
            "5. 不要在正文末尾列出来源；程序会统一追加唯一的参考链接列表。\n"
            "6. 【强制】报告最末尾必须单独写一行：Overall Confidence: 0.XX（0-1 之间的数字）。不可省略。\n"
            "</format>\n\n"
            "<audience>\n"
            "报告面向最终读者，读者看不到也不关心你的内部工作过程。严禁在报告中出现：\n"
            "- 任何内部编号或标签：「材料 N」「Result N」「子任务」「验证任务」「task_」等；\n"
            "- 任何关于你如何整合、核对、处理矛盾的章节或段落（如「来源矛盾的解决」「处理方式」）；\n"
            "- 任何「本报告采用……标注」之类的元叙述。\n"
            "矛盾的结论必须直接体现在正文措辞里（区间、存疑标注、弱化表述），而不是单独解释。\n"
            "</audience>"
        )

    def _build_synthesis_prompt(
        self,
        query: str,
        results: list[AgentResult],
        sources: list[dict[str, Any]] | None = None,
    ) -> str:
        """构建合成 prompt，按置信度降序排列结果。

        内容完全相同的子结果只保留一份（Planner 可能拆出重复任务），
        避免模型把重复确认误读为多方独立佐证，或在报告中解释重复现象。
        """
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        def _output_text(r: AgentResult) -> str:
            return r.output if isinstance(r.output, str) else json.dumps(r.output, ensure_ascii=False, default=str)

        seen_outputs: dict[str, int] = {}
        unique_results: list[AgentResult] = []
        duplicate_of: dict[int, int] = {}
        for r in sorted_results:
            key = re.sub(r"\s+", "", _output_text(r))
            if key in seen_outputs:
                duplicate_of[id(r)] = seen_outputs[key]
                continue
            seen_outputs[key] = len(unique_results) + 1
            unique_results.append(r)
        dup_counts: dict[int, int] = {}
        for target in duplicate_of.values():
            dup_counts[target] = dup_counts.get(target, 0) + 1

        sources = sources if sources is not None else self._collect_sources(results)
        source_id_by_url = {source["url"]: source["citation_id"] for source in sources}
        parts = [
            f"# Research Question\n{query}\n",
            f"# 研究材料（共 {len(unique_results)} 份）\n",
        ]
        for i, r in enumerate(unique_results, 1):
            status_icon = "✓" if r.status == AgentStatus.SUCCESS else "✗"
            dup_note = f"（另有 {dup_counts[i]} 份材料内容与此完全相同，视为同一来源的重复确认）\n" if dup_counts.get(i) else ""
            material_ids = sorted({
                source_id_by_url[url]
                for url in self._source_urls(r)
                if url in source_id_by_url
            })
            citation_note = (
                "该材料关联的可用引用：" + ", ".join(f"[{sid}]" for sid in material_ids) + "\n"
                if material_ids else "该材料没有已注册来源，不得为其中结论虚构引用。\n"
            )
            parts.append(
                f"## 材料 {i} [{status_icon}] (confidence: {r.confidence:.2f})\n"
                f"{dup_note}"
                f"{citation_note}"
                f"内容：\n{_output_text(r)}\n"
            )

        parts.append("\n# 可用引用注册表\n")
        if sources:
            for source in sources:
                parts.append(
                    f"[{source['citation_id']}] {source.get('title', '')}\n"
                    f"URL: {source['url']}\n"
                    f"摘要: {source.get('snippet', '')}\n"
                )
        else:
            parts.append("无已注册来源。正文不得生成数字引用。\n")

        parts.append(
            "\n# Instructions\n"
            "1. 【强制】报告最末尾必须单独写一行：Overall Confidence: 0.XX（根据材料置信度和信息完整度给出一个0-1之间的数字，不要省略）。这一行必须在报告正文全部结束后另起一行单独出现。\n"
            "2. 直接基于上述材料撰写综合报告，不要说'我将进行合成'。\n"
            "3. 报告必须全面且详细（至少 3000 中文字符或 2000 英文单词）。\n"
            "4. 结构：执行摘要 → 背景 → 关键发现（附细节）→ 分析 → 比较 → 影响 → 结论。\n"
            "5. 静默解决材料之间的矛盾：直接输出最终结论，数据有分歧时在行内标注（如「存疑」「来源存在分歧」），不要描述解决过程，不要为矛盾单设章节。\n"
            "6. 严禁在报告中引用内部标签（「材料 N」「Result N」「子任务」「验证任务」等），也不要提及材料的数量、重复情况或你的整合方式。\n"
            "7. 需要来源支持的事实，使用与该材料关联的已有编号 [N]；不得创造编号，"
            "不得在正文中直接写 URL，也不得生成参考文献章节。"
        )
        return "\n".join(parts)

    def _source_urls(self, result: AgentResult) -> list[str]:
        """Extract source URLs associated with one research result."""
        urls: list[str] = []
        for step in result.trajectory:
            if step.get("role") != "tool":
                continue
            res = step.get("result")
            items: list[dict] = []
            if isinstance(res, list):
                items = [item for item in res if isinstance(item, dict)]
            elif isinstance(res, dict):
                if isinstance(res.get("results"), list):
                    items = [item for item in res["results"] if isinstance(item, dict)]
                elif isinstance(res.get("papers"), list):
                    items = [item for item in res["papers"] if isinstance(item, dict)]
            for item in items:
                url = str(item.get("url") or item.get("pdf_url") or "").strip()
                if url:
                    urls.append(url)
        return urls

    def _collect_sources(self, results: list[AgentResult]) -> list[dict[str, Any]]:
        """Collect, filter, deduplicate and number sources before synthesis."""
        collected: list[dict[str, Any]] = []
        for result in results:
            if result.status != AgentStatus.SUCCESS:
                continue
            for step in result.trajectory:
                if step.get("role") != "tool":
                    continue
                res = step.get("result")
                items: list[dict] = []
                if isinstance(res, list):
                    items = [item for item in res if isinstance(item, dict)]
                elif isinstance(res, dict) and isinstance(res.get("results"), list):
                    items = [item for item in res["results"] if isinstance(item, dict)]
                elif isinstance(res, dict) and isinstance(res.get("papers"), list):
                    items = [item for item in res["papers"] if isinstance(item, dict)]
                for item in items:
                    url = str(item.get("url") or item.get("pdf_url") or "").strip()
                    if not url or self._is_noise_url(url):
                        continue
                    collected.append({
                        "url": url,
                        "title": item.get("title", ""),
                        "snippet": str(
                            item.get("snippet") or item.get("summary") or ""
                        )[:500],
                        "task_id": result.task_id,
                    })
        return prepare_sources(collected)

    _NOISE_DOMAINS = {
        # 社交媒体
        "tiktok.com", "instagram.com", "facebook.com", "twitter.com", "x.com",
        "youtube.com", "pinterest.com", "reddit.com",
        # 登录/认证页
        "login", "signin", "signup", "/auth", "accounts.google.com",
        "appleid.apple.com",
        # 应用商店
        "apps.microsoft.com", "play.google.com", "apps.apple.com",
        # 帮助/客服页
        "support.microsoft.com", "help.netflix.com", "support.google.com",
        # 电商/购物
        "oliveyoung", "amazon.com/dp", "shopify",
        # 地图/门票/其他
        "map.naver.com", "map.google.com", "e-tix.jp", "ticket",
        # 非技术类网站
        "niapune.org.in", "myut.ut.ac.id", "tasks.google.com",
        "cambridge.org/dictionary", "dictionary.cambridge.org",
        "yahoo.com/search", "tw.dictionary",
    }

    @classmethod
    def _is_noise_url(cls, url: str) -> bool:
        """判断 URL 是否属于噪音来源（社交媒体、登录页、应用商店等）。"""
        url_lower = url.lower()
        return any(b in url_lower for b in cls._NOISE_DOMAINS)

    def _parse_report(
        self,
        query: str,
        content: str,
        results: list[AgentResult],
        sources: list[dict[str, Any]] | None = None,
    ) -> ResearchReport:
        """从 LLM 输出中解析 ResearchReport，并基于子任务成功率校准置信度。"""
        # 1. 从文本中提取 LLM 自评置信度（支持中英文格式）
        # 如果 LLM 遗漏，fallback 到子任务平均置信度，而非硬编码 0.5
        success_results = [r for r in results if r.status == AgentStatus.SUCCESS]
        avg_sub_confidence = (
            sum(r.confidence for r in success_results) / len(success_results)
            if success_results else 0.5
        )
        llm_confidence = avg_sub_confidence
        patterns = [
            r"(?:overall\s+)?confidence[^\d]*?(0\.\d+|1\.0|1)",
            r"(?:整体|总体|综合)?置信度[^\d]*?(0\.\d+|1\.0|1)",
        ]
        for pat in patterns:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                try:
                    llm_confidence = float(m.group(1))
                    break
                except ValueError:
                    continue

        # 2. 基于子任务成功率计算客观置信度
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)

        # 3. 综合置信度 = LLM 自评 × 成功率开根（降低成功率的影响权重）
        confidence = llm_confidence * (success_rate ** 0.5)
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # 置信度行只是给程序解析用的，剥离后再交付给用户
        content = re.sub(
            r"(?:\n|^)\s*(?:#+\s*)?(?:Overall\s+Confidence|(?:整体|总体|综合)?置信度)\s*[：:][^\n]*\s*$",
            "",
            content,
            flags=re.IGNORECASE,
        ).rstrip()
        content = strip_reference_sections(content)
        sources = prepare_sources(sources if sources is not None else self._collect_sources(results))
        content = remove_invalid_citations(content, sources)

        # 统计实际工具调用次数
        num_searches = sum(
            len([t for t in r.trajectory if t.get("role") == "tool"])
            for r in results
        )

        return ResearchReport(
            query=query,
            content=content,
            sources=sources,
            confidence=confidence,
            num_searches=num_searches,
        )

    async def _persist_report(self, report: ResearchReport, context: dict) -> None:
        """Legacy hook for persisting the final report.

        Session Memory writing is now managed by the Orchestrator.
        """
        pass
