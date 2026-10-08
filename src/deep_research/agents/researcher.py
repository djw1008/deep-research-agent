"""ResearchAgent — 执行单个子任务的 Agent（多轮 tool-calling）。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..compressor.context_compressor import ContextCompressor
from ..core.citations import normalize_source_citations
from ..core.schema import AgentResult, AgentStatus, SubTask
from ..observability import EventSink, ObservableTrajectory
from .base_agent import BaseAgent


# 模型上下文窗口映射（从 LLM 配置推断 budget）
_MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "claude-3-5-sonnet": 200000,
    "claude-3-5": 200000,
    "claude-3": 200000,
    "gemini-1.5": 1000000,
    "gemini-1.5-flash": 1000000,
    "qwen2.5": 128000,
}


class ResearchAgent(BaseAgent):
    """
    研究员 Agent：负责搜索、分析、验证类任务。

    核心设计：多轮 LLM ↔ 工具 交互循环。
      1. LLM 决定调用什么工具
      2. Agent 执行工具
      3. 工具结果回写给 LLM
      4. 重复直到 LLM 不再调用工具或达到 max_turns
    """

    # 表示用户关注当前/最近时间的查询关键词；命中时跳过记忆召回，避免返回过时结果。
    _TIME_SENSITIVE_KEYWORDS = (
        "最近", "现在", "当前", "最新", "今天", "本月", "今年", "今年来",
        "时下", "近期", "此刻", "目前", "刚刚", "前不久", "近阶段",
    )

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        max_turns: int = 10,
        config: dict | None = None,
        knowledge_base=None,
        event_sink: EventSink | None = None,
    ):
        super().__init__(name, policy, tools)
        self.max_turns = max_turns
        self.tool_map: dict[str, Any] = {t.name: t for t in (tools or [])}
        self.config = config or {}
        researcher_cfg = self.config.get("researcher", {})
        self.max_turns = int(researcher_cfg.get("max_llm_turns", max_turns))
        self.max_total_tool_calls = int(researcher_cfg.get("max_total_tool_calls", 8))
        self.max_web_search_calls = int(researcher_cfg.get("max_web_search_calls", 3))
        self.max_browser_calls = int(researcher_cfg.get("max_browser_calls", 4))
        self.knowledge_base = knowledge_base
        self.event_sink = event_sink
        # 从 yaml 配置读取 compressor 参数
        comp_cfg = self.config.get("compressor", {})
        self.browser_post_l2_max_chars = int(comp_cfg.get("browser_post_l2_max_chars", 8000))
        # 复用 knowledge_base 的 embedder，避免同时加载两个 sentence-transformers 模型。
        self.compressor = ContextCompressor(
            l2_threshold=comp_cfg.get("l2_threshold", 0.10),
            embedder=getattr(knowledge_base, "embedder", None),
        )

    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行 Researcher 任务。"""
        trajectory: list[dict] = ObservableTrajectory(task.id, self.event_sink)
        total_tokens: int = 0
        consecutive_empty: int = 0  # 连续空结果计数器
        consecutive_tool_failures: int = 0  # 工具内部重试耗尽后，按调用计数
        force_summary_after_failures: bool = False
        seen_search_urls: set[str] = set()
        source_labels: dict[str, str] = {}

        # 启发式判断：无法通过网络搜索获取答案的任务
        if self._is_non_searchable(task, context):
            return await self._run_direct_analysis(task, context, trajectory, total_tokens)

        # search 类型任务工具调用多、token 消耗大，执行前先尝试从记忆召回。
        # analyze/verify 通常依赖上游上下文，不单独召回，避免上下文缺失导致错误。
        # 若原始问题含"最近/现在/当前"等时效性词，跳过召回，直接执行子任务获取最新信息。
        if (
            task.task_type == "search"
            and self.knowledge_base is not None
            and not self._is_time_sensitive_query(context)
        ):
            recalled = await self._recall_from_memory(
                task, context.get("query", "")
            )
            if recalled is not None:
                return recalled

        task_prompt = self._build_task_prompt(task, context)
        messages = [
            {"role": "system", "content": self._system_prompt() + "\n\n【当前任务】\n" + task_prompt},
        ]

        for turn in range(self.max_turns):
            remaining = self.max_turns - turn
            is_final = remaining <= 2

            # 每轮根据已经使用的工具预算动态注册剩余工具。
            search_count = sum(1 for t in trajectory if t.get("role") == "tool" and t.get("name") == "web_search")
            browser_count = sum(
                1 for t in trajectory
                if t.get("role") == "tool" and t.get("name") in {"browser", "browser_batch"}
            )
            total_tool_calls = sum(1 for t in trajectory if t.get("role") == "tool")
            force_no_tools = (
                total_tool_calls >= self.max_total_tool_calls
                or is_final
                or force_summary_after_failures
            )

            if hasattr(self.policy, "set_tools"):
                if force_no_tools:
                    self.policy.set_tools(None)
                else:
                    schemas = []
                    for tool in self.tools or []:
                        if tool.name == "web_search" and search_count >= self.max_web_search_calls:
                            continue
                        if tool.name in {"browser", "browser_batch"} and browser_count >= self.max_browser_calls:
                            continue
                        schemas.append(getattr(tool, "get_schema", lambda: {
                            "type": "function", "function": {"name": tool.name}
                        })())
                    self.policy.set_tools(schemas or None)

            # 轮次感知提示（每轮都注入）
            if force_no_tools:
                messages.append({
                    "role": "user",
                    "content": (
                        f"[系统提示] 第 {turn + 1}/{self.max_turns} 轮。"
                        f"{'连续两次工具调用失败，已触发兜底总结。' if force_summary_after_failures else '工具预算已用完。'}"
                        "禁止再调用任何工具，必须基于已有信息和失败记录写出最终总结（中文，含置信度0-1）。"
                        f"【强制】禁止输出任何XML标签、工具调用格式或伪代码，只输出纯文本总结。"
                    ),
                })
            elif is_final:
                messages.append({
                    "role": "user",
                    "content": (
                        f"[系统提示] 第 {turn + 1}/{self.max_turns} 轮（最后阶段）。"
                        f"禁止调用任何工具，必须直接基于已有信息写出最终总结（中文，含置信度0-1）。"
                        f"【强制】禁止输出任何XML标签、工具调用格式或伪代码，只输出纯文本总结。"
                    ),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        f"[系统提示] 第 {turn + 1}/{self.max_turns} 轮，还剩 {remaining - 1} 轮。"
                        f"工具预算：总计剩余 {self.max_total_tool_calls - total_tool_calls} 次，"
                        f"web_search 剩余 {max(0, self.max_web_search_calls - search_count)} 次，"
                        f"browser/browser_batch 剩余 {max(0, self.max_browser_calls - browser_count)} 次。"
                        "可继续搜索、用 refined query 补充信息，或用 browser_batch 同时读取多个高价值来源；"
                        "如果不再调用工具，则直接输出最终总结。"
                    ),
                })

            # 调用 LLM（最后两轮或已搜索2次强制不给工具）
            try:
                if force_no_tools:
                    resp = await asyncio.to_thread(self.policy.chat, messages, tool_choice="none")
                else:
                    resp = await asyncio.to_thread(self.policy.chat, messages)
            except RuntimeError as e:
                trajectory.append({"turn": turn, "error": str(e)})
                return AgentResult(
                    task_id=task.id, status=AgentStatus.FAILED, output=str(e),
                    trajectory=trajectory, token_usage=total_tokens, confidence=0.0,
                )

            content = resp.content or ""
            tool_calls = resp.tool_calls or []
            tool_calls = self._limit_tool_calls(
                tool_calls,
                remaining_total=max(0, self.max_total_tool_calls - total_tool_calls),
                remaining_search=max(0, self.max_web_search_calls - search_count),
                remaining_browser=max(0, self.max_browser_calls - browser_count),
            )

            trajectory.append({"turn": turn, "role": "assistant", "content": content, "tool_calls": tool_calls})
            total_tokens += len(json.dumps(messages, ensure_ascii=False)) // 3

            # 无 tool_calls → 任务完成
            if not tool_calls:
                # 检测并处理 XML 伪工具调用污染
                if self._contains_pseudo_tool_calls(content):
                    if turn < self.max_turns - 1:
                        messages.append({
                            "role": "user",
                            "content": (
                                "你的回答中包含未执行的工具调用格式（XML/invoke标签）。"
                                "工具已被禁用，禁止输出任何工具调用格式。"
                                "请直接基于已有信息给出最终总结（中文，含置信度0-1）。"
                            ),
                        })
                        trajectory.append({
                            "turn": turn, "role": "system",
                            "content": "检测到XML伪工具调用，强制重试",
                        })
                        continue
                    else:
                        cleaned = self._strip_pseudo_tool_calls(content)
                        if not cleaned:
                            logging.warning(
                                "[%s] 最后一轮仅含工具调用，无有效总结", task.id
                            )
                            return AgentResult(
                                task_id=task.id, status=AgentStatus.FAILED,
                                output="达到最大轮次仍未获得最终总结。",
                                trajectory=trajectory, token_usage=total_tokens,
                                confidence=0.0,
                                metadata={
                                    "from_memory": False,
                                    "task_description": task.description,
                                },
                            )
                        logging.warning(
                            "[%s] 最后一轮仍含XML伪工具调用，清洗后返回", task.id
                        )
                        cleaned, metadata = self._prepare_result(
                            task, cleaned, trajectory, context, from_memory=False
                        )
                        return AgentResult(
                            task_id=task.id, status=AgentStatus.SUCCESS,
                            output=cleaned, trajectory=trajectory,
                            token_usage=total_tokens,
                            confidence=self._extract_confidence(cleaned),
                            metadata=metadata,
                        )

                content, metadata = self._prepare_result(
                    task, content, trajectory, context, from_memory=False
                )
                return AgentResult(
                    task_id=task.id, status=AgentStatus.SUCCESS, output=content,
                    trajectory=trajectory, token_usage=total_tokens,
                    confidence=self._extract_confidence(content),
                    metadata=metadata,
                )

            # 执行工具
            tool_results = []
            for tc in tool_calls:
                compression_info = None
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                raw_args = func.get("arguments", "{}")
                try:
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif isinstance(raw_args, str):
                        args = json.loads(raw_args) if raw_args.strip() else {}
                    else:
                        args = {}
                except Exception:
                    args = {}

                result = await self._execute_tool(tool_name, args)

                # browser：每次走 L2 段落级过滤（文章摘要刚需）
                if tool_name == "browser" and isinstance(result, dict) and "content" in result:
                    original_len = len(result["content"])
                    original_content = result["content"]
                    if original_len == 0:
                        logging.warning(
                            "[%s] browser 返回空内容，跳过压缩。URL: %s",
                            task.id, result.get("url", "unknown"),
                        )
                    else:
                        current_tokens = self._estimate_tokens(messages)
                        budget = self._get_budget()
                        result["content"] = self.compressor.l2(
                            result["content"],
                            query=task.description,
                            current_tokens=current_tokens,
                            budget=budget,
                        )
                        l2_len = len(result["content"])
                        safety_truncated = l2_len > self.browser_post_l2_max_chars
                        if safety_truncated:
                            result["content"] = (
                                result["content"][: self.browser_post_l2_max_chars]
                                + f"\n\n[压缩结果过长，已安全截断至 {self.browser_post_l2_max_chars} 字符]"
                            )
                        compressed_len = len(result["content"])
                        compression_info = {
                            "source_turn": turn + 1,
                            "strategy": "L2 paragraph filtering",
                            "tool": "browser",
                            "before_chars": original_len,
                            "after_chars": compressed_len,
                            "l2_output_chars": l2_len,
                            "safety_limit_chars": self.browser_post_l2_max_chars,
                            "safety_truncated": safety_truncated,
                            "saved_chars": max(0, original_len - compressed_len),
                            "retention_ratio": round(compressed_len / max(original_len, 1), 4),
                            "estimated_tokens_before": original_len // 3,
                            "estimated_tokens_after": compressed_len // 3,
                            "context_tokens_before": current_tokens,
                            "context_budget": budget,
                            "query": task.description,
                            "original_content": original_content,
                            "retained_content": result["content"],
                        }
                        logging.info(
                            "[%s] L2 browser compressed: %d -> %d chars (%.1f%%)",
                            task.id, original_len, compressed_len,
                            (1 - compressed_len / max(original_len, 1)) * 100,
                        )

                elif (
                    tool_name == "browser_batch"
                    and isinstance(result, dict)
                    and isinstance(result.get("results"), list)
                ):
                    current_tokens = self._estimate_tokens(messages)
                    budget = self._get_budget()
                    originals = []
                    page_metrics = []
                    total_before = 0
                    total_after = 0
                    for page in result["results"]:
                        if not isinstance(page, dict) or not isinstance(page.get("content"), str):
                            continue
                        original_content = page["content"]
                        original_len = len(original_content)
                        originals.append({"url": page.get("url", ""), "content": original_content})
                        if original_len:
                            page["content"] = self.compressor.l2(
                                original_content,
                                query=task.description,
                                current_tokens=current_tokens,
                                budget=budget,
                            )
                        l2_len = len(page["content"])
                        safety_truncated = l2_len > self.browser_post_l2_max_chars
                        if safety_truncated:
                            page["content"] = (
                                page["content"][: self.browser_post_l2_max_chars]
                                + f"\n\n[压缩结果过长，已安全截断至 {self.browser_post_l2_max_chars} 字符]"
                            )
                        after_len = len(page["content"])
                        total_before += original_len
                        total_after += after_len
                        page_metrics.append({
                            "url": page.get("url", ""),
                            "before_chars": original_len,
                            "l2_output_chars": l2_len,
                            "after_chars": after_len,
                            "safety_truncated": safety_truncated,
                        })
                    compression_info = {
                        "source_turn": turn + 1,
                        "strategy": "L2 batch paragraph filtering",
                        "tool": "browser_batch",
                        "before_chars": total_before,
                        "after_chars": total_after,
                        "saved_chars": max(0, total_before - total_after),
                        "retention_ratio": round(total_after / max(total_before, 1), 4),
                        "items_before": len(result["results"]),
                        "items_after": len(result["results"]),
                        "pages": page_metrics,
                        "safety_limit_chars": self.browser_post_l2_max_chars,
                        "estimated_tokens_before": total_before // 3,
                        "estimated_tokens_after": total_after // 3,
                        "context_tokens_before": current_tokens,
                        "context_budget": budget,
                        "query": task.description,
                        "original_content": originals,
                        "retained_content": result["results"],
                    }

                # web_search：超过 80% 才触发 L1 文章级过滤
                elif tool_name == "web_search" and isinstance(result, list):
                    received_result = result
                    result = self._deduplicate_search_results(result, seen_search_urls)
                    deduplicated_count = len(received_result) - len(result)
                    current_tokens = self._estimate_tokens(messages)
                    budget = self._get_budget()
                    tool_tokens = len(json.dumps(result, ensure_ascii=False, default=str)) // 3
                    if current_tokens + tool_tokens > budget * 0.8:
                        logging.info(
                            "[%s] L1 web_search triggered: current=%d + tool=%d > budget*0.8=%d",
                            task.id, current_tokens, tool_tokens, int(budget * 0.8),
                        )
                        raw_result = result
                        articles = [f"{r.get('title', '')}\n{r.get('snippet', '')}" for r in result]
                        filtered = self.compressor.l1(
                            articles,
                            query=task.description,
                            budget=int(budget * 0.2),
                        )
                        filtered_set = set(filtered)
                        result = [raw_result[i] for i in range(len(raw_result)) if articles[i] in filtered_set]
                        before_chars = len(json.dumps(raw_result, ensure_ascii=False, default=str))
                        after_chars = len(json.dumps(result, ensure_ascii=False, default=str))
                        compression_info = {
                            "source_turn": turn + 1,
                            "strategy": "L1 article filtering",
                            "tool": "web_search",
                            "before_chars": before_chars,
                            "after_chars": after_chars,
                            "saved_chars": max(0, before_chars - after_chars),
                            "retention_ratio": round(after_chars / max(before_chars, 1), 4),
                            "items_before": len(raw_result),
                            "items_after": len(result),
                            "items_received": len(received_result),
                            "duplicates_removed": deduplicated_count,
                            "estimated_tokens_before": before_chars // 3,
                            "estimated_tokens_after": after_chars // 3,
                            "context_tokens_before": current_tokens,
                            "context_budget": budget,
                            "trigger_threshold": 0.8,
                            "query": task.description,
                            "original_content": raw_result,
                            "retained_content": result,
                        }
                        logging.info(
                            "[%s] L1 web_search filtered: %d -> %d articles",
                            task.id, len(raw_result), len(result),
                        )
                    elif deduplicated_count:
                        before_chars = len(json.dumps(received_result, ensure_ascii=False, default=str))
                        after_chars = len(json.dumps(result, ensure_ascii=False, default=str))
                        compression_info = {
                            "source_turn": turn + 1,
                            "strategy": "URL deduplication",
                            "tool": "web_search",
                            "before_chars": before_chars,
                            "after_chars": after_chars,
                            "saved_chars": max(0, before_chars - after_chars),
                            "retention_ratio": round(after_chars / max(before_chars, 1), 4),
                            "items_before": len(received_result),
                            "items_after": len(result),
                            "duplicates_removed": deduplicated_count,
                            "estimated_tokens_before": before_chars // 3,
                            "estimated_tokens_after": after_chars // 3,
                            "context_tokens_before": current_tokens,
                            "context_budget": budget,
                            "query": task.description,
                            "original_content": received_result,
                            "retained_content": result,
                        }

                tool_failed = self._is_tool_result_failed(result)
                if tool_failed:
                    consecutive_tool_failures += 1
                    err_msg = self._tool_error_message(result)
                    logging.warning(
                        "[%s] 工具 '%s' 内部重试耗尽 (%d consecutive invocation): %s",
                        task.id, tool_name, consecutive_tool_failures, err_msg,
                    )
                    if isinstance(result, dict):
                        result = {
                            **result,
                            "fallback": "请改用其他 URL、其他工具，或基于已有资料总结。",
                        }
                    if consecutive_tool_failures >= 2:
                        force_summary_after_failures = True
                else:
                    consecutive_tool_failures = 0

                # 为模型实际看到的 URL 分配稳定局部标签，供最终总结引用。
                self._attach_source_labels(result, source_labels)

                tool_results.append({"tool_call_id": tc.get("id", ""), "name": tool_name, "result": result})
                tool_event = {"turn": turn, "role": "tool", "tool_call_id": tc.get("id", ""), "name": tool_name, "result": result}
                tool_event["failed"] = tool_failed
                if compression_info is not None:
                    tool_event["compression"] = compression_info
                trajectory.append(tool_event)


            # 连续空结果检测：所有工具返回空结果 → 计数 +1，否则重置
            usable_results = [
                tr["result"] for tr in tool_results
                if not self._is_tool_result_failed(tr["result"])
            ]
            if usable_results and all(self._is_tool_result_empty(result) for result in usable_results):
                consecutive_empty += 1
                logging.warning("[%s] 第 %d 轮所有工具返回空结果 (连续 %d/2)",
                                task.id, turn + 1, consecutive_empty)
            elif usable_results:
                consecutive_empty = 0  # 有非空结果，重置计数器

            # 连续 2 轮全部空结果 → 强制终止，搜索无效
            if consecutive_empty >= 2:
                logging.warning("[%s] 连续 %d 轮工具返回空结果，强制终止", task.id, consecutive_empty)
                return AgentResult(
                    task_id=task.id, status=AgentStatus.FAILED,
                    output="连续 2 轮工具调用均返回空结果，搜索无法获取有效信息，任务终止。",
                    trajectory=trajectory, token_usage=total_tokens, confidence=0.0,
                )

            # 追加消息
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            for tr in tool_results:
                msg_content = json.dumps(tr["result"], ensure_ascii=False, default=str)
                messages.append({"role": "tool", "tool_call_id": tr["tool_call_id"], "content": msg_content})

        return AgentResult(
            task_id=task.id, status=AgentStatus.TIMEOUT,
            output="达到最大轮次仍未获得最终答案。",
            trajectory=trajectory, token_usage=total_tokens, confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Token 估算 & Budget 推断
    # ------------------------------------------------------------------

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """估算当前 messages 的 token 数：字符数 / 3。"""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 3

    def _get_budget(self) -> int:
        """从 yaml 配置的 compressor.max_context_length 读取 budget。"""
        comp_cfg = self.config.get("compressor", {})
        max_ctx = comp_cfg.get("max_context_length", 128000)
        reserve = comp_cfg.get("output_reserve_tokens", 2048)
        return max(0, max_ctx - reserve)

    @staticmethod
    def _limit_tool_calls(
        tool_calls: list[dict],
        remaining_total: int,
        remaining_search: int,
        remaining_browser: int,
    ) -> list[dict]:
        """Run at most one budget-eligible tool call in each LLM loop."""
        accepted: list[dict] = []
        search_left = remaining_search
        browser_left = remaining_browser
        for call in tool_calls:
            if len(accepted) >= remaining_total:
                break
            name = call.get("function", {}).get("name", "")
            if name == "web_search":
                if search_left <= 0:
                    continue
                search_left -= 1
            elif name in {"browser", "browser_batch"}:
                if browser_left <= 0:
                    continue
                browser_left -= 1
            accepted.append(call)
            break
        return accepted

    # ------------------------------------------------------------------
    # 不可搜索任务：直接分析路径
    # ------------------------------------------------------------------
    async def _run_direct_analysis(
        self, task: SubTask, context: dict, trajectory: list, total_tokens: int
    ) -> AgentResult:
        """针对无法通过网络搜索获取答案的任务（个人隐私、主观建议）。"""
        messages = [
            {"role": "system", "content": self._system_prompt_direct_analysis()},
            {"role": "user", "content": task.description},
        ]
        try:
            resp = await asyncio.to_thread(self.policy.chat, messages)
        except RuntimeError as e:
            return AgentResult(
                task_id=task.id, status=AgentStatus.FAILED, output=str(e),
                trajectory=trajectory, token_usage=total_tokens, confidence=0.0,
            )

        content = resp.content or ""
        trajectory.append({"role": "assistant", "content": content})
        content, metadata = self._prepare_result(
            task, content, trajectory, context, from_memory=False
        )
        return AgentResult(
            task_id=task.id, status=AgentStatus.SUCCESS, output=content,
            trajectory=trajectory, token_usage=len(content) // 3,
            confidence=self._extract_confidence(content),
            metadata=metadata,
        )

    def _is_time_sensitive_query(self, context: dict) -> bool:
        """判断原始问题是否包含当前/最近等时效性关键词。"""
        query = (context.get("query", "") or "").lower()
        return any(kw in query for kw in self._TIME_SENSITIVE_KEYWORDS)

    async def _recall_from_memory(
        self, task: SubTask, query: str
    ) -> AgentResult | None:
        """高阈值召回历史 search 任务结果，命中则直接返回避免重复搜索。"""
        if self.knowledge_base is None:
            return None

        cfg = self.config.get("memory", {}).get("recall_before_research", {})
        if not cfg.get("enabled", True):
            return None

        threshold = cfg.get("threshold", 0.75)
        top_k = cfg.get("top_k", 1)

        # 只比较当前子任务。原始问题在同一批子任务中高度重复，会淹没
        # GPT/Claude/Gemini/Qwen 等真正用于区分研究对象的关键词。
        search_query = task.description.strip()

        try:
            matches = await self.knowledge_base.search(
                search_query,
                task_type="search",
                top_k=top_k,
                threshold=threshold,
            )
        except Exception:
            logging.exception("子任务执行前知识库召回失败 [%s]", task.id)
            return None

        if not matches:
            return None

        best = matches[0]
        logging.info(
            "[%s] 命中知识库 (score=%.2f)，直接返回已有结果: %s",
            task.id, best.score, best.entry.id
        )
        return AgentResult(
            task_id=task.id,
            status=AgentStatus.SUCCESS,
            output=best.entry.content,
            trajectory=[
                {
                    "role": "system",
                    "content": (
                        f"从知识库召回历史结果 (id={best.entry.id}, "
                        f"score={best.score:.2f}, confidence={best.entry.confidence:.2f})"
                    ),
                }
            ],
            token_usage=0,
            confidence=best.entry.confidence,
            metadata={
                "from_memory": True,
                "memory_id": best.entry.id,
                "memory_score": round(best.score, 4),
                "task_description": task.description,
                "sources": list(best.entry.sources or []),
            },
        )

    # ------------------------------------------------------------------
    # Prompt 工程
    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        return (
            "你是一位严谨的研究助手。针对用户问题，使用正确的工具收集和分析信息。\n\n"
            "可用工具:\n"
            "- web_search: 通用网页搜索，用于新闻、市场数据、行业报告、时事。\n"
            "- browser: 打开 URL 提取正文。**搜索返回通用结果时，必须立即用 browser 访问 Wikipedia、官方博客等高质量页面。**\n"
            "- browser_batch: 一次并发打开多个 URL，适合交叉核对搜索结果中的多个高价值来源。\n"
            "- arxiv_reader: 学术论文检索（ArXiv）。涉及论文、publication、学术引用时使用。\n"
            "- code_sandbox: Python 代码执行。计算 FLOPs、内存、统计分析、数据转换时使用。\n"
            "- calculator: 轻量数学计算（+ - * /）。简单计算时代替 code_sandbox。\n"
            "- notepad: 记录中间结论/搜索策略，避免多步研究中遗忘发现。\n"
            "- file_reader: 读取本地文件（txt/pdf/csv/json/docx）。任务明确引用本地文件时使用。\n\n"
            "规则:\n"
            "1. 【最高优先级】任务描述是你的唯一目标。所有搜索、分析和结论必须严格限定在任务描述范围内，禁止擅自扩展或替换研究对象。\n"
            "2. 必须使用工具获取事实信息，禁止凭记忆回答。\n"
            "3. 首次调用 web_search 后检查结果；信息不足时可使用 refined query 再搜索，或用 browser/browser_batch 阅读高价值原文。\n"
            "4. 高质量来源优先级：Wikipedia > 官方技术博客 > ArXiv > 通用搜索。涉及 benchmark/评测数据时，优先访问 Wikipedia 和 Papers with Code。\n"
            f"5. web_search 最多使用 {self.max_web_search_calls} 次，优先让不同 query 分别覆盖宽泛检索、信息缺口和交叉验证。\n"
            "6. 涉及数字/计算，用 calculator 或 code_sandbox。\n"
            f"7. 工具总预算为 {self.max_total_tool_calls} 次；搜索返回多个高价值 URL 时，优先用 browser_batch 批量读取。\n"
            "8. 每一轮最多调用一个工具；等待该工具返回后，再决定下一轮动作。\n"
            "9. 最终总结用中文，包含置信度 (0-1) 和关键数据点。\n"
            "10. 禁止问候用户，直接执行。\n"
            "11. 当系统提示'最后阶段'时，必须立即输出最终总结，绝对禁止调用工具。\n"
            "12. 工具调用失败时，如实报告失败原因，禁止编造数据或基于训练数据回答。\n"
            "13. 【反范围蔓延】如果你在搜索中发现了与任务描述无关但'更有趣'的信息，必须忽略它，不得偏离当前任务。\n"
            "14. 工具结果中的 URL 会带有 source_label（如 SRC-1）。最终总结中的事实和数据必须在相关句末引用对应标签，格式为 [SRC-1]；"
            "只能引用实际出现过的标签，不得自行编造标签，也不要输出参考文献列表。"
        )

    @staticmethod
    def _source_items(result: Any) -> list[dict[str, Any]]:
        """提取工具结果中所有带 URL 的来源项。"""
        items: list[dict[str, Any]] = []
        if isinstance(result, list):
            items = [item for item in result if isinstance(item, dict)]
        elif isinstance(result, dict):
            if result.get("url") or result.get("pdf_url"):
                items.append(result)
            for key in ("results", "papers"):
                nested = result.get(key)
                if isinstance(nested, list):
                    items.extend(item for item in nested if isinstance(item, dict))
        return items

    @classmethod
    def _attach_source_labels(
        cls, result: Any, labels_by_url: dict[str, str]
    ) -> None:
        """为单个 ResearchAgent 看到的 URL 分配稳定局部标签。"""
        for item in cls._source_items(result):
            url = str(item.get("url") or item.get("pdf_url") or "").strip()
            if not url:
                continue
            label = labels_by_url.get(url)
            if label is None:
                label = f"SRC-{len(labels_by_url) + 1}"
                labels_by_url[url] = label
            item["source_label"] = label

    @classmethod
    def _build_result_metadata(
        cls,
        task: SubTask,
        output: str,
        trajectory: list[dict[str, Any]],
        *,
        from_memory: bool,
    ) -> dict[str, Any]:
        """构造持久化和报告合成共用的来源元数据。"""
        return {
            "from_memory": from_memory,
            "task_description": task.description,
            "sources": cls._select_cited_sources(output, trajectory),
        }

    @classmethod
    def _prepare_result(
        cls,
        task: SubTask,
        output: str,
        trajectory: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        *,
        from_memory: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Normalize citations before output and source metadata are persisted."""
        available_sources = cls._available_sources(trajectory, context)
        available_labels = set(available_sources)
        parsed = normalize_source_citations(output, available_labels)
        if parsed.invalid_labels:
            logging.warning(
                "[%s] 输出包含不存在的来源标签: %s",
                task.id,
                sorted(parsed.invalid_labels),
            )
        selected_sources = cls._select_sources_by_labels(
            parsed.cited_labels, available_sources
        )
        metadata = {
            "from_memory": from_memory,
            "task_description": task.description,
            "sources": selected_sources,
        }
        metadata["citation_diagnostics"] = {
            "normalized_count": parsed.normalized_count,
            "invalid_labels": sorted(parsed.invalid_labels),
        }
        return parsed.text, metadata

    @classmethod
    def _available_sources(
        cls,
        trajectory: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, str]]:
        """Index local and dependency sources by their unambiguous labels."""
        indexed: dict[str, dict[str, str]] = {}
        for step in trajectory:
            if step.get("role") != "tool":
                continue
            for item in cls._source_items(step.get("result")):
                label = str(item.get("source_label", "")).upper()
                url = str(item.get("url") or item.get("pdf_url") or "").strip()
                if label and url:
                    indexed[label] = cls._merge_source_metadata(indexed.get(label), item)

        for key, sources in (context or {}).items():
            if not key.startswith("dep_sources:") or not isinstance(sources, list):
                continue
            dep_id = key.removeprefix("dep_sources:")
            for item in sources:
                if not isinstance(item, dict):
                    continue
                local_label = str(item.get("source_label", "")).upper()
                url = str(item.get("url") or item.get("pdf_url") or "").strip()
                if local_label and url:
                    indexed[cls._dependency_label(dep_id, local_label)] = item
        return indexed

    @staticmethod
    def _merge_source_metadata(
        existing: dict[str, Any] | None,
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge repeated tool results without replacing useful text with empty/mojibake fields."""
        if existing is None:
            return dict(incoming)

        merged = dict(existing)
        for key, value in incoming.items():
            if key in {"title", "snippet", "summary"}:
                text = str(value or "").strip()
                if text and "�" not in text and not any("\x80" <= char <= "\x9f" for char in text):
                    merged[key] = value
            elif value not in (None, ""):
                merged[key] = value
        return merged

    @staticmethod
    def _dependency_label(dep_id: str, label: str) -> str:
        """Keep original provenance when a source crosses multiple DAG layers."""
        normalized = label.upper()
        if ":" in normalized:
            return normalized
        return f"{dep_id}:{normalized}".upper()

    @staticmethod
    def _select_sources_by_labels(
        cited_labels: set[str],
        available_sources: dict[str, dict[str, Any]],
    ) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        seen_bindings: set[tuple[str, str]] = set()
        for label, item in available_sources.items():
            url = str(item.get("url") or item.get("pdf_url") or "").strip()
            binding = (label, url)
            if label not in cited_labels or not url or binding in seen_bindings:
                continue
            seen_bindings.add(binding)
            selected.append({
                "source_label": label,
                "url": url,
                "title": str(item.get("title", "")),
                "snippet": str(item.get("snippet") or item.get("summary") or "")[:500],
            })
        return selected

    @classmethod
    def _available_source_labels(
        cls, trajectory: list[dict[str, Any]]
    ) -> set[str]:
        labels: set[str] = set()
        for step in trajectory:
            if step.get("role") != "tool":
                continue
            for item in cls._source_items(step.get("result")):
                label = str(item.get("source_label", "")).upper()
                if label:
                    labels.add(label)
        return labels

    @classmethod
    def _select_cited_sources(
        cls, output: str, trajectory: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """只保留最终 output 明确引用的 URL 来源。"""
        parsed = normalize_source_citations(
            output, cls._available_source_labels(trajectory)
        )
        cited_labels = parsed.cited_labels
        if not cited_labels:
            return []

        selected: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for step in trajectory:
            if step.get("role") != "tool":
                continue
            for item in cls._source_items(step.get("result")):
                label = str(item.get("source_label", "")).upper()
                url = str(item.get("url") or item.get("pdf_url") or "").strip()
                if label not in cited_labels or not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                selected.append({
                    "source_label": label,
                    "url": url,
                    "title": str(item.get("title", "")),
                    "snippet": str(
                        item.get("snippet") or item.get("summary") or ""
                    )[:500],
                })
        return selected

    def _system_prompt_direct_analysis(self) -> str:
        return (
            "你是一位深思熟虑的分析师。用户提出的问题无法通过网络搜索获得答案"
            "（例如分析特定私人个体、个人建议、主观判断）。"
            "你的职责是仅基于上下文中已提供的信息进行推理分析。"
            "禁止编造事实。明确指出已知信息、合理推断和未知之处。"
            "结尾给出置信度 (0-1)。"
        )

    def _build_task_prompt(self, task: SubTask, context: dict) -> str:
        """根据 SubTask 和上游 agent 结果构建 user prompt。"""
        desc_lower = (task.description or "").lower()
        tool_recommendations = []

        # 关键词驱动的工具推荐
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        if any(kw in desc_lower for kw in academic_keywords):
            tool_recommendations.append("arxiv_reader")

        calc_keywords = ["计算", "flops", "显存", "内存", "参数量", "延迟", "成本", "公式", "数值", "统计", "数学", "推导"]
        if any(kw in desc_lower for kw in calc_keywords):
            tool_recommendations.append("calculator")
            tool_recommendations.append("code_sandbox")

        browser_keywords = ["详细", "原文", "全文", "深度", "详细内容", "网页内容", "文章正文"]
        if any(kw in desc_lower for kw in browser_keywords):
            tool_recommendations.append("browser")

        file_keywords = ["文件", "文档", "dataset", "数据集", "pdf", "csv", "json"]
        if any(kw in desc_lower for kw in file_keywords):
            tool_recommendations.append("file_reader")

        # 确定首选和备选工具
        if "arxiv_reader" in tool_recommendations:
            tool_recommendations = ["arxiv_reader"] + [t for t in tool_recommendations if t != "arxiv_reader"]
        elif not tool_recommendations:
            tool_recommendations.insert(0, "web_search")

        primary_tool = tool_recommendations[0]
        secondary_tools = tool_recommendations[1:]

        original_query = context.get("query", "")
        lines = []
        if original_query:
            lines.append(f"## 原始研究问题: {original_query}")
        lines.extend([
            f"## 当前子任务: {task.description}",
            f"类型: {task.task_type}",
            f"期望输出: {task.expected_type}",
            "",
            f"## 推荐工具（按优先级）: {', '.join(tool_recommendations)}",
        ])

        if secondary_tools:
            lines.append(f"先用 '{primary_tool}'。如涉及数字/计算，再用 {', '.join(secondary_tools)}。")
        else:
            lines.append(f"使用 '{primary_tool}' 收集信息。")

        lines.extend([
            "",
            "## 范围约束（必须遵守）:",
            "1. 你的唯一目标是回答上面的【任务描述】。",
            "2. 搜索和引用的事实必须直接服务于该目标。",
            "3. 禁止因为'觉得某个相关话题更有价值'而偏离当前任务。",
            "4. 如果任务描述要求研究 X，即使 Y 更热门或信息更多，也绝不允许用 Y 替代 X。",
            "5. 当发现信息不足以完整回答任务时，如实报告缺口，禁止擅自更换研究对象来'凑数'。",
            "",
            "## 执行指令:",
            f"1. 首先调用 '{primary_tool}' 工具，使用相关 query 收集信息。",
            "2. 检查结果。",
            f"3. 如需 refined query，可继续调用 '{primary_tool}'，但不得超过系统给出的剩余预算。",
            "4. 搜索结果包含多个高价值 URL 时，优先用 'browser_batch' 批量读原文；只需单页时使用 'browser'。",
            "5. 需计算时，用 'calculator' 或 'code_sandbox'（算 1 次）。",
            "6. 最后用中文总结发现并给出置信度 (0-1)。",
            "7. 禁止问候用户，直接执行。",
            "8. query 必须直接针对任务描述。",
        ])

        if task.search_hints:
            lines.insert(1, f"搜索提示（必须使用这些关键词）: {', '.join(task.search_hints)}")

        # 自动根据 DAG 依赖关系拉取上游子任务结果
        dep_parts = []
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_key = f"dep:{dep_id}"
                dep_output = context.get(dep_key)
                if dep_output is not None:
                    dep_sources = context.get(f"dep_sources:{dep_id}", [])
                    local_labels = {
                        str(source.get("source_label", "")).upper()
                        for source in dep_sources
                        if isinstance(source, dict) and source.get("source_label")
                    }
                    parsed = normalize_source_citations(str(dep_output), local_labels)
                    namespaced_output = re.sub(
                        r"\[(SRC-\d+)\]",
                        lambda match: (
                            f"[{self._dependency_label(dep_id, match.group(1))}]"
                        ),
                        parsed.text,
                        flags=re.IGNORECASE,
                    )
                    registry = []
                    registry_labels = []
                    for source in dep_sources:
                        if not isinstance(source, dict):
                            continue
                        label = str(source.get("source_label", "")).upper()
                        if not label:
                            continue
                        inherited_label = self._dependency_label(dep_id, label)
                        registry_labels.append(inherited_label)
                        registry.append(
                            f"[{inherited_label}] {source.get('title', '')} "
                            f"URL: {source.get('url') or source.get('pdf_url') or ''}"
                        )
                    registry_text = "\n".join(registry) or "（无可继承来源）"
                    label_example = (
                        registry_labels[0] if registry_labels else f"{dep_id}:SRC-1"
                    )
                    dep_parts.append(
                        f"### 上游任务 {dep_id} 的结果:\n{namespaced_output}\n"
                        f"#### 上游来源绑定:\n{registry_text}\n"
                        f"引用该上游证据时必须保留完整标签，如 [{label_example}]。"
                    )
                else:
                    dep_parts.append(f"### 上游任务 {dep_id} 的结果:\n（暂不可用）")
        if dep_parts:
            lines.append("\n## 上游任务结果（必须基于以下结果回答当前任务）:")
            lines.extend(dep_parts)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 启发式判断
    # ------------------------------------------------------------------
    def _is_non_searchable(self, task: SubTask, context: dict) -> bool:
        """判断任务是否无法通过网络搜索获取答案。"""
        desc = (task.description or "").lower()
        query = context.get("query", "").lower()
        combined = desc + " " + query

        if "朋友" in combined or "同学" in combined or "同事" in combined:
            if any(w in combined for w in ["分析", "评价", "是什么样", "性格", "人品"]):
                return True

        if any(w in combined for w in ["建议我", "我该怎么", "适合我吗", "要不要"]):
            if "朋友" in combined or "我" in query:
                return True

        if "叫" in combined and any(w in combined for w in ["分析", "评价", "是什么样"]):
            return True

        return False

    def _determine_fallback_tool(self, task: SubTask) -> str:
        """确定 fallback 时的首选工具。"""
        desc_lower = (task.description or "").lower()
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr"]
        if any(kw in desc_lower for kw in academic_keywords):
            return "arxiv_reader"
        return "web_search"

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {"error": f"工具 '{tool_name}' 未找到"}
        try:
            return await tool.execute(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _is_tool_result_failed(result: Any) -> bool:
        if isinstance(result, dict) and result.get("error"):
            return True
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            pages = result["results"]
            return bool(pages) and all(
                isinstance(page, dict) and page.get("error") for page in pages
            )
        if isinstance(result, list):
            return bool(result) and all(
                isinstance(item, dict) and item.get("error") for item in result
            )
        return False

    @staticmethod
    def _tool_error_message(result: Any) -> str:
        if isinstance(result, dict) and result.get("error"):
            return str(result["error"])
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            errors = [
                str(page.get("error")) for page in result["results"]
                if isinstance(page, dict) and page.get("error")
            ]
            return "; ".join(errors) or "all batch requests failed"
        if isinstance(result, list):
            errors = [
                str(item.get("error")) for item in result
                if isinstance(item, dict) and item.get("error")
            ]
            return "; ".join(errors) or "tool returned only errors"
        return "unknown tool error"

    @staticmethod
    def _canonical_result_url(value: Any) -> str:
        """Build a comparison key without changing the URL shown to the model/UI."""
        raw = str(value or "").strip()
        markdown_link = re.fullmatch(r"\[[^\]]*\]\((https?://[^)]+)\)", raw)
        if markdown_link:
            raw = markdown_link.group(1).strip()

        try:
            parts = urlsplit(raw)
        except ValueError:
            return raw.casefold().rstrip("/")
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return raw.casefold().rstrip("/")

        hostname = parts.hostname.casefold()
        port = parts.port
        if port and not (
            parts.scheme.lower() == "http" and port == 80
            or parts.scheme.lower() == "https" and port == 443
        ):
            hostname = f"{hostname}:{port}"

        tracking_keys = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}
        query = [
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_") and key.casefold() not in tracking_keys
        ]
        query.sort()
        path = parts.path.rstrip("/") or "/"
        # Treat HTTP/HTTPS variants as the same result. Fragment and tracking
        # parameters do not identify a different source page.
        return urlunsplit(("", hostname, path, urlencode(query, doseq=True), ""))

    @classmethod
    def _deduplicate_search_results(
        cls, results: list[Any], seen: set[str] | None = None
    ) -> list[Any]:
        """Keep first-seen results within and across search rounds."""
        unique: list[Any] = []
        seen_keys = seen if seen is not None else set()
        for index, item in enumerate(results):
            if isinstance(item, dict):
                key = cls._canonical_result_url(item.get("url"))
                if not key:
                    # URL-less entries are only duplicates when their visible
                    # title and snippet are identical.
                    key = "text:" + json.dumps(
                        [item.get("title", ""), item.get("snippet", "")],
                        ensure_ascii=False,
                        default=str,
                    )
            else:
                key = f"value:{item!r}"

            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _is_tool_result_empty(result: dict | list | None) -> bool:
        """判断工具返回结果是否为空/无效。"""
        if result is None:
            return True
        if isinstance(result, dict):
            # error 字段有实际值时才视为无效（null/None 不算）
            if result.get("error"):
                return True
            # 含 papers/results 但为空列表
            if "papers" in result and not result["papers"]:
                return True
            if "results" in result and not result["results"]:
                return True
            return False
        if isinstance(result, list):
            # 空列表，或所有元素都是含有效 error 的 dict
            if not result:
                return True
            if all(isinstance(item, dict) and item.get("error") for item in result):
                return True
            return False
        return False

    def _is_tool_failure_explanation(self, content: str) -> bool:
        if not content:
            return False
        c = content.lower()
        failure_keywords = [
            "无法通过", "无法执行", "无法使用", "无法获取", "无法访问",
            "额度已用尽", "cannot search", "unable to search", "quota exceeded",
        ]
        return any(kw in c for kw in failure_keywords)

    _PSEUDO_TOOL_PATTERNS = [
        r"<invoke\s+name\s*=",
        r"<tool_calls>",
        r"<DSML_tool_calls>",
        r"<function_calls>",
        # DeepSeek 实际返回的 tool-call 分隔符格式
        r"<\uff5c\uff5cDSML\uff5c\uff5ctool_calls>",
        r"<\uff5c\uff5cDSML\uff5c\uff5cinvoke\s+name",
    ]

    def _contains_pseudo_tool_calls(self, content: str) -> bool:
        """检测 content 中是否包含未执行的 XML 伪工具调用。"""
        return any(re.search(p, content) for p in self._PSEUDO_TOOL_PATTERNS)

    def _strip_pseudo_tool_calls(self, content: str) -> str:
        """从 content 中剥离 XML 伪工具调用部分。"""
        strip_patterns = [
            r"<tool_calls>.*?</tool_calls>",
            r"<invoke\s+name[^>]*>.*?</invoke>",
            r"<DSML_tool_calls>.*?</DSML_tool_calls>",
            r"<function_calls>.*?</function_calls>",
            # DeepSeek 实际返回的 tool-call 分隔符格式
            r"<\uff5c\uff5cDSML\uff5c\uff5ctool_calls>.*?<\uff5c\uff5c/DSML\uff5c\uff5ctool_calls>",
            r"<\uff5c\uff5cDSML\uff5c\uff5cinvoke\s+name[^>]*>.*?<\uff5c\uff5c/DSML\uff5c\uff5cinvoke>",
        ]
        cleaned = content
        for p in strip_patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def _extract_confidence(self, content: str) -> float:
        for line in content.splitlines():
            line_stripped = line.strip()
            if "置信度" in line_stripped or "confidence" in line_stripped.lower():
                m = re.search(
                    r"(?:置信度|confidence)[^\d]*?(0\.\d+|1\.0|1)",
                    line_stripped,
                    re.IGNORECASE,
                )
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        continue
        return 0.6
