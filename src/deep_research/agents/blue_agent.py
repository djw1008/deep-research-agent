"""Blue Team Agent — 根据 Red Team 攻击结果批量修复研究报告。

修复策略：
  - 将 issues 按 fix_type 分组，按 REMOVAL → SEARCH → IN_PLACE 顺序批量修复。
  - 同一 fix_type 的一批 issue 只调一次 LLM，SEARCH 类型启用 function calling，
    由模型自己决定 web_search 的 query 和次数（最多 2 次）。
  - 修复后的完整报告交给下一轮独立 Red 复评。
  - 修复时绝不编造来源或事实；无法验证的内容删除或标注 [未经证实]/[来源待补充]。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from typing import Any

from json_repair import loads as json_repair_loads

from ..core.schema import (
    AgentResult,
    AgentStatus,
    AttackDimension,
    FixType,
    Issue,
    ResearchReport,
    Severity,
    SubTask,
)
from ..core.report_content import (
    citation_ids,
    compact_cited_sources,
    prepare_sources,
    remap_citation_ids,
    strip_reference_sections,
)
from .base_agent import BaseAgent


try:
    from ..compressor.context_compressor import ContextCompressor
except Exception:
    ContextCompressor = None  # type: ignore


logger = logging.getLogger(__name__)

__all__ = ["BlueTeamAgent"]


SYSTEM_BLUE_AGENT = (
    "你是一位严谨的研究报告编辑（Blue Agent）。你的任务是根据 Red Agent 指出的问题，"
    "对研究报告进行最小化、安全的修改。核心原则：\n"
    "1. 绝不能为了修复问题而编造来源或事实。\n"
    "2. 如果已有来源能够支持论断，即使权威性较低，也应保留引用并注明来源性质；只有来源缺失或不支持论断时，才删除或弱化该论断。不得输出 [来源待补充] 等占位标记。\n"
    "3. 每次只修改 issue 明确指向的位置，不要扩大范围。\n"
    "4. 输出必须是严格的 JSON 格式，包含完整的修改后报告。\n"
    "5. 来源清单由程序维护。不得创建或修改引用来源/参考文献/参考链接章节。"
    "正文只能使用可用 Sources 或搜索工具返回的既有数字编号 [N]，不得自行创建编号。"
)


PROMPT_REMOVAL = """请根据以下 issue 修改研究报告。

修改要求：
- 删除 issue 中指出的无依据、错误或过时的论断。
- 只删除明确提到的内容，不要扩大范围。
- 删除后保持上下文连贯、段落衔接自然。
- 如果删除的是关键结论，用一句话说明"该结论因缺乏依据已删除"。
- 不要添加新的来源或事实。

输出必须是 JSON：
{
  "content": "修改后的完整报告内容",
  "changes": "简要说明做了什么修改"
}

--- Issue ---
{issue_json}

--- 当前报告 ---
{content}
"""


PROMPT_SEARCH = """请根据以下 issue 修改研究报告。

修改要求：
- issue 指出某 claim 需要来源支撑。
- 优先在"可用 Sources"和"新增搜索结果"中查找能支撑该 claim 的来源；若找到，补充引用并改写表述。
- 如果现有来源能够支持该 claim，只是权威性较低，应保留引用并注明来源性质，或用更权威来源替换；不得将其当作无来源内容。
- 若均无支撑来源，请删除该具体 claim，或删去无法支撑的精确数据并改写为不确定性表述；不得添加 [来源待补充] 等占位标记。
- 不要编造 URL、数据或研究结论。

输出必须是 JSON：
{
  "content": "修改后的完整报告内容",
  "changes": "简要说明做了什么修改"
}

--- Issue ---
{issue_json}

--- 当前报告 ---
{content}

--- 可用 Sources ---
{sources}

--- 新增搜索结果 ---
{search_results}
"""


PROMPT_IN_PLACE = """请根据以下 issue 修改研究报告。

修改要求：
- 修复 issue 指出的逻辑矛盾、表述不当、覆盖不足或立场偏差。
- 可进行原地重写、补充反方观点、降低绝对化表述、标注不确定性等。
- 不能编造新的事实或来源。
- 如果无法确定正确表述，使用 [未经证实] 或 [待进一步核实] 标注。

输出必须是 JSON：
{
  "content": "修改后的完整报告内容",
  "changes": "简要说明做了什么修改"
}

--- Issue ---
{issue_json}

--- 当前报告 ---
{content}
"""


PROMPT_TEMPLATES: dict[FixType, str] = {
    FixType.REMOVAL: PROMPT_REMOVAL,
    FixType.SEARCH: PROMPT_SEARCH,
    FixType.IN_PLACE: PROMPT_IN_PLACE,
}


# fix_type 执行优先级：先删除，再补来源，最后改措辞
_FIX_TYPE_ORDER = {
    FixType.REMOVAL: 0,
    FixType.SEARCH: 1,
    FixType.IN_PLACE: 2,
}


_SEVERITY_RANK = {
    Severity.CRITICAL: 3,
    Severity.MAJOR: 2,
    Severity.MINOR: 1,
}


class BlueTeamAgent(BaseAgent):
    """蓝队 Agent：按 fix_type 批量修复 issue，由下一轮 Red 独立复评。"""

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(name, policy, tools)
        self.config = config or {}

        # 搜索结果压缩器（L2），失败则回退到无压缩
        self.compressor: Any | None = None
        if ContextCompressor is not None:
            try:
                comp_cfg = self.config.get("compressor", {})
                self.compressor = ContextCompressor(
                    l2_threshold=comp_cfg.get("l2_threshold", 0.10),
                )
            except Exception:
                logger.warning("BlueTeamAgent 初始化压缩器失败，将禁用搜索结果压缩")

    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行修复任务。

        Args:
            task: task_type 应为 "blue_agent"
            context: 必须包含
                - report: ResearchReport
                - query: str
                - dimension: AttackDimension
                - issues: List[Issue]（已过滤后的待修复问题）

        Returns:
            AgentResult，output 为修复后的 ResearchReport。
        """
        report = context.get("report")
        query = context.get("query", "")
        dimension = context.get("dimension")
        issues = context.get("issues", [])

        if not isinstance(report, ResearchReport):
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.FAILED,
                output="context 中缺少有效的 ResearchReport",
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        if not isinstance(dimension, AttackDimension):
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.FAILED,
                output="context 中缺少有效的 AttackDimension",
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        filtered_issues = self._filter_issues(issues)
        if not filtered_issues:
            logger.info(
                "BlueTeamAgent: 维度 %s 无需要修复的 issue，直接返回原报告",
                dimension.value,
            )
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.SUCCESS,
                output=report,
                trajectory=[
                    {
                        "turn": 0,
                        "role": "assistant",
                        "log": True,
                        "content": "本批次没有需要修复的 issue，跳过修复。",
                    }
                ],
                token_usage=0,
                confidence=report.confidence,
                metadata={"dimension": dimension.value, "action": "no_issues_to_fix"},
            )

        current_report = copy.deepcopy(report)
        fixes: list[dict[str, Any]] = []

        # 按 fix_type 分组批量修复：先删除、再补来源、最后改措辞
        fix_type_groups = self._group_by_fix_type(filtered_issues)
        ordered_fix_types = sorted(fix_type_groups.keys(), key=lambda ft: _FIX_TYPE_ORDER.get(ft, 99))
        batch_desc = "、".join(f"{ft.value}×{len(fix_type_groups[ft])}" for ft in ordered_fix_types)

        trajectory: list[dict[str, Any]] = [
            {
                "turn": 0,
                "role": "assistant",
                "log": True,
                "content": (
                    f"收到修复任务：{len(ordered_fix_types)} 个批次（{batch_desc}），"
                    f"共 {len(filtered_issues)} 个 issue，基于当前报告（{len(report.content)} 字符）。"
                ),
            },
            {
                "turn": 0,
                "role": "tool",
                "name": "blue_input",
                "result": {
                    "fix_type_batches": {ft.value: len(fix_type_groups[ft]) for ft in ordered_fix_types},
                    "report_length": len(report.content),
                    "issue_count": len(filtered_issues),
                    "issues": filtered_issues,
                },
            },
        ]
        token_usage = 0

        try:
            for batch_idx, fix_type in enumerate(ordered_fix_types):
                group = self._sort_group_by_severity(fix_type_groups[fix_type])
                original_content = current_report.content
                batch_turn = batch_idx + 1

                fix_result = await self._apply_fix_batch(
                    current_report, group, query, dimension
                )
                token_usage += fix_result.get("token_usage", 0)

                current_report.sources = self._merge_sources(
                    current_report.sources,
                    fix_result.get("sources", []),
                )
                current_report.content, current_report.sources = compact_cited_sources(
                    strip_reference_sections(fix_result["content"]),
                    current_report.sources,
                )
                fix_record = {
                    "dimension": dimension.value,
                    "fix_type": fix_type.value,
                    "issue_count": len(group),
                    "changes": fix_result.get("changes", ""),
                }
                fixes.append(fix_record)
                trajectory.append(
                    {
                        "turn": batch_turn,
                        "role": "assistant",
                        "log": True,
                        "content": (
                            f"批次修复 [{fix_type.value}]：处理 {len(group)} 个 issue，"
                            f"报告长度 {len(original_content)} → {len(current_report.content)}。"
                            f"\n\n修改说明：{fix_result.get('changes', '')}"
                        ),
                    }
                )

                logger.info(
                    "BlueTeamAgent %s 批量修复完成：issue 数 %d，报告长度 %d -> %d，修改说明：%s",
                    fix_type.value,
                    len(group),
                    len(original_content),
                    len(current_report.content),
                    fix_result.get("changes", "")[:100],
                )

            logger.info(
                "BlueTeamAgent: 维度 %s 完成 %d 批 issue 修复（共 %d 个）",
                dimension.value,
                len(fixes),
                len(filtered_issues),
            )

            summary_turn = len(fixes) + 1
            trajectory.append(
                {
                    "turn": summary_turn,
                    "role": "assistant",
                    "log": True,
                    "content": (
                        f"修复完成：{len(fixes)} 批修复"
                        f"（共 {len(filtered_issues)} 个 issue）。"
                    ),
                }
            )
            trajectory.append(
                {
                    "turn": summary_turn,
                    "role": "tool",
                    "name": "blue_fixes",
                    "result": {
                        "fixes": fixes,
                    },
                }
            )

            return AgentResult(
                task_id=task.id,
                status=AgentStatus.SUCCESS,
                output=current_report,
                trajectory=trajectory,
                token_usage=token_usage,
                confidence=current_report.confidence,
                metadata={
                    "dimension": dimension.value,
                    "fixes": fixes,
                },
            )

        except Exception as e:
            logger.exception("BlueTeamAgent 修复失败")
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.FAILED,
                output=f"Blue repair failed: {type(e).__name__}: {e}",
                trajectory=[
                    {"turn": 0, "role": "assistant", "log": True, "content": f"修复失败：{type(e).__name__}: {e}"}
                ],
                token_usage=token_usage,
                confidence=report.confidence,
                metadata={"dimension": dimension.value, "error": str(e)},
            )

    def _filter_issues(self, issues: list[Issue]) -> list[Issue]:
        """过滤出有效 issue：排除 severity 解析失败或 description 为空的问题。"""
        valid: list[Issue] = []
        for issue in issues:
            if not isinstance(issue, Issue):
                continue
            if not issue.description or not issue.description.strip():
                continue
            valid.append(issue)
        return valid

    def _group_by_fix_type(self, issues: list[Issue]) -> dict[FixType, list[Issue]]:
        """按 fix_type 对 issues 分组。"""
        groups: dict[FixType, list[Issue]] = {}
        for issue in issues:
            groups.setdefault(issue.fix_type, []).append(issue)
        return groups

    def _sort_group_by_severity(self, issues: list[Issue]) -> list[Issue]:
        """组内按 severity 降序排列，让更严重的 issue 排在前面。"""
        return sorted(
            issues,
            key=lambda i: -_SEVERITY_RANK.get(i.severity, 0),
        )

    def _sort_issues(self, issues: list[Issue]) -> list[Issue]:
        """按 fix_type 优先级排序，相同优先级按 severity 降序（兼容旧接口）。"""
        return sorted(
            issues,
            key=lambda i: (
                _FIX_TYPE_ORDER.get(i.fix_type, 99),
                -_SEVERITY_RANK.get(i.severity, 0),
            ),
        )

    async def _apply_fix(
        self,
        report: ResearchReport,
        issue: Issue,
        query: str,
        dimension: AttackDimension,
    ) -> dict[str, Any]:
        """对单个 issue 调用 LLM 修复（兼容旧接口，实际走批量修复）。"""
        return await self._apply_fix_batch(report, [issue], query, dimension)

    async def _apply_fix_batch(
        self,
        report: ResearchReport,
        issues: list[Issue],
        query: str,
        dimension: AttackDimension,
    ) -> dict[str, Any]:
        """对一批相同 fix_type 的 issue 调用一次 LLM 修复。

        SEARCH 类型启用 function calling：把 web_search 注册给模型，由模型自己决定
        搜索关键词和次数（最多 2 次），再把结果拿回后一次性输出修复后的报告。
        """
        if not issues:
            return {"content": report.content, "changes": "", "token_usage": 0}

        fix_type = issues[0].fix_type
        web_search_tool = self._get_tool("web_search") if fix_type == FixType.SEARCH else None
        old_tools = self._register_search_tool(web_search_tool)

        try:
            messages = self._build_repair_messages(
                report, issues, query, dimension, web_search_tool
            )
            content, token_usage, new_sources = await self._run_repair_loop(
                messages, fix_type, web_search_tool, query, report.sources
            )

            parsed = self._parse_fix_json(content)
            revised_content, cited_new_sources = self._select_cited_new_sources(
                parsed.get("content", report.content),
                report.sources,
                new_sources,
            )
            logger.info(
                "BlueTeamAgent %s 修复 LLM 输出解析完成：changes=%s",
                fix_type.value,
                parsed.get("changes", "")[:100],
            )
            return {
                "content": revised_content,
                "changes": parsed.get("changes", ""),
                "token_usage": token_usage,
                "sources": cited_new_sources,
            }

        finally:
            self._restore_tools(old_tools)

    def _register_search_tool(self, web_search_tool: Any | None) -> Any | None:
        """为 SEARCH 类型修复注册 web_search 工具，返回旧 tools 以便恢复。"""
        if web_search_tool is None or not hasattr(self.policy, "set_tools"):
            return None
        old_tools = getattr(self.policy, "tools", None)
        schema = getattr(
            web_search_tool,
            "get_schema",
            lambda: {"type": "function", "function": {"name": web_search_tool.name}},
        )()
        self.policy.set_tools([schema])
        return old_tools

    def _restore_tools(self, old_tools: Any | None) -> None:
        """恢复 policy 的 tools 设置。"""
        if not hasattr(self.policy, "set_tools"):
            return
        if old_tools is not None:
            self.policy.set_tools(old_tools)
        else:
            self.policy.set_tools(None)

    def _build_repair_messages(
        self,
        report: ResearchReport,
        issues: list[Issue],
        query: str,
        dimension: AttackDimension,
        web_search_tool: Any | None,
    ) -> list[dict[str, Any]]:
        """构造修复 LLM 的 messages。"""
        fix_type = issues[0].fix_type
        template = PROMPT_TEMPLATES.get(fix_type, PROMPT_IN_PLACE)
        sources_text = self._format_sources(report.sources)

        issue_dicts = [
            {
                "dimension": issue.dimension.value,
                "severity": issue.severity.value,
                "location": issue.location,
                "description": issue.description,
                "fix_type": issue.fix_type.value,
                "evidence": issue.evidence,
            }
            for issue in issues
        ]
        issue_json = (
            f"本次共需修复 {len(issues)} 个 {fix_type.value} issue：\n"
            + json.dumps(issue_dicts, ensure_ascii=False, indent=2)
        )

        if fix_type == FixType.SEARCH:
            if web_search_tool is not None:
                search_results_text = (
                    "（你可以调用 web_search 工具获取补充来源，最多 2 次；"
                    "如果搜索无果，请删除无支撑论断或弱化表述，不得添加引用占位标记。）"
                )
            else:
                search_results_text = "（未配置 web_search 工具，无法补充来源。）"
        else:
            search_results_text = "无新增搜索结果"

        user_prompt = self._format_prompt(
            template,
            issue_json=issue_json,
            content=report.content,
            sources=sources_text,
            search_results=search_results_text,
            query=query,
            dimension=dimension.value,
        )

        logger.info(
            "BlueTeamAgent 开始批量调用 %s/%s 修复 LLM，%d 个 issue，报告长度 %d 字符",
            fix_type.value,
            dimension.value,
            len(issues),
            len(report.content),
        )

        return [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": user_prompt},
        ]

    async def _run_repair_loop(
        self,
        messages: list[dict[str, Any]],
        fix_type: FixType,
        web_search_tool: Any | None,
        query: str,
        existing_sources: list[dict[str, Any]],
        max_tool_turns: int = 2,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """运行修复 LLM 调用循环，处理可选的 function calling。"""
        total_token_usage = 0
        content = ""
        collected_sources: list[dict[str, Any]] = []

        for turn in range(max_tool_turns + 1):
            resp = await asyncio.to_thread(self.policy.chat, messages)
            content = resp.content or ""
            total_token_usage += getattr(
                getattr(resp, "usage", None), "total_tokens", 0
            ) or 0
            tool_calls = getattr(resp, "tool_calls", []) or []

            if not tool_calls:
                break

            if turn >= max_tool_turns:
                logger.warning(
                    "BlueTeamAgent SEARCH 达到最大工具轮次，强制结束工具调用"
                )
                break

            # 记录 assistant 的工具调用请求
            messages.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )

            collected_sources.extend(
                await self._execute_tool_calls(
                    tool_calls,
                    web_search_tool,
                    messages,
                    query,
                    existing_sources + collected_sources,
                )
            )

            # 提示模型基于搜索结果继续
            messages.append({
                "role": "user",
                "content": "搜索结果已返回，请基于上述结果继续修复并输出 JSON。",
            })

        return content, total_token_usage, collected_sources

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        web_search_tool: Any | None,
        messages: list[dict[str, Any]],
        query: str,
        existing_sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """执行一批 tool_calls，把结果以 tool 消息形式追加到 messages。"""
        collected_sources: list[dict[str, Any]] = []
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            tool_call_id = tc.get("id", "")

            if tool_name != "web_search" or web_search_tool is None:
                result = {"error": f"工具 '{tool_name}' 不可用"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                continue

            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except Exception:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            search_query = args.get("query", "").strip()
            num_results = args.get("num_results", 5)
            if not search_query:
                result = [{"error": "搜索 query 为空"}]
            else:
                result = await web_search_tool.execute(search_query, num_results=num_results)

            if isinstance(result, list):
                candidates = [
                    item for item in result
                    if isinstance(item, dict) and item.get("url")
                ]
                registry = prepare_sources(existing_sources + collected_sources + candidates)
                citation_by_url = {
                    source["url"]: source["citation_id"] for source in registry
                }
                annotated_result = []
                for item in result:
                    annotated = dict(item) if isinstance(item, dict) else item
                    if isinstance(annotated, dict) and annotated.get("url") in citation_by_url:
                        annotated["citation_id"] = citation_by_url[annotated["url"]]
                    annotated_result.append(annotated)
                result = annotated_result
                collected_sources.extend(
                    source for source in registry
                    if source["url"] not in {
                        str(existing.get("url", "")).strip() for existing in existing_sources
                    }
                )

            # 对搜索结果做 L2 压缩，避免上下文爆炸
            current_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 3
            budget = self.config.get("compressor", {}).get("max_context_length", 128000)
            result = self._compress_search_results(
                result, query=query, current_tokens=current_tokens, budget=budget
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": "web_search",
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        return collected_sources

    def _get_tool(self, name: str) -> Any | None:
        """按名称获取已注册的工具实例。"""
        for tool in self.tools or []:
            if getattr(tool, "name", None) == name:
                return tool
        return None

    def _format_sources(self, sources: list[dict]) -> str:
        """将 sources 列表格式化为文本块。"""
        if not sources:
            return "无来源"
        lines = []
        for s in prepare_sources(sources):
            title = s.get("title", "")
            url = s.get("url", "")
            snippet = s.get("snippet", "")
            lines.append(
                f"[{s['citation_id']}] {title}\nURL: {url}\n摘要: {snippet}\n"
            )
        return "\n".join(lines)

    def _merge_sources(
        self,
        existing: list[dict],
        discovered: list[dict],
    ) -> list[dict]:
        """Merge tool-discovered sources into report metadata, deduplicated by URL."""
        normalized_discovered = [
            {
                "url": str(source.get("url", "")).strip(),
                "title": str(source.get("title", "")),
                "snippet": str(source.get("snippet", "")),
                "task_id": "blue_agent",
            }
            for source in discovered or []
        ]
        return prepare_sources(list(existing or []) + normalized_discovered)

    def _select_cited_new_sources(
        self,
        content: str,
        existing: list[dict],
        discovered: list[dict],
    ) -> tuple[str, list[dict]]:
        """Register only newly discovered sources actually cited by the repair.

        Search results receive temporary numbers before L2 compression. If only
        a subset is cited, compact those temporary numbers so the final registry
        stays contiguous and the正文编号 remains aligned with ``report.sources``.
        """
        existing_count = len(prepare_sources(existing))
        used_ids = citation_ids(content)
        selected = [
            source for source in prepare_sources(list(existing) + list(discovered))[existing_count:]
            if int(source["citation_id"]) in used_ids
        ]
        mapping = {
            int(source["citation_id"]): existing_count + index
            for index, source in enumerate(selected, 1)
        }
        remapped_content = remap_citation_ids(content, mapping)
        return remapped_content, selected

    def _format_prompt(self, template: str, **kwargs) -> str:
        """安全格式化 prompt：转义 JSON 中的花括号，只保留已知占位符。"""
        escaped = template.replace("{", "{{").replace("}", "}}")
        for key in kwargs:
            escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")
        return escaped.format(**kwargs)

    def _compress_search_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        current_tokens: int = 0,
        budget: int = 128000,
    ) -> list[dict[str, Any]]:
        """对 web_search 返回的列表做 L2 段落级压缩，防止塞爆上下文。

        压缩失败时回退到原始结果。
        """
        if not results or self.compressor is None:
            return results

        valid_results = [
            r for r in results if isinstance(r, dict) and "error" not in r
        ]
        if not valid_results:
            return results

        try:
            full_text = "\n\n---\n\n".join(
                f"[{r.get('citation_id', i + 1)}] {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n{r.get('snippet', '')}"
                for i, r in enumerate(valid_results)
            )

            # 内容较短时直接保留原文，不强行压缩
            if len(full_text) < 800:
                return results

            compressed = self.compressor.l2(
                full_text,
                query=query,
                current_tokens=current_tokens,
                budget=budget,
            )

            if not compressed:
                return results

            return [
                {
                    "title": f"web_search 结果摘要（原 {len(valid_results)} 条）",
                    "url": "",
                    "snippet": compressed,
                }
            ]
        except Exception:
            logger.warning("BlueTeamAgent 搜索结果压缩失败，使用原始结果", exc_info=True)
            return results

    def _parse_fix_json(self, text: str) -> dict[str, Any]:
        """解析修复 LLM 的 JSON 输出。"""
        raw = self._extract_json(text)
        try:
            data = json.loads(raw) if raw is not None else json_repair_loads(text)
        except json.JSONDecodeError as e:
            try:
                data = json_repair_loads(raw if raw is not None else text)
            except Exception:
                logger.warning("BlueTeamAgent 修复 JSON 解码失败: %s", e)
                return {}
        return data if isinstance(data, dict) else {}

    def _extract_json(self, text: str) -> str | None:
        """从文本中提取第一个 JSON 对象。

        优先匹配 ```json 代码块；否则用 json.JSONDecoder.raw_decode 定位第一个
        合法对象；最后再回退到简单的花括号匹配。
        """
        if not text:
            return None

        stripped = text.strip()

        # 1. 优先匹配 markdown 代码块（非贪婪）
        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?)\s*```", stripped, re.DOTALL
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # 2. 用 JSONDecoder 找第一个合法对象
        decoder = json.JSONDecoder()
        start_positions = [idx for idx, ch in enumerate(stripped) if ch == "{"]

        # 如果文本以 { 开头，优先从开头解析；失败直接回退正则
        if start_positions and start_positions[0] == 0:
            try:
                _, end = decoder.raw_decode(stripped, 0)
                return stripped[:end]
            except json.JSONDecodeError:
                brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
                if brace_match:
                    return brace_match.group(0).strip()
                return None

        # 文本不以 { 开头：从所有 { 位置找第一个能解析的合法对象
        for idx in start_positions:
            try:
                _, end = decoder.raw_decode(stripped, idx)
                return stripped[idx:end]
            except json.JSONDecodeError:
                continue

        # 3. 兜底：第一个 { 到最后一个 } 的片段
        brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if brace_match:
            return brace_match.group(0).strip()

        return None
