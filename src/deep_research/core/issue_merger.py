"""Issue 合并器：对 Red Agent 产出的 issues 进行去重、合并、冲突仲裁。

支持可选的 LLM 二次仲裁：在 rule-based 合并后，调用一个 Judge LLM 判断真伪、
消解跨 fix_type 冲突，保证后续按 REMOVAL → SEARCH → IN_PLACE 分批修复时不会互相矛盾。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from typing import Any

from json_repair import loads as json_repair_loads

from .schema import AttackDimension, FixType, Issue, Severity

logger = logging.getLogger(__name__)


SYSTEM_ISSUE_ARBITER = (
    "你是一位研究报告问题仲裁专家。Red Team 已经从多个维度审查了研究报告并提出了一些问题。"
    "你的任务是在 Blue Team 修复之前，对这些问题进行最终审核、整合与冲突消解。\n\n"
    "Blue Team 会按照以下顺序按 fix_type 分批修复：\n"
    "1. REMOVAL：删除无依据、错误或过时的论断。\n"
    "2. SEARCH：为需要来源支撑的 claim 补充来源。\n"
    "3. IN_PLACE：原地改写、软化表述、补充反方观点等。\n\n"
    "请你遵循以下原则：\n"
    "- 剔除误报（false positive）：issue 描述与报告实际内容不符、证据不足、或 Red Team 过度推断的，直接删除。\n"
    "- 消解冲突：如果同一 claim 既被 hallucination/factual 要求删除/修改，又被 coverage/source 要求补来源，"
    "优先保留 REMOVAL/IN_PLACE，删除 SEARCH；不要同时让 Blue 删了又补。\n"
    "- 合并重复：同一位置或同一描述的多个 issue 只保留一个最准确、最可执行的。\n"
    "- 保证批次一致性：确保留下的 issue 在按 REMOVAL → SEARCH → IN_PLACE 顺序执行后不会互相矛盾。\n"
    "- 不要修改 issue 的实质内容，只做取舍；如果多个 issue 应保留，可以精炼 description 和 evidence。\n\n"
    "输出必须是严格 JSON 数组，每个元素包含字段：\n"
    "dimension（字符串：factual/hallucination/logic/source/coverage）、"
    "severity（字符串：critical/major/minor）、location（字符串）、"
    "description（字符串）、fix_type（字符串：removal/search/in_place）、evidence（字符串）。\n"
    "如果所有 issue 都是误报，输出空数组 []。"
)

PROMPT_ISSUE_ARBITER = """用户研究问题：{query}

--- 当前报告内容 ---
{report_content}

--- 待仲裁 issues（共 {count} 个） ---
{issues_json}

请基于完整的报告内容和研究问题，对 issues 进行最终仲裁。只输出 JSON 数组，不要解释。
"""


class IssueMerger:
    """对跨维度、跨位置的 issues 进行合并与冲突仲裁。"""

    # 维度优先级：数字越小越基础，合并时优先保留
    DIMENSION_ORDER: dict[AttackDimension, int] = {
        AttackDimension.HALLUCINATION: 0,  # 幻觉最基础，涉及内容真实性
        AttackDimension.FACTUAL: 1,
        AttackDimension.SOURCE: 2,
        AttackDimension.LOGIC: 3,
        AttackDimension.COVERAGE: 4,
    }

    # severity 排序：数字越小越严重
    SEVERITY_RANK: dict[Severity, int] = {
        Severity.CRITICAL: 0,
        Severity.MAJOR: 1,
        Severity.MINOR: 2,
    }

    # fix_type 保守程度：数字越小越保守
    FIX_TYPE_RANK: dict[FixType, int] = {
        FixType.REMOVAL: 0,
        FixType.SEARCH: 1,
        FixType.IN_PLACE: 2,
    }

    SIMILARITY_THRESHOLD: float = 0.5
    REPORT_CONTENT_MAX_LEN: int = 20000

    @classmethod
    async def merge_issues(
        cls,
        issues: list[Issue],
        llm_client: Any | None = None,
        query: str = "",
        report_content: str = "",
    ) -> list[Issue]:
        """合并 issues：先 rule-based 去重/仲裁，再可选 LLM 二次仲裁。"""
        if not issues:
            return []

        merged = cls._static_merge(issues)
        if llm_client is None:
            return merged

        try:
            arbitrated = await cls._llm_arbitrate(
                merged, llm_client, query, report_content
            )
            logger.info(
                "IssueMerger LLM 仲裁完成：%d -> %d 个 issue",
                len(merged),
                len(arbitrated),
            )
            return arbitrated
        except Exception:
            logger.exception("IssueMerger LLM 仲裁失败，回退到 rule-based 结果")
            return merged

    @classmethod
    def _static_merge(cls, issues: list[Issue]) -> list[Issue]:
        """Rule-based 合并：按 location 分组、description 去重、冲突仲裁、排序。"""
        # 1. 按 location 分组
        location_groups: dict[str, list[Issue]] = defaultdict(list)
        no_location_group: list[Issue] = []

        for issue in issues:
            loc = cls._normalize_text(issue.location)
            if loc:
                location_groups[loc].append(issue)
            else:
                no_location_group.append(issue)

        merged: list[Issue] = []

        # 2. 有 location 的组内去重合并
        for loc, group in location_groups.items():
            subgroups = cls._group_by_description_similarity(group)
            for sg in subgroups:
                merged.append(cls._merge_issue_group(sg))

        # 3. 无 location 的按 description 去重合并
        if no_location_group:
            subgroups = cls._group_by_description_similarity(no_location_group)
            for sg in subgroups:
                merged.append(cls._merge_issue_group(sg))

        # 4. 冲突仲裁
        merged = cls._arbitrate_conflicts(merged)

        # 5. 排序：维度优先级 -> severity 降序
        merged.sort(
            key=lambda i: (
                cls.DIMENSION_ORDER.get(i.dimension, 99),
                cls.SEVERITY_RANK.get(i.severity, 99),
            )
        )

        return merged

    @classmethod
    async def _llm_arbitrate(
        cls,
        issues: list[Issue],
        llm_client: Any,
        query: str,
        report_content: str,
    ) -> list[Issue]:
        """调用 Judge LLM 对 issues 做真伪判断和冲突消解。"""
        if not issues:
            return []

        truncated_report = report_content
        if len(truncated_report) > cls.REPORT_CONTENT_MAX_LEN:
            truncated_report = (
                truncated_report[: cls.REPORT_CONTENT_MAX_LEN]
                + "\n...（报告已截断）"
            )

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

        user_prompt = PROMPT_ISSUE_ARBITER.format(
            query=query,
            report_content=truncated_report,
            count=len(issue_dicts),
            issues_json=json.dumps(issue_dicts, ensure_ascii=False, indent=2),
        )

        messages = [
            {"role": "system", "content": SYSTEM_ISSUE_ARBITER},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("IssueMerger 开始调用 LLM 进行 issue 仲裁")
        resp = await asyncio.to_thread(llm_client.chat, messages)
        content = resp.content or ""
        logger.info("IssueMerger LLM 仲裁调用返回")

        parsed = cls._parse_arbitration_json(content)
        return cls._validate_issue_dicts(parsed)

    @classmethod
    def _parse_arbitration_json(cls, content: str) -> list[dict[str, Any]]:
        """解析 LLM 返回的 JSON 数组。"""
        # 先尝试直接提取 JSON 块
        raw = cls._extract_json(content)
        try:
            data = json.loads(raw) if raw is not None else json_repair_loads(content)
        except json.JSONDecodeError:
            data = json_repair_loads(raw if raw is not None else content)

        if isinstance(data, dict) and "issues" in data:
            data = data["issues"]
        if not isinstance(data, list):
            logger.warning("IssueMerger LLM 仲裁返回不是 JSON 数组：%s", type(data))
            return []
        return data

    @classmethod
    def _validate_issue_dicts(cls, data: list[dict[str, Any]]) -> list[Issue]:
        """把 LLM 返回的 dict 列表转成 Issue 对象，过滤无效项。"""
        valid: list[Issue] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                issue = Issue(
                    dimension=AttackDimension(str(item.get("dimension", "factual")).lower()),
                    severity=Severity(str(item.get("severity", "major")).lower()),
                    location=str(item.get("location", "")),
                    description=str(item.get("description", "")),
                    fix_type=FixType(str(item.get("fix_type", "in_place")).lower()),
                    evidence=str(item.get("evidence", "")),
                )
                if not issue.description:
                    continue
                valid.append(issue)
            except Exception:
                logger.warning("IssueMerger LLM 仲裁返回的 issue 格式无效：%s", item)
                continue
        return valid

    @classmethod
    def _extract_json(cls, content: str) -> str | None:
        """从文本中提取第一个 JSON 数组或对象块。"""
        content = content.strip()
        if content.startswith("["):
            # 找匹配的闭合 ]
            depth = 0
            in_str = False
            escape = False
            for i, ch in enumerate(content):
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            return content[: i + 1]
        if content.startswith("{"):
            depth = 0
            in_str = False
            escape = False
            for i, ch in enumerate(content):
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return content[: i + 1]
        return None

    @classmethod
    def _group_by_description_similarity(cls, issues: list[Issue]) -> list[list[Issue]]:
        """把 issues 按 description 相似度分成若干组。"""
        subgroups: list[list[Issue]] = []
        for issue in issues:
            placed = False
            for sg in subgroups:
                if cls._jaccard_similarity(issue.description, sg[0].description) >= cls.SIMILARITY_THRESHOLD:
                    sg.append(issue)
                    placed = True
                    break
            if not placed:
                subgroups.append([issue])
        return subgroups

    @classmethod
    def _merge_issue_group(cls, group: list[Issue]) -> Issue:
        """合并一组相似 issue。"""
        # severity 取最严重的
        base_severity = min(
            group,
            key=lambda i: cls.SEVERITY_RANK.get(i.severity, 99),
        ).severity

        # dimension 取最基础的
        base_dimension = min(
            group,
            key=lambda i: cls.DIMENSION_ORDER.get(i.dimension, 99),
        ).dimension

        # fix_type 取最保守的
        base_fix = min(
            group,
            key=lambda i: cls.FIX_TYPE_RANK.get(i.fix_type, 99),
        ).fix_type

        # description 取最长的（通常信息最完整）
        description = max(group, key=lambda i: len(i.description)).description

        # evidence 合并去重
        evidences: list[str] = []
        for i in group:
            if i.evidence and i.evidence not in evidences:
                evidences.append(i.evidence)
        evidence = "\n---\n".join(evidences)

        # location 取第一个非空的
        location = next((i.location for i in group if i.location), "")

        return Issue(
            severity=base_severity,
            dimension=base_dimension,
            description=description,
            location=location,
            fix_type=base_fix,
            evidence=evidence,
        )

    @classmethod
    def _arbitrate_conflicts(cls, issues: list[Issue]) -> list[Issue]:
        """仲裁互相冲突的 issues。"""
        loc_map: dict[str, list[tuple[int, Issue]]] = defaultdict(list)
        for idx, issue in enumerate(issues):
            loc = cls._normalize_text(issue.location)
            key = loc if loc else f"__no_loc_{idx}__"
            loc_map[key].append((idx, issue))

        to_remove: set[int] = set()

        for loc, group in loc_map.items():
            if loc.startswith("__no_loc_"):
                continue

            has_hallucination_removal = any(
                issue.dimension == AttackDimension.HALLUCINATION
                and issue.fix_type in (FixType.REMOVAL, FixType.IN_PLACE)
                for _, issue in group
            )
            has_coverage_supplement = any(
                issue.dimension == AttackDimension.COVERAGE
                and issue.fix_type == FixType.SEARCH
                for _, issue in group
            )

            # 冲突：hallucination 要删/改，coverage 要补来源 -> 删掉 coverage 的 supplement
            if has_hallucination_removal and has_coverage_supplement:
                for idx, issue in group:
                    if (
                        issue.dimension == AttackDimension.COVERAGE
                        and issue.fix_type == FixType.SEARCH
                    ):
                        to_remove.add(idx)

        return [issue for idx, issue in enumerate(issues) if idx not in to_remove]

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def _jaccard_similarity(cls, a: str, b: str) -> float:
        set_a = set(cls._normalize_text(a).split())
        set_b = set(cls._normalize_text(b).split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
