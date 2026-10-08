"""Red Team Agent — 对研究报告进行五维度单轮对抗攻击。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from json_repair import loads as json_repair_loads

from ..core.schema import (
    AgentResult,
    AgentStatus,
    AttackDimension,
    DimensionAttack,
    FixType,
    Issue,
    RedAttackResult,
    ResearchReport,
    Severity,
    SubTask,
)
from ..core.report_content import prepare_sources
from .base_agent import BaseAgent


logger = logging.getLogger(__name__)

__all__ = ["RedTeamAgent"]


SYSTEM_RED_AGENT = (
    "你是一位极其严苛的研究报告审查员（Red Agent）。你的任务是以批判性思维深度审查研究报告，"
    "找出所有事实错误、幻觉、逻辑漏洞、来源缺陷和覆盖缺失。你必须基于客观证据给出评分，"
    "不能因报告写作流畅而放松标准。评分标准要严格——多数研究报告默认只有 5-6 分而非 8-9 分。"
    "输出必须是严格的 JSON 格式。"
)


PROMPT_FACTUAL = """请对以下研究报告进行【事实核查】评分。

评分标准（0-10分）：
- 10分：所有可验证的事实（数字、日期、人名、机构名、统计数据）表述准确、前后一致，无明显的常识性错误。
- 7-9分：个别非核心事实表述不够精确，但不影响整体可信度。
- 4-6分：存在明显事实错误（日期错误、数据引用错误、前后矛盾）但核心论点仍成立。
- 1-3分：多处核心事实错误，严重损害报告可信度。
- 0分：大量事实完全错误，报告基本不可信。

审查要求：
1. 逐条提取报告中的 factual claims（数字、日期、比例、排名、人名、机构名等）。
2. 检查这些 claims 是否存在：
   - 前后矛盾（前文说A，后文说非A）
   - 明显违背常识
   - 数据/日期格式异常
   - 比例、排名与上下文不一致
3. 本维度不做外部来源比对，只检查报告内部一致性和常识合理性。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": 6.5,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "具体问题描述",
      "location": "问题位置，如第3段",
      "fix_type": "in_place|search|removal",
      "evidence": "原文引用或推理依据"
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}
"""


PROMPT_HALLUCINATION = """请对以下研究报告进行【幻觉检测】评分。

评分标准（0-10分）：
- 10分：报告中所有具体信息都有明确依据或已标注为推断，无疑似编造内容。
- 7-9分：存在少量"合理的推断"但未明确标注，可能误导读者。
- 4-6分：存在明显的无依据断言，尤其是具体数字、事件细节、精确日期、直接引语或因果关系。
- 1-3分：大量段落包含疑似编造信息。
- 0分：报告充斥着模型幻觉，几乎无可信内容。

审查要求：
1. 逐段检查是否存在以下疑似幻觉特征：
   - 具体数字、精确日期、百分比、排名无来源或上下文支撑
   - 直接引语未标注出处
   - 强因果关系缺乏证据
   - 看似真实但无法从报告中推导出的细节
2. 区分"合理推断"与"无依据断言"：推断应使用"可能"、" reportedly"等弱化表述，否则视为疑似幻觉。
3. 结合下方来源列表核对正文引用。引用编号存在且来源标题或摘要能够支持论断时，不得仅因来源权威性较低而判定为幻觉；来源权威性问题交由来源可信度维度处理。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": 6.5,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "具体问题描述",
      "location": "问题位置，如第3段",
      "fix_type": "in_place|search|removal",
      "evidence": "原文引用"
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}

--- 来源列表 ---
{sources}
"""


PROMPT_LOGICAL = """请对以下研究报告进行【逻辑一致性】评分。

评分标准（0-10分）：
- 10分：论证链条完整，前提与结论一致，无矛盾陈述。
- 7-9分：个别推断稍显跳跃，但不影响整体结论。
- 4-6分：存在内部矛盾（如前文说A，后文说非A）或因果谬误。
- 1-3分：多处逻辑断裂、自相矛盾，核心论点无法自洽。
- 0分：报告逻辑混乱，论证完全不可信。

审查要求：
1. 检查是否存在前后矛盾的陈述。
2. 检查因果关系是否合理（避免 post hoc / 因果倒置）。
3. 检查样本推断总体是否存在以偏概全。
4. 检查比较类论述的基准是否一致。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": 6.5,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "具体问题描述",
      "location": "问题位置，如第3段或引用标记",
      "fix_type": "in_place|search|removal",
      "evidence": "支撑证据或原文引用"
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}
"""


PROMPT_SOURCE_CREDIBILITY = """请对以下研究报告的【来源可信度】评分。

评分标准（0-10分）：
- 10分：所有来源均为高权威的一手资料（政府官网、顶级期刊、官方财报），且时效性强。
- 7-9分：以权威二手资料为主，个别来源时效稍旧但非核心数据。
- 4-6分：混有低权威来源（匿名论坛、未验证自媒体）且未做交叉验证。
- 1-3分：主要依赖低质量来源，或存在来源循环引用。
- 0分：无来源或来源完全不可信。

审查要求：
1. 评估每个 source 的域名权威性（.gov / .edu / 顶级媒体 / 自媒体 / 未知）。
2. 评估内容类型（一手数据 / 分析报道 / 社论 / 用户生成内容）。
3. 评估时效性：对于快速变化领域（科技、股市），1年以上为陈旧。
4. 检查一手程度：优先一手数据，二手分析需标注原始来源。
5. 核对正文中的数字引用 [N] 是否存在于下方去重来源列表，并判断该来源的标题、摘要与附近论断是否匹配。
6. 关键事实或精确数字缺少正文引用时，应降低来源可信度评分并报告具体位置。
7. 必须区分三种情况：没有来源、来源不支持论断、来源支持论断但权威性较低。第三种仅属于来源质量问题，不能判定为幻觉或无依据；可建议保留引用并注明来源性质，或寻找更权威来源交叉验证。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": 6.5,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "具体问题描述",
      "location": "问题位置，如第3段或引用标记",
      "fix_type": "in_place|search|removal",
      "evidence": "支撑证据或 source 原文"
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}

--- 来源列表 ---
{sources}
"""


PROMPT_COVERAGE = """请对以下研究报告的【覆盖完整度】评分。

评分标准（0-10分）：
- 10分：完全覆盖 query 要求的所有子话题，无重要遗漏，正反方观点均衡呈现，且每个子话题的讨论都基于相关搜索结果。
- 7-9分：覆盖了主要子话题，个别边缘视角缺失，但不影响核心结论。搜索结果与查询基本相关。
- 4-6分：遗漏了 query 隐含的关键子话题，或只呈现单方面观点。部分搜索结果可能与查询无关。
- 1-3分：严重跑题（例如搜索内容与查询主题无关）或大量子话题未覆盖。
- 0分：完全未回答 query。

审查要求：
1. 将 query 拆解为应覆盖的子话题列表。
2. 逐一检查每个子话题是否在报告中得到充分讨论。
3. 检查是否存在明显的立场偏差（只呈现正方而忽略反方）。
4. 检查时间维度是否覆盖（历史背景、现状、未来趋势，视 query 需求而定）。
5. CRITICAL: 检查报告中的 sources（搜索来源）是否与 query 主题相关。如果 sources 全是与 query 无关的网页（如搜"实习"却返回"科技趋势"），必须标记为 major/critical issue，并说明搜索内容与查询意图不匹配。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": 6.5,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "具体问题描述",
      "location": "问题位置，如第3段或引用标记",
      "fix_type": "in_place|search|removal",
      "evidence": "支撑证据或原文引用"
    }
  ]
}

--- 原始问题 ---
{query}

--- 研究报告 ---
{content}
"""


PROMPT_TEMPLATES = {
    AttackDimension.FACTUAL: PROMPT_FACTUAL,
    AttackDimension.HALLUCINATION: PROMPT_HALLUCINATION,
    AttackDimension.LOGIC: PROMPT_LOGICAL,
    AttackDimension.SOURCE: PROMPT_SOURCE_CREDIBILITY,
    AttackDimension.COVERAGE: PROMPT_COVERAGE,
}


class RedTeamAgent(BaseAgent):
    """红队 Agent：对最终报告进行事实/幻觉/逻辑/来源/覆盖五维单轮攻击。

    每个维度一次 system prompt + 一次 user prompt，直接输出 JSON。
    Red Agent 不调用任何工具。
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(name, policy, tools)
        self.config = config or {}

    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行 Red 攻击。

        支持两种模式：
          - 全维度模式：context 不含 "dimension"，依次攻击 5 个维度，返回 RedAttackResult。
          - 单维度模式：context 包含 "dimension" (AttackDimension)，只攻击该维度，
            返回 DimensionAttack。

        Args:
            task: task_type 应为 "red_agent"
            context: 必须包含 "report" (ResearchReport) 和 "query"
                可选包含 "round_no" (int)、"dimension" (AttackDimension)

        Returns:
            AgentResult，output 为 RedAttackResult 或 DimensionAttack。
        """
        report = context.get("report")
        query = context.get("query", "")
        round_no = context.get("round_no", 1)
        dimension = context.get("dimension")

        if not isinstance(report, ResearchReport):
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.FAILED,
                output="context 中缺少有效的 ResearchReport",
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        try:
            # Red Agent 禁用工具，单轮对话
            old_tools = getattr(self.policy, "tools", None)
            if hasattr(self.policy, "set_tools"):
                self.policy.set_tools(None)

            if isinstance(dimension, AttackDimension):
                # 单维度模式（Orchestrator 按维度串行调用）
                attack = await self.attack_dimension(report, query, dimension)
                if hasattr(self.policy, "set_tools") and old_tools is not None:
                    self.policy.set_tools(old_tools)

                return AgentResult(
                    task_id=task.id,
                    status=AgentStatus.SUCCESS,
                    output=attack,
                    trajectory=[
                        {
                            "turn": 0,
                            "role": "assistant",
                            "log": True,
                            "content": (
                                f"第 {round_no} 轮 · {dimension.value} 维度攻击完成："
                                f"评分 {attack.dimension_score:.1f}/10，"
                                f"发现 {len(attack.issues)} 个问题。\n\n{attack.analysis_summary}"
                            ),
                        },
                        {"turn": 0, "role": "tool", "name": "dimension_attack", "result": attack},
                    ],
                    token_usage=len(str(attack)) // 6,
                    confidence=attack.dimension_score / 10.0,
                )

            # 全维度模式（兼容旧测试/入口）
            dimension_attacks: dict[AttackDimension, DimensionAttack] = {}
            for dim in AttackDimension:
                attack = await self.attack_dimension(report, query, dim)
                dimension_attacks[dim] = attack

            if hasattr(self.policy, "set_tools") and old_tools is not None:
                self.policy.set_tools(old_tools)

            red_result = RedAttackResult(
                round_no=round_no,
                dimension_attacks=dimension_attacks,
            )
            red_result.overall_score = red_result.compute_overall_score()
            red_result.overall_summary = self._build_overall_summary(dimension_attacks)

            token_usage = sum(len(str(da)) for da in dimension_attacks.values()) // 6

            return AgentResult(
                task_id=task.id,
                status=AgentStatus.SUCCESS,
                output=red_result,
                trajectory=[
                    {
                        "turn": 0,
                        "role": "assistant",
                        "log": True,
                        "content": (
                            f"第 {round_no} 轮全维度攻击完成："
                            f"综合评分 {red_result.overall_score:.2f}/10。\n\n{red_result.overall_summary}"
                        ),
                    },
                    {"turn": 0, "role": "tool", "name": "red_attack", "result": red_result},
                ],
                token_usage=token_usage,
                confidence=red_result.overall_score / 10.0,
            )

        except Exception as e:
            logger.exception("RedTeamAgent 攻击失败")
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.FAILED,
                output=f"Red attack failed: {type(e).__name__}: {e}",
                trajectory=[{"turn": 0, "role": "assistant", "log": True, "content": f"攻击失败：{type(e).__name__}: {e}"}],
                token_usage=0,
                confidence=0.0,
            )

    async def attack_dimension(
        self,
        report: ResearchReport,
        query: str,
        dimension: AttackDimension,
    ) -> DimensionAttack:
        """对单个维度执行一次单轮攻击（供 Orchestrator 按维度串行调用）。"""
        template = PROMPT_TEMPLATES[dimension]
        sources_text = self._format_sources(report.sources)

        # 幻觉和来源维度需要核对引用；其他维度只做轻量级内部审查
        if dimension in {AttackDimension.HALLUCINATION, AttackDimension.SOURCE}:
            user_prompt = self._format_prompt(
                template, query=query, content=report.content, sources=sources_text
            )
        else:
            user_prompt = self._format_prompt(template, query=query, content=report.content)

        messages = [
            {"role": "system", "content": SYSTEM_RED_AGENT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            "RedTeamAgent 开始调用 %s 维度 LLM，报告长度 %d 字符",
            dimension.value,
            len(report.content),
        )
        resp = await asyncio.to_thread(self.policy.chat, messages)
        logger.info("RedTeamAgent %s 维度 LLM 调用返回", dimension.value)
        content = resp.content or ""

        parsed = self._parse_dimension_json(content, dimension)
        return parsed

    def _format_prompt(self, template: str, **kwargs) -> str:
        """安全格式化 prompt：转义 JSON 中的花括号，只保留已知占位符。"""
        escaped = template.replace("{", "{{").replace("}", "}}")
        for key in kwargs:
            escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")
        return escaped.format(**kwargs)

    def _format_sources(self, sources: list[dict]) -> str:
        """按 URL 去重后，将真实来源列表格式化给来源维度评分。"""
        if not sources:
            return "无来源"
        unique_sources = prepare_sources(sources)
        if not unique_sources:
            return "无来源"
        lines = []
        for s in unique_sources:
            title = s.get("title", "")
            url = s.get("url", "")
            snippet = s.get("snippet", "")
            lines.append(
                f"[{s['citation_id']}] {title}\nURL: {url}\n摘要: {snippet}\n"
            )
        return "\n".join(lines)

    def _parse_dimension_json(self, content: str, dimension: AttackDimension) -> DimensionAttack:
        """解析单个维度的 JSON 输出，带容错清洗。"""
        raw = self._extract_json(content)
        if raw is None:
            logger.warning("RedTeamAgent 无法从 %s 维度输出中提取 JSON", dimension.value)
            logger.debug("RedTeamAgent %s 维度原始输出: %s", dimension.value, content[:2000])
            return DimensionAttack(
                dimension=dimension,
                dimension_score=0.0,
                analysis_summary="JSON 解析失败",
                issues=[],
            )

        # 先尝试直接解析；失败则用 json_repair 容错修复，再走手动清洗
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            try:
                data = json_repair_loads(raw)
            except Exception:
                repaired = self._repair_json(raw)
                if repaired is None:
                    logger.warning(
                        "RedTeamAgent %s 维度 JSON 解码失败: %s", dimension.value, e
                    )
                    logger.debug(
                        "RedTeamAgent %s 维度提取片段: %s", dimension.value, raw[:2000]
                    )
                    return DimensionAttack(
                        dimension=dimension,
                        dimension_score=0.0,
                        analysis_summary=f"JSON 解码失败: {e}",
                        issues=[],
                    )
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError as e2:
                    logger.warning(
                        "RedTeamAgent %s 维度 JSON 修复后仍解码失败: %s", dimension.value, e2
                    )
                    logger.debug(
                        "RedTeamAgent %s 维度提取片段: %s", dimension.value, raw[:2000]
                    )
                    return DimensionAttack(
                        dimension=dimension,
                        dimension_score=0.0,
                        analysis_summary=f"JSON 解码失败: {e2}",
                        issues=[],
                    )

        if not isinstance(data, dict):
            # json_repair 偶尔会把全角标点文本解析成 list，再用手动清洗兜底
            repaired = self._repair_json(raw)
            if repaired is not None:
                try:
                    data = json.loads(repaired)
                except Exception:
                    pass
            if not isinstance(data, dict):
                logger.debug(
                    "RedTeamAgent %s 维度解析结果不是 JSON 对象: %s",
                    dimension.value,
                    type(data),
                )
                return DimensionAttack(
                    dimension=dimension,
                    dimension_score=0.0,
                    analysis_summary="JSON 解析结果类型错误",
                    issues=[],
                )

        dimension_score = float(data.get("score", 0.0))

        issues: list[Issue] = []
        for item in data.get("issues", []):
            if not isinstance(item, dict):
                continue
            try:
                issue = Issue(
                    dimension=dimension,
                    severity=Severity(str(item.get("severity", "minor")).lower()),
                    location=str(item.get("location", "")),
                    description=str(item.get("description", "")),
                    fix_type=FixType(str(item.get("fix_type", "in_place")).lower()),
                    evidence=str(item.get("evidence", "")),
                )
                issues.append(issue)
            except (ValueError, TypeError) as e:
                logger.warning("RedTeamAgent 解析 issue 失败: %s", e)
                continue

        # 限制每个维度返回的 issue 数量，优先保留高严重程度的问题
        max_issues = int(self.config.get("adversarial", {}).get("max_issues_per_dimension", 3))
        if len(issues) > max_issues:
            severity_rank = {Severity.CRITICAL: 3, Severity.MAJOR: 2, Severity.MINOR: 1}
            issues = sorted(
                issues,
                key=lambda i: severity_rank.get(i.severity, 0),
                reverse=True,
            )[:max_issues]
            logger.info(
                "RedTeamAgent %s 维度 issue 过多，已按严重程度截断至 %d 个",
                dimension.value,
                max_issues,
            )

        return DimensionAttack(
            dimension=dimension,
            dimension_score=dimension_score,
            issues=issues,
        )

    def _extract_json(self, text: str) -> str | None:
        """从文本中提取第一个 JSON 对象。

        优先匹配 ```json 代码块；否则用花括号平衡匹配定位最外层对象；
        最后再回退到简单正则，给 _repair_json 兜底。也会尝试全角花括号。
        """
        if not text:
            return None

        stripped = text.strip()

        # 1. 优先匹配 markdown 代码块（非贪婪）
        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # 2. 平衡匹配最外层花括号对象（避免 raw_decode 命中内层合法子对象）
        outer = self._find_balanced_braces(stripped)
        if outer is not None:
            return outer.strip()

        # 3. 兜底：第一个 { 到最后一个 } 的片段
        brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if brace_match:
            return brace_match.group(0).strip()

        # 4. 全角花括号兜底：先归一化再递归重试一次
        normalized = stripped.replace("｛", "{").replace("｝", "}")
        if normalized != stripped:
            return self._extract_json(normalized)

        return None

    def _find_balanced_braces(self, text: str) -> str | None:
        """找到文本中第一个平衡的花括号对象（简单处理字符串转义）。"""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _repair_json(self, raw: str) -> str | None:
        """修复 LLM 常见的 JSON 格式噪声。

        依次尝试：去 markdown 围栏、去尾部逗号、去 // 行注释、全角标点归一化。
        返回第一个能成功解析的候选字符串，都不行返回 None。
        """
        if not raw:
            return None

        candidates = []

        # 候选 1：基本清洗（去掉残留代码块围栏 + 尾部逗号）
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        candidates.append(cleaned)

        # 候选 2：去除 // 行注释
        no_comment_lines = []
        for line in cleaned.splitlines():
            if "//" in line:
                line = line[: line.index("//")]
            no_comment_lines.append(line)
        candidates.append("\n".join(no_comment_lines))

        # 候选 3：全角标点归一化（兜底，可能轻微改变字符串内容）
        normalized = cleaned
        for src, dst in (
            ("｛", "{"),
            ("｝", "}"),
            ("［", "["),
            ("］", "]"),
            ("：", ":"),
            ("，", ","),
        ):
            normalized = normalized.replace(src, dst)
        # 全角引号（最后兜底）
        normalized = (
            normalized.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        candidates.append(normalized)

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

        return None

    def _build_overall_summary(self, dimension_attacks: dict[AttackDimension, DimensionAttack]) -> str:
        """构建整体攻击总结。"""
        parts = ["本轮 Red 攻击汇总："]
        for dim in AttackDimension:
            da = dimension_attacks.get(dim)
            if da is None:
                continue
            issue_count = len(da.issues)
            parts.append(
                f"- {dim.value}: 维度得分 {da.dimension_score:.1f}/10，发现 {issue_count} 个问题"
            )
        return "\n".join(parts)
