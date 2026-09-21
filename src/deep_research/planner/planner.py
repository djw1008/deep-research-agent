"""自适应规划器 — 将研究问题分解为子任务 DAG。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from ..core.schema import SubTask
from ..models import LLMClient
from .dag import DAG, DAGCycleError

logger = logging.getLogger(__name__)


class PlanParseError(Exception):
    """规划结果解析失败时抛出。"""


_INITIAL_PLAN_PROMPT = """\
你是一位研究规划专家。将一个复杂研究问题分解为子任务的有向无环图（DAG）。

## 输入
研究问题: {query}

{memory_context}

## 输出格式
返回严格如下结构的 JSON（不要 markdown，不要额外文字）:
{{
  "sub_tasks": [
    {{
      "task_id": "task_1",
      "task_type": "search",
      "description": "具体可执行的任务描述",
      "dependencies": [],
      "search_hints": ["关键词1", "关键词2"],
      "timeout_seconds": 360,
      "priority": 1,
      "expected_type": "factual"
    }}
  ]
}}

## 规则
1. task_type 只能是 search、analyze、verify 之一
2. dependencies 必须引用已存在的 task_id
3. 图必须是 DAG，不能有循环依赖
4. 生成 3 到 8 个子任务
5. 基础信息收集任务依赖应最少（放底层），验证任务依赖分析任务（放上层）
6. 每个子任务描述必须直接回应研究问题，禁止生成无关任务
7. search_hints 必须从用户问题中直接提取关键词
8. 当研究问题涉及多个对象的比较时，底层收集任务应按对象聚合（每个对象一个任务，收集该对象的全部相关信息），上层再安排跨对象对比/分析任务。禁止在底层按维度交叉拆分多个对象，否则会导致信息碎片化、来源混杂。
9. 如果提供了"历史相关研究记忆"，请参考已有结论避免重复搜索，并重点关注当前问题的新角度或未覆盖内容；如果历史记忆与当前问题关联不大，请忽略。

## 反例
- 用户问"如何在大厂找实习" → 错误: "2025科技趋势""年度科学新闻"
- 用户问"如何准备LLM后训练工程师实习" → 正确: "大厂后训练实习生JD要求""LLM后训练实习面经"
"""

_REPLAN_PROMPT = """\
你是一位研究规划专家。部分子任务执行失败，需要重新规划。

## 原始问题
{query}

## 失败任务
{failed_tasks_json}

## 需要保留的成功结果
{preserved_results_json}

## 失败原因
{reason}

## 输出格式
返回新的 sub_tasks JSON。你可以:
1. 修改失败任务（可换 task_id，可改描述）
2. 添加新任务填补空缺
3. 删除不再需要的任务
4. 保持依赖关系一致合法

只返回 JSON，不要 markdown，不要额外文字。
"""

_CONTINUATION_PLAN_PROMPT = """\
直接输出一个合法 JSON 对象，用于继续深入研究。禁止输出 markdown 代码块、禁止解释、禁止分析、禁止搜索、禁止闲聊。只输出 JSON。

## 输出格式
返回严格如下结构的 JSON:
{{
  "sub_tasks": [
    {{
      "task_id": "task_1",
      "task_type": "search",
      "description": "具体可执行的任务描述",
      "dependencies": [],
      "search_hints": ["关键词1", "关键词2"],
      "timeout_seconds": 360,
      "priority": 1,
      "expected_type": "factual"
    }}
  ],
  "reasoning": "简要说明本次规划如何承接历史研究，重点补充哪些方向"
}}

## 历史研究上下文
{continuation_context}

{memory_context}

## 当前新的研究方向
{query}

## 规则
1. task_type 只能是 search、analyze、verify 之一
2. dependencies 必须引用已存在的 task_id
3. 图必须是 DAG，不能有循环依赖
4. 生成 2 到 6 个新子任务
5. 避免重复历史中已经充分研究的子任务
6. 重点针对历史报告的薄弱点、遗漏点、用户新方向设计任务
7. 如需验证或更新旧结论，请明确设计验证任务
8. 如果历史上下文与当前问题关联不大，可以当作全新问题规划
"""


class Planner:
    """自适应规划器。"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._last_raw_json: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_plan(
        self, query: str, memory_context: str = ""
    ) -> tuple[DAG, list[SubTask]]:
        """生成初始执行计划。

        Args:
            query: 用户研究问题。
            memory_context: 从持久记忆召回的相关上下文，供 Planner 参考。
        """
        prompt = _INITIAL_PLAN_PROMPT.format(
            query=query,
            memory_context=memory_context,
        )
        messages = [
            {"role": "system", "content": "你是一位研究规划专家。你的唯一输出必须是一个合法 JSON 对象。禁止输出 markdown 代码块、禁止解释、禁止分析、禁止搜索、禁止闲聊。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]

        logger.info("【规划提示词】\n%s", prompt)

        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            resp = await asyncio.to_thread(self.client.chat, messages)
            if resp.content.startswith("Error:"):
                raise PlanParseError(f"LLM 调用失败: {resp.content}")

            self._last_raw_json = resp.content
            try:
                return self._parse_plan(resp.content)
            except PlanParseError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "规划解析失败（第 %d/%d 次），重试中... 错误: %s",
                        attempt + 1, max_retries + 1, str(e)[:200],
                    )
                    # 重试时提示模型修正
                    messages.append({
                        "role": "user",
                        "content": (
                            "你上次的输出无法解析为有效 JSON。"
                            "请只输出纯 JSON 对象（不要 markdown 代码块、不要解释），"
                            "格式必须为: {\"sub_tasks\": [...]}"
                        ),
                    })
                else:
                    raise last_error

    async def generate_continuation_plan(
        self,
        query: str,
        continuation_context: str,
        memory_context: str = "",
    ) -> tuple[DAG, list[SubTask]]:
        """Generate a plan that continues from previous research rounds.

        Args:
            query: The new research direction from the user.
            continuation_context: Formatted history of previous rounds
                (query + DAG + report for each round).
            memory_context: Optional knowledge-base context.
        """
        if not memory_context or not memory_context.strip():
            memory_context = "（无额外知识库上下文）"
        prompt = _CONTINUATION_PLAN_PROMPT.format(
            query=query,
            continuation_context=continuation_context,
            memory_context=memory_context,
        )
        messages = [
            {"role": "system", "content": "你是一位研究规划专家。当前研究是对同一话题的继续深入。你的唯一输出必须是一个合法 JSON 对象。禁止输出 markdown 代码块、禁止解释、禁止分析、禁止搜索、禁止闲聊。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]

        logger.info("【继续研究规划提示词】\n%s", prompt)

        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            resp = await asyncio.to_thread(self.client.chat, messages)
            if resp.content.startswith("Error:"):
                raise PlanParseError(f"LLM 调用失败: {resp.content}")

            self._last_raw_json = resp.content
            try:
                return self._parse_plan(resp.content)
            except PlanParseError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "继续规划解析失败（第 %d/%d 次），重试中...",
                        attempt + 1, max_retries + 1,
                    )
                    messages.append({
                        "role": "user",
                        "content": "你上次的输出无法解析。请只输出纯 JSON: {\"sub_tasks\": [...]}",
                    })
                else:
                    raise last_error

    async def replan(
        self,
        query: str,
        failed_tasks: list[SubTask],
        preserved_results: dict[str, Any],
        reason: str,
    ) -> tuple[DAG, list[SubTask]]:
        """增量重规划：保留成功结果，修改失败任务。"""
        failed_json = json.dumps(
            [{"task_id": t.id, "description": t.description, "type": t.task_type} for t in failed_tasks],
            ensure_ascii=False,
            indent=2,
        )
        preserved_json = json.dumps(
            [{"task_id": k, "output": str(v)[:500]} for k, v in preserved_results.items()],
            ensure_ascii=False,
            indent=2,
        )

        prompt = _REPLAN_PROMPT.format(
            query=query,
            failed_tasks_json=failed_json,
            preserved_results_json=preserved_json,
            reason=reason,
        )
        messages = [
            {"role": "system", "content": "你是一位研究规划专家。部分子任务执行失败，需要重新规划。你的唯一输出必须是一个合法 JSON 对象。禁止输出 markdown 代码块、禁止解释、禁止分析、禁止搜索、禁止闲聊。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]

        logger.info("【重规划提示词】\n%s", prompt)

        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            resp = await asyncio.to_thread(self.client.chat, messages)
            if resp.content.startswith("Error:"):
                raise PlanParseError(f"LLM 调用失败: {resp.content}")

            self._last_raw_json = resp.content
            try:
                return self._parse_plan(resp.content)
            except PlanParseError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "重规划解析失败（第 %d/%d 次），重试中...",
                        attempt + 1, max_retries + 1,
                    )
                    messages.append({
                        "role": "user",
                        "content": "你上次的输出无法解析。请只输出纯 JSON: {\"sub_tasks\": [...]}",
                    })
                else:
                    raise last_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_plan(self, json_str: str) -> tuple[DAG, list[SubTask]]:
        """健壮性 JSON 解析：处理 markdown 代码块、尾部逗号等噪声。"""
        raw = json_str.strip()

        # 去 markdown 代码块
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()

        # 提取 JSON 对象：第一个 '{' 到最后一个 '}'
        obj_match = re.search(r"(\{.*\})", raw, re.DOTALL)
        if obj_match:
            raw = obj_match.group(1).strip()
        else:
            # replan 等场景可能直接返回数组
            array_match = re.search(r"(\[.*\])", raw, re.DOTALL)
            if array_match:
                raw = array_match.group(1).strip()

        # 去尾部逗号
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)

        # 解析 JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            cleaned_lines = []
            for line in raw.splitlines():
                if "//" in line:
                    line = line[: line.index("//")]
                cleaned_lines.append(line)
            try:
                data = json.loads("\n".join(cleaned_lines))
            except json.JSONDecodeError:
                if "{" not in json_str and "[" not in json_str:
                    raise PlanParseError(
                        f"规划模型未返回 JSON，而是返回了自然语言。原始片段: {json_str[:1000]}"
                    ) from e
                raise PlanParseError(
                    f"无法解析规划输出为 JSON。原始片段: {json_str[:1000]}"
                ) from e

        # replan 时 LLM 可能直接返回数组，需要包装
        if isinstance(data, list):
            data = {"sub_tasks": data}

        if not isinstance(data, dict) or "sub_tasks" not in data:
            raise PlanParseError(f"规划输出缺少 'sub_tasks' 键。键: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        logger.debug("Planner 原始 JSON:\n%s", json.dumps(data, ensure_ascii=False, indent=2)[:2000])

        sub_tasks_raw = data["sub_tasks"]
        if not isinstance(sub_tasks_raw, list):
            raise PlanParseError(f"'sub_tasks' 必须是列表，得到 {type(sub_tasks_raw)}")

        # 构建 DAG
        dag = DAG()
        subtasks: list[SubTask] = []

        # 第一遍：加节点
        for item in sub_tasks_raw:
            task = self._deserialize_subtask(item)
            dag.add_node(task.id)
            subtasks.append(task)

        # 第二遍：加边
        for item in sub_tasks_raw:
            task_id = item.get("task_id", "")
            for dep in item.get("dependencies", []):
                if not dag.has_node(dep):
                    dag.add_node(dep)
                dag.add_edge(dep, task_id)

        # 验证无环
        try:
            dag.topological_sort()
        except DAGCycleError as e:
            raise PlanParseError(f"规划器生成了循环依赖图: {e}") from e

        return dag, subtasks

    def _deserialize_subtask(self, item: dict[str, Any]) -> SubTask:
        """将 JSON dict 反序列化为 SubTask 对象。"""
        task_type = item.get("task_type", "search")
        if task_type not in ("search", "analyze", "verify"):
            task_type = "search"

        return SubTask(
            id=item.get("task_id", "unknown"),
            description=item.get("description", ""),
            task_type=task_type,
            dependencies=list(item.get("dependencies", [])),
            search_hints=list(item.get("search_hints", [])),
            timeout_seconds=int(item.get("timeout_seconds", 360)),
            priority=int(item.get("priority", 1)),
            expected_type=item.get("expected_type", "factual"),
        )
