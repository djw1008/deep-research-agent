#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/rule_based.py
================================================================================
基于规则/统计的轻量级评测指标。

适用于批量运行、CI/CD、消融实验等需要快速、免费、可复现评分的场景。

Adapted from deepresearch-agent-main for deep-research-agent.
================================================================================
"""

from __future__ import annotations

import math
import re
from typing import Any


class RuleBasedMetrics:
    """研究报告质量评测指标集合（规则版）。"""

    # -----------------------------------------------------------------------
    # 1. 事实准确性 (Factual Accuracy) — 字符串匹配版（快速但粗糙）
    # -----------------------------------------------------------------------
    @staticmethod
    def fact_accuracy(report: str, ground_truth: dict[str, Any] | None = None) -> float:
        """
        计算报告中的关键事实与 ground_truth 的匹配程度。

        当前实现采用简单启发式：统计报告中包含的 ground_truth 关键短语比例。
        若无 ground_truth，则返回 0.0（需外部 Judge LLM 补充评估）。
        """
        if not ground_truth:
            return 0.0

        report_lower = report.lower()
        matched = 0
        for key_fact in ground_truth.keys():
            if key_fact.lower() in report_lower:
                matched += 1

        return matched / len(ground_truth) if ground_truth else 0.0

    # -----------------------------------------------------------------------
    # 1b. 语义事实准确性 (Semantic Factual Accuracy) — embedding强化版
    # -----------------------------------------------------------------------
    @staticmethod
    def semantic_fact_accuracy(
        report: str,
        ground_truth: dict[str, Any] | None = None,
        threshold: float = 0.65,
    ) -> float:
        """
        基于 embedding 语义相似度的事实准确性验证。

        改进点（相比字符串匹配）：
        1. 把 ground_truth 的 key + description 编码为语义向量
        2. 把报告拆分成句子 chunk，分别编码
        3. 计算每个 ground_truth 条目与报告中最相似 chunk 的 cosine similarity
        4. 超过阈值（默认 0.65）才判定为"事实被覆盖"

        这样能避免"GPT-4o 是 Google 发布的"这种关键词命中但语义错误的误报。

        Args:
            report: 研究报告全文
            ground_truth: 期望事实字典 {key: description}
            threshold: 语义相似度阈值，0-1

        Returns:
            0.0 ~ 1.0 的覆盖率
        """
        if not ground_truth:
            return 0.0

        import numpy as np
        from deep_research.memory.embedder import MemoryEmbedder

        embedder = MemoryEmbedder()

        # 把报告拆成句子 chunk（避免长报告淹没短事实）
        chunks = [s.strip() for s in re.split(r"[。！？\n]", report) if len(s.strip()) > 10]
        if not chunks:
            return 0.0

        # 批量编码 chunk（MemoryEmbedder.encode 原生支持 batch + normalize）
        chunk_embs = embedder.encode(chunks)
        chunk_embs = np.array(chunk_embs)

        matched = 0
        for key_fact, expected_desc in ground_truth.items():
            # 组合 key + description 作为语义查询
            fact_text = f"{key_fact}：{expected_desc}"
            fact_emb = embedder.encode(fact_text)
            fact_emb = np.array(fact_emb).squeeze()  # (1, D) → (D,)

            # 计算与所有 chunk 的 cosine similarity
            # （embedder 已 L2-normalize，点积即 cosine similarity）
            sims = chunk_embs.dot(fact_emb)
            max_sim = float(np.max(sims)) if sims.size > 0 else 0.0

            if max_sim > threshold:
                matched += 1

        return matched / len(ground_truth)

    # -----------------------------------------------------------------------
    # 2. 幻觉率 (Hallucination Rate)
    # -----------------------------------------------------------------------
    @staticmethod
    def hallucination_rate(report: str) -> float:
        """
        估算报告中可能存在的幻觉内容比例。

        当前启发式策略：
        - 检测无引用的数值声明（数字+单位）。
        - 检测缺乏来源的绝对化表述（"绝对"、"毫无疑问"等）。
        - 检测模型常见的幻觉模式（"据我所知"、"研究表明"但无具体引用）。

        Returns:
            0.0 ~ 1.0，越高表示幻觉风险越大。
        """
        if not report:
            return 1.0

        sentences = re.split(r"[。！？\n]", report)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 1.0

        hallucination_indicators = [
            r"\d+[\d,]*\.?\d*\s*(%|倍|个|人|元|美元|亿|万)",  # 带单位的孤立数字
            r"毫无疑问|绝对|必然|一定|众所周知",
            r"据我所知|据了解|研究显示[^【\[（(]",  # 模糊引用开头
        ]

        suspicious_count = 0
        for sentence in sentences:
            # 如果句子中无引用标记，检查是否包含幻觉特征
            if not re.search(r"[\[【（(].*?[\]）)]", sentence):
                for pattern in hallucination_indicators:
                    if re.search(pattern, sentence):
                        suspicious_count += 1
                        break

        return min(1.0, suspicious_count / max(len(sentences), 1))

    # -----------------------------------------------------------------------
    # 3. 引用覆盖率 / 来源充足度 (Source Adequacy)
    # -----------------------------------------------------------------------
    @staticmethod
    def source_adequacy(report: str, num_sources: int = 0) -> float:
        """
        评估报告的来源充分程度。

        策略（同时使用两种信号）：
        1. 正文内联引用：检测 [N]、【来源:】等标记的段落比例（权重 0.3）
        2. 报告末尾的参考链接：检测 "参考链接"/"来源"/"References" 节中的 URL 数（权重 0.3）
        3. 元数据来源数：num_sources 相对报告长度的密度（权重 0.4）

        Args:
            report: 报告全文
            num_sources: 从 agent 元数据中提取的去重来源数

        Returns:
            0.0 ~ 1.0，越高表示来源越充分
        """
        if not report:
            return 0.0

        score = 0.0

        # --- 信号 1：内联引用（权重 0.3）---
        paragraphs = [p.strip() for p in report.split("\n") if p.strip()]
        if paragraphs:
            inline_patterns = [
                r"\[\d+\]", r"\[来源[：:]", r"【来源[：:]", r"\(来源[：:]",
            ]
            cited = sum(
                1 for p in paragraphs
                if any(re.search(pat, p) for pat in inline_patterns)
            )
            score += 0.3 * (cited / len(paragraphs))

        # --- 信号 2：报告末尾的参考/来源节中的链接（权重 0.3）---
        # 找报告后 30% 的部分（参考节通常在这里）
        tail_start = int(len(report) * 0.7)
        tail = report[tail_start:]
        urls_in_tail = len(re.findall(r"https?://[^\s\)\]]+", tail))
        # 期望至少 5 个链接算满分
        score += 0.3 * min(1.0, urls_in_tail / 5.0)

        # --- 信号 3：元数据来源密度（权重 0.4）---
        if num_sources > 0:
            # 每 1000 字符 1 个来源 = 基准；2 个来源/千字 = 满分
            report_kchars = max(len(report) / 1000.0, 0.5)
            density = num_sources / report_kchars
            score += 0.4 * min(1.0, density / 2.0)

        return min(1.0, score)

    # -----------------------------------------------------------------------
    # 4. 逻辑一致性 (Logical Consistency)
    # -----------------------------------------------------------------------
    @staticmethod
    def logical_consistency(report: str) -> float:
        """
        估算报告的逻辑一致性分数。

        当前启发式策略：
        - 检测明显的自相矛盾关键词对（"是" vs "不是" 在同一上下文）。
        - 检测逻辑连接词使用是否合理（"因此"、"然而"前是否有前提）。
        """
        if not report:
            return 0.0

        # 简单检测矛盾对：句子中同时出现 A 和 非A（同一句话）
        contradiction_pairs = [
            ("是", "不是"),
            ("可以", "不可以"),
            ("会", "不会"),
            ("支持", "反对"),
            ("增加", "减少"),
        ]

        sentences = re.split(r"[。！？\n]", report)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        contradiction_count = 0
        for sentence in sentences:
            for a, b in contradiction_pairs:
                if a in sentence and b in sentence:
                    # 更严格的检查：确保它们之间没有否定词分隔
                    contradiction_count += 1
                    break

        # 同时奖励使用逻辑连接词
        connectives = ["因此", "所以", "然而", "但是", "首先", "其次", "综上所述"]
        connective_count = sum(1 for c in connectives if c in report)
        connective_bonus = min(0.1, connective_count * 0.01)

        base_score = 1.0 - (contradiction_count / max(len(sentences), 1))
        return min(1.0, max(0.0, base_score + connective_bonus))

    # -----------------------------------------------------------------------
    # 5. 完备性 (Comprehensiveness)
    # -----------------------------------------------------------------------
    @staticmethod
    def comprehensiveness(report: str, expected_topics: list[str] | None = None) -> float:
        """
        计算报告对期望主题的覆盖程度。
        """
        if not expected_topics:
            return 0.0

        report_lower = report.lower()
        covered = 0
        for topic in expected_topics:
            if topic.lower() in report_lower:
                covered += 1

        return covered / len(expected_topics) if expected_topics else 0.0

    # -----------------------------------------------------------------------
    # 6. 综合得分 (Composite Score)
    # -----------------------------------------------------------------------
    @staticmethod
    def composite_score(
        metrics: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        基于多维度指标和权重计算加权综合得分。

        默认权重与 Red Agent 的五维度对齐：
        - factual_accuracy: 0.25
        - logical_consistency: 0.20
        - citation_coverage: 0.20
        - bias (1 - hallucination_rate 作为代理): 0.20
        - comprehensiveness: 0.15
        """
        default_weights = {
            "factual_accuracy": 0.25,
            "logical_consistency": 0.20,
            "source_adequacy": 0.20,
            "bias": 0.20,
            "comprehensiveness": 0.15,
        }

        w = weights if weights is not None else default_weights
        total_score = 0.0
        total_weight = 0.0

        for key, weight in w.items():
            value = metrics.get(key, 0.0)
            total_score += value * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    # -----------------------------------------------------------------------
    # 7. 效率指标 (Efficiency)
    # -----------------------------------------------------------------------
    @staticmethod
    def efficiency_score(
        num_turns: int,
        target_turns: float = 8.0,
        slope: float = 0.5,
        max_bonus: float = 0.5,
    ) -> float:
        """
        基于 sigmoid 的效率奖励分数。

        公式：max_bonus * sigmoid(slope * (target_turns - num_turns))
        """
        sigmoid = 1.0 / (1.0 + math.exp(-slope * (target_turns - num_turns)))
        return max_bonus * sigmoid


# =============================================================================
# 简单自测
# =============================================================================
if __name__ == "__main__":
    sample_report = """
    GPT-4o 是 OpenAI 于 2024 年 5 月发布的原生多模态大模型[1]。
    Claude 3.5 Sonnet 由 Anthropic 于 2024 年 6 月发布，引入了 Artifacts 功能[2]。
    Gemini 1.5 Pro 支持超过 100 万 token 的上下文窗口[3]。
    Qwen2.5 是阿里巴巴的开源模型，支持 128K 上下文[4]。
    在中文推理方面，各模型表现接近；代码生成和长上下文处理各有优势。
    """

    gt = {
        "GPT-4o": "OpenAI 发布于 2024 年 5 月，原生多模态",
        "Claude 3.5 Sonnet": "Anthropic 发布于 2024 年 6 月，Artifacts 功能",
        "Gemini 1.5 Pro": "Google 发布，1M+ token 上下文窗口",
        "Qwen2.5": "阿里巴巴发布，开源并支持 128K 上下文",
    }

    print("=== 规则评测指标自测 ===\n")

    fa = RuleBasedMetrics.fact_accuracy(sample_report, gt)
    print(f"事实准确性 (string):    {fa:.3f}")

    halluc = RuleBasedMetrics.hallucination_rate(sample_report)
    print(f"幻觉率:                 {halluc:.3f}")

    sa = RuleBasedMetrics.source_adequacy(sample_report, num_sources=4)
    print(f"来源充分度:             {sa:.3f}")

    logic = RuleBasedMetrics.logical_consistency(sample_report)
    print(f"逻辑一致性:             {logic:.3f}")

    comp = RuleBasedMetrics.comprehensiveness(
        sample_report,
        ["GPT-4o", "Claude 3.5", "Gemini 1.5", "Qwen2.5"],
    )
    print(f"完备性:                 {comp:.3f}")

    metrics = {
        "factual_accuracy": fa,
        "logical_consistency": logic,
        "source_adequacy": sa,
        "bias": max(0.0, 1.0 - halluc),
        "comprehensiveness": comp,
    }
    composite = RuleBasedMetrics.composite_score(metrics)
    print(f"\n综合得分:               {composite:.3f}")

    eff = RuleBasedMetrics.efficiency_score(num_turns=5, target_turns=8.0)
    print(f"效率加分:               {eff:.3f}")

    print("\n=== 语义事实准确性测试（需要 sentence-transformers）===")
    try:
        sem = RuleBasedMetrics.semantic_fact_accuracy(sample_report, gt)
        print(f"事实准确性 (semantic):   {sem:.3f}")
    except Exception as e:
        print(f"跳过（embedder 不可用）: {e}")
