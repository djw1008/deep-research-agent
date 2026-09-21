"""三级上下文压缩器。

L1: 文章级 —— 输入 list[str]，对多篇文本做列表级操作
L2: 段落级 —— 输入 str，在单篇文章内部逐句过滤 (TextRank + query-biased + 高价值 bonus)
L3: LLM 摘要级 —— 输入 list[str]，调用 LLM 两阶段生成摘要（单篇摘要 → 聚合综述）
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# L3: LLM 摘要提示词
# ---------------------------------------------------------------------------

_DOCUMENT_SUMMARY_PROMPT = """请对以下文档进行摘要，要求：
1. 保留所有关键数字、日期、统计数据
2. 保留来源引用和作者信息
3. 保留不确定性表述（如"约""可能""据报道""初步结果显示"）
4. 摘要长度不超过 {max_length} 字
5. 使用中文输出

文档内容：
{doc}

摘要："""

_AGGREGATE_SUMMARY_PROMPT = """请将以下多篇文档摘要整合为一份连贯的综述，要求：
1. 合并重复信息，保留不同文档间的互补内容
2. 保留所有关键数字、日期、统计数据
3. 保留来源引用
4. 保留不确定性表述
5. 若文档间存在矛盾，请分别列出不同观点并标注来源
6. 总长度不超过 {max_length} 字
7. 使用中文输出

用户查询背景：{query}

文档摘要列表：
{docs}

综述："""


# ---------------------------------------------------------------------------
# 轻量级 n-gram 向量化器（默认 embedder，零外部依赖）
# ---------------------------------------------------------------------------

class _NGramEmbedder:
    """字符级 n-gram 向量化器。

    先 fit_transform 构建词表，再 transform 编码新文本。
    所有输出向量均已 L2 归一化。
    """

    def __init__(self, n: int = 2) -> None:
        self.n = n
        self.vocab: list[str] | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """构建词表并编码所有文本，返回 L2 归一化矩阵。"""
        vocab = set()
        raw_vecs = []
        for text in texts:
            text = text.lower().strip()
            vec = {}
            for i in range(len(text) - self.n + 1):
                gram = text[i : i + self.n]
                vec[gram] = vec.get(gram, 0) + 1
                vocab.add(gram)
            raw_vecs.append(vec)

        self.vocab = sorted(vocab)
        matrix = np.zeros((len(texts), len(self.vocab)))
        for i, vec in enumerate(raw_vecs):
            for j, gram in enumerate(self.vocab):
                matrix[i, j] = vec.get(gram, 0)

        # L2 归一化
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def transform(self, text: str) -> np.ndarray:
        """在已有词表下编码单个文本，返回 L2 归一化向量。"""
        if self.vocab is None:
            raise RuntimeError("先调用 fit_transform 构建词表")

        text = text.lower().strip()
        vec = {}
        for i in range(len(text) - self.n + 1):
            gram = text[i : i + self.n]
            vec[gram] = vec.get(gram, 0) + 1

        arr = np.zeros(len(self.vocab))
        for j, gram in enumerate(self.vocab):
            arr[j] = vec.get(gram, 0)

        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr


# ---------------------------------------------------------------------------
# 压缩器主类
# ---------------------------------------------------------------------------

class ContextCompressor:
    """三级上下文压缩器。L1 处理列表，L2/L3 处理单篇。

    L2 默认使用中文向量模型 BAAI/bge-small-zh-v1.5 做向量化，
    也可通过 embedder 参数传入自定义模型。
    """

    _HIGH_VALUE_PATTERN = re.compile(
        r"(\d+[\d,]*\.?\d*\s*%?|"
        r"\d{4}-\d{2}-\d{2}|"
        r"https?://|www\.|"
        r"\[[\d\w]+\]|"
        r"according to|cited|reported|found that|"
        r"结论|结果表明|数据显示)",
        re.IGNORECASE,
    )

    _DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(
        self,
        l1_max_items: int = 5,
        l1_max_length: int = 3000,
        l2_threshold: float = 0.10,
        l2_keep_top_k: int = 3,
        l2_min_para_length: int = 20,
        l3_enabled: bool = False,
        l3_trigger_length: int = 2000,
        l3_max_length: int = 800,
        l3_doc_max_length: int = 300,
        l3_aggregate_max_length: int = 800,
        embedder: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.l1_max_items = l1_max_items
        self.l1_max_length = l1_max_length
        self.l2_threshold = l2_threshold
        self.l2_keep_top_k = l2_keep_top_k
        self.l2_min_para_length = l2_min_para_length
        self.l3_enabled = l3_enabled
        self.l3_trigger_length = l3_trigger_length
        self.l3_max_length = l3_max_length
        self.l3_doc_max_length = l3_doc_max_length
        self.l3_aggregate_max_length = l3_aggregate_max_length
        self._embedder = embedder  # 外部传入的 embedder
        self._st_model: Any | None = None  # 懒加载的 sentence-transformers 模型
        self._llm_client = llm_client  # L3 使用的 LLM 客户端

    # ===================================================================
    # L1: 文章级
    # ===================================================================

    def l1(
        self,
        texts: list[str],
        query: str = "",
        budget: int = 128000,
        threshold_start: float = 0.25,
        threshold_step: float = 0.05,
    ) -> list[str]:
        """L1: 文章级列表过滤 —— 自适应阈值 + 余弦相似度。

        算法：
          1. 计算每篇文章（只取前1000字）与 query 的余弦相似度
          2. 阈值从 0.25 开始，按原始顺序过滤出相似度 >= 阈值的文章
          3. 计算保留文章的 token 总和，若 <= budget * 0.8 则接受并结束
          4. 若全部过滤光也结束
          5. 否则阈值 += 0.05 继续
          6. 循环结束后：
             - 若始终没找到 budget 内结果，回退到最后一个非空过滤结果
             - 若彻底为空，保底保留相似度最高的 1 篇

        Args:
            texts: 多篇文本（如搜索结果 snippets）。
            query: 当前子任务描述，用于计算相关性。
            budget: token 预算上限。
            threshold_start: 起始相似度阈值（默认 0.25）。
            threshold_step: 每次提高的步长（默认 0.05）。

        Returns:
            过滤后的文本列表（保持原始顺序）。
        """
        if not texts:
            return texts
        if len(texts) == 1:
            return texts
        if not query:
            # 无 query 时按硬截断 + 数量控制
            return [t[: self.l1_max_length] for t in texts[: self.l1_max_items]]

        # 1. 只取前 1000 字计算相似度
        truncated = [t[:1000] for t in texts]

        # 2. 编码 + 算余弦相似度
        embs = self._encode_sentences(truncated)
        query_emb = self._encode_query(query)
        similarities = embs.dot(query_emb)

        # 3. 自适应阈值循环（按原始顺序过滤，不排序）
        best_result: list[int] = []       # 满足 budget 的结果
        last_non_empty: list[int] = []    # 最后一个非空过滤结果
        threshold = threshold_start
        target_budget = int(budget * 0.8)

        while threshold <= 0.95:
            # 按原始顺序过滤出相似度 >= 阈值的文章
            kept = [i for i in range(len(texts)) if similarities[i] >= threshold]

            if kept:
                last_non_empty = kept

            est_tokens = sum(len(texts[i]) for i in kept) // 3

            # 满足 budget，接受并结束
            if est_tokens <= target_budget:
                best_result = kept
                break

            # 全部过滤光了，结束
            if not kept:
                break

            threshold += threshold_step

        # 4. 回退策略
        if not best_result and last_non_empty:
            best_result = last_non_empty

        if not best_result and texts:
            best_result = [int(np.argmax(similarities))]

        # 5. 返回（保持原始顺序）
        return [texts[i] for i in sorted(best_result)]

    # ===================================================================
    # L2: 段落级 —— TextRank + query-biased + 高价值 bonus
    # ===================================================================

    def l2(
        self,
        text: str,
        query: str,
        current_tokens: int = 0,
        budget: int = 128000,
    ) -> str:
        """L2: 单篇文章的段落级精过滤。

        完整流程：
          1. 分句 → 过滤短句碎片
          2. TextRank 得分（句子间重要性投票）
          3. query-biased 得分（与 query 的余弦相似度）
          4. 高价值句子 bonus（数字/日期/URL/引用/结论词 ×1.2）
          5. 融合三因子：combined = textrank * query_sims * value_bonus
          6. 选 top_k 句，按原文顺序拼接
        """
        # 第 1 步：分句
        sentences = self._l2_tokenize_sentences(text)
        if not sentences:
            return text

        # 短路保护：短内容或句子太少时直接返回原文
        if len(text) < 500 or len(sentences) <= 3:
            return text

        # 第 2 步：TextRank 得分
        textrank = self._l2_compute_textrank(sentences)

        # 第 3 步：query-biased 得分
        query_sims = self._l2_compute_query_similarity(sentences, query)

        # 第 4 步：高价值 bonus
        value_bonus = self._l2_compute_value_bonus(sentences)

        # 第 5 步：融合三因子
        combined = textrank * query_sims * value_bonus

        # 第 6 步：选句 + 按原文顺序拼接
        return self._l2_select_sentences(
            sentences, combined, current_tokens, budget
        )

    def _l2_tokenize_sentences(self, text: str) -> list[str]:
        """第 1 步：分句。用正则拆分，过滤 ≤ 8 字符的碎片。"""
        # 匹配中文句号、问号、感叹号、英文句号、问号、感叹号、分号
        # 中文标点可直接作为句末；英文句点仅在“字母 + 句点 + 空白 +
        # 新句首”时切分。不能按每个 `.` 切分，否则 Qwen2.5、70.3、
        # v1.1 等版本号/小数会被破坏，表格也会退化成孤立数字碎片。
        boundary = re.compile(
            r"(?<=[。！？!?；;])[ \t]*|"
            r"(?<=[A-Za-z])\.(?=[ \t]+[A-Z\u4e00-\u9fff])[ \t]*"
        )
        raw = boundary.split(text)
        sentences = []
        for s in raw:
            s = s.strip()
            if len(s) >= self.l2_min_para_length:
                sentences.append(s)
        return sentences

    def _l2_compute_textrank(self, sentences: list[str]) -> np.ndarray:
        """第 2 步：TextRank 得分。

        流程：
          1. 批量编码句子为 embedding
          2. 计算句子间余弦相似度矩阵（点积，因为已归一化）
          3. 只保留相似度 > 0.1 的边（去噪声）
          4. 行归一化转概率分布
          5. PageRank 迭代 30 次
        """
        n = len(sentences)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # 编码句子
        embs = self._encode_sentences(sentences)

        # 余弦相似度矩阵（已归一化，点积即余弦相似度）
        sim_matrix = embs @ embs.T

        # 只保留 > 0.1 的边
        adj = sim_matrix.copy()
        adj[adj < 0.1] = 0

        # 行归一化
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition = adj / row_sums

        # PageRank 迭代
        damping = 0.85
        rank = np.ones(n) / n
        for _ in range(30):
            rank = (1 - damping) / n + damping * transition.T.dot(rank)

        return rank

    def _l2_compute_query_similarity(
        self, sentences: list[str], query: str
    ) -> np.ndarray:
        """第 3 步：query-biased 得分。

        计算每句话与 query 的归一化余弦相似度。
        """
        n = len(sentences)
        if n == 0:
            return np.array([])

        # query 编码
        query_emb = self._encode_query(query)

        # 句子编码（复用 TextRank 时的词表）
        embs = self._encode_sentences(sentences)

        # 点积 = 余弦相似度（均已归一化）
        sims = embs.dot(query_emb)
        return sims

    def _l2_compute_value_bonus(self, sentences: list[str]) -> np.ndarray:
        """第 4 步：高价值句子 bonus。

        匹配到数字、日期、URL、引用标记、结论性词汇的句子 × 1.2，其余 × 1.0。
        """
        bonuses = np.ones(len(sentences))
        for i, s in enumerate(sentences):
            if self._HIGH_VALUE_PATTERN.search(s):
                bonuses[i] = 1.2
        return bonuses

    def _l2_select_sentences(
        self,
        sentences: list[str],
        combined: np.ndarray,
        current_tokens: int,
        budget: int,
    ) -> str:
        """第 6 步：选句 + 按原文顺序拼接。

        动态计算 top_ratio：
            target_ratio = max(0.15, min(0.40, 0.50 - (current_tokens / budget) * 0.35))

        选 top_k = max(1, int(n * target_ratio)) 句，
        但输出时按原文顺序拼接，不按得分排序。
        """
        n = len(sentences)
        if n == 0:
            return ""

        # 动态 top_ratio
        ratio = max(
            0.15,
            min(0.40, 0.50 - (current_tokens / max(budget, 1)) * 0.35),
        )
        k = max(1, int(n * ratio))
        k = min(k, n)

        # 选 top_k 的索引
        top_indices = set(np.argsort(combined)[::-1][:k])

        # 按原文顺序拼接
        result = [s for i, s in enumerate(sentences) if i in top_indices]
        return "\n".join(result)

    # ------------------------------------------------------------------
    # L2 辅助：编码
    # ------------------------------------------------------------------

    def _get_embedder(self) -> Any:
        """获取 embedder 实例。懒加载 sentence-transformers 模型。"""
        if self._embedder is not None:
            return self._embedder
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "使用 L2 压缩需要安装 sentence-transformers: "
                    "pip install sentence-transformers"
                ) from e
            self._st_model = SentenceTransformer(self._DEFAULT_MODEL)
        return self._st_model

    def _encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """批量编码句子。使用 sentence-transformers，输出 L2 归一化向量。"""
        model = self._get_embedder()
        embs = model.encode(sentences, convert_to_numpy=True)
        if not isinstance(embs, np.ndarray):
            embs = np.array(embs)
        # L2 归一化
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embs / norms

    def _encode_query(self, query: str) -> np.ndarray:
        """编码 query。使用 sentence-transformers，输出 L2 归一化向量。"""
        model = self._get_embedder()
        emb = model.encode(query, convert_to_numpy=True)
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb)
        # 处理不同 shape（encode 单条可能返回 1D）
        if emb.ndim == 2:
            emb = emb.flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # ===================================================================
    # L3: LLM 摘要级 —— 两阶段：单篇摘要 → 聚合综述
    # ===================================================================

    def l3(self, texts: list[str], query: str = "") -> str:
        """L3: 多篇文档的 LLM 摘要级压缩。

        两阶段流程：
          1. 单篇摘要：对每篇超长文档调用 LLM 生成摘要
          2. 聚合综述：将所有单篇摘要整合为一段连贯综述

        若 LLM 未配置或调用失败，自动 fallback 到截断拼接。
        """
        if not texts:
            return ""

        # 未启用 L3 或未配置 LLM 时 fallback
        if not self.l3_enabled or self._llm_client is None:
            combined = "\n\n".join(texts)
            return combined[: self.l3_max_length]

        # ---------- 阶段 1：单篇摘要 ----------
        summaries: list[str] = []
        for text in texts:
            # 低于触发阈值直接保留原文
            if len(text) < self.l3_trigger_length:
                summaries.append(text)
                continue

            prompt = _DOCUMENT_SUMMARY_PROMPT.format(
                max_length=self.l3_doc_max_length,
                doc=text,
            )
            messages = [{"role": "user", "content": prompt}]
            try:
                resp = self._llm_client.chat(
                    messages,
                    max_tokens=self.l3_doc_max_length,
                    temperature=0.3,
                )
                summary = getattr(resp, "content", "") or ""
                summaries.append(summary.strip())
            except Exception:
                # LLM 失败时保留原文截断版
                summaries.append(text[: self.l3_doc_max_length])

        # ---------- 阶段 2：聚合综述 ----------
        docs_text = "\n\n---\n\n".join(
            f"【文档{i + 1}】\n{s}" for i, s in enumerate(summaries)
        )
        prompt = _AGGREGATE_SUMMARY_PROMPT.format(
            max_length=self.l3_aggregate_max_length,
            query=query,
            docs=docs_text,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = self._llm_client.chat(
                messages,
                max_tokens=self.l3_aggregate_max_length,
                temperature=0.3,
            )
            aggregate = getattr(resp, "content", "") or ""
            return aggregate.strip()
        except Exception:
            # 聚合失败时返回拼接的单篇摘要
            combined = "\n\n".join(summaries)
            return combined[: self.l3_max_length]

    # ===================================================================
    # 工具方法
    # ===================================================================

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """从文本中提取关键词（简单分词，去掉停用词）。"""
        stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "and", "or",
        }
        # 中文按字符，英文按空格
        tokens = re.findall(r"[a-zA-Z]+|\S", text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]
