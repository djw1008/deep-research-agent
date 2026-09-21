"""Embedding helper for the memory layer.

Defaults to a Chinese sentence-transformers model (BAAI/bge-small-zh-v1.5,
512-dim) for better Chinese recall/compression quality.  Falls back to the
project's n-gram embedder if sentence-transformers is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

from ..compressor.context_compressor import _NGramEmbedder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


class MemoryEmbedder:
    """Embedder for memory summaries.

    Args:
        model_name: sentence-transformers model name.  Use ``None`` to force the
            n-gram fallback (useful in tests or offline environments).
        fallback_n: n-gram size when falling back to the n-gram embedder.
    """

    def __init__(
        self,
        model_name: str | None = _DEFAULT_MODEL,
        fallback_n: int = 2,
    ) -> None:
        self.model_name = model_name
        self._st_model: Any | None = None
        self._ngram = _NGramEmbedder(n=fallback_n)
        self._ngram_fitted = False
        self._backend: str | None = None

    def _load_st_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name)
            self._backend = "sentence_transformers"
        return self._st_model

    def encode(
        self,
        texts: str | list[str],
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a single text or a batch into L2-normalized float32 vectors.

        Accepts ``convert_to_numpy`` for compatibility with callers that mirror
        the ``sentence_transformers.SentenceTransformer.encode`` signature.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([], dtype=np.float32)

        # Try sentence-transformers first.
        if self.model_name is not None:
            try:
                model = self._load_st_model()
                embs = model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    **kwargs,
                )
                if not isinstance(embs, np.ndarray):
                    embs = np.array(embs)
                return embs.astype(np.float32, copy=False)
            except Exception as exc:
                logger.warning(
                    "sentence-transformers failed (%s), falling back to n-gram",
                    exc,
                )
                self.model_name = None  # avoid repeated failures

        # Fallback: n-gram character embedder.
        if not self._ngram_fitted:
            embs = self._ngram.fit_transform(texts)
            self._ngram_fitted = True
        else:
            embs = np.stack([self._ngram.transform(t) for t in texts])

        self._backend = "ngram"
        return embs.astype(np.float32, copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode(text)[0]

    @property
    def backend(self) -> str | None:
        return self._backend
