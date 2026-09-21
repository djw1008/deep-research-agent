"""多后端 LLM 路由器 — 从配置动态创建 LLMClient。"""

from __future__ import annotations

import os
from typing import Optional

from .llm_client import LLMClient

# 全局缓存：后端名 → LLMClient 实例
_BACKEND_CACHE: dict[str, LLMClient] = {}


def _get_env(key: str, default: str | None = None) -> str | None:
    """安全读取环境变量。"""
    return os.getenv(key, default)


class ModelRouter:
    """
    多后端 LLM 路由器。

    不需要实例化，全部用静态方法调用：
      >>> client = ModelRouter.create_backend("deepseek")
      >>> client = ModelRouter.create_backend()  # 读取 DEFAULT_LLM_BACKEND
    """

    @staticmethod
    def create_backend(
        backend_name: str | None = None,
        **override_kwargs,
    ) -> LLMClient:
        """
        创建指定名称的 LLM 后端。

        配置来源优先级：
          1. 调用时传入的 override_kwargs
          2. 环境变量（按 {PREFIX}_{PARAM} 命名规范）
          3. 内置默认值
        """
        name = (backend_name or _get_env("DEFAULT_LLM_BACKEND", "vllm")).lower().strip()

        cache_key = f"{name}:{hash(tuple(sorted(override_kwargs.items())))}"
        if cache_key in _BACKEND_CACHE:
            return _BACKEND_CACHE[cache_key]

        config = ModelRouter._load_backend_config(name)
        config.update(override_kwargs)

        client = LLMClient(**config)
        _BACKEND_CACHE[cache_key] = client
        return client

    @staticmethod
    def get_all_backends(backend_names: list[str] | None = None) -> dict[str, LLMClient]:
        """预加载并返回所有已配置的后端。"""
        backends: dict[str, LLMClient] = {}

        if backend_names is None:
            backend_names = ["deepseek", "vllm", "openai"]
            for key in os.environ:
                if key.endswith("_API_KEY"):
                    prefix = key[: -len("_API_KEY")].lower()
                    if prefix not in backend_names:
                        backend_names.append(prefix)

        for name in backend_names:
            if ModelRouter._is_backend_configured(name):
                try:
                    backends[name] = ModelRouter.create_backend(name)
                except ValueError:
                    pass
        return backends

    @staticmethod
    def clear_cache() -> None:
        """清空后端缓存。"""
        _BACKEND_CACHE.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_backend_configured(name: str) -> bool:
        prefix = name.upper()
        return (
            _get_env(f"{prefix}_API_KEY") is not None
            or _get_env(f"{prefix}_BASE_URL") is not None
        )

    @staticmethod
    def _load_backend_config(name: str) -> dict:
        """从环境变量加载指定后端的配置字典。"""
        prefix = name.upper()

        api_key = _get_env(f"{prefix}_API_KEY")
        base_url = _get_env(f"{prefix}_BASE_URL")
        model = _get_env(f"{prefix}_MODEL")

        if api_key is None and base_url is None:
            raise ValueError(
                f"后端 '{name}' 未配置。请设置 {prefix}_API_KEY 和/或 {prefix}_BASE_URL。"
            )

        config: dict = {}
        if model is not None:
            config["model_name"] = model
        if base_url is not None:
            config["base_url"] = base_url
        if api_key is not None:
            config["api_key"] = api_key

        temp = _get_env(f"{prefix}_TEMPERATURE")
        if temp is not None:
            config["temperature"] = float(temp)

        max_tok = _get_env(f"{prefix}_MAX_TOKENS")
        if max_tok is not None:
            config["max_tokens"] = int(max_tok)

        timeout = _get_env(f"{prefix}_TIMEOUT")
        if timeout is not None:
            config["timeout"] = float(timeout)

        max_retries = _get_env(f"{prefix}_MAX_RETRIES")
        if max_retries is not None:
            config["max_retries"] = int(max_retries)

        # 内置默认值
        if name == "vllm":
            config.setdefault("model_name", "Qwen/Qwen2.5-7B-Instruct")
            config.setdefault("base_url", "http://localhost:8000/v1")
            config.setdefault("api_key", "EMPTY")

        if name == "deepseek":
            config.setdefault("model_name", "deepseek-chat")
            config.setdefault("base_url", "https://api.deepseek.com/v1")

        if name == "openai":
            config.setdefault("model_name", "gpt-4o")
            config.setdefault("base_url", "https://api.openai.com/v1")

        if name == "kimi":
            config.setdefault("model_name", "moonshot-v1-8k")
            config.setdefault("base_url", "https://api.moonshot.cn/v1")

        if name == "mimo":
            # MiMo 对长 prompt 响应较慢，默认给 5 分钟超时且不重试
            config.setdefault("timeout", 300.0)
            config.setdefault("max_retries", 0)

        return config
