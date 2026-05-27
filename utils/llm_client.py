import os
import logging
import time

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """
    支持两类生成端:
    1) openai_compatible: /v1/chat/completions
    2) ollama: /api/chat
    """

    def __init__(
        self,
        provider: str = "openai_compatible",
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 90,
    ):
        self.provider = provider
        self.timeout_sec = timeout_sec

        if provider == "openai_compatible":
            self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
            self.api_base = (api_base or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not self.api_base:
                raise ValueError("openai_compatible 需要 OPENAI_BASE_URL 或 --llm-api-base")
            if not self.api_key:
                raise ValueError("openai_compatible 需要 OPENAI_API_KEY 或 --llm-api-key")
            return

        if provider == "ollama":
            self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")
            self.api_base = (api_base or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
            self.api_key = ""
            return

        raise ValueError(f"不支持的 llm provider: {provider}")

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        started_at = time.perf_counter()
        if self.provider == "openai_compatible":
            url = f"{self.api_base}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            logger.info(
                "llm_request_started provider=%s model=%s url=%s system_chars=%s user_chars=%s temperature=%s timeout_sec=%s",
                self.provider,
                self.model,
                url,
                len(system_prompt),
                len(user_prompt),
                temperature,
                self.timeout_sec,
            )
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                logger.info(
                    "llm_request_finished provider=%s model=%s latency_ms=%s answer_chars=%s",
                    self.provider,
                    self.model,
                    latency_ms,
                    len(content or ""),
                )
                return content
            except Exception as exc:
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                logger.exception(
                    "llm_request_failed provider=%s model=%s url=%s latency_ms=%s error=%s",
                    self.provider,
                    self.model,
                    url,
                    latency_ms,
                    exc,
                )
                raise

        url = f"{self.api_base}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        logger.info(
            "llm_request_started provider=%s model=%s url=%s system_chars=%s user_chars=%s temperature=%s timeout_sec=%s",
            self.provider,
            self.model,
            url,
            len(system_prompt),
            len(user_prompt),
            temperature,
            self.timeout_sec,
        )
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            body = resp.json()
            message = body.get("message", {})
            content = message.get("content", "")
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "llm_request_finished provider=%s model=%s latency_ms=%s answer_chars=%s",
                self.provider,
                self.model,
                latency_ms,
                len(content or ""),
            )
            return content
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.exception(
                "llm_request_failed provider=%s model=%s url=%s latency_ms=%s error=%s",
                self.provider,
                self.model,
                url,
                latency_ms,
                exc,
            )
            raise
