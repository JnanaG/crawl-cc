import os

import requests


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
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"]

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
        resp = requests.post(url, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        body = resp.json()
        message = body.get("message", {})
        return message.get("content", "")
