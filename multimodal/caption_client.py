from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path

import requests


def _normalize_text_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


class MultimodalCaptionClient:
    """
    支持三类 caption 生成方式:
    1) heuristic: 基于图片元数据做规则化 caption，便于离线调试
    2) openai_compatible: /v1/chat/completions，多模态 image_url
    3) ollama: /api/chat，需先拉取图片并转 base64
    """

    def __init__(
        self,
        provider: str = "heuristic",
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 90,
        image_download_timeout_sec: int = 20,
    ):
        self.provider = provider
        self.timeout_sec = timeout_sec
        self.image_download_timeout_sec = image_download_timeout_sec

        if provider == "heuristic":
            self.model = model or "heuristic-caption-v1"
            self.api_base = ""
            self.api_key = ""
            return

        if provider == "openai_compatible":
            self.model = model or os.getenv("MM_CAPTION_MODEL", "gpt-4o-mini")
            self.api_base = (api_base or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not self.api_base:
                raise ValueError("openai_compatible 需要 OPENAI_BASE_URL 或 --caption-api-base")
            if not self.api_key:
                raise ValueError("openai_compatible 需要 OPENAI_API_KEY 或 --caption-api-key")
            return

        if provider == "ollama":
            self.model = model or os.getenv("MM_CAPTION_MODEL", "qwen2.5vl:7b")
            self.api_base = (api_base or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
            self.api_key = ""
            return

        raise ValueError(f"不支持的 caption provider: {provider}")

    def _download_image_base64(self, image_url: str) -> str:
        response = requests.get(image_url, timeout=self.image_download_timeout_sec)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("ascii")

    def _load_local_image_base64(self, local_path: str) -> str:
        with Path(local_path).open("rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def caption_from_image_url(
        self,
        *,
        image_url: str,
        local_path: str = "",
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        started_at = time.perf_counter()
        if self.provider == "heuristic":
            raise RuntimeError("heuristic provider 不支持直接处理图像输入，请在上层使用规则 fallback")

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
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        image_url
                                        if not (local_path and Path(local_path).exists())
                                        else (
                                            f"data:{mimetypes.guess_type(local_path)[0] or 'image/jpeg'};base64,"
                                            f"{self._load_local_image_base64(local_path)}"
                                        )
                                    )
                                },
                            },
                        ],
                    },
                ],
                "temperature": temperature,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            body = resp.json()
            content = _normalize_text_content(body["choices"][0]["message"]["content"])
            if not content:
                raise RuntimeError("openai_compatible caption 返回为空")
            _ = int((time.perf_counter() - started_at) * 1000)
            return content

        image_base64 = (
            self._load_local_image_base64(local_path)
            if local_path and Path(local_path).exists()
            else self._download_image_base64(image_url)
        )
        url = f"{self.api_base}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_base64],
                },
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(url, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        body = resp.json()
        content = _normalize_text_content(body.get("message", {}).get("content", ""))
        if not content:
            raise RuntimeError("ollama caption 返回为空")
        return content
