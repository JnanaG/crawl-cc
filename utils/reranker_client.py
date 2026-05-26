from loguru import logger


class RerankerClient:
    """
    可选重排客户端：
    - none: 不启用模型重排
    - cross_encoder: 使用 sentence-transformers CrossEncoder 打分
    """

    def __init__(
        self,
        provider: str = "none",
        model_name: str | None = None,
        device: str | None = None,
        fail_open: bool = True,
    ):
        self.provider = provider
        self.model_name = model_name
        self.device = device
        self.fail_open = fail_open
        self._model = None

        if provider == "none":
            return

        if provider == "cross_encoder":
            self.model_name = model_name or "BAAI/bge-reranker-base"
            try:
                from sentence_transformers import CrossEncoder
            except Exception as e:
                if self.fail_open:
                    logger.warning(
                        "加载 CrossEncoder 失败，已自动降级为轻量重排: "
                        f"{e}。可继续运行，若需模型重排请修复 sentence-transformers/torch 环境。"
                    )
                    self.provider = "none"
                    self._model = None
                    return
                raise RuntimeError(
                    f"加载 CrossEncoder 失败: {e}。请检查 sentence-transformers/torch 环境，"
                    "或改用 --reranker-provider none / --reranker-fail-open"
                ) from e
            kwargs = {}
            if device:
                kwargs["device"] = device
            try:
                self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception as e:
                if self.fail_open:
                    logger.warning(
                        "初始化 CrossEncoder 失败，已自动降级为轻量重排: "
                        f"{e}。可继续运行，后续可修复 torch DLL 环境。"
                    )
                    self.provider = "none"
                    self._model = None
                    return
                raise RuntimeError(
                    f"初始化 CrossEncoder 失败: {e}。可改用 --reranker-provider none 或启用 --reranker-fail-open"
                ) from e
            return

        raise ValueError(f"不支持的 reranker provider: {provider}")

    def enabled(self) -> bool:
        return self.provider != "none" and self._model is not None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not self.enabled():
            return [0.0] * len(texts)

        pairs = [[query, t] for t in texts]
        values = self._model.predict(pairs)
        return [float(v) for v in values]
