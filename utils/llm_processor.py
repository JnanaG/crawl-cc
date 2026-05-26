import tiktoken
import json
import os
import re
from loguru import logger


class LLMDataProcessor:
    def __init__(self, model_name: str = "cl100k_base"):
        """
        model_name: "cl100k_base" 是 OpenAI GPT-3.5/GPT-4 系列的分词器
        """
        try:
            self.encoding = tiktoken.get_encoding(model_name)
        except Exception as e:
            logger.error(f"加载 tiktoken 分词器失败: {e}")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """计算一段文本的 Token 数量"""
        if not self.encoding or not text:
            return 0
        return len(self.encoding.encode(text))

    def _measure_length(self, text: str) -> int:
        """统一长度度量：有分词器时用 token，无分词器时用字符长度。"""
        if self.encoding:
            return self.count_tokens(text)
        return len(text or "")

    def _split_markdown_sections(self, text: str) -> list[tuple[str, str]]:
        """按 Markdown 标题拆分 section，返回 [(heading, body), ...]。"""
        heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+\S+")
        sections = []
        current_heading = ""
        current_body_lines = []

        for raw_line in (text or "").splitlines():
            line = raw_line.rstrip()
            if heading_pattern.match(line):
                # flush previous section
                body = "\n".join(current_body_lines).strip()
                if current_heading or body:
                    sections.append((current_heading.strip(), body))
                current_heading = line.strip()
                current_body_lines = []
            else:
                current_body_lines.append(line)

        body = "\n".join(current_body_lines).strip()
        if current_heading or body:
            sections.append((current_heading.strip(), body))

        if not sections and (text or "").strip():
            sections = [("", (text or "").strip())]
        return sections

    def _split_to_semantic_units(self, text: str) -> list[str]:
        """先按段落，再按句子切分，尽量保持语义完整。"""
        units = []
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]
        for para in paragraphs:
            # 含列表/表格痕迹时优先保持段落完整，避免打散结构
            if re.search(r"^\s*[-*]\s+|^\s*\d+\.\s+", para, flags=re.M):
                units.append(para)
                continue

            sentences = [
                s.strip()
                for s in re.split(r"(?<=[。！？!?；;])\s+", para)
                if s.strip()
            ]
            if len(sentences) <= 1:
                units.append(para)
            else:
                units.extend(sentences)
        return units

    def _token_window_chunks(self, text: str, max_chunk_tokens: int, overlap_tokens: int) -> list[str]:
        """窗口切分兜底：用于超长句/超长段。"""
        if not text:
            return []

        if self.encoding:
            token_ids = self.encoding.encode(text)
            if not token_ids:
                return []
            step = max(1, max_chunk_tokens - overlap_tokens)
            chunks = []
            for start in range(0, len(token_ids), step):
                end = start + max_chunk_tokens
                token_slice = token_ids[start:end]
                if not token_slice:
                    continue
                piece = self.encoding.decode(token_slice).strip()
                if piece:
                    chunks.append(piece)
                if end >= len(token_ids):
                    break
            return chunks

        char_step = max(1, max_chunk_tokens - overlap_tokens)
        return [
            text[i:i + max_chunk_tokens].strip()
            for i in range(0, len(text), char_step)
            if text[i:i + max_chunk_tokens].strip()
        ]

    def chunk_text(self, text: str, max_chunk_tokens: int = 500, overlap_tokens: int | None = None) -> list:
        """
        标题+语义切分策略：
        1) 按 Markdown 标题拆 section；
        2) section 内按段落/句子切分语义单元；
        3) 在 token 上限内进行合并；
        4) 超长单元退化为窗口切分。
        """
        if not text:
            return []
        if max_chunk_tokens <= 0:
            max_chunk_tokens = 500

        if overlap_tokens is None:
            overlap_tokens = max(20, int(max_chunk_tokens * 0.2))
        overlap_tokens = max(0, min(overlap_tokens, max_chunk_tokens - 1))

        sections = self._split_markdown_sections(text)
        if not sections:
            return self._token_window_chunks(text, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)

        chunks = []
        for heading, body in sections:
            prefix = f"{heading}\n" if heading else ""
            prefix_len = self._measure_length(prefix)

            # 防止标题太长导致可用窗口为0
            local_limit = max_chunk_tokens
            available = max(20, local_limit - prefix_len) if prefix else local_limit

            section_text = body.strip()
            if not section_text:
                # 只有标题也要保留
                if heading:
                    chunks.append(heading.strip())
                continue

            semantic_units = self._split_to_semantic_units(section_text)
            current_units = []
            current_len = 0

            for unit in semantic_units:
                unit_len = self._measure_length(unit)

                # 单个语义单元过长：先落当前，再对该单元做窗口兜底
                if unit_len > available:
                    if current_units:
                        merged = "\n\n".join(current_units).strip()
                        chunk = f"{prefix}{merged}".strip()
                        if chunk:
                            chunks.append(chunk)
                        current_units = []
                        current_len = 0

                    fallback_parts = self._token_window_chunks(
                        unit,
                        max_chunk_tokens=available,
                        overlap_tokens=min(overlap_tokens, max(available // 4, 0)),
                    )
                    for part in fallback_parts:
                        chunk = f"{prefix}{part}".strip()
                        if chunk:
                            chunks.append(chunk)
                    continue

                # 正常合并到当前 chunk
                projected_len = current_len + unit_len
                if current_units:
                    projected_len += self._measure_length("\n\n")

                if projected_len > available:
                    merged = "\n\n".join(current_units).strip()
                    chunk = f"{prefix}{merged}".strip()
                    if chunk:
                        chunks.append(chunk)
                    current_units = [unit]
                    current_len = unit_len
                else:
                    current_units.append(unit)
                    if current_len == 0:
                        current_len = unit_len
                    else:
                        current_len += self._measure_length("\n\n") + unit_len

            if current_units:
                merged = "\n\n".join(current_units).strip()
                chunk = f"{prefix}{merged}".strip()
                if chunk:
                    chunks.append(chunk)

        # 兜底：极端情况下切分为空，回退窗口切分
        if not chunks:
            return self._token_window_chunks(text, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)
        return chunks

    def format_as_jsonl(self, filepath: str, data: list):
        """将结构化数据保存为 JSONL 格式（大模型微调最常用格式）"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"成功将 {len(data)} 条数据保存至 {filepath}")
