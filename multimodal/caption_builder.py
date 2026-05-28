from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .caption_client import MultimodalCaptionClient


PROMPT_VERSION = "image-caption-v1"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _category_focus(category_name: str) -> str:
    mapping = {
        "外观": "重点识别视角、车身颜色、前脸/侧面/尾部设计、轮毂、车身姿态与运动感",
        "内饰": "重点识别方向盘、中控屏、仪表台、档把、座椅、内饰材质与布局风格",
        "空间": "重点识别后排空间、头部空间、腿部空间、后备厢、座椅放倒等空间表现",
        "车展": "重点识别展示环境、展台状态、车辆外观与陈列特征",
        "官方图": "重点识别官方宣传图中的主体车辆、角度和突出卖点",
        "封面": "重点识别主体车型和最明显的视觉特征",
    }
    return mapping.get(category_name or "", "重点识别车辆主体、颜色、视角与明显设计特征")


def build_caption_prompts(asset: dict[str, Any]) -> tuple[str, str]:
    category_name = asset.get("category_name") or "未分类"
    system_prompt = (
        "你是汽车图片理解助手。"
        "请基于图片内容生成适合检索的中文描述，不要编造看不见的参数。"
        "输出 3 到 5 句自然语言，尽量包含车系、图片类别、视角、颜色和设计特征。"
    )
    user_prompt = (
        f"车系: {asset.get('series_name') or '未知车系'}\n"
        f"品牌: {asset.get('brand_name') or '未知品牌'}\n"
        f"图片类别: {category_name}\n"
        f"来源角色: {asset.get('image_role') or 'unknown'}\n"
        f"本地原图: {asset.get('local_path') or '未落盘'}\n"
        f"补充要求: {_category_focus(category_name)}\n\n"
        "请生成适合向量检索和问答引用的中文 caption。"
    )
    return system_prompt, user_prompt


def heuristic_caption(asset: dict[str, Any]) -> str:
    series_name = asset.get("series_name") or "未知车系"
    brand_name = asset.get("brand_name") or "未知品牌"
    category_name = asset.get("category_name") or "未分类"
    color_name = asset.get("color_name") or "颜色信息未标注"
    image_role = asset.get("image_role") or "图片"
    source_section = asset.get("source_section") or "unknown"

    category_templates = {
        "外观": "这是一张{series_name}的外观图片，主体为{brand_name}车型，适合观察车身姿态、前脸和侧面线条。",
        "内饰": "这是一张{series_name}的内饰图片，主体为{brand_name}车型，适合观察中控布局、方向盘和座舱风格。",
        "空间": "这是一张{series_name}的空间展示图片，主体为{brand_name}车型，适合观察乘坐空间或储物空间表现。",
        "车展": "这是一张{series_name}的车展场景图片，主体为{brand_name}车型，可用于补充真实展示环境下的视觉特征。",
        "官方图": "这是一张{series_name}的官方图片，主体为{brand_name}车型，可用于观察宣传图中的核心设计卖点。",
        "封面": "这是一张{series_name}的封面图片，主体为{brand_name}车型，可作为该车系的视觉代表图。",
    }
    main_sentence = category_templates.get(
        category_name,
        "这是一张{series_name}的车辆图片，主体为{brand_name}车型，可用于补充视觉描述信息。",
    ).format(series_name=series_name, brand_name=brand_name)

    detail_parts = [
        f"图片类别为{category_name}，来源于{source_section}，角色为{image_role}。",
        f"已知颜色信息为{color_name}。",
    ]
    if asset.get("width") and asset.get("height"):
        detail_parts.append(
            f"图片分辨率约为{asset.get('width')}x{asset.get('height')}。"
        )
    if asset.get("quality_score") is not None:
        detail_parts.append(
            f"该图片在离线规则中被评估为质量分{float(asset.get('quality_score')):.2f}。"
        )
    return " ".join([main_sentence, *detail_parts]).strip()


def normalize_caption_text(asset: dict[str, Any], raw_caption: str) -> str:
    prefix = (
        f"车系: {asset.get('series_name') or '未知车系'}。"
        f"品牌: {asset.get('brand_name') or '未知品牌'}。"
        f"图片类别: {asset.get('category_name') or '未分类'}。"
    )
    return f"{prefix}{raw_caption.strip()}".strip()


def asset_to_caption_record(
    asset: dict[str, Any],
    caption_text: str,
    *,
    provider: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    series_name = asset.get("series_name") or "未知车系"
    category_name = asset.get("category_name") or "未分类"
    return {
        "metadata": {
            "source": "dongchedi_image_caption",
            "url": asset.get("image_url", ""),
            "title": f"{series_name} {category_name} 图片描述",
            "series_id": asset.get("series_id", ""),
            "brand_name": asset.get("brand_name", ""),
            "car_type": asset.get("car_type", ""),
            "asset_id": asset.get("asset_id", ""),
            "image_url": asset.get("image_url", ""),
            "image_local_path": asset.get("local_path", ""),
            "image_category": asset.get("category", ""),
            "image_category_name": category_name,
            "image_role": asset.get("image_role", ""),
            "image_source_section": asset.get("source_section", ""),
            "image_quality_score": asset.get("quality_score", 0.0),
            "modality": "image_caption",
            "content_type": "image_caption",
            "caption_provider": provider,
            "caption_model": model,
            "caption_prompt_version": prompt_version,
        },
        "text": caption_text,
    }


def _rows_for_parquet(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    return normalized_rows


def build_image_caption_corpus(
    image_assets: list[dict[str, Any]],
    *,
    provider: str = "heuristic",
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    only_usable: bool = True,
    limit: int | None = None,
    fail_open: bool = True,
    temperature: float = 0.1,
) -> dict[str, Any]:
    client = MultimodalCaptionClient(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
    )

    candidates = []
    for asset in image_assets:
        if only_usable and not asset.get("is_usable"):
            continue
        candidates.append(asset)
    if limit is not None:
        candidates = candidates[: max(limit, 0)]

    corpus: list[dict[str, Any]] = []
    generation_log: list[dict[str, Any]] = []
    modality_counter = Counter()

    for asset in candidates:
        status = "success"
        error = ""
        raw_caption = ""
        try:
            if provider == "heuristic":
                raw_caption = heuristic_caption(asset)
            else:
                system_prompt, user_prompt = build_caption_prompts(asset)
                raw_caption = client.caption_from_image_url(
                    image_url=asset.get("image_url", ""),
                    local_path=asset.get("local_path", ""),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                )
        except Exception as exc:
            if not fail_open:
                raise
            status = "fallback"
            error = str(exc)
            raw_caption = heuristic_caption(asset)

        caption_text = normalize_caption_text(asset, raw_caption)
        corpus.append(
            asset_to_caption_record(
                asset,
                caption_text,
                provider=provider,
                model=client.model,
            )
        )
        generation_log.append(
            {
                "asset_id": asset.get("asset_id", ""),
                "series_id": asset.get("series_id", ""),
                "series_name": asset.get("series_name", ""),
                "image_url": asset.get("image_url", ""),
                "local_path": asset.get("local_path", ""),
                "status": status,
                "error": error,
                "caption_preview": caption_text[:200],
            }
        )
        modality_counter[asset.get("category_name") or "未分类"] += 1

    summary = {
        "input_asset_count": len(image_assets),
        "caption_candidate_count": len(candidates),
        "caption_record_count": len(corpus),
        "fallback_count": sum(1 for row in generation_log if row["status"] == "fallback"),
        "category_distribution": dict(modality_counter),
        "caption_provider": provider,
        "caption_model": client.model,
        "prompt_version": PROMPT_VERSION,
    }

    return {
        "caption_corpus": corpus,
        "generation_log": generation_log,
        "summary": summary,
    }


def write_image_caption_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "image_caption_corpus.jsonl"
    corpus_parquet_path = output_dir / "image_caption_corpus.parquet"
    log_path = output_dir / "image_caption_generation_log.jsonl"
    summary_path = output_dir / "image_caption_summary.json"

    with corpus_path.open("w", encoding="utf-8") as f:
        for row in result["caption_corpus"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with log_path.open("w", encoding="utf-8") as f:
        for row in result["generation_log"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)

    pd.DataFrame(_rows_for_parquet(result["caption_corpus"])).to_parquet(
        corpus_parquet_path,
        index=False,
        engine="pyarrow",
    )
    return {
        "corpus_path": str(corpus_path),
        "corpus_parquet_path": str(corpus_parquet_path),
        "log_path": str(log_path),
        "summary_path": str(summary_path),
    }
