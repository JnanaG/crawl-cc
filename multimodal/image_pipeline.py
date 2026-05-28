from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


CATEGORY_NAME_MAP = {
    "wg": "外观",
    "ns": "内饰",
    "kj": "空间",
    "cz": "车展",
    "gft": "官方图",
    "xt": "细节",
    "zj": "证件照",
}

CORE_IMAGE_SECTIONS = {
    "series_cover",
    "series_gallery",
    "image_floor",
    "image_floor_cover",
    "module_wg",
    "module_ns",
}

SUPPORTED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_url(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    normalized = url.strip()
    if not normalized:
        return ""
    if normalized.startswith("//"):
        return f"https:{normalized}"
    return normalized


def _looks_like_image_url(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    if path.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
        return True
    return any(marker in path for marker in (".image", "/img/", "img/", "image", "byteimg"))


def _category_name(category: str, category_name: str = "") -> str:
    if category_name:
        return category_name
    if category in CATEGORY_NAME_MAP:
        return CATEGORY_NAME_MAP[category]
    return category or "未分类"


def _make_asset_id(series_id: str, image_url: str, source_section: str, image_role: str) -> str:
    base = "|".join([series_id or "", image_url, source_section, image_role])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _infer_extension(url: str, content_type: str = "") -> str:
    if content_type:
        content_type = content_type.split(";")[0].strip().lower()
        if content_type in SUPPORTED_CONTENT_TYPES:
            return SUPPORTED_CONTENT_TYPES[content_type]
    path = (urlparse(url).path or "").lower()
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        if path.endswith(suffix):
            return suffix
    if ".image" in path:
        return ".jpg"
    return ".bin"


def _source_reliability(source_section: str, image_role: str) -> float:
    if source_section in {"module_wg", "module_ns"}:
        return 0.98
    if source_section in {"series_gallery", "image_floor"}:
        return 0.95
    if source_section in {"series_cover", "image_floor_cover"}:
        return 0.88
    if source_section in {"news_cover", "news_image"}:
        return 0.62
    if source_section in {"community_image", "related_series_cover", "recommended_series_cover"}:
        return 0.55
    if image_role == "ugc_image":
        return 0.45
    return 0.5


def _score_asset(asset: dict[str, Any]) -> tuple[float, list[str], bool]:
    score = 0.0
    flags: list[str] = []
    url = asset.get("image_url", "")
    width = _safe_int(asset.get("width"))
    height = _safe_int(asset.get("height"))
    category = asset.get("category", "")
    source_section = asset.get("source_section", "")
    image_role = asset.get("image_role", "")

    if url.startswith(("http://", "https://")):
        score += 0.25
    else:
        flags.append("invalid_url")

    if _looks_like_image_url(url):
        score += 0.15
    else:
        flags.append("not_image_like_url")

    if asset.get("series_id"):
        score += 0.2
    else:
        flags.append("missing_series_id")

    if category:
        score += 0.1
    else:
        flags.append("missing_category")

    if width > 0 and height > 0:
        score += 0.1
        long_edge = max(width, height)
        if long_edge >= 960:
            score += 0.1
        elif long_edge >= 640:
            score += 0.05
        else:
            flags.append("low_resolution")
    else:
        flags.append("missing_resolution")

    score += _source_reliability(source_section, image_role) * 0.2

    if source_section not in CORE_IMAGE_SECTIONS:
        flags.append("non_core_source")
    if image_role in {"ugc_image", "news_cover", "news_image"}:
        score -= 0.05

    score = max(0.0, min(round(score, 4), 1.0))
    is_usable = score >= 0.6 and source_section in CORE_IMAGE_SECTIONS
    return score, flags, is_usable


def _classification_split(asset_id: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(asset_id[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def _append_asset(
    assets: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    *,
    series_id: str,
    series_name: str,
    brand_name: str,
    image_url: Any,
    source_section: str,
    image_role: str,
    category: str = "",
    category_name: str = "",
    rank: int = 0,
    width: int = 0,
    height: int = 0,
    car_id: Any = None,
    color_id: Any = None,
    color_name: str = "",
    raw_ref: str = "",
    local_path: str = "",
    local_exists: bool = False,
    raw_image_status: str = "",
    raw_content_type: str = "",
    raw_bytes: int = 0,
) -> None:
    normalized_url = _normalize_url(image_url)
    if not normalized_url:
        return
    dedupe_key = (normalized_url, source_section, str(rank))
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)

    asset = {
        "asset_id": _make_asset_id(series_id, normalized_url, source_section, image_role),
        "series_id": series_id,
        "series_name": series_name,
        "brand_name": brand_name,
        "image_url": normalized_url,
        "category": category or "",
        "category_name": _category_name(category or "", category_name or ""),
        "source_section": source_section,
        "image_role": image_role,
        "rank": rank,
        "width": _safe_int(width),
        "height": _safe_int(height),
        "car_id": str(car_id) if car_id not in (None, "") else "",
        "color_id": str(color_id) if color_id not in (None, "") else "",
        "color_name": color_name or "",
        "raw_ref": raw_ref,
        "local_path": local_path or "",
        "local_exists": bool(local_exists),
        "raw_image_status": raw_image_status or "",
        "raw_content_type": raw_content_type or "",
        "raw_bytes": _safe_int(raw_bytes),
    }
    quality_score, quality_flags, is_usable = _score_asset(asset)
    asset["quality_score"] = quality_score
    asset["quality_flags"] = quality_flags
    asset["is_usable"] = is_usable
    assets.append(asset)


def _series_meta(raw_json: dict[str, Any], clean_record: dict[str, Any] | None) -> tuple[str, str, str]:
    clean_record = clean_record or {}
    clean_series = _as_dict(clean_record.get("series"))
    page_props = _as_dict(_as_dict(raw_json.get("props")).get("pageProps"))
    series_head = _as_dict(page_props.get("seriesHomeHead"))

    series_id = str(
        clean_series.get("series_id")
        or page_props.get("seriesId")
        or series_head.get("series_id")
        or ""
    )
    series_name = (
        clean_series.get("series_name")
        or page_props.get("seriesName")
        or series_head.get("series_name")
        or "未知车系"
    )
    brand_name = clean_series.get("brand_name") or series_head.get("brand_name") or ""
    return series_id, series_name, brand_name


def _load_raw_image_manifest(raw_path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    manifest_path = raw_path.with_name(f"{raw_path.stem}_images.json")
    if not manifest_path.exists():
        return {}, {}
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in _as_list(manifest.get("items")):
        item = _as_dict(item)
        key = (_normalize_url(item.get("image_url")), item.get("source_section", ""), str(item.get("rank", 0)))
        lookup[key] = item
    return lookup, manifest


def extract_series_image_assets(
    raw_json: dict[str, Any],
    clean_record: dict[str, Any] | None = None,
    *,
    include_contextual_images: bool = False,
    raw_image_lookup: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    page_props = _as_dict(_as_dict(raw_json.get("props")).get("pageProps"))
    series_head = _as_dict(page_props.get("seriesHomeHead"))
    image_floor_data = _as_dict(page_props.get("imageFloorData"))
    series_id, series_name, brand_name = _series_meta(raw_json, clean_record)

    assets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    raw_image_lookup = raw_image_lookup or {}

    def local_kwargs(image_url: Any, source_section: str, rank: int) -> dict[str, Any]:
        item = raw_image_lookup.get((_normalize_url(image_url), source_section, str(rank)), {})
        local_path = item.get("local_path", "")
        return {
            "local_path": local_path,
            "local_exists": bool(local_path and Path(local_path).exists()),
            "raw_image_status": item.get("status", ""),
            "raw_content_type": item.get("content_type", ""),
            "raw_bytes": item.get("bytes", 0),
        }

    _append_asset(
        assets,
        seen,
        series_id=series_id,
        series_name=series_name,
        brand_name=brand_name,
        image_url=series_head.get("cover_url"),
        source_section="series_cover",
        image_role="cover",
        category="cover",
        category_name="封面",
        raw_ref="pageProps.seriesHomeHead.cover_url",
        **local_kwargs(series_head.get("cover_url"), "series_cover", 0),
    )

    selected_modules = _as_dict(page_props.get("selectedImageModules"))
    if selected_modules:
        for module_key in ("wg", "ns"):
            module = _as_dict(selected_modules.get(module_key))
            if not module:
                continue
            source_section = f"module_{module_key}"
            for idx, sample in enumerate(_as_list(module.get("selected_images"))):
                sample = _as_dict(sample)
                _append_asset(
                    assets,
                    seen,
                    series_id=series_id,
                    series_name=series_name,
                    brand_name=brand_name,
                    image_url=sample.get("image_url"),
                    source_section=source_section,
                    image_role="module_sample",
                    category=module_key,
                    category_name=module.get("category_name") or "",
                    rank=idx,
                    car_id=sample.get("car_id"),
                    color_name=",".join(_as_list(sample.get("available_color_names"))[:3]),
                    raw_ref=f"pageProps.selectedImageModules.{module_key}.selected_images",
                    **local_kwargs(sample.get("image_url"), source_section, idx),
                )

        category_counter = Counter(asset.get("category_name") or "未分类" for asset in assets)
        usable_assets = [asset for asset in assets if asset.get("is_usable")]
        classification_candidates = []
        for asset in usable_assets:
            label = asset.get("category_name") or ""
            if not label or label in {"封面", "未分类"}:
                continue
            candidate = dict(asset)
            candidate["task_type"] = "image_category_classification"
            classification_candidates.append(candidate)

        return {
            "series": {
                "series_id": series_id,
                "series_name": series_name,
                "brand_name": brand_name,
            },
            "assets": assets,
            "classification_candidates": classification_candidates,
            "summary": {
                "asset_count": len(assets),
                "usable_asset_count": len(usable_assets),
                "category_distribution": dict(category_counter),
                "classification_candidate_count": len(classification_candidates),
            },
        }

    for idx, item in enumerate(_as_list(series_head.get("series_image_info_list"))):
        item = _as_dict(item)
        _append_asset(
            assets,
            seen,
            series_id=series_id,
            series_name=series_name,
            brand_name=brand_name,
            image_url=item.get("image_url"),
            source_section="series_gallery",
            image_role="gallery",
            category=str(item.get("category") or ""),
            rank=idx,
            car_id=item.get("car_id"),
            color_id=item.get("color_id"),
            raw_ref="pageProps.seriesHomeHead.series_image_info_list",
            **local_kwargs(item.get("image_url"), "series_gallery", idx),
        )

    for idx, item in enumerate(_as_list(series_head.get("pics_summary_info"))):
        item = _as_dict(item)
        _append_asset(
            assets,
            seen,
            series_id=series_id,
            series_name=series_name,
            brand_name=brand_name,
            image_url=item.get("CoverPicUrl"),
            source_section="image_floor_cover",
            image_role="group_cover",
            category=str(item.get("Category") or "").lower(),
            rank=idx,
            raw_ref="pageProps.seriesHomeHead.pics_summary_info",
            **local_kwargs(item.get("CoverPicUrl"), "image_floor_cover", idx),
        )

    for idx, item in enumerate(_as_list(image_floor_data.get("floor_image_list"))):
        item = _as_dict(item)
        category = str(item.get("category") or "")
        category_name = str(item.get("text") or "")
        for image_idx, image_info in enumerate(_as_list(item.get("image_list"))):
            image_info = _as_dict(image_info)
            _append_asset(
                assets,
                seen,
                series_id=series_id,
                series_name=series_name,
                brand_name=brand_name,
                image_url=image_info.get("image_url") or image_info.get("url") or image_info.get("cover_url"),
                source_section="image_floor",
                image_role="gallery",
                category=category,
                category_name=category_name,
                rank=(idx * 1000) + image_idx,
                width=_safe_int(image_info.get("width")),
                height=_safe_int(image_info.get("height")),
                car_id=image_info.get("car_id"),
                color_id=image_info.get("color_id"),
                color_name=image_info.get("color_name") or "",
                raw_ref="pageProps.imageFloorData.floor_image_list",
                **local_kwargs(
                    image_info.get("image_url") or image_info.get("url") or image_info.get("cover_url"),
                    "image_floor",
                    (idx * 1000) + image_idx,
                ),
            )

    for idx, item in enumerate(_as_list(image_floor_data.get("floor_head_list"))):
        item = _as_dict(item)
        category = str(item.get("category") or "")
        category_name = str(item.get("text") or "")
        color_list = _as_list(item.get("color_list"))
        for color_idx, color in enumerate(color_list):
            color = _as_dict(color)
            _append_asset(
                assets,
                seen,
                series_id=series_id,
                series_name=series_name,
                brand_name=brand_name,
                image_url=color.get("image_url") or color.get("cover_url"),
                source_section="image_floor_cover",
                image_role="color_cover",
                category=category,
                category_name=category_name,
                rank=(idx * 100) + color_idx,
                color_id=color.get("color_id"),
                color_name=color.get("color_name") or "",
                raw_ref="pageProps.imageFloorData.floor_head_list",
                **local_kwargs(
                    color.get("image_url") or color.get("cover_url"),
                    "image_floor_cover",
                    (idx * 100) + color_idx,
                ),
            )

    if include_contextual_images:
        for idx, item in enumerate(_as_list(_as_dict(page_props.get("sameBrandData")).get("list"))):
            item = _as_dict(item)
            _append_asset(
                assets,
                seen,
                series_id=series_id,
                series_name=series_name,
                brand_name=brand_name,
                image_url=item.get("cover_uri"),
                source_section="related_series_cover",
                image_role="related_cover",
                category="related",
                category_name="同品牌推荐",
                rank=idx,
                raw_ref="pageProps.sameBrandData.list",
                **local_kwargs(item.get("cover_uri"), "related_series_cover", idx),
            )

        for idx, item in enumerate(_as_list(_as_dict(page_props.get("recommendSeriesData")).get("list"))):
            item = _as_dict(item)
            _append_asset(
                assets,
                seen,
                series_id=series_id,
                series_name=series_name,
                brand_name=brand_name,
                image_url=item.get("cover_url"),
                source_section="recommended_series_cover",
                image_role="related_cover",
                category="recommended",
                category_name="推荐车系",
                rank=idx,
                raw_ref="pageProps.recommendSeriesData.list",
                **local_kwargs(item.get("cover_url"), "recommended_series_cover", idx),
            )

        news_sections = [
            ("newest", _as_list(page_props.get("newestStaticNews"))),
            ("guide", _as_list(page_props.get("guideStaticNews"))),
            ("newcar", _as_list(page_props.get("newcarStaticNews"))),
            ("evaluating", _as_list(page_props.get("evaluatingStaticNews"))),
            ("original", _as_list(page_props.get("originalStaticNews"))),
        ]
        for section_name, items in news_sections:
            for idx, item in enumerate(items):
                item = _as_dict(item)
                video_info = _as_dict(item.get("video_info"))
                _append_asset(
                    assets,
                    seen,
                    series_id=series_id,
                    series_name=series_name,
                    brand_name=brand_name,
                    image_url=video_info.get("cover_url"),
                    source_section="news_cover",
                    image_role="news_cover",
                    category="news",
                    category_name=f"{section_name}新闻封面",
                    rank=idx,
                    width=_safe_int(video_info.get("width")),
                    height=_safe_int(video_info.get("height")),
                    raw_ref=f"pageProps.{section_name}StaticNews[].video_info.cover_url",
                    **local_kwargs(video_info.get("cover_url"), "news_cover", idx),
                )

                for image_idx, image_info in enumerate(_as_list(item.get("image_list")) + _as_list(item.get("image_urls"))):
                    image_info = _as_dict(image_info)
                    _append_asset(
                        assets,
                        seen,
                        series_id=series_id,
                        series_name=series_name,
                        brand_name=brand_name,
                        image_url=image_info.get("url") or image_info.get("image_url"),
                        source_section="news_image",
                        image_role="ugc_image",
                        category="news",
                        category_name=f"{section_name}新闻配图",
                        rank=(idx * 100) + image_idx,
                        width=_safe_int(image_info.get("width")),
                        height=_safe_int(image_info.get("height")),
                        raw_ref=f"pageProps.{section_name}StaticNews[].image_urls",
                        **local_kwargs(
                            image_info.get("url") or image_info.get("image_url"),
                            "news_image",
                            (idx * 100) + image_idx,
                        ),
                    )

    category_counter = Counter(asset.get("category_name") or "未分类" for asset in assets)
    usable_assets = [asset for asset in assets if asset.get("is_usable")]
    classification_candidates = []
    for asset in usable_assets:
        label = asset.get("category_name") or ""
        if not label or label in {"封面", "未分类"}:
            continue
        candidate = dict(asset)
        candidate["task_type"] = "image_category_classification"
        classification_candidates.append(candidate)

    return {
        "series": {
            "series_id": series_id,
            "series_name": series_name,
            "brand_name": brand_name,
        },
        "assets": assets,
        "summary": {
            "asset_count": len(assets),
            "usable_asset_count": len(usable_assets),
            "classification_candidate_count": len(classification_candidates),
            "category_distribution": dict(category_counter),
        },
    }


def build_image_dataset(
    raw_dir: str | Path,
    *,
    cleaned_dir: str | Path | None = None,
    include_contextual_images: bool = False,
    limit: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir) if cleaned_dir else None

    image_assets: list[dict[str, Any]] = []
    classification_manifest: list[dict[str, Any]] = []
    series_summaries: list[dict[str, Any]] = []
    error_page_count = 0
    series_with_assets = 0

    raw_files = sorted(
        path
        for path in raw_dir.glob("series_*.json")
        if re.fullmatch(r"series_\d+\.json", path.name)
    )
    if limit is not None:
        raw_files = raw_files[: max(limit, 0)]

    for raw_path in raw_files:
        with raw_path.open("r", encoding="utf-8") as f:
            raw_json = json.load(f)
        if raw_json.get("page") == "/_error":
            error_page_count += 1
        raw_image_lookup, raw_manifest = _load_raw_image_manifest(raw_path)

        clean_record = None
        if cleaned_dir:
            clean_path = cleaned_dir / raw_path.name
            if clean_path.exists():
                with clean_path.open("r", encoding="utf-8") as f:
                    clean_record = json.load(f)

        extracted = extract_series_image_assets(
            raw_json,
            clean_record=clean_record,
            include_contextual_images=include_contextual_images,
            raw_image_lookup=raw_image_lookup,
        )
        series_meta = extracted["series"]
        summary = dict(extracted["summary"])
        summary.update(series_meta)
        summary["raw_file"] = str(raw_path)
        summary["raw_image_manifest"] = str(raw_path.with_name(f"{raw_path.stem}_images.json"))
        summary["raw_image_saved_count"] = _safe_int(raw_manifest.get("success_count", 0))
        summary["raw_image_failed_count"] = _safe_int(raw_manifest.get("failed_count", 0))
        series_summaries.append(summary)
        if summary["asset_count"] > 0:
            series_with_assets += 1

        for asset in extracted["assets"]:
            record = dict(asset)
            record["raw_file"] = str(raw_path)
            record["raw_image_manifest"] = str(raw_path.with_name(f"{raw_path.stem}_images.json"))
            image_assets.append(record)
            if record.get("is_usable") and record.get("category_name") not in {"", "封面", "未分类"}:
                candidate = dict(record)
                candidate["task_type"] = "image_category_classification"
                candidate["dataset_split"] = _classification_split(
                    candidate["asset_id"],
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
                classification_manifest.append(candidate)

    overall_category_distribution = Counter(
        item.get("category_name") or "未分类" for item in image_assets
    )
    split_distribution = Counter(item.get("dataset_split") or "unknown" for item in classification_manifest)
    source_distribution = Counter(item.get("source_section") or "unknown" for item in image_assets)
    per_series_counts = defaultdict(int)
    for item in classification_manifest:
        per_series_counts[item.get("series_id") or ""] += 1

    summary = {
        "raw_files_scanned": len(raw_files),
        "error_page_count": error_page_count,
        "series_count": len(series_summaries),
        "series_with_assets": series_with_assets,
        "asset_count": len(image_assets),
        "asset_with_local_file_count": sum(1 for item in image_assets if item.get("local_exists")),
        "usable_asset_count": sum(1 for item in image_assets if item.get("is_usable")),
        "classification_candidate_count": len(classification_manifest),
        "category_distribution": dict(overall_category_distribution),
        "source_distribution": dict(source_distribution),
        "split_distribution": dict(split_distribution),
        "series_with_classification_candidates": sum(1 for count in per_series_counts.values() if count > 0),
    }

    return {
        "image_assets": image_assets,
        "classification_manifest": classification_manifest,
        "series_summaries": series_summaries,
        "summary": summary,
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


def write_image_dataset_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_assets_path = output_dir / "image_assets.jsonl"
    classification_path = output_dir / "image_classification_manifest.jsonl"
    summary_path = output_dir / "image_dataset_summary.json"
    series_summary_path = output_dir / "image_series_summary.parquet"
    asset_parquet_path = output_dir / "image_assets.parquet"
    classification_parquet_path = output_dir / "image_classification_manifest.parquet"

    with image_assets_path.open("w", encoding="utf-8") as f:
        for row in result["image_assets"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with classification_path.open("w", encoding="utf-8") as f:
        for row in result["classification_manifest"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)

    pd.DataFrame(_rows_for_parquet(result["series_summaries"])).to_parquet(
        series_summary_path,
        index=False,
        engine="pyarrow",
    )
    pd.DataFrame(_rows_for_parquet(result["image_assets"])).to_parquet(
        asset_parquet_path,
        index=False,
        engine="pyarrow",
    )
    pd.DataFrame(_rows_for_parquet(result["classification_manifest"])).to_parquet(
        classification_parquet_path,
        index=False,
        engine="pyarrow",
    )

    return {
        "image_assets_path": str(image_assets_path),
        "classification_path": str(classification_path),
        "summary_path": str(summary_path),
        "series_summary_path": str(series_summary_path),
        "asset_parquet_path": str(asset_parquet_path),
        "classification_parquet_path": str(classification_parquet_path),
    }


def download_image_assets(
    image_assets: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    limit_per_series: int = 20,
    timeout_sec: int = 15,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    per_series_downloaded: dict[str, int] = defaultdict(int)
    download_manifest: list[dict[str, Any]] = []

    for asset in image_assets:
        if not asset.get("is_usable"):
            continue
        series_id = asset.get("series_id") or "unknown"
        if per_series_downloaded[series_id] >= limit_per_series:
            continue

        image_url = asset.get("image_url", "")
        file_ext = _infer_extension(image_url)
        target_dir = output_dir / series_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{asset['asset_id']}{file_ext}"

        result = {
            "asset_id": asset["asset_id"],
            "series_id": series_id,
            "image_url": image_url,
            "status": "failed",
            "file_path": str(target_path),
            "content_type": "",
            "bytes": 0,
            "error": "",
        }

        try:
            response = session.get(image_url, timeout=timeout_sec, stream=True)
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith("image/"):
                raise ValueError(f"unexpected_content_type={content_type}")

            if not content_type and not _looks_like_image_url(image_url):
                raise ValueError("content_type_missing_and_url_not_image_like")

            with target_path.open("wb") as f:
                total_bytes = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total_bytes += len(chunk)

            result["status"] = "success"
            result["content_type"] = content_type
            result["bytes"] = total_bytes
            per_series_downloaded[series_id] += 1
        except Exception as exc:
            result["error"] = str(exc)
        download_manifest.append(result)

    manifest_path = output_dir / "download_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in download_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "download_manifest_path": str(manifest_path),
        "downloaded_count": sum(1 for row in download_manifest if row["status"] == "success"),
        "failed_count": sum(1 for row in download_manifest if row["status"] != "success"),
        "output_dir": str(output_dir),
    }
