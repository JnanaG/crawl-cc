import json
import os
import statistics
from datetime import datetime
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DataQualityEngine:
    def __init__(self):
        self.required_series_fields = [
            "series_id",
            "series_name",
            "brand_name",
            "car_type",
        ]
        self.required_training_meta = [
            "source",
            "url",
            "title",
            "series_id",
            "chunk_index",
            "total_chunks",
            "tokens",
        ]

    def _is_probably_garbled(self, text: str) -> bool:
        if not text:
            return True
        mojibake_chars = set("ÃÂåæçïðþ¤�")
        hit = sum(1 for ch in text if ch in mojibake_chars)
        ratio = hit / max(len(text), 1)
        return ratio > 0.03

    def validate_clean_record(self, record: dict[str, Any]) -> dict[str, Any]:
        issues = []
        warnings = []
        series = record.get("series", {})
        pricing = record.get("pricing", {})
        scores = record.get("scores", {})
        stats = record.get("stats", {})
        models = record.get("models", [])
        news = record.get("news", [])

        for field in self.required_series_fields:
            if not series.get(field):
                issues.append(f"series.{field} 缺失")

        if not pricing.get("dealer_price_range") and not pricing.get("official_price_range"):
            issues.append("pricing.dealer_price_range 与 pricing.official_price_range 均缺失")

        if stats.get("model_count", 0) <= 0:
            issues.append("stats.model_count <= 0")

        score_value = scores.get("total_score")
        if score_value is not None and isinstance(score_value, (int, float)):
            if score_value < 0 or score_value > 500:
                warnings.append(f"scores.total_score={score_value} 超出常见区间")

        # 车型去重检测
        car_ids = [m.get("car_id") for m in models if m.get("car_id")]
        duplicate_model_count = len(car_ids) - len(set(car_ids))
        if duplicate_model_count > 0:
            warnings.append(f"models 中存在重复 car_id 数量: {duplicate_model_count}")

        # 标题乱码检测
        if self._is_probably_garbled(series.get("series_name", "")):
            issues.append("series.series_name 疑似乱码")

        # 新闻质量检测
        empty_news_title = sum(1 for n in news if not n.get("title"))
        if empty_news_title > 0:
            warnings.append(f"news 中空标题数量: {empty_news_title}")

        return {
            "series_id": series.get("series_id"),
            "series_name": series.get("series_name"),
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    def validate_training_item(self, item: dict[str, Any]) -> dict[str, Any]:
        errors = []
        warnings = []
        meta = item.get("metadata", {})
        text = item.get("text", "")

        for field in self.required_training_meta:
            if meta.get(field) in (None, ""):
                errors.append(f"metadata.{field} 缺失")

        if len(text.strip()) < 80:
            errors.append("text 长度过短(<80)")

        tokens = meta.get("tokens", 0)
        if isinstance(tokens, int):
            if tokens <= 0:
                errors.append("metadata.tokens <= 0")
            elif tokens < 30:
                warnings.append("metadata.tokens 过小(<30)")
            elif tokens > 1500:
                warnings.append("metadata.tokens 偏大(>1500)")
        else:
            errors.append("metadata.tokens 不是整数")

        if self._is_probably_garbled(text):
            errors.append("text 疑似乱码")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _calc_token_stats(self, training_items: list[dict[str, Any]]) -> dict[str, Any]:
        token_list = [
            item.get("metadata", {}).get("tokens", 0)
            for item in training_items
            if isinstance(item.get("metadata", {}).get("tokens", 0), int)
        ]
        if not token_list:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        return {
            "min": min(token_list),
            "max": max(token_list),
            "avg": round(sum(token_list) / len(token_list), 2),
            "median": round(statistics.median(token_list), 2),
        }

    def _calc_field_coverage(self, clean_records: list[dict[str, Any]]) -> dict[str, float]:
        target_fields = [
            ("series.series_id", lambda r: r.get("series", {}).get("series_id")),
            ("series.series_name", lambda r: r.get("series", {}).get("series_name")),
            ("series.brand_name", lambda r: r.get("series", {}).get("brand_name")),
            ("series.car_type", lambda r: r.get("series", {}).get("car_type")),
            ("pricing.dealer_price_range", lambda r: r.get("pricing", {}).get("dealer_price_range")),
            ("pricing.official_price_range", lambda r: r.get("pricing", {}).get("official_price_range")),
            ("scores.total_score", lambda r: r.get("scores", {}).get("total_score")),
            ("stats.model_count", lambda r: r.get("stats", {}).get("model_count")),
            ("stats.news_count", lambda r: r.get("stats", {}).get("news_count")),
            ("dimensions", lambda r: r.get("dimensions")),
            ("images", lambda r: r.get("images")),
            ("models", lambda r: r.get("models")),
            ("news", lambda r: r.get("news")),
        ]
        total = len(clean_records)
        if total == 0:
            return {name: 0.0 for name, _ in target_fields}

        coverage = {}
        for name, getter in target_fields:
            hit = 0
            for record in clean_records:
                value = getter(record)
                if value in (None, "", [], {}):
                    continue
                hit += 1
            coverage[name] = round((hit / total) * 100, 2)
        return coverage

    def _make_distribution_charts(self, training_items: list[dict[str, Any]], clean_records: list[dict[str, Any]], output_dir: str) -> dict[str, str]:
        charts = {}
        os.makedirs(output_dir, exist_ok=True)

        # 1) token 分布图
        token_values = [
            item.get("metadata", {}).get("tokens", 0)
            for item in training_items
            if isinstance(item.get("metadata", {}).get("tokens", 0), int)
        ]
        if token_values:
            plt.figure(figsize=(8, 5))
            plt.hist(token_values, bins=10, color="#4C72B0", edgecolor="black")
            plt.title("Token Distribution")
            plt.xlabel("tokens")
            plt.ylabel("count")
            token_chart = os.path.join(output_dir, "token_distribution.png")
            plt.tight_layout()
            plt.savefig(token_chart, dpi=120)
            plt.close()
            charts["token_distribution"] = token_chart

        # 2) 每车系车型数分布图
        model_counts = []
        series_names = []
        for record in clean_records:
            series = record.get("series", {})
            stats = record.get("stats", {})
            series_names.append(series.get("series_name") or series.get("series_id") or "unknown")
            model_counts.append(stats.get("model_count", 0))
        if model_counts:
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(model_counts)), model_counts, color="#55A868")
            plt.title("Model Count Per Series")
            plt.xlabel("series index")
            plt.ylabel("model_count")
            plt.xticks(range(len(series_names)), [str(i + 1) for i in range(len(series_names))])
            model_chart = os.path.join(output_dir, "model_count_distribution.png")
            plt.tight_layout()
            plt.savefig(model_chart, dpi=120)
            plt.close()
            charts["model_count_distribution"] = model_chart

        return charts

    def generate_quality_report(
        self,
        clean_validations: list[dict[str, Any]],
        clean_records: list[dict[str, Any]],
        training_items: list[dict[str, Any]],
        dropped_training_items: int,
        output_dir: str,
    ) -> dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.now().isoformat()

        invalid_clean = [v for v in clean_validations if not v.get("is_valid")]
        warn_clean = [v for v in clean_validations if v.get("warnings")]

        series_ids = set()
        for item in training_items:
            sid = item.get("metadata", {}).get("series_id")
            if sid:
                series_ids.add(str(sid))

        report = {
            "generated_at": now,
            "summary": {
                "series_total": len(clean_validations),
                "series_valid": len(clean_validations) - len(invalid_clean),
                "series_invalid": len(invalid_clean),
                "series_with_warnings": len(warn_clean),
                "training_records_total": len(training_items),
                "training_records_dropped": dropped_training_items,
                "covered_series_count": len(series_ids),
            },
            "training_token_stats": self._calc_token_stats(training_items),
            "field_coverage_percent": self._calc_field_coverage(clean_records),
            "invalid_series_details": invalid_clean,
            "warning_series_details": warn_clean[:20],
        }

        charts = self._make_distribution_charts(training_items, clean_records, output_dir)
        report["charts"] = charts

        json_path = os.path.join(output_dir, "quality_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        md_lines = [
            "# 数据质量报告",
            "",
            f"- 生成时间: {now}",
            f"- 车系总数: {report['summary']['series_total']}",
            f"- 车系有效数: {report['summary']['series_valid']}",
            f"- 车系无效数: {report['summary']['series_invalid']}",
            f"- 训练记录总数: {report['summary']['training_records_total']}",
            f"- 训练记录丢弃数: {report['summary']['training_records_dropped']}",
            "",
            "## Token 统计",
            f"- min: {report['training_token_stats']['min']}",
            f"- max: {report['training_token_stats']['max']}",
            f"- avg: {report['training_token_stats']['avg']}",
            f"- median: {report['training_token_stats']['median']}",
            "",
            "## 字段覆盖率(%)",
        ]
        for field_name, pct in report["field_coverage_percent"].items():
            md_lines.append(f"- {field_name}: {pct}%")

        md_lines.extend(
            [
                "",
                "## 分布图文件",
                f"- token 分布图: {charts.get('token_distribution', '未生成')}",
                f"- 车型数量分布图: {charts.get('model_count_distribution', '未生成')}",
                "",
            "## 无效车系列表",
            ]
        )
        if invalid_clean:
            for item in invalid_clean:
                md_lines.append(
                    f"- {item.get('series_id')} {item.get('series_name')}: {'; '.join(item.get('issues', []))}"
                )
        else:
            md_lines.append("- 无")

        md_path = os.path.join(output_dir, "quality_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return report
