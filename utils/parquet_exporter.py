import os


class ParquetExporter:
    def __init__(self):
        pass

    def export_training_items(self, training_items: list[dict], output_path: str) -> None:
        """将 training JSONL 结构导出为 Parquet。"""
        import pandas as pd

        rows = []
        for item in training_items:
            meta = item.get("metadata", {})
            rows.append(
                {
                    "source": meta.get("source"),
                    "url": meta.get("url"),
                    "title": meta.get("title"),
                    "series_id": meta.get("series_id"),
                    "brand_name": meta.get("brand_name"),
                    "car_type": meta.get("car_type"),
                    "model_count": meta.get("model_count"),
                    "news_count": meta.get("news_count"),
                    "crawl_timestamp": meta.get("crawl_timestamp"),
                    "chunk_index": meta.get("chunk_index"),
                    "total_chunks": meta.get("total_chunks"),
                    "tokens": meta.get("tokens"),
                    "text": item.get("text"),
                }
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False, engine="pyarrow")

    def export_clean_summary(self, clean_records: list[dict], output_path: str) -> None:
        """将 cleaned 层摘要导出为 Parquet（每车系一行）。"""
        import pandas as pd

        rows = []
        for record in clean_records:
            series = record.get("series", {})
            pricing = record.get("pricing", {})
            scores = record.get("scores", {})
            stats = record.get("stats", {})
            rows.append(
                {
                    "schema_version": record.get("schema_version"),
                    "source": record.get("source"),
                    "entity_type": record.get("entity_type"),
                    "series_id": series.get("series_id"),
                    "series_name": series.get("series_name"),
                    "brand_name": series.get("brand_name"),
                    "sub_brand_name": series.get("sub_brand_name"),
                    "car_type": series.get("car_type"),
                    "city_name": series.get("city_name"),
                    "dealer_price_range": pricing.get("dealer_price_range"),
                    "official_price_range": pricing.get("official_price_range"),
                    "latest_owner_price": pricing.get("latest_owner_price"),
                    "lowest_owner_price": pricing.get("lowest_owner_price"),
                    "total_score": scores.get("total_score"),
                    "total_review_count": scores.get("total_review_count"),
                    "model_count": stats.get("model_count"),
                    "dimension_group_count": stats.get("dimension_group_count"),
                    "image_group_count": stats.get("image_group_count"),
                    "news_count": stats.get("news_count"),
                }
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False, engine="pyarrow")
