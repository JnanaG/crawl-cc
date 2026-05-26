class DongchediCleaner:
    def __init__(self):
        pass

    @staticmethod
    def _as_dict(value) -> dict:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_list(value) -> list:
        return value if isinstance(value, list) else []

    def _extract_models(self, page_props: dict) -> list[dict]:
        models = []
        tab_list = self._as_list(self._as_dict(page_props.get("carModelsData")).get("tab_list"))
        for tab in tab_list:
            tab = self._as_dict(tab)
            for item in self._as_list(tab.get("data")):
                item = self._as_dict(item)
                info = self._as_dict(item.get("info"))
                if not info.get("car_id"):
                    continue
                car_config = self._as_dict(info.get("car_config"))
                models.append(
                    {
                        "car_id": info.get("car_id"),
                        "name": info.get("name"),
                        "year": info.get("year"),
                        "brand_name": info.get("brand_name"),
                        "series_name": info.get("series_name"),
                        "official_price": info.get("official_price_str") or info.get("price"),
                        "dealer_price": info.get("dealer_price"),
                        "owner_price": info.get("owner_price"),
                        "tags": self._as_list(info.get("tags")),
                        "base_config": self._as_list(car_config.get("base_config")),
                        "highlights_config": self._as_list(car_config.get("highlights_config")),
                        "follower_rate": self._as_dict(info.get("follower_rate")).get("text"),
                        "picture_count": info.get("picture_count", 0),
                        "is_new": bool(info.get("new_car_tag")),
                        "is_hot": bool(info.get("hot_car_tag")),
                    }
                )
        return models

    def _extract_dimensions(self, page_props: dict) -> list[dict]:
        dimensions = []
        seen = set()
        overview_data = self._as_dict(page_props.get("overviewData"))
        for item in self._as_list(overview_data.get("space")):
            item = self._as_dict(item)
            dimension = {
                "length_mm": item.get("length"),
                "width_mm": item.get("width"),
                "height_mm": item.get("height"),
                "wheelbase_mm": item.get("wheelbase"),
                "car_count": len(self._as_list(item.get("car_id_list"))),
            }
            dedupe_key = tuple(dimension.items())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            dimensions.append(dimension)
        return dimensions

    def _extract_images(self, page_props: dict) -> list[dict]:
        images = []
        image_floor_data = self._as_dict(page_props.get("imageFloorData"))
        for item in self._as_list(image_floor_data.get("floor_head_list")):
            item = self._as_dict(item)
            color_list = self._as_list(item.get("color_list"))
            images.append(
                {
                    "category": item.get("category"),
                    "category_name": item.get("text"),
                    "color_count": len(color_list),
                    "sample_colors": [
                        color.get("color_name")
                        for color in color_list[:5]
                        if isinstance(color, dict) and color.get("color_name")
                    ],
                }
            )
        return images

    def _extract_news(self, page_props: dict) -> list[dict]:
        news_sections = [
            ("newest", self._as_list(page_props.get("newestStaticNews"))),
            ("guide", self._as_list(page_props.get("guideStaticNews"))),
            ("newcar", self._as_list(page_props.get("newcarStaticNews"))),
            ("evaluating", self._as_list(page_props.get("evaluatingStaticNews"))),
            ("original", self._as_list(page_props.get("originalStaticNews"))),
        ]
        news_items = []
        for category, items in news_sections:
            for item in items[:10]:
                item = self._as_dict(item)
                user_info = self._as_dict(item.get("user_info"))
                news_items.append(
                    {
                        "category": category,
                        "title": item.get("title"),
                        "publish_time": item.get("publish_time"),
                        "watch_or_read_count": item.get("watch_or_read_count", 0),
                        "has_video": item.get("has_video", False),
                        "author": user_info.get("name"),
                        "author_verified": user_info.get("verified_content"),
                    }
                )
        return news_items

    def extract_clean_series_record(self, raw_json: dict) -> dict:
        """将懂车帝原始 SSR JSON 规范化为 cleaned 层结构。"""
        page_props = self._as_dict(self._as_dict(raw_json.get("props")).get("pageProps"))
        series_head = self._as_dict(page_props.get("seriesHomeHead"))
        price_card = self._as_dict(page_props.get("carPriceCard"))
        score_info = self._as_dict(page_props.get("scoreSimpleInfo"))
        series_name = page_props.get("seriesName") or series_head.get("series_name") or "未知车系"
        series_id = page_props.get("seriesId") or series_head.get("series_id")

        models = self._extract_models(page_props)
        dimensions = self._extract_dimensions(page_props)
        images = self._extract_images(page_props)
        news_items = self._extract_news(page_props)

        return {
            "schema_version": "v1",
            "source": "dongchedi",
            "entity_type": "car_series",
            "series": {
                "series_id": str(series_id) if series_id is not None else "",
                "series_name": series_name,
                "brand_name": series_head.get("brand_name"),
                "sub_brand_name": series_head.get("sub_brand_name"),
                "car_type": series_head.get("car_type"),
                "city_name": page_props.get("cityName", "全国"),
                "cover_url": series_head.get("cover_url") or price_card.get("cover_url"),
                "car_id_list": self._as_list(series_head.get("car_id_list")),
            },
            "pricing": {
                "dealer_price_range": series_head.get("dealer_price"),
                "official_price_range": series_head.get("official_price"),
                "latest_owner_price": price_card.get("latest_owner_price"),
                "lowest_owner_price": price_card.get("lowest_owner_price"),
                "lowest_owner_city_name": price_card.get("lowest_owner_city_name"),
                "query_price_count": price_card.get("query_price_count_en"),
            },
            "scores": {
                "total_score": score_info.get("score"),
                "total_review_count": score_info.get("total_review_count", 0),
                "comfort_score": score_info.get("comfort_score"),
                "appearance_score": score_info.get("appearance_score"),
                "configuration_score": score_info.get("configuration_score"),
                "control_score": score_info.get("control_score"),
                "power_score": score_info.get("power_score"),
                "space_score": score_info.get("space_score"),
                "interiors_score": score_info.get("interiors_score"),
            },
            "dimensions": dimensions,
            "images": images,
            "models": models,
            "news": news_items,
            "stats": {
                "model_count": len(models),
                "dimension_group_count": len(dimensions),
                "image_group_count": len(images),
                "news_count": len(news_items),
            },
        }

    def clean_record_to_markdown(self, clean_record: dict) -> str:
        """将 cleaned 层结构转换为适合 LLM 消费的 Markdown。"""
        series = clean_record.get("series", {})
        pricing = clean_record.get("pricing", {})
        scores = clean_record.get("scores", {})
        dimensions = clean_record.get("dimensions", [])
        images = clean_record.get("images", [])
        models = clean_record.get("models", [])
        news_items = clean_record.get("news", [])
        stats = clean_record.get("stats", {})

        md = []
        md.append(f"# {series.get('series_name', '未知车系')} 车型资料")
        md.append("")
        md.append("## 基础信息")
        md.append(f"- 品牌: {series.get('brand_name') or '未知'}")
        md.append(f"- 子品牌: {series.get('sub_brand_name') or '未知'}")
        md.append(f"- 车型类别: {series.get('car_type') or '未知'}")
        md.append(f"- 城市基准: {series.get('city_name') or '全国'}")
        md.append("")
        md.append("## 价格信息")
        md.append(f"- 经销商报价区间: {pricing.get('dealer_price_range') or '暂无报价'}")
        md.append(f"- 官方指导价区间: {pricing.get('official_price_range') or '暂无指导价'}")
        md.append(f"- 最新车主成交价: {pricing.get('latest_owner_price') or '暂无'}")
        md.append(f"- 最低车主成交价: {pricing.get('lowest_owner_price') or '暂无'}")
        md.append(f"- 最低成交城市: {pricing.get('lowest_owner_city_name') or '暂无'}")
        md.append(f"- 询价量级: {pricing.get('query_price_count') or '暂无'}")
        md.append("")
        md.append("## 评分信息")
        md.append(f"- 综合评分: {scores.get('total_score') or 0}")
        md.append(f"- 评价人数: {scores.get('total_review_count') or 0}")
        md.append(f"- 舒适性: {scores.get('comfort_score') or 0}")
        md.append(f"- 外观: {scores.get('appearance_score') or 0}")
        md.append(f"- 配置: {scores.get('configuration_score') or 0}")
        md.append(f"- 操控: {scores.get('control_score') or 0}")
        md.append(f"- 动力: {scores.get('power_score') or 0}")
        md.append(f"- 空间: {scores.get('space_score') or 0}")
        md.append(f"- 内饰: {scores.get('interiors_score') or 0}")
        md.append("")
        md.append("## 尺寸信息")
        if not dimensions:
            md.append("- 暂无尺寸信息")
        else:
            for item in dimensions:
                md.append(
                    "- 长宽高/轴距: "
                    f"{item.get('length_mm')}/{item.get('width_mm')}/{item.get('height_mm')}/{item.get('wheelbase_mm')} mm"
                    f"，覆盖车型数: {item.get('car_count')}"
                )
        md.append("")
        md.append("## 图片与颜色信息")
        if not images:
            md.append("- 暂无图片分类信息")
        else:
            for item in images:
                md.append(
                    f"- {item.get('category_name') or item.get('category')}: "
                    f"颜色数 {item.get('color_count')}, 示例颜色 {', '.join(item.get('sample_colors', [])) or '暂无'}"
                )
        md.append("")
        md.append(f"## 车型列表 ({stats.get('model_count', 0)} 款)")
        if not models:
            md.append("- 暂无具体车型列表信息")
        else:
            for model in models[:30]:
                tags = " | ".join(model.get("tags", [])) or "暂无"
                base_config = " | ".join(model.get("base_config", [])) or "暂无"
                highlights = " | ".join(model.get("highlights_config", [])) or "暂无"
                md.append(f"### {model.get('name') or '未知车型'}")
                md.append(f"- 年款: {model.get('year') or '未知'}")
                md.append(f"- 官方价格: {model.get('official_price') or '暂无'}")
                md.append(f"- 经销商价格: {model.get('dealer_price') or '暂无'}")
                md.append(f"- 标签: {tags}")
                md.append(f"- 基础配置: {base_config}")
                md.append(f"- 亮点配置: {highlights}")
                md.append(f"- 关注度: {model.get('follower_rate') or '暂无'}")
                md.append(f"- 图片数: {model.get('picture_count') or 0}")
                md.append("")
        md.append(f"## 新闻内容 ({stats.get('news_count', 0)} 条)")
        if not news_items:
            md.append("- 暂无新闻内容")
        else:
            for item in news_items[:20]:
                md.append(
                    f"- [{item.get('category')}] {item.get('title')} | 作者: {item.get('author') or '未知'} | "
                    f"阅读/播放: {item.get('watch_or_read_count') or 0} | 视频: {'是' if item.get('has_video') else '否'}"
                )
        return "\n".join(md).strip()

    def parse_series_data_to_markdown(self, raw_json: dict) -> str:
        """兼容旧调用方式：从原始数据直接生成 Markdown。"""
        try:
            clean_record = self.extract_clean_series_record(raw_json)
            return self.clean_record_to_markdown(clean_record)
        except Exception as e:
            return f"数据解析异常: {e}"
