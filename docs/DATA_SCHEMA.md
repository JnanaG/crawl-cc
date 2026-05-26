# Data Schema

本项目将懂车帝数据拆分为三层，分别服务于采集、清洗和大模型下游任务。

## 1. Raw Layer

目录:

- `data/raw/dongchedi/series_<series_id>.html`
- `data/raw/dongchedi/series_<series_id>.json`

用途:

- 保留页面原始 HTML，便于回放、调试和选择新的提取字段
- 保留页面 `__NEXT_DATA__` 原始 JSON，便于稳定重跑清洗逻辑

字段约定:

- 文件名中的 `series_id` 是车系主键
- HTML 与 JSON 必须一一对应

## 2. Cleaned Layer

目录:

- `data/cleaned/dongchedi/json/series_<series_id>.json`
- `data/cleaned/dongchedi/markdown/series_<series_id>.md`

用途:

- `json` 用于结构化分析、数据校验、二次加工
- `markdown` 用于 RAG、摘要、问答、切块等 LLM 任务

`series_<series_id>.json` 顶层字段:

- `schema_version`: 当前 schema 版本
- `source`: 数据源，固定为 `dongchedi`
- `entity_type`: 实体类型，当前为 `car_series`
- `series`: 车系基础信息
- `pricing`: 价格相关信息
- `scores`: 评分相关信息
- `dimensions`: 尺寸分组信息
- `images`: 图片/颜色分类信息
- `models`: 车型配置列表
- `news`: 新闻资讯列表
- `stats`: 统计摘要

`series` 字段:

- `series_id`
- `series_name`
- `brand_name`
- `sub_brand_name`
- `car_type`
- `city_name`
- `cover_url`
- `car_id_list`

`pricing` 字段:

- `dealer_price_range`
- `official_price_range`
- `latest_owner_price`
- `lowest_owner_price`
- `lowest_owner_city_name`
- `query_price_count`

`scores` 字段:

- `total_score`
- `total_review_count`
- `comfort_score`
- `appearance_score`
- `configuration_score`
- `control_score`
- `power_score`
- `space_score`
- `interiors_score`

`dimensions[]` 字段:

- `length_mm`
- `width_mm`
- `height_mm`
- `wheelbase_mm`
- `car_count`

`images[]` 字段:

- `category`
- `category_name`
- `color_count`
- `sample_colors`

`models[]` 字段:

- `car_id`
- `name`
- `year`
- `brand_name`
- `series_name`
- `official_price`
- `dealer_price`
- `owner_price`
- `tags`
- `base_config`
- `highlights_config`
- `follower_rate`
- `picture_count`
- `is_new`
- `is_hot`

`news[]` 字段:

- `category`
- `title`
- `publish_time`
- `watch_or_read_count`
- `has_video`
- `author`
- `author_verified`

`stats` 字段:

- `model_count`
- `dimension_group_count`
- `image_group_count`
- `news_count`

## 3. Training Layer

目录:

- `data/processed/dongchedi_training_data.jsonl`

用途:

- 面向 LLM/RAG 的最终消费层
- 每条记录是一个 chunk，包含元数据与正文

JSONL 字段:

- `metadata.source`
- `metadata.url`
- `metadata.title`
- `metadata.series_id`
- `metadata.brand_name`
- `metadata.car_type`
- `metadata.model_count`
- `metadata.news_count`
- `metadata.crawl_timestamp`
- `metadata.chunk_index`
- `metadata.total_chunks`
- `metadata.tokens`
- `text`

## 命名规范

- 原始层只保存“不可逆”的源数据，不做字段裁剪
- 清洗层只保存“结构化、可复用”的业务字段
- 训练层只保存“适合大模型直接消费”的文本与必要元数据

## 后续演进建议

- `schema_version` 从 `v1` 开始，后续字段变更必须升级版本
- 新增字段优先进入 cleaned layer，再决定是否进入 training layer
- 删除字段前先检查是否被评估脚本、RAG 或微调任务依赖
