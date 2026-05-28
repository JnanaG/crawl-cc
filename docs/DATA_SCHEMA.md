# Data Schema

本项目将懂车帝数据拆分为三层，分别服务于采集、清洗和大模型下游任务。

## 1. Raw Layer

目录:

- `data/raw/dongchedi/series_<series_id>.html`
- `data/raw/dongchedi/series_<series_id>.json`
- `data/raw/dongchedi/series_<series_id>_images.json`
- `data/raw/dongchedi/images/series_<series_id>/*`

用途:

- 保留页面原始 HTML，便于回放、调试和选择新的提取字段
- 保留页面 `__NEXT_DATA__` 原始 JSON，便于稳定重跑清洗逻辑
- 保留车系页相关原始图片文件，供多模态 caption、视觉训练和人工质检复用
- 保留图片保存 manifest，供图片资产层回填 `local_path`

字段约定:

- 文件名中的 `series_id` 是车系主键
- HTML 与 JSON 必须一一对应
- `series_<series_id>_images.json` 记录每张原始图片的 URL、来源、保存状态、本地路径和字节数

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

## 4. Multimodal Layer

目录:

- `data/multimodal/image_assets.jsonl`
- `data/multimodal/image_assets.parquet`
- `data/multimodal/image_classification_manifest.jsonl`
- `data/multimodal/image_classification_manifest.parquet`
- `data/multimodal/image_series_summary.parquet`
- `data/multimodal/image_dataset_summary.json`

用途:

- 从懂车帝原始车系页中抽取可复用图片资产
- 为图像分类、小模型训练、人工质检、图文检索提供统一 manifest
- 在不依赖图像解码的前提下先做第一阶段质量打分和筛选

`image_assets` 字段:

- `asset_id`
- `series_id`
- `series_name`
- `brand_name`
- `image_url`
- `category`
- `category_name`
- `source_section`
- `image_role`
- `rank`
- `width`
- `height`
- `car_id`
- `color_id`
- `color_name`
- `raw_ref`
- `local_path`
- `local_exists`
- `raw_image_status`
- `raw_content_type`
- `raw_bytes`
- `quality_score`
- `quality_flags`
- `is_usable`

`image_classification_manifest` 额外字段:

- `task_type`
- `dataset_split`

`image_caption_corpus` 字段:

- `metadata.source`
- `metadata.url`
- `metadata.title`
- `metadata.series_id`
- `metadata.brand_name`
- `metadata.asset_id`
- `metadata.image_url`
- `metadata.image_category`
- `metadata.image_category_name`
- `metadata.image_role`
- `metadata.image_source_section`
- `metadata.image_quality_score`
- `metadata.modality`
- `metadata.content_type`
- `metadata.caption_provider`
- `metadata.caption_model`
- `metadata.caption_prompt_version`
- `text`

说明:

- `image_assets` 是图片主资产表，面向多模态任务复用
- `image_classification_manifest` 是面向视觉分类的候选训练集清单
- `image_caption_corpus` 是将图片理解结果文本化后的 RAG 语料，可直接并入现有 embedding / vector store 流程
- `dataset_split` 由 `asset_id` 做稳定切分，默认生成 `train / val / test`

## 后续演进建议

- `schema_version` 从 `v1` 开始，后续字段变更必须升级版本
- 新增字段优先进入 cleaned layer，再决定是否进入 training layer
- 删除字段前先检查是否被评估脚本、RAG 或微调任务依赖
- 多模态新增字段优先写入 `image_assets`，稳定后再考虑沉淀进 cleaned layer
