# 图片数据链路说明

## 1. 目标

本链路在现有文本数据流水线之外，补充一条面向汽车图片数据的离线数据生产流程，用于支持以下场景：

- 多模态数据治理与资产沉淀
- 汽车图片分类训练样本构造
- 多模态 caption 生成与文本化入库
- 图片质量筛选与人工质检
- 后续图文检索、视觉分类、小模型训练

## 2. 输入来源

输入以 `data/raw/dongchedi/series_*.json` 为主，必要时结合 `data/cleaned/dongchedi/json/series_*.json` 补充车系元数据。

抓取阶段会同步保存以下原始文件：

- `data/raw/dongchedi/series_<series_id>.html`
- `data/raw/dongchedi/series_<series_id>.json`
- `data/raw/dongchedi/series_<series_id>_images.json`
- `data/raw/dongchedi/images/series_<series_id>/*`

说明：

- `series_<series_id>_images.json` 是该车系原始图片保存 manifest
- `images/series_<series_id>/` 下是实际下载落盘的原始图片文件

当前优先抽取以下图片来源：

- `pageProps.seriesHomeHead.cover_url`
- `pageProps.seriesHomeHead.series_image_info_list`
- `pageProps.seriesHomeHead.pics_summary_info`
- `pageProps.imageFloorData.floor_image_list`
- `pageProps.imageFloorData.floor_head_list`

可选纳入的上下文图片来源：

- 新闻封面图
- 新闻配图
- 同品牌推荐车系封面
- 推荐车系封面

## 3. 输出产物

脚本入口：`scripts/build_image_dataset.py`

默认输出目录：`data/multimodal/`

核心产物：

- `image_assets.jsonl`
- `image_assets.parquet`
- `image_classification_manifest.jsonl`
- `image_classification_manifest.parquet`
- `image_series_summary.parquet`
- `image_dataset_summary.json`
- `image_caption_corpus.jsonl`
- `image_caption_corpus.parquet`
- `image_caption_generation_log.jsonl`
- `image_caption_summary.json`

如果开启下载：

- `downloaded_images/<series_id>/<asset_id>.<ext>`
- `downloaded_images/download_manifest.jsonl`

## 4. 字段说明

### 4.1 `image_assets`

每条记录代表一张归一化图片资产，主要字段包括：

- `asset_id`: 图片资产唯一 ID
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

### 4.2 `image_classification_manifest`

这是面向视觉分类任务的候选样本清单，在 `image_assets` 基础上增加：

- `task_type`: 固定为 `image_category_classification`
- `dataset_split`: `train / val / test`

默认只保留满足以下条件的图片：

- 来自核心图片源
- 质量分达到可用阈值
- 具有明确分类标签
- 非纯封面图、非未分类图

### 4.3 `image_caption_corpus`

这是将图片内容文本化后的 RAG 语料，每条记录的结构与现有文本 `processed jsonl` 保持兼容，核心字段包括：

- `metadata.source`: 固定为 `dongchedi_image_caption`
- `metadata.url`: 图片 URL
- `metadata.title`: 例如 `某车系 外观 图片描述`
- `metadata.series_id`
- `metadata.brand_name`
- `metadata.asset_id`
- `metadata.image_category`
- `metadata.image_category_name`
- `metadata.image_role`
- `metadata.image_source_section`
- `metadata.image_quality_score`
- `metadata.modality`: 固定为 `image_caption`
- `metadata.content_type`: 固定为 `image_caption`
- `metadata.caption_provider`
- `metadata.caption_model`
- `metadata.caption_prompt_version`
- `text`: 最终用于 embedding 和检索的 caption 文本

## 5. 质量规则

当前采用轻量规则打分，不依赖图像解码：

- URL 是否合法
- URL 是否像图片链接
- 是否绑定到车系
- 是否具备图片分类
- 是否带有分辨率信息
- 分辨率是否达到较高档位
- 来源是否属于高可信核心图片源

说明：

- 核心图片源包括 `series_cover`、`series_gallery`、`image_floor`、`image_floor_cover`
- 新闻和 UGC 图片默认只作为上下文扩展，不优先进入训练候选
- 如果存在抓取阶段下载的原始图片，`image_assets` 会优先回填 `local_path`

## 6. Caption 资产化

脚本入口：`scripts/build_image_caption_corpus.py`

支持三种 caption 方式：

- `heuristic`: 基于图片元数据的规则化 caption，适合本地离线调试和测试
- `openai_compatible`: 通过兼容 OpenAI 的多模态接口直接读取图片 URL 生成 caption
- `ollama`: 先下载图片，再通过 Ollama 多模态模型生成 caption

caption 处理思路：

- 对高质量可用图片做 caption
- 将 caption 与 `series_id / image_url / category / quality_score` 绑定
- 生成 `image_caption_corpus.jsonl`
- 后续在 `rag_llm_demo.py build` 时通过 `--extra-inputs` 并入现有文本语料

## 7. 使用方式

### 6.1 只生成 manifest

```bash
python scripts/build_image_dataset.py
```

说明：

- 该命令会优先读取 `data/raw/dongchedi/series_<series_id>_images.json`
- 如果 manifest 中存在已落盘图片，会自动把 `local_path` 回填到 `image_assets`

### 6.2 只处理部分 raw 数据

```bash
python scripts/build_image_dataset.py --limit 50
```

### 6.2.1 为已有 raw json 回填原始图片

```bash
python scripts/backfill_raw_images.py --skip-existing --limit 50
```

说明：

- 这个脚本会扫描已有 `series_<id>.json`
- 为每个车系补齐 `series_<id>_images.json`
- 并把原始图片落到 `data/raw/dongchedi/images/series_<id>/`

### 6.3 纳入新闻等上下文图片

```bash
python scripts/build_image_dataset.py --include-contextual-images
```

### 6.4 下载可用图片到本地

```bash
python scripts/build_image_dataset.py --download-assets --download-limit-per-series 10
```

### 7.5 生成图片 caption 语料

```bash
python scripts/build_image_caption_corpus.py --caption-provider heuristic
```

### 7.6 将 caption 语料并入现有 RAG

```bash
python rag_llm_demo.py build ^
  --input data/processed/dongchedi_training_data.jsonl ^
  --extra-inputs data/multimodal/image_caption_corpus.jsonl ^
  --embedding-provider hash
```

## 8. 后续演进建议

- 接入 Pillow/OpenCV，增加真实图像分辨率、模糊度、重复图检测
- 增加感知哈希、CLIP embedding，支持视觉去重和语义聚类
- 引入 PyTorch 训练脚本，对 `外观/内饰/空间` 等类别做轻量分类
- 优化多模态 prompt，将 caption 提升为更适合汽车检索的结构化描述
- 补充图文对齐样本，将图片资产与车型配置、价格、描述文本关联
- 将图片链路接入统一 workflow，纳入质量报告和资产血缘
