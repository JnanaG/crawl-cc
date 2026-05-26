# Quality Rules

本文件定义当前项目在 `cleaned` 和 `training` 两层执行的数据质量规则。

## Cleaned Layer 规则

执行对象: `data/cleaned/dongchedi/json/series_<series_id>.json`

强校验（不通过记为 invalid）:

- `series.series_id` 不能为空
- `series.series_name` 不能为空
- `series.brand_name` 不能为空
- `series.car_type` 不能为空
- `pricing.dealer_price_range` 与 `pricing.official_price_range` 不能同时为空
- `stats.model_count` 必须大于 0
- `series.series_name` 不能疑似乱码

弱校验（通过但记 warning）:

- `scores.total_score` 超出常见区间（<0 或 >500）
- `models` 内存在重复 `car_id`
- `news` 中存在空标题

## Training Layer 规则

执行对象: `data/processed/dongchedi_training_data.jsonl` 的每条记录

强校验（不通过直接过滤）:

- `metadata` 必填字段缺失:
  - `source`
  - `url`
  - `title`
  - `series_id`
  - `chunk_index`
  - `total_chunks`
  - `tokens`
- `text` 长度小于 80
- `metadata.tokens <= 0` 或非整数
- `text` 疑似乱码

弱校验（通过但记 warning）:

- `metadata.tokens < 30`
- `metadata.tokens > 1500`

## 质量报告产出

每次主流程运行后自动产出:

- `data/reports/quality_report.json`
- `data/reports/quality_report.md`
- `data/reports/task_summary.json`
- `data/reports/task_summary.md`

报告包含:

- 车系有效/无效统计
- 训练样本总量与丢弃量
- token 分布统计（min/max/avg/median）
- 关键字段覆盖率（百分比）
- 分布图文件（token 分布、车型数量分布）
- 无效样本原因明细

## Parquet 输出

每次主流程运行后还会输出:

- `data/processed/dongchedi_training_data.parquet`（训练层）
- `data/cleaned/dongchedi/cleaned_series.parquet`（cleaned 摘要层）

## 强工程能力

当前流水线已支持:

- 断点续跑: `data/state/job_state.json` 记录每个车系任务状态
- 失败重试: 抓取请求支持重试和指数退避
- 限速: 请求前统一节流并增加随机抖动
- 日志分级: INFO/DEBUG/WARNING/ERROR，运行日志落盘到 `data/logs/`
- 抓取统计: 成功/失败/跳过、重试次数、耗时与状态码分布
- 任务汇总: 自动输出 `task_summary.json/md`
