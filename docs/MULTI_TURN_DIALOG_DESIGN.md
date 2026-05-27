# 多轮对话设计

## 1. 目标

在不接知识图谱的前提下，先落一个可运行的多轮对话 MVP，复用现有 `RAG + Agent + 审计` 能力，优先解决：

- 会话状态持久化
- 短期记忆与用户偏好记忆
- 多轮 query rewrite
- 多轮上下文构建
- 复用现有 RAG 检索链
- 提供最小 CLI Demo 与测试脚本
- 澄清追问策略
- 推荐/比较场景的结构化回答
- 最小多轮评测样例

## 2. 模块结构

新增目录：

- `conversation/`
- `memory/`

核心文件：

- `conversation/types.py`
- `conversation/dialog_router.py`
- `conversation/query_rewriter.py`
- `conversation/context_builder.py`
- `conversation/rag_backend.py`
- `conversation/clarification.py`
- `conversation/response_generator.py`
- `conversation/service.py`
- `memory/store.py`
- `scripts/chat_multi_turn_demo.py`
- `scripts/test_multi_turn_dialog.py`
- `scripts/eval_multi_turn_dialog.py`
- `conversation/api.py`
- `scripts/serve_multi_turn_api.py`
- `scripts/test_multi_turn_api.py`

## 3. 数据契约

### 3.1 Message

- `role`: `user | assistant | system`
- `content`: 消息正文
- `created_at`: ISO 时间
- `metadata`: 扩展字段

### 3.2 UserPreference

- `budget_text`
- `budget_min`
- `budget_max`
- `energy_type`
- `car_type`
- `brand_preference`
- `focus_points`

### 3.3 TaskMemory

- `task_type`
- `stage`
- `current_focus_series`
- `candidate_series`
- `last_rewritten_query`
- `notes`

### 3.4 ResponsePayload

- `session_id`
- `turn_id`
- `route`
- `rewritten_query`
- `answer`
- `memory_summary`
- `preferences`
- `task_memory`
- `hits`
- `should_clarify`
- `clarification_question`

## 4. 执行流程

每一轮对话执行顺序如下：

1. 读取或创建 `session`
2. 写入用户消息
3. 加载最近几轮消息
4. 合并用户偏好记忆
5. 加载任务记忆
6. 识别本轮 `route`
7. 生成澄清问题或执行 query rewrite
8. 构建 memory summary
9. 调用 RAG 后端
10. 更新任务记忆
11. 写入助手回复
12. 返回统一 payload

## 5. 当前策略

### 5.1 路由

当前支持：

- `fact_qa`
- `recommend`
- `compare`
- `follow_up`
- `clarify`

### 5.2 记忆

当前优先做：

- 短期会话记忆：最近几轮对话
- 用户偏好记忆：预算/能源/车型/关注点
- 任务记忆：当前焦点车系、候选车系、当前阶段

### 5.3 Query Rewrite

当前处理：

- 代词消解：`它/这款/那款`
- 比较补全：`那和银河L7比呢`
- 偏好补齐：预算/能源/车型/关注点

## 6. 存储

第一阶段使用 SQLite：

- `sessions`
- `messages`
- `preferences`
- `task_memory`

默认路径：

- `data/conversation/chat_memory.sqlite3`

## 7. 运行方式

不启用 LLM：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/chat_multi_turn_demo.py `
  --backend-mode hash `
  --message "推荐20万内的插混SUV，重点看空间和油耗"
```

继续同一个会话：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/chat_multi_turn_demo.py `
  --session-id <上一步返回的session_id> `
  --backend-mode hash `
  --message "它的价格怎么样" `
  --show-session
```

启用 LLM：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/chat_multi_turn_demo.py `
  --backend-mode hash `
  --use-llm `
  --llm-provider ollama `
  --llm-model qwen2.5:3b `
  --llm-api-base http://localhost:11434 `
  --message "推荐20万内的插混SUV，重点看空间和油耗"
```

测试：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/test_multi_turn_dialog.py
```

## 8. 第二阶段增强

当前已补齐：

- 推荐场景缺关键槽位时主动追问
- 比较场景缺少比较对象时追问
- 追问场景缺少焦点车型时追问
- 推荐回答结构化输出
- 比较回答结构化输出
- 最小多轮评测集与评测脚本

## 9. 运行与评测

运行多轮 demo：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/chat_multi_turn_demo.py `
  --backend-mode hash `
  --message "推荐一款家用车"
```

运行多轮回归测试：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/test_multi_turn_dialog.py
```

运行多轮评测：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/eval_multi_turn_dialog.py
```

评测样例位置：

- `data/eval/multi_turn_dialog_eval.jsonl`

## 10. 下一步

下一阶段建议继续补：

- 推荐/比较专用 prompt 进一步细化
- 多轮评测集
- workflow 集成
- 鉴权与限流

## 11. 在线 API

当前已提供 `FastAPI` 封装，核心入口：

- `conversation/api.py`
- `scripts/serve_multi_turn_api.py`

当前接口：

- `GET /health`
- `GET /`
- `GET /app`
- `GET /api/v1/runtime-config`
- `POST /api/v1/sessions`
- `GET /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `DELETE /api/v1/sessions/{session_id}`
- `POST /api/v1/chat`
- `GET /api/v1/overview`
- `GET /api/v1/modules/{module_name}`

启动方式：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/serve_multi_turn_api.py `
  --backend-mode hash `
  --host 127.0.0.1 `
  --port 8001
```

启动后可访问：

- 前端控制台：`http://127.0.0.1:8001/`
- Swagger：`http://127.0.0.1:8001/docs`

启用真实 LLM：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/serve_multi_turn_api.py `
  --backend-mode hash `
  --use-llm `
  --llm-provider ollama `
  --llm-model qwen2.5:3b `
  --llm-api-base http://localhost:11434 `
  --host 127.0.0.1 `
  --port 8001
```

接口测试：

```powershell
& D:\workplace\Crawl\.venv\Scripts\python.exe scripts/test_multi_turn_api.py
```

### 11.1 前端控制台

当前提供一个最小单页前端，位置：

- `conversation/web/index.html`
- `conversation/web/app.js`
- `conversation/web/styles.css`

页面包含：

- 多轮聊天区
- 会话列表
- 问答后端切换区
- 链路总览卡片
- 数据资产 / 评测 / 反馈 / workflow 摘要面板

问答后端切换区当前支持：

- `backend_mode`: `hash / faiss`
- `use_llm`: 开 / 关
- `llm_provider`
- `llm_model`
- `llm_api_base`
- `llm_api_key`

这些配置按“本次提问”动态生效，不影响会话记忆本身。

`llm_api_key` 额外约束：

- 仅作为前端临时输入
- 仅随当前请求发送到后端
- 不写入本地会话数据库
- 不出现在 `runtime-config` 接口响应里
- 不回显在 `chat` 响应里

### 11.2 模块最小接入

当前前端已接入这些后端链路的最小摘要：

- `sessions`
- `chat`
- `reports`
- `assets`
- `evaluation`
- `feedback`
- `workflow`

这一步的目标不是把所有模块都做成完整可操作后台，而是先打通：

- 前端可看
- 后端可取
- 各链路状态可串联

补充说明：

- `hash` 后端默认离线可跑，适合验证多轮记忆和 query rewrite。
- `faiss` 后端复用现有真实向量库，更适合正式问答效果验证，但依赖 embedding 运行环境。
