const state = {
  sessionId: "",
  userId: "demo-user",
  requestInFlight: false,
  requestStartedAt: 0,
  requestTimer: null,
  runtimeConfig: {
    backend_mode: "hash",
    use_llm: false,
    llm_provider: "ollama",
    llm_model: "",
    llm_api_base: "",
    llm_api_key: "",
  },
};

function $(id) {
  return document.getElementById(id);
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function appendMessage(role, content, meta = "") {
  const box = $("chatWindow");
  const node = document.createElement("div");
  node.className = `message ${role}`;
  const metaNode = document.createElement("small");
  metaNode.textContent = meta || (role === "user" ? "用户" : "助手");
  const contentNode = document.createElement("div");
  contentNode.className = "message-content";
  contentNode.textContent = String(content);
  node.appendChild(metaNode);
  node.appendChild(contentNode);
  box.appendChild(node);
  box.scrollTop = box.scrollHeight;
  return node;
}

function updateMessage(node, content, meta = "") {
  if (!node) return;
  const metaNode = node.querySelector("small");
  const contentNode = node.querySelector(".message-content");
  if (metaNode) {
    metaNode.textContent = meta;
  }
  if (contentNode) {
    contentNode.textContent = String(content);
  }
  const box = $("chatWindow");
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const contentType = resp.headers.get("content-type") || "";
  let payload = null;
  let text = "";
  if (contentType.includes("application/json")) {
    payload = await resp.json();
  } else {
    text = await resp.text();
  }
  if (!resp.ok) {
    const detail = payload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message || payload?.message || text || `${resp.status}`;
    const error = new Error(message);
    error.status = resp.status;
    error.detail = detail || payload || text;
    throw error;
  }
  return payload;
}

function renderRequestStatus(message) {
  const sessionText = state.sessionId ? `当前会话: ${state.sessionId}` : "当前未选择会话";
  $("chatMeta").textContent = `${sessionText} | ${message}`;
}

function setSendingState(sending, runtimeConfig = null, pendingNode = null) {
  state.requestInFlight = sending;
  const sendBtn = $("sendBtn");
  const input = $("messageInput");
  sendBtn.disabled = sending;
  input.disabled = sending;
  sendBtn.textContent = sending ? "发送中..." : "发送";

  if (state.requestTimer) {
    clearInterval(state.requestTimer);
    state.requestTimer = null;
  }

  if (!sending) {
    renderRequestStatus("空闲");
    return;
  }

  state.requestStartedAt = Date.now();
  const backend = runtimeConfig?.backend_mode || state.runtimeConfig.backend_mode || "hash";
  const llmText = runtimeConfig?.use_llm
    ? `${runtimeConfig?.llm_provider || "ollama"}:${runtimeConfig?.llm_model || "default"}`
    : "off";

  const update = () => {
    const elapsedSec = Math.max(0, Math.round((Date.now() - state.requestStartedAt) / 1000));
    renderRequestStatus(`请求处理中 ${elapsedSec}s | backend=${backend} | llm=${llmText} | 请查看后端终端日志`);
    if (pendingNode) {
      updateMessage(
        pendingNode,
        `请求已发送，后端正在处理中...\n已等待 ${elapsedSec}s\nbackend=${backend}\nllm=${llmText}\n如果长时间无响应，请查看后端终端中的 trace_id 和错误日志。`,
        "系统"
      );
      pendingNode.classList.add("pending");
    }
  };

  update();
  state.requestTimer = setInterval(update, 1000);
}

function formatSendError(error) {
  const detail = error?.detail || {};
  const traceId = detail?.trace_id ? `\ntrace_id: ${detail.trace_id}` : "";
  const stage = detail?.stage ? `\nstage: ${detail.stage}` : "";
  const message = error?.message || "未知错误";
  return `发送失败。\n原因: ${message}${stage}${traceId}\n请同时查看后端终端中的同名日志。`;
}

async function createSession() {
  const userId = $("userIdInput").value.trim() || "demo-user";
  const title = $("titleInput").value.trim();
  const data = await fetchJson("/api/v1/sessions", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, title }),
  });
  state.sessionId = data.session_id;
  state.userId = userId;
  $("sessionIdInput").value = state.sessionId;
  $("chatWindow").innerHTML = "";
  $("chatMeta").textContent = `当前会话: ${state.sessionId}`;
  await loadSessions();
}

async function sendMessage() {
  if (state.requestInFlight) return;
  const message = $("messageInput").value.trim();
  if (!message) return;
  const sessionId = $("sessionIdInput").value.trim();
  const userId = $("userIdInput").value.trim() || "demo-user";
  if (!sessionId) {
    alert("请先新建或选择一个会话");
    return;
  }
  state.sessionId = sessionId;
  state.userId = userId;
  appendMessage("user", message);
  $("messageInput").value = "";
  const runtimeConfig = collectRuntimeConfig();
  const pendingNode = appendMessage("assistant", "请求已发送，后端正在处理中...", "系统");
  pendingNode.classList.add("system");
  setSendingState(true, runtimeConfig, pendingNode);
  try {
    const data = await fetchJson("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        user_id: userId,
        message,
        runtime_config: runtimeConfig,
      }),
    });
    const meta = [
      `route=${data.route}`,
      `rewrite=${data.rewritten_query}`,
      `backend=${data.runtime_config_used?.backend_mode || state.runtimeConfig.backend_mode}`,
      data.runtime_config_used?.use_llm
        ? `llm=${data.runtime_config_used?.llm_provider || state.runtimeConfig.llm_provider}:${data.runtime_config_used?.llm_model || state.runtimeConfig.llm_model || "default"}`
        : "llm=off",
      data.should_clarify ? "需要澄清" : "正常回答",
    ].join(" | ");
    pendingNode.classList.remove("system", "pending", "error");
    pendingNode.classList.add("assistant");
    updateMessage(pendingNode, data.answer, meta);
    await loadSessionDetail(sessionId);
    await loadOverview();
  } catch (error) {
    console.error(error);
    pendingNode.classList.remove("pending");
    pendingNode.classList.add("error");
    updateMessage(pendingNode, formatSendError(error), "系统");
    throw error;
  } finally {
    setSendingState(false);
  }
}

async function loadSessions() {
  const items = await fetchJson("/api/v1/sessions?limit=20");
  const box = $("sessionList");
  box.innerHTML = "";
  for (const item of items) {
    const node = document.createElement("div");
    node.className = `session-item ${item.session_id === state.sessionId ? "active" : ""}`;
    node.innerHTML = `
      <div class="title">${escapeHtml(item.title || item.session_id.slice(0, 8))}</div>
      <div class="meta">${escapeHtml(item.user_id || "anonymous")} | ${escapeHtml(item.updated_at)}</div>
    `;
    node.onclick = async () => {
      state.sessionId = item.session_id;
      $("sessionIdInput").value = item.session_id;
      if (item.user_id) $("userIdInput").value = item.user_id;
      await loadSessionDetail(item.session_id);
      await loadSessions();
    };
    box.appendChild(node);
  }
}

async function loadSessionDetail(sessionId) {
  const data = await fetchJson(`/api/v1/sessions/${sessionId}`);
  $("chatWindow").innerHTML = "";
  $("chatMeta").textContent = `当前会话: ${data.session_id} | 用户: ${data.user_id || "anonymous"} | 标题: ${data.title || "-"}`;
  for (const msg of data.messages) {
    appendMessage(msg.role === "user" ? "user" : "assistant", msg.content, msg.role === "user" ? "用户" : "助手");
  }
}

function renderCards(cards) {
  const box = $("cards");
  box.innerHTML = "";
  const entries = [
    ["会话数", cards.session_count],
    ["任务总数", cards.task_total],
    ["任务成功", cards.task_success_count],
    ["训练样本", cards.training_records_total],
    ["RAG 行数", cards.rag_rows],
    ["训练行数", cards.training_rows],
    ["评测候选", cards.eval_candidate_rows],
    ["faithfulness", cards.faithfulness],
    ["answer_relevancy", cards.answer_relevancy],
    ["反馈总数", cards.feedback_total],
    ["workflow_success", cards.workflow_success],
  ];
  for (const [label, value] of entries) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value ?? "-")}</div>`;
    box.appendChild(card);
  }
}

async function loadOverview() {
  const data = await fetchJson("/api/v1/overview");
  renderCards(data.cards || {});
  $("assetsBox").textContent = formatJson(data.modules.assets || {});
  $("evaluationBox").textContent = formatJson(data.modules.evaluation || {});
  $("feedbackBox").textContent = formatJson(data.modules.feedback || {});
  $("workflowBox").textContent = formatJson(data.modules.workflow || {});
}

async function loadRuntimeConfig() {
  const data = await fetchJson("/api/v1/runtime-config");
  const cfg = data.default_runtime_config || {};
  state.runtimeConfig = {
    backend_mode: cfg.backend_mode || "hash",
    use_llm: Boolean(cfg.use_llm),
    llm_provider: cfg.llm_provider || "ollama",
    llm_model: cfg.llm_model || "",
    llm_api_base: cfg.llm_api_base || "",
    llm_api_key: "",
  };
  applyRuntimeConfigToForm();
}

function applyRuntimeConfigToForm() {
  $("backendModeSelect").value = state.runtimeConfig.backend_mode || "hash";
  $("useLlmCheckbox").checked = Boolean(state.runtimeConfig.use_llm);
  $("llmProviderSelect").value = state.runtimeConfig.llm_provider || "ollama";
  $("llmModelInput").value = state.runtimeConfig.llm_model || "";
  $("llmApiBaseInput").value = state.runtimeConfig.llm_api_base || "";
  $("llmApiKeyInput").value = state.runtimeConfig.llm_api_key || "";
}

function collectRuntimeConfig() {
  state.runtimeConfig = {
    backend_mode: $("backendModeSelect").value,
    use_llm: $("useLlmCheckbox").checked,
    llm_provider: $("llmProviderSelect").value,
    llm_model: $("llmModelInput").value.trim(),
    llm_api_base: $("llmApiBaseInput").value.trim(),
    llm_api_key: $("llmApiKeyInput").value,
  };
  return { ...state.runtimeConfig };
}

window.addEventListener("DOMContentLoaded", async () => {
  $("createSessionBtn").addEventListener("click", async () => {
    try {
      await createSession();
    } catch (error) {
      alert(`创建会话失败: ${error.message}`);
    }
  });
  $("sendBtn").addEventListener("click", async () => {
    try {
      await sendMessage();
    } catch (error) {
      alert(`发送失败: ${error.message}`);
    }
  });
  $("refreshOverviewBtn").addEventListener("click", async () => {
    try {
      await loadOverview();
      await loadSessions();
    } catch (error) {
      alert(`刷新失败: ${error.message}`);
    }
  });

  try {
    await loadRuntimeConfig();
    await loadOverview();
    await loadSessions();
  } catch (error) {
    console.error(error);
  }
});
