"use strict";

const $ = (id) => document.getElementById(id);
const state = { sessions: [], current: null, views: new Map(), busy: false, runId: null,
  deleting: false, creating: false, deleteTarget: null };

function icon(name, className = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", `icon ${className}`);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(svg.namespaceURI, "use");
  use.setAttribute("href", `#${name}`);
  svg.append(use);
  return svg;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function notice(text = "") {
  $("notice").textContent = text;
  $("notice").hidden = !text;
}

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function updateControls() {
  const blocked = state.busy || state.deleting || state.creating;
  $("new-session").disabled = blocked;
  $("message-input").disabled = blocked || !state.current;
  $("send").disabled = blocked || !state.current || !$("message-input").value.trim();
  $("send").hidden = state.busy;
  $("stop").hidden = !state.busy;
  $("stop").disabled = !state.runId;
  for (const button of document.querySelectorAll(".session-item, .session-delete, .suggestion")) button.disabled = blocked;
}

function renderSessions() {
  $("sessions").replaceChildren();
  $("session-count").textContent = state.sessions.length;
  for (const id of state.sessions) {
    const row = element("div", "session-row");
    const button = element("button", `session-item${id === state.current ? " active" : ""}`);
    button.title = id;
    button.setAttribute("aria-current", id === state.current ? "true" : "false");
    button.append(icon("chat"), element("span", "", `会话 · ${id.replace(/^session_/, "").slice(0, 8)}`));
    button.addEventListener("click", () => selectSession(id));
    const remove = element("button", "session-delete");
    const label = `删除会话 ${id.replace(/^session_/, "").slice(0, 8)}`;
    remove.title = "删除对话和该会话的所有记忆";
    remove.setAttribute("aria-label", label);
    remove.append(icon("trash"));
    remove.addEventListener("click", () => openDeleteDialog(id));
    row.append(button, remove);
    $("sessions").append(row);
  }
  updateControls();
}

function selectSession(id) {
  if (state.busy || state.deleting) return;
  state.current = id;
  if (id) {
    if (!state.views.has(id)) state.views.set(id, element("div", "session-view"));
    $("messages").replaceChildren(state.views.get(id));
  } else $("messages").replaceChildren();
  $("welcome").hidden = id ? state.views.get(id).children.length > 0 : false;
  $("message-input").value = "";
  renderSessions();
  scrollToBottom();
  $("message-input").focus();
}

async function newSession() {
  if (state.busy || state.deleting || state.creating) return;
  state.creating = true;
  updateControls();
  try {
    const { session_id } = await api("/api/sessions", {});
    state.sessions.unshift(session_id);
    selectSession(session_id);
    notice();
  } catch (error) { notice(`无法创建会话：${error.message}`); }
  finally {
    state.creating = false;
    updateControls();
    if (state.current) $("message-input").focus();
  }
}

function openDeleteDialog(id) {
  if (state.busy || state.deleting || state.creating) return;
  state.deleteTarget = id;
  $("delete-target").textContent = id;
  $("delete-error").hidden = true;
  $("confirm-delete").textContent = "删除对话和记忆";
  $("delete-dialog").showModal();
}

async function deleteSession() {
  if (!state.deleteTarget || state.busy || state.deleting || state.creating) return;
  const id = state.deleteTarget;
  state.deleting = true;
  $("confirm-delete").disabled = true;
  $("cancel-delete").disabled = true;
  $("confirm-delete").textContent = "正在清空记忆…";
  $("delete-error").hidden = true;
  updateControls();
  try {
    await api("/api/sessions/delete", { session_id: id });
    state.sessions = state.sessions.filter((session) => session !== id);
    state.views.delete(id);
    state.deleting = false;
    if (state.current === id) selectSession(state.sessions[0] || null);
    else renderSessions();
    $("delete-dialog").close();
    if (state.current) $("message-input").focus();
    state.deleteTarget = null;
    notice("已删除对话，并清空该 session 的记忆。");
  } catch (error) {
    $("delete-error").textContent = `删除未完成：${error.message}。会话仍保留，可重试；已清除的记忆不会恢复。`;
    $("delete-error").hidden = false;
  } finally {
    state.deleting = false;
    $("confirm-delete").disabled = false;
    $("cancel-delete").disabled = false;
    $("confirm-delete").textContent = "删除对话和记忆";
    updateControls();
  }
}

function scrollToBottom(force = true) {
  const area = $("conversation");
  if (force || area.scrollHeight - area.scrollTop - area.clientHeight < 180) area.scrollTop = area.scrollHeight;
}

function assistantMessage() {
  const row = element("article", "message assistant");
  const heading = element("div", "assistant-label");
  const status = element("span", "assistant-status", "正在准备");
  status.setAttribute("role", "status");
  heading.append(icon("spark"), element("span", "", "Agent"), status);
  const trace = element("details", "trace");
  trace.open = true;
  trace.hidden = !$("show-thoughts").checked;
  const summary = element("summary", "");
  const summaryText = element("span", "", "思考与执行过程");
  summary.append(icon("activity"), summaryText, icon("chevron", "chevron"));
  const traceBody = element("div", "trace-body");
  trace.append(summary, traceBody);
  const answer = element("div", "answer");
  row.append(heading, trace, answer);
  state.views.get(state.current).append(row);
  return { row, status, trace, traceBody, summaryText, answer, steps: new Map(), finished: false };
}

function getStep(view, number) {
  if (!view.steps.has(number)) {
    const step = element("div", "trace-step");
    step.append(element("div", "step-label", `STEP ${String(number).padStart(2, "0")}`));
    view.traceBody.append(step);
    view.steps.set(number, step);
  }
  return view.steps.get(number);
}

function toolBlock(step, label, content) {
  const block = element("div", "tool-block");
  block.append(element("div", "tool-label", label), element("pre", "tool-content", content));
  step.append(block);
}

function handleEvent(event, view) {
  switch (event.type) {
    case "started": state.runId = event.run_id; updateControls(); break;
    case "status": view.status.textContent = "正在初始化 agent"; break;
    case "step":
      view.status.textContent = `思考中 · ${event.step} / ${event.max_steps}`;
      getStep(view, event.step);
      break;
    case "thought": getStep(view, event.step).append(element("div", "thought-content", event.content)); break;
    case "action":
      view.status.textContent = `正在调用 ${event.tool}`;
      toolBlock(getStep(view, event.step), `工具调用 · ${event.tool}`, event.input);
      break;
    case "observation": toolBlock(getStep(view, event.step), "工具返回", event.content); break;
    case "final":
      view.answer.textContent = event.content;
      view.status.textContent = event.status === "exhausted" ? "已达到步骤上限" : "已完成";
      view.finished = true;
      addCopyButton(view);
      break;
    case "cancelled":
      view.status.textContent = "已停止";
      view.answer.textContent = "本轮任务已停止。已经执行的工具操作会保留。";
      view.finished = true;
      break;
    case "error":
      view.status.textContent = "运行失败";
      view.answer.classList.add("error");
      view.answer.textContent = `运行失败：${event.content}`;
      view.finished = true;
      break;
  }
  if (event.type !== "heartbeat") scrollToBottom(false);
}

function addCopyButton(view) {
  const button = element("button", "copy-button");
  button.type = "button";
  const label = element("span", "", "复制回答");
  button.append(icon("copy"), label);
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(view.answer.textContent);
      label.textContent = "已复制";
      setTimeout(() => { label.textContent = "复制回答"; }, 1800);
    } catch { notice("复制失败，请选中回答文本手动复制。"); }
  });
  view.row.append(button);
}

async function sendMessage(event) {
  event.preventDefault();
  const message = $("message-input").value.trim();
  if (!message || state.busy || state.deleting || state.creating || !state.current) return;
  state.busy = true;
  state.runId = null;
  $("stop").textContent = "■ 停止";
  notice();
  $("welcome").hidden = true;
  const user = element("article", "message user");
  user.append(element("div", "user-text", message));
  state.views.get(state.current).append(user);
  const view = assistantMessage();
  const started = performance.now();
  $("message-input").value = "";
  $("message-input").style.height = "auto";
  updateControls();
  scrollToBottom();
  let done = false;
  try {
    const response = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.current, message }),
    });
    if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const chunk = await reader.read();
      buffer += decoder.decode(chunk.value, { stream: !chunk.done });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const data = JSON.parse(line);
        handleEvent(data, view);
        if (data.type === "done") done = true;
      }
      if (chunk.done) break;
    }
    if (!done) throw new Error("连接中断，未收到任务完成确认。请等待后重试。");
    view.summaryText.textContent = `思考与执行过程 · ${view.steps.size} 步 · ${((performance.now() - started) / 1000).toFixed(1)} 秒`;
  } catch (error) {
    if (view.finished) notice(`连接异常：${error.message}`);
    else handleEvent({ type: "error", content: error.message }, view);
  } finally {
    state.busy = false;
    state.runId = null;
    updateControls();
    $("message-input").focus();
    scrollToBottom(false);
  }
}

$("new-session").addEventListener("click", newSession);
$("confirm-delete").addEventListener("click", deleteSession);
$("cancel-delete").addEventListener("click", () => {
  if (!state.deleting) $("delete-dialog").close();
});
$("delete-dialog").addEventListener("cancel", (event) => {
  if (state.deleting) event.preventDefault();
});
$("chat-form").addEventListener("submit", sendMessage);
$("message-input").addEventListener("input", () => {
  const input = $("message-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  updateControls();
});
$("message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});
$("show-thoughts").addEventListener("change", () => {
  for (const view of state.views.values()) {
    for (const trace of view.querySelectorAll(".trace")) trace.hidden = !$("show-thoughts").checked;
  }
});
$("stop").addEventListener("click", async () => {
  if (!state.runId) return;
  $("stop").disabled = true;
  try {
    const result = await api("/api/cancel", { run_id: state.runId });
    if (result.cancelled) {
      $("stop").textContent = "正在停止…";
      notice("停止请求已发送；当前模型请求或工具调用结束后停止后续步骤。");
    }
  } catch (error) { notice(`无法停止：${error.message}`); $("stop").disabled = false; }
});
for (const button of document.querySelectorAll(".suggestion")) {
  button.addEventListener("click", () => {
    $("message-input").value = button.dataset.prompt;
    updateControls();
    $("message-input").focus();
  });
}

async function initialize() {
  try {
    const { sessions } = await api("/api/sessions");
    state.sessions = sessions;
    $("connection-status").replaceChildren(element("span", "status-dot"), document.createTextNode("本地服务已连接"));
    if (sessions.length) selectSession(sessions[0]);
    else await newSession();
  } catch (error) {
    notice(`无法连接本地服务：${error.message}`);
    $("connection-status").textContent = "服务连接失败，请刷新重试";
  }
}
initialize();
