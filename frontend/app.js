// frontend/app.js
"use strict";

/**
 * Agent Server のベースURL
 * - langgraph dev のデフォルトは http://127.0.0.1:2024
 * - ?api=http://127.0.0.1:2024 のようにクエリで上書き可能
 */
const API_BASE = (() => {
  const qs = new URLSearchParams(location.search);
  const fromQuery = qs.get("api");
  const fromStorage = localStorage.getItem("agent_api_base");
  const v = (fromQuery || fromStorage || "http://127.0.0.1:2024").trim();
  localStorage.setItem("agent_api_base", v);
  return v;
})();

// 公開した graph 名（langgraph.json の graphs のキー）
const ASSISTANT_ID = "business_plan";

// ログ最大件数（最新が上）
const MAX_LOG_ITEMS = 80;

// thread_id 保存キー
const THREAD_ID_KEY = "agent_thread_id";

// ノイズのSSE(生JSON/metadata等)を出すか（普段は false 推奨）
const SHOW_NOISY_SSE = false;

let threadId = null;
let uiState = "idle"; // idle | starting | waiting_approval | resuming | done | error
let logItems = [];   // [{timeISO, tag, agent, summary, detail, raw}]

// SSE受信中断のためのAbortController
let currentController = null;

// ---- labels ----
const NODE_LABEL = {
  research_agent: "調査（research_agent）",
  tools: "検索（tools）",
  summary_agent: "要約（summary_agent）",
  market_agent: "市場分析（market_agent）",
  technical_agent: "技術検討（technical_agent）",
  human_approval: "承認待ち（human_approval）",
  report_agent: "レポート作成（report_agent）",
};

const UI_LABEL = {
  idle: "待機中",
  starting: "実行中（ストリーミング…）",
  waiting_approval: "承認待ち（HITL入力待ち）",
  resuming: "再開中（ストリーミング…）",
  done: "完了",
  error: "エラー",
};

// ---- DOM ----
const themeEl = document.getElementById("theme");
const startBtn = document.getElementById("startBtn");
const clearBtn = document.getElementById("clearBtn");

const statusCard = document.getElementById("statusCard");
const statusEl = document.getElementById("status");
const statusDetails = document.getElementById("statusDetails");
const stateBadge = document.getElementById("stateBadge");
const spinnerEl = document.getElementById("spinner");
const errorBanner = document.getElementById("errorBanner");
const logListEl = document.getElementById("logList");
const clearLogBtn = document.getElementById("clearLogBtn");
const cancelBtn = document.getElementById("cancelBtn");

const workBadge = document.getElementById("workBadge");

const approvalCard = document.getElementById("approvalCard");
const questionEl = document.getElementById("question");
const previewEl = document.getElementById("preview");
const approveBtn = document.getElementById("approveBtn");
const retryBtn = document.getElementById("retryBtn");
const rejectBtn = document.getElementById("rejectBtn");

const reportCard = document.getElementById("reportCard");
const reportEl = document.getElementById("report");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");

// ---- UI helpers ----
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function setError(message) {
  if (!message) {
    hide(errorBanner);
    errorBanner.textContent = "";
    return;
  }
  show(statusCard);
  show(errorBanner);
  errorBanner.textContent = String(message);
}

function setBadge(state) {
  stateBadge.textContent = UI_LABEL[state] || state;
  stateBadge.dataset.state = state;
}

function setWork(nodeKeyOrLabel) {
  const s = nodeKeyOrLabel || "-";
  workBadge.textContent = NODE_LABEL[s] || s;
}

function setUiState(state) {
  uiState = state;
  setBadge(state);

  const busy = (state === "starting" || state === "resuming");
  const waiting = (state === "waiting_approval");

  startBtn.disabled = busy || waiting;
  clearBtn.disabled = busy || waiting;

  approveBtn.disabled = !waiting;
  retryBtn.disabled = !waiting;
  rejectBtn.disabled = !waiting;

  if (busy) show(spinnerEl);
  else hide(spinnerEl);

  if (busy) show(cancelBtn);
  else hide(cancelBtn);

  if (waiting) statusDetails.open = false;
  else if (state === "idle") statusDetails.open = false;
  else statusDetails.open = true;
}

function setStatus(obj) {
  show(statusCard);
  statusEl.textContent = JSON.stringify(obj, null, 2);
}

function toTextMaybe(x) {
  if (x == null) return "";
  if (typeof x === "string") return x;
  try { return JSON.stringify(x, null, 2); } catch { return String(x); }
}

function firstLines(text, n = 4) {
  const s = String(text || "");
  const lines = s.split(/\r?\n/);
  const head = lines.slice(0, n).join("\n");
  const more = lines.length > n;
  return { head, more };
}

function renderPreview(analysisPreview) {
  previewEl.innerHTML = "";
  const list = Array.isArray(analysisPreview) ? analysisPreview : [];

  if (list.length === 0) {
    const p = document.createElement("p");
    p.className = "hint";
    p.style.margin = "6px 0 0";
    p.textContent = "（プレビュー情報がありません。必要ならサーバ側で analysis_preview を付与してください。）";
    previewEl.appendChild(p);
    return;
  }

  const frag = document.createDocumentFragment();

  list.forEach((x, idx) => {
    const type = String(x?.type || `Message ${idx + 1}`);
    const full = toTextMaybe(x?.content ?? x);

    const { head, more } = firstLines(full, 4);
    const headOneLine = head.replace(/\n+/g, " ⏎ ");

    // details: 初期は「先頭数行」を見せ、クリックで全文を展開
    const details = document.createElement("details");
    details.className = "preview-item";
    details.open = false;

    const summary = document.createElement("summary");
    summary.style.cursor = "pointer";

    const title = document.createElement("strong");
    title.textContent = type;

    const snippet = document.createElement("span");
    snippet.style.marginLeft = "10px";
    snippet.style.color = "#d7deee";
    snippet.textContent = headOneLine + (more ? " …" : "");

    summary.appendChild(title);
    summary.appendChild(snippet);

    const pre = document.createElement("pre");
    pre.textContent = full || "";

    details.appendChild(summary);
    details.appendChild(pre);

    frag.appendChild(details);
  });

  previewEl.appendChild(frag);
}

function restoreTheme() {
  const saved = localStorage.getItem("agent_theme");
  if (saved) themeEl.value = saved;
}
function persistTheme(value) { localStorage.setItem("agent_theme", value); }

function restoreThreadId() {
  const saved = localStorage.getItem(THREAD_ID_KEY);
  return saved || null;
}
function persistThreadId(id) { localStorage.setItem(THREAD_ID_KEY, id); }
function clearThreadId() { localStorage.removeItem(THREAD_ID_KEY); }

function nowISO() { return new Date().toISOString(); }
function fmtLocalTime(iso) {
  try {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  } catch {
    return iso;
  }
}

function truncate(s, n = 120) {
  const t = String(s ?? "");
  if (t.length <= n) return t;
  return t.slice(0, n) + "…";
}

function addLogEntry({ tag, agent, summary, detail, raw }) {
  logItems.unshift({
    timeISO: nowISO(),
    tag: String(tag || "LOG"),
    agent: agent ? String(agent) : "",
    summary: String(summary || ""),
    detail: String(detail || ""),
    raw: raw ?? null,
  });
  if (logItems.length > MAX_LOG_ITEMS) logItems = logItems.slice(0, MAX_LOG_ITEMS);
  renderLog();
}

function renderLog() {
  logListEl.innerHTML = "";
  const frag = document.createDocumentFragment();

  if (logItems.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.style.margin = "0";
    empty.textContent = "まだログはありません。";
    frag.appendChild(empty);
    logListEl.appendChild(frag);
    return;
  }

  for (const item of logItems) {
    const details = document.createElement("details");
    details.className = "log-item";
    details.open = false; // 初期は折りたたみ

    const summary = document.createElement("summary");

    const left = document.createElement("div");
    left.className = "log-left";

    const tag = document.createElement("span");
    tag.className = "log-tag";
    tag.textContent = item.tag;

    left.appendChild(tag);

    if (item.agent) {
      const agent = document.createElement("span");
      agent.className = "log-agent";
      agent.textContent = item.agent;
      left.appendChild(agent);
    }

    const sum = document.createElement("span");
    sum.className = "log-summary";
    sum.textContent = item.summary;
    left.appendChild(sum);

    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = fmtLocalTime(item.timeISO);

    summary.appendChild(left);
    summary.appendChild(time);

    const body = document.createElement("div");
    body.className = "log-body";

    const pre = document.createElement("pre");
    pre.textContent = item.detail || item.summary || "";
    body.appendChild(pre);

    if (item.raw) {
      const rawDetails = document.createElement("details");
      rawDetails.className = "raw-toggle";
      rawDetails.open = false;

      const rawSummary = document.createElement("summary");
      rawSummary.textContent = "生JSON（SSE）を表示";
      rawDetails.appendChild(rawSummary);

      const rawPre = document.createElement("pre");
      rawPre.textContent = JSON.stringify(item.raw, null, 2);
      rawDetails.appendChild(rawPre);

      body.appendChild(rawDetails);
    }

    details.appendChild(summary);
    details.appendChild(body);
    frag.appendChild(details);
  }

  logListEl.appendChild(frag);
}

function clearLog() {
  logItems = [];
  renderLog();
  addLogEntry({ tag: "UI", summary: "ログをクリアしました。", detail: "ログをクリアしました。" });
}

// ---- API helpers ----
async function createThread() {
  const res = await fetch(`${API_BASE}/threads`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify({ metadata: {} }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`thread create failed: ${res.status} ${text}`);
  }
  const data = await res.json();
  if (!data?.thread_id) throw new Error("thread create response has no thread_id");
  return data.thread_id;
}

/**
 * start は新規 thread を作る（状態混入を防ぐ）
 * resume は既存 thread を使う（HITL 再開のため）
 */
async function ensureThreadIdFromServer({ forceNew } = { forceNew: false }) {
  if (!forceNew && threadId) return threadId;

  const id = await createThread();
  threadId = id;
  persistThreadId(threadId);
  addLogEntry({ tag: "THREAD", summary: `作成: thread_id=${threadId}`, detail: `作成: thread_id=${threadId}` });
  return threadId;
}

async function getThread(tid) {
  const res = await fetch(`${API_BASE}/threads/${tid}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`get thread failed: ${res.status} ${text}`);
  }
  return await res.json();
}

/**
 * interrupt payload は場所が揺れるので、複数候補から拾う
 * - obj.__interrupt__
 * - obj.values.__interrupt__
 * - obj.interrupts
 * - obj.values.interrupts
 */
function extractInterruptPayload(obj) {
  if (!obj) return null;

  const candidates = [
    obj.__interrupt__,
    obj?.values?.__interrupt__,
    obj?.interrupts,
    obj?.values?.interrupts,
  ].filter(Boolean);

  for (const c of candidates) {
    const first = Array.isArray(c) ? c[0] : c;
    if (!first) continue;

    const payload = first?.value ?? first;
    if (!payload) continue;

    // それっぽいキーがあれば採用
    if (payload.kind === "approval_request" || payload.question || payload.analysis_preview) {
      return payload;
    }
  }

  // ステータスだけ "interrupted" になっていてペイロードが無い場合の保険
  if (String(obj.status || "") === "interrupted") {
    return { kind: "approval_request", question: "承認しますか？", options: ["y", "n", "retry"], analysis_preview: [] };
  }

  return null;
}

function extractFinalReportFromThread(threadObj) {
  const values = threadObj?.values;
  if (!values) return "";
  if (typeof values.final_report === "string" && values.final_report.trim()) {
    return values.final_report;
  }
  return "";
}

function extractCurrentStepFromThread(threadObj) {
  const values = threadObj?.values;
  if (!values) return "";
  return String(values.current_step || "");
}

// SSEパース（event/data 形式）
function parseSseFrame(frameText) {
  const normalized = frameText.replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  let eventName = "";
  const dataLines = [];
  let id = "";

  for (const line of lines) {
    if (!line) continue;
    if (line.startsWith(":")) continue; // comment line
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    else if (line.startsWith("id:")) id = line.slice(3).trim();
  }

  const dataRaw = dataLines.join("\n");
  let data = dataRaw;
  if (dataRaw) {
    try { data = JSON.parse(dataRaw); } catch { /* keep raw */ }
  }
  return { id, event: eventName, data };
}

// runs/stream 呼び出し（POSTでSSEを受け取る）
async function runStream({ tid, body, onEvent }) {
  // 既存ストリームがあれば中断
  if (currentController) {
    try { currentController.abort(); } catch {}
  }
  const controller = new AbortController();
  currentController = controller;

  let eventCount = 0;

  const res = await fetch(`${API_BASE}/threads/${tid}/runs/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
      "Cache-Control": "no-cache",
    },
    body: JSON.stringify(body),
    signal: controller.signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`runs/stream failed: ${res.status} ${text}`);
  }

  const contentType = res.headers.get("content-type") || "";
  if (SHOW_NOISY_SSE) {
    addLogEntry({
      tag: "HTTP",
      summary: `runs/stream content-type=${contentType}`,
      detail: `runs/stream content-type=${contentType}`,
    });
  } else {
    // ログに出さず、詳細(status)だけに残したいならここで setStatus に含めてもOK
    // setStatus({ ... , http: { contentType } });
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");

      let sepIndex;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sepIndex).trim();
        buffer = buffer.slice(sepIndex + 2);

        if (!frame) continue;

        const ev = parseSseFrame(frame);
        eventCount += 1;
        onEvent?.(ev);
      }
    }
  } finally {
    if (currentController === controller) currentController = null;
  }

  if (eventCount === 0) {
    addLogEntry({
      tag: "WARN",
      summary: "SSEイベントが1件も届きませんでした。",
      detail: "SSEイベントが1件も届きませんでした。サーバがSSEを返していない/バッファされている/プロキシの影響などが考えられます。",
    });
  }
}

// ---- SSE interpretation helpers ----
function pickNodeKeyFromUpdates(obj) {
  if (!obj || typeof obj !== "object") return "";
  const u = (obj.updates && typeof obj.updates === "object") ? obj.updates : obj;
  if (!u || typeof u !== "object") return "";

  for (const k of Object.keys(u)) {
    if (NODE_LABEL[k]) return k;
  }
  return "";
}

/**
 * updates の中から current_step を拾う
 * 例: {"research_start":{"current_step":"research_agent"}}
 */
function pickCurrentStepFromUpdates(obj) {
  if (!obj || typeof obj !== "object") return "";
  const u = (obj.updates && typeof obj.updates === "object") ? obj.updates : obj;
  if (!u || typeof u !== "object") return "";

  for (const k of Object.keys(u)) {
    const v = u[k];
    if (v && typeof v === "object" && typeof v.current_step === "string" && v.current_step.trim()) {
      return v.current_step.trim();
    }
  }
  return "";
}

function extractReadableOutputFromNodeState(nodeState) {
  if (!nodeState || typeof nodeState !== "object") return "";

  if (typeof nodeState.final_report === "string" && nodeState.final_report.trim()) {
    return nodeState.final_report;
  }

  const keys = ["analysis_messages", "research_messages"];
  for (const key of keys) {
    const arr = nodeState[key];
    if (Array.isArray(arr) && arr.length > 0) {
      const last = arr[arr.length - 1];
      const content = last?.content ?? last?.text ?? last?.message ?? last;
      if (typeof content === "string") return content;
      try { return JSON.stringify(content, null, 2); } catch { return String(content); }
    }
  }

  try { return JSON.stringify(nodeState, null, 2); } catch { return String(nodeState); }
}

function showApprovalIfNeeded(payload) {
  if (!payload) return;
  if (uiState === "waiting_approval") return;

  setUiState("waiting_approval");
  show(approvalCard);

  questionEl.textContent = payload.question || "承認しますか？";
  renderPreview(payload.analysis_preview || []);

  addLogEntry({
    tag: "HITL",
    agent: NODE_LABEL.human_approval,
    summary: "承認待ちになりました（入力待ち）",
    detail: JSON.stringify(payload, null, 2),
    raw: payload,
  });
}

function handleSseEvent(ev) {
  const raw = ev?.data;
  const eventName = ev?.event || "";

  setStatus({ last_event: ev, thread_id: threadId });

  // interrupt を SSE 側で拾える場合は先に反映（プレビューもここで出る）
  if (raw && typeof raw === "object") {
    const interruptPayload = extractInterruptPayload(raw);
    if (interruptPayload) {
      showApprovalIfNeeded(interruptPayload);
      // interrupt が来たら、通常これ以上は進まない想定なのでここで戻る
      return;
    }
  }

  // metadata / keepalive などは基本ノイズなので抑制
  if (!SHOW_NOISY_SSE && (eventName === "metadata" || eventName === "ping" || eventName === "keepalive")) {
    return;
  }

  // values に current_step があれば最優先で反映
  if (raw && typeof raw === "object") {
    const valuesObj = raw.values && typeof raw.values === "object" ? raw.values : null;
    const step = valuesObj?.current_step ? String(valuesObj.current_step) : "";
    if (step) setWork(step);
  }

  // updates: 「開始マーカー」の current_step を拾って work を更新し、SSE生ログは出さない
  if (eventName === "updates" && raw && typeof raw === "object") {
    const stepFromUpdates = pickCurrentStepFromUpdates(raw);
    if (stepFromUpdates) {
      setWork(stepFromUpdates);
      return;
    }
  }

  // updates: “結果”ログ（読みやすく残す）
  if (eventName === "updates") {
    const nodeKey = pickNodeKeyFromUpdates(raw);
    if (nodeKey) {
      const u = raw.updates && typeof raw.updates === "object" ? raw.updates : raw;
      const nodeState = u[nodeKey];
      const text = extractReadableOutputFromNodeState(nodeState || {});
      addLogEntry({
        tag: "AGENT",
        agent: NODE_LABEL[nodeKey] || nodeKey,
        summary: truncate(text, 140) || "(更新)",
        detail: text || "(更新)",
        raw,
      });
      return;
    }

    // nodeKey が特定できない updates は基本ノイズ：デバッグ時だけ表示
    if (!SHOW_NOISY_SSE) return;
  }

  // values: 状態変化の軽ログ（current_step中心）
  if (eventName === "values") {
    const current = raw?.values?.current_step ? String(raw.values.current_step) : "";
    if (current) {
      addLogEntry({
        tag: "STATE",
        agent: NODE_LABEL[current] || current,
        summary: "状態更新（values）",
        detail: JSON.stringify(raw?.values ?? raw, null, 2),
        raw,
      });
      return;
    }

    // current_step の無い values はノイズなので抑制
    if (!SHOW_NOISY_SSE) return;

    addLogEntry({
      tag: "STATE",
      summary: "状態更新（values）",
      detail: JSON.stringify(raw, null, 2),
      raw,
    });
    return;
  }

  // ここから下は「本当に必要なときだけ」SSE生ログを出す
  const isEmpty =
    raw == null ||
    raw === "" ||
    (typeof raw === "string" && raw.trim() === "");

  if (!SHOW_NOISY_SSE) {
    if (!eventName || isEmpty) return;
    return;
  }

  const asText = (typeof raw === "string")
    ? raw
    : (() => { try { return JSON.stringify(raw); } catch { return String(raw); } })();

  addLogEntry({
    tag: eventName ? `SSE:${eventName}` : "SSE",
    summary: truncate(asText, 140) || "(SSE)",
    detail: (typeof raw === "string") ? raw : JSON.stringify(raw, null, 2),
    raw: (raw && typeof raw === "object") ? raw : null,
  });
}

// ---- app logic ----
function themeValue() { return themeEl.value.trim() || "宇宙ゴミの回収事業"; }

function resetViewForRun() {
  setError("");
  hide(approvalCard);
  hide(reportCard);
  show(statusCard);
}

function cancelInFlight() {
  if (!currentController) return;
  addLogEntry({ tag: "UI", summary: "ユーザーがストリーム受信を中断しました。", detail: "ユーザーがストリーム受信を中断しました。" });
  try { currentController.abort(); } catch {}
}

async function start() {
  resetViewForRun();

  const theme = themeValue();
  persistTheme(theme);

  const tid = await ensureThreadIdFromServer({ forceNew: true });
  setWork("-");

  setUiState("starting");
  setStatus({ step: "start", theme, thread_id: tid, api_base: API_BASE, assistant_id: ASSISTANT_ID });
  addLogEntry({ tag: "START", summary: `開始: 「${theme}」`, detail: `開始: 「${theme}」 / thread_id=${tid}` });

  await runStream({
    tid,
    body: {
      assistant_id: ASSISTANT_ID,
      input: {
        research_messages: [{ type: "human", content: `テーマ: ${theme}` }],
        analysis_messages: [],
        loop_count: 0,
        current_step: "start",
        approval_decision: "",
        final_report: "",
      },
      stream_mode: ["updates", "values"],
      on_disconnect: "cancel",
    },
    onEvent: (ev) => handleSseEvent(ev),
  });

  const threadObj = await getThread(tid);

  const payload = extractInterruptPayload(threadObj);
  const currentStep = extractCurrentStepFromThread(threadObj);
  if (currentStep) setWork(currentStep);

  if (payload) {
    showApprovalIfNeeded(payload);
    setStatus({ thread: threadObj, thread_id: tid });
    return;
  }

  const report = extractFinalReportFromThread(threadObj);
  if (!report) {
    setUiState("done");
    show(reportCard);
    reportEl.textContent = "（最終レポートはまだ生成されていません。承認待ちになっていない場合はサーバ側ログを確認してください。）";
    setStatus({ thread: threadObj, thread_id: tid });
    addLogEntry({ tag: "DONE", summary: "完了（最終レポートなし）", detail: "完了（最終レポートなし）" });
    return;
  }

  setUiState("done");
  show(reportCard);
  reportEl.textContent = report;
  setStatus({ thread: threadObj, thread_id: tid });
  addLogEntry({ tag: "DONE", summary: "完了（最終レポートが生成されました）", detail: "完了（最終レポートが生成されました）" });
}

async function resume(decision) {
  resetViewForRun();

  const tid = await ensureThreadIdFromServer({ forceNew: false });

  setUiState("resuming");
  setStatus({ step: "resume", decision, thread_id: tid, api_base: API_BASE, assistant_id: ASSISTANT_ID });
  addLogEntry({ tag: "RESUME", summary: `入力: ${decision}`, detail: `入力: ${decision} / thread_id=${tid}` });

  await runStream({
    tid,
    body: {
      assistant_id: ASSISTANT_ID,
      command: { resume: decision },
      stream_mode: ["updates", "values"],
      on_disconnect: "cancel",
    },
    onEvent: (ev) => handleSseEvent(ev),
  });

  const threadObj = await getThread(tid);

  const payload = extractInterruptPayload(threadObj);
  const currentStep = extractCurrentStepFromThread(threadObj);
  if (currentStep) setWork(currentStep);

  if (payload) {
    showApprovalIfNeeded(payload);
    setStatus({ thread: threadObj, thread_id: tid });
    return;
  }

  const report = extractFinalReportFromThread(threadObj);
  if (!report) {
    setUiState("done");
    show(reportCard);
    reportEl.textContent = "（最終レポートはまだ生成されていません。）";
    setStatus({ thread: threadObj, thread_id: tid });
    addLogEntry({ tag: "DONE", summary: "完了（最終レポートなし）", detail: "完了（最終レポートなし）" });
    return;
  }

  setUiState("done");
  show(reportCard);
  reportEl.textContent = report;
  setStatus({ thread: threadObj, thread_id: tid });
  addLogEntry({ tag: "DONE", summary: "完了（最終レポートが生成されました）", detail: "完了（最終レポートが生成されました）" });
}

async function runWithUi(fn) {
  try {
    setError("");
    await fn();
  } catch (e) {
    setUiState("error");
    const msg = String(e?.message || e);
    setError(msg);
    setStatus({ error: msg, uiState, thread_id: threadId ?? null, api_base: API_BASE, assistant_id: ASSISTANT_ID });
    addLogEntry({ tag: "ERROR", summary: msg, detail: msg });
  }
}

function handleDecision(decision) {
  approveBtn.disabled = true;
  retryBtn.disabled = true;
  rejectBtn.disabled = true;
  startBtn.disabled = true;
  clearBtn.disabled = true;

  setUiState("resuming");
  runWithUi(() => resume(decision));
}

// ---- copy / download ----
function copyReport() {
  const text = reportEl.textContent || "";
  if (!text) {
    addLogEntry({ tag: "UI", summary: "コピー対象のレポートが空です。", detail: "コピー対象のレポートが空です。" });
    return;
  }

  navigator.clipboard?.writeText(text)
    .then(() => {
      setError("");
      addLogEntry({ tag: "UI", summary: "レポートをコピーしました。", detail: "レポートをクリップボードにコピーしました。" });
    })
    .catch(() => {
      setError("コピーに失敗しました（ブラウザ権限をご確認ください）。");
      addLogEntry({ tag: "ERROR", summary: "コピーに失敗しました。", detail: "コピーに失敗しました。" });
    });
}

function sanitizeFilename(s) {
  return String(s)
    .replace(/[\\/:*?"<>|]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 60);
}

function downloadReport() {
  const text = reportEl.textContent || "";
  if (!text) {
    addLogEntry({ tag: "UI", summary: "保存対象のレポートが空です。", detail: "保存対象のレポートが空です。" });
    return;
  }

  const theme = sanitizeFilename(themeValue());
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");

  const filename = `business_plan_${yyyy}${mm}${dd}_${hh}${mi}_${theme || "theme"}.txt`;
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
  addLogEntry({ tag: "UI", summary: `保存: ${filename}`, detail: `レポートを保存しました: ${filename}` });
}

// ---- clear / init ----
function clearAll() {
  themeEl.value = "";
  persistTheme("");
  threadId = null;
  clearThreadId();

  hide(approvalCard);
  hide(reportCard);
  hide(statusCard);

  setError("");
  setUiState("idle");
  setWork("-");
  statusEl.textContent = "";
  logItems = [];
  renderLog();
}

async function initAsync() {
  restoreTheme();

  setUiState("idle");
  setWork("-");
  renderLog();
  addLogEntry({ tag: "UI", summary: `起動しました。API: ${API_BASE}`, detail: `起動しました。API: ${API_BASE}` });

  const savedTid = restoreThreadId();
  if (!savedTid) return;

  try {
    const t = await getThread(savedTid);
    const payload = extractInterruptPayload(t);
    const currentStep = extractCurrentStepFromThread(t);
    if (currentStep) setWork(currentStep);

    if (payload) {
      threadId = savedTid;
      addLogEntry({ tag: "UI", summary: `復元: 承認待ち thread_id=${threadId}`, detail: `復元: 承認待ち thread_id=${threadId}` });
      setUiState("waiting_approval");
      show(approvalCard);
      questionEl.textContent = payload.question || "承認しますか？";
      renderPreview(payload.analysis_preview || []);
      setStatus({ thread: t, thread_id: threadId });
    } else {
      clearThreadId();
      addLogEntry({ tag: "UI", summary: "前回threadは承認待ちではないため破棄しました", detail: `前回threadは承認待ちではないため破棄しました: thread_id=${savedTid}` });
    }
  } catch (e) {
    clearThreadId();
    addLogEntry({ tag: "WARN", summary: "保存済みthreadの確認に失敗→破棄", detail: `保存済みthreadの確認に失敗したため破棄しました: ${String(e?.message || e)}` });
  }
}

// ---- Events ----
startBtn.addEventListener("click", () => runWithUi(start));
clearBtn.addEventListener("click", clearAll);
clearLogBtn.addEventListener("click", clearLog);
cancelBtn.addEventListener("click", cancelInFlight);

approveBtn.addEventListener("click", () => handleDecision("y"));
retryBtn.addEventListener("click", () => handleDecision("retry"));
rejectBtn.addEventListener("click", () => handleDecision("n"));

copyBtn.addEventListener("click", copyReport);
downloadBtn.addEventListener("click", downloadReport);

// Enterで実行（IME変換中は無視）
themeEl.addEventListener("keydown", (e) => {
  if (e.isComposing) return;
  if (e.key === "Enter") {
    e.preventDefault();
    if (!startBtn.disabled) runWithUi(start);
  }
});

initAsync();
