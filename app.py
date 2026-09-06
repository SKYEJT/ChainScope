"""ChainScope - Autonomous On-Chain Investigation Agent (Streamlit Frontend).

This UI drives the real LangGraph agent: the LLM plans, calls tools, observes
results, reflects/self-corrects across many steps, and concludes with an on-chain
attestation. The investigation is NOT a fixed pipeline - every action is chosen
by the model at runtime, and the long-horizon run is streamed live.

Visual system ("monochrome premium"): a near-black canvas, hairline rules, one
light primary button, large type and generous whitespace. Semantic colour is
spent only on risk (red / amber / green) and information (blue). Most of the
look is carried by .streamlit/config.toml (Streamlit >= 1.58 theme options);
the stylesheet below shapes the custom typographic blocks and finishes widgets.
"""
import hmac
import html as _html
import math
import os
import time

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

from chainscope import config as cs_config
from chainscope.agent.graph import build_investigation_graph
from chainscope.agent.case_file import new_case
from chainscope.agent.prompts import SYSTEM_PROMPT, DEFAULT_GOAL
from chainscope.tools.chainscout import default_scout
from chainscope.utils import run_trace

st.set_page_config(page_title="ChainScope", layout="wide", page_icon=":material/radar:",
                   initial_sidebar_state="expanded")


# ─────────────────────────────────────────────────────────────────────────────
# Stylesheet. Injected before the access gate so every screen shares the look.
# Colours mirror .streamlit/config.toml; keep the two in step.
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
:root {
    --bg: #0B0C0F; --surface: #13151A; --surface-2: #191C22;
    --line: #23262E; --line-2: #30343E;
    --text: #E8EAEE; --text-2: #9AA1AE; --text-3: #6A7280;
    --ok: #55CFA4; --warn: #E3AE5C; --bad: #F0716B; --info: #7FB6FF;
    --mono: 'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

/* ── Chrome ─────────────────────────────────────────────────────────────── */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 1160px; padding: 2.2rem 2.6rem 4rem 2.6rem;
}
[data-testid="stSidebarUserContent"] { padding-top: 1.4rem; }
::selection { background: rgba(232,234,238,.22); }
* { scrollbar-width: thin; scrollbar-color: var(--line-2) transparent; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

/* ── Type helpers ───────────────────────────────────────────────────────── */
.cs-eyebrow {
    font-size: 11px; font-weight: 500; letter-spacing: .14em; text-transform: uppercase;
    color: var(--text-3); font-family: var(--mono);
}
.cs-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
          background: var(--ok); flex: none; }
.cs-dot.off { background: var(--text-3); }

/* ── Top bar ────────────────────────────────────────────────────────────── */
.cs-top {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 0 0 18px 0; border-bottom: 1px solid var(--line); margin-bottom: 40px;
}
.cs-wordmark { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.cs-wordmark b { font-size: 17px; font-weight: 600; letter-spacing: -.02em; color: var(--text); }
.cs-wordmark span { font-size: 13px; color: var(--text-3); }
.cs-status { display: flex; gap: 22px; flex-wrap: wrap; justify-content: flex-end; }
.cs-status span {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 12px; color: var(--text-2); white-space: nowrap;
}

/* ── Hero (empty state) ─────────────────────────────────────────────────── */
.cs-hero { margin: 0 0 34px 0; }
.cs-loopline { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.cs-loopline i { font-style: normal; color: var(--line-2); }
.cs-h1 {
    font-size: 56px; line-height: 1.04; letter-spacing: -.035em; font-weight: 600;
    color: var(--text); margin: 18px 0 20px 0; max-width: 22ch;
}
.cs-hero p { font-size: 16.5px; line-height: 1.65; color: var(--text-2); max-width: 66ch; margin: 0; }
.cs-field-label { margin: 0 0 8px 0; }

/* ── Capabilities (empty state) ─────────────────────────────────────────── */
.cs-caps {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 44px;
    margin: 60px 0 0 0; padding-top: 28px; border-top: 1px solid var(--line);
}
.cs-cap-title { font-size: 15.5px; font-weight: 600; letter-spacing: -.01em; margin: 14px 0 8px 0; color: var(--text); }
.cs-cap p { font-size: 13.5px; line-height: 1.62; color: var(--text-2); margin: 0; }
.cs-footnote { margin: 34px 0 0 0; font-size: 12.5px; color: var(--text-3); }
.cs-try { font-size: 12.5px; color: var(--text-3); white-space: nowrap; }

/* ── Sections ───────────────────────────────────────────────────────────── */
.cs-section {
    display: flex; align-items: baseline; gap: 16px;
    margin: 46px 0 14px 0; padding-top: 26px; border-top: 1px solid var(--line);
}
.cs-section .cs-idx { font-family: var(--mono); font-size: 12px; color: var(--text-3); min-width: 26px; }
.cs-title { font-size: 22px; font-weight: 600; letter-spacing: -.02em; color: var(--text); }
.cs-sub { font-size: 15px; font-weight: 600; letter-spacing: -.01em; color: var(--text); margin: 8px 0 4px 0; }
.cs-desc { color: var(--text-2); font-size: 13.5px; line-height: 1.6; max-width: 84ch; margin: 0 0 16px 0; }

/* ── Verdict / stats strip ──────────────────────────────────────────────── */
.cs-stats {
    display: grid; grid-template-columns: 1.2fr 1.3fr 1fr 1fr;
    border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin-top: 6px;
}
.cs-stats > div { padding: 22px 24px; border-left: 1px solid var(--line); }
.cs-stats > div:first-child { border-left: none; padding-left: 0; }
.cs-stat {
    font-family: var(--mono); font-size: 42px; font-weight: 500; letter-spacing: -.03em;
    line-height: 1; margin-top: 14px; color: var(--text); font-variant-numeric: tabular-nums;
}
.cs-stat.small {
    font-family: inherit; font-size: 22px; font-weight: 600; letter-spacing: -.02em;
    display: flex; align-items: center; gap: 10px; height: 42px; margin-top: 14px;
}
.risk-high .cs-risk, .risk-high .cs-risk .cs-dot { color: var(--bad); }
.risk-med  .cs-risk, .risk-med  .cs-risk .cs-dot { color: var(--warn); }
.risk-low  .cs-risk, .risk-low  .cs-risk .cs-dot { color: var(--ok); }
.cs-risk .cs-dot { background: currentColor; }
.cs-bar { position: relative; height: 2px; background: var(--line-2); margin: 28px 0 8px 0; }
.cs-bar .fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--text); }
.risk-high .cs-bar .fill { background: var(--bad); }
.risk-med  .cs-bar .fill { background: var(--warn); }
.risk-low  .cs-bar .fill { background: var(--ok); }
.cs-bar .tick { position: absolute; left: 60%; top: -5px; height: 12px; width: 1px; background: var(--text-3); }
.cs-bar-legend {
    display: flex; justify-content: space-between;
    font-family: var(--mono); font-size: 11px; color: var(--text-3);
}

/* ── Fund paths ─────────────────────────────────────────────────────────── */
.cs-path { padding: 12px 0; border-bottom: 1px solid var(--line); }
.cs-path .dir { font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--text-2); }
.cs-path .meta { font-family: var(--mono); font-size: 12px; color: var(--text-3); margin-left: 12px; }
.cs-path .route { font-family: var(--mono); font-size: 12.5px; color: var(--text); margin-top: 6px; word-break: break-all; }

/* ── Live stream ────────────────────────────────────────────────────────── */
.cs-who { font-family: var(--mono); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-3); }
.cs-tool {
    font-family: var(--mono); font-size: 12.5px; color: var(--text-2);
    padding: 3px 0 3px 14px; border-left: 1px solid var(--line-2); margin: 2px 0 2px 2px; word-break: break-all;
}
.cs-tool b { color: var(--text); font-weight: 500; }
.cs-phase {
    display: flex; align-items: center; gap: 12px; margin: 18px 0 8px 0;
    font-family: var(--mono); font-size: 12px; color: var(--text-2);
}
.cs-phase .idx { color: var(--text-3); }
.cs-phase::after { content: ""; flex: 1; height: 1px; background: var(--line); }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
.cs-side-brand { display: flex; flex-direction: column; gap: 3px; margin: 0 0 14px 0; }
.cs-side-brand b { font-size: 16px; font-weight: 600; letter-spacing: -.02em; color: var(--text); }
.cs-side-brand span { font-size: 12px; color: var(--text-3); }
.cs-side-label {
    font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--text-3); margin: 22px 0 2px 0;
}
.cs-side-foot { margin-top: 30px; padding-top: 14px; border-top: 1px solid var(--line);
                font-size: 12px; color: var(--text-3); }
section[data-testid="stSidebar"] label p { font-size: 13px; color: var(--text-2); }

/* ── Widgets ────────────────────────────────────────────────────────────── */
.st-key-address_input [data-baseweb="input"], .st-key-address_input [data-baseweb="base-input"],
.st-key-address_input input { height: 52px !important; }
.st-key-address_input input {
    font-family: var(--mono) !important; font-size: 16px !important; letter-spacing: .01em;
}
.st-key-goal_input input { font-size: 14px !important; }
.st-key-investigate_btn button { min-height: 52px; }
.st-key-probe_btn { margin-left: auto; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    font-weight: 500; letter-spacing: .005em; white-space: nowrap;
    transition: background-color .15s ease, border-color .15s ease, color .15s ease;
}
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: var(--text) !important; border: 1px solid var(--text) !important;
    color: var(--bg) !important; font-weight: 600 !important;
}
button[data-testid="stBaseButton-primary"] *, button[data-testid="stBaseButton-primaryFormSubmit"] * {
    color: var(--bg) !important;
}
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background: #FFFFFF !important; border-color: #FFFFFF !important;
}
button[data-testid="stBaseButton-secondary"] {
    background: transparent !important; border: 1px solid var(--line-2) !important; color: var(--text-2) !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--text-2) !important; color: var(--text) !important;
}
button[data-testid="stBaseButton-secondary"]:hover * { color: var(--text) !important; }
button[data-testid="stBaseButton-tertiary"] {
    color: var(--text-2) !important; font-family: var(--mono); font-size: 12.5px !important;
    padding: 2px 8px !important; min-height: 0 !important;
}
button[data-testid="stBaseButton-tertiary"]:hover {
    color: var(--text) !important; background: var(--surface) !important;
}
[data-testid="stExpander"] details { background: transparent !important; }
[data-testid="stExpander"] summary { font-family: var(--mono); font-size: 12.5px; color: var(--text-2); }
[data-testid="stExpander"] summary:hover { color: var(--text); }
[data-testid="stAlertContainer"] { border: 1px solid var(--line); }
[data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre { border: 1px solid var(--line); }
[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] { color: var(--text-3); }
[data-testid="stDataFrame"] { font-size: 13px; }
h1, h2, h3 { letter-spacing: -.02em; }

/* ── Access gate ────────────────────────────────────────────────────────── */
.cs-gate { text-align: center; margin: 14vh 0 22px 0; }
.cs-gate-title { font-size: 30px; font-weight: 600; letter-spacing: -.03em; margin: 12px 0 8px 0; color: var(--text); }
.cs-gate p { color: var(--text-2); font-size: 14px; margin: 0; }

/* ── Footer ─────────────────────────────────────────────────────────────── */
.cs-footer {
    margin-top: 72px; padding-top: 18px; border-top: 1px solid var(--line);
    display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
    font-family: var(--mono); font-size: 12px; color: var(--text-3);
}
.cs-footer a { color: var(--text-2); text-decoration: none; }
.cs-footer a:hover { color: var(--text); }

@media (max-width: 900px) {
    .cs-h1 { font-size: 40px; }
    .cs-caps { grid-template-columns: 1fr; gap: 26px; }
    .cs-stats { grid-template-columns: 1fr 1fr; }
    .cs-stats > div:nth-child(3) { border-left: none; padding-left: 0; }
    .cs-top { flex-direction: column; align-items: flex-start; }
    .cs-status { justify-content: flex-start; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Access gate for public deployments (Streamlit Community Cloud etc.).
# Every investigation burns paid LLM tokens, so a public URL can be closed off
# with APP_PASSWORD (env var or Streamlit secret). Without it the app is open.
# ─────────────────────────────────────────────────────────────────────────────
def _expected_password() -> str:
    pwd = os.environ.get("APP_PASSWORD", "")
    if pwd:
        return pwd
    try:  # no secrets.toml on a dev box -> keep the app open
        return str(st.secrets.get("APP_PASSWORD", "") or "")
    except Exception:
        return ""


def require_access() -> None:
    expected = _expected_password()
    if not expected or st.session_state.get("_access_granted"):
        return
    st.markdown(
        "<div class='cs-gate'><div class='cs-eyebrow'>ChainScope</div>"
        "<div class='cs-gate-title'>Private deployment</div>"
        "<p>Enter the access password to continue · 输入访问密码以继续</p></div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        with st.form("access_gate"):
            supplied = st.text_input("Access password / 访问密码", type="password")
            submitted = st.form_submit_button("Enter / 进入", type="primary")
        if submitted:
            if hmac.compare_digest(supplied.encode(), expected.encode()):
                st.session_state["_access_granted"] = True
                st.rerun()
            st.error("Wrong password / 密码错误")
    st.stop()


require_access()

# ─────────────────────────────────────────────────────────────────────────────
# i18n: every user-visible string lives here as {key: {"en": ..., "cn": ...}}.
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "tagline":            {"en": "Autonomous On-Chain Investigation Agent",
                           "cn": "自主链上调查智能体"},
    "model_label":        {"en": "LLM model", "cn": "LLM 模型"},
    "model_help":         {"en": "Set by LLM_MODEL / LLM_BASE_URL / LLM_API_KEY in .env (any OpenAI-compatible endpoint with tool calling). Change .env and restart to switch.",
                           "cn": "由 .env 中的 LLM_MODEL / LLM_BASE_URL / LLM_API_KEY 决定（任何支持工具调用的 OpenAI 兼容接口）。修改 .env 并重启即可切换。"},
    "window_label":       {"en": "Time window (days)", "cn": "时间窗口（天）"},
    "window_help":        {"en": "How far back (in days) to pull on-chain transactions. No 90-day cap (up to 6000); larger windows pull deeper history and the graph is auto-capped to stay fast. Tip: click 'Probe activity span' first so the window actually captures this address's transactions - otherwise the graph can be empty.",
                           "cn": "向前回溯多少天的链上交易（已取消 90 天上限，最大 6000）；窗口越大回溯的历史越多，图会自动裁剪到上限以保持流畅。建议先点“探测交易活跃区间”，确保窗口能覆盖到该地址的交易，否则图可能为空。"},
    "activity_probe_btn": {"en": "Probe activity span", "cn": "探测交易活跃区间"},
    "activity_no_address":{"en": "Enter an address above first, then probe.",
                           "cn": "请先在上方输入要调查的地址，再点击探测。"},
    "activity_none":      {"en": "No normal transactions found for this address in the available sources - any time window will yield an empty graph. Double-check the address or try another.",
                           "cn": "在可查数据源中没有找到该地址的普通交易——任何时间窗口都会得到空图。请核对地址或更换一个。"},
    "activity_summary":   {"en": "Latest tx ~{dl} days ago · first tx ~{de} days ago (exact span).",
                           "cn": "最近一笔交易约 {dl} 天前 · 首笔交易约 {de} 天前（精确区间）。"},
    "activity_suggest":   {"en": "Time window set to {sug} days (covers the full history with buffer) to avoid an empty graph.",
                           "cn": "已自动把时间窗口设为 {sug} 天（覆盖完整历史并留余量），以避免空图。"},
    "graph_capped":       {"en": "Raw graph had {orig} nodes; capped to the {n} most-connected (target always kept) to keep detection & rendering fast.",
                           "cn": "原始图有 {orig} 个节点，已裁剪为关联度最高的 {n} 个（始终保留目标地址），以保证检测与渲染速度。"},
    "max_steps_label":    {"en": "Max agent steps", "cn": "最大执行步数"},
    "max_steps_help":     {"en": "Long-horizon budget. The agent self-terminates when confident.",
                           "cn": "长程预算。智能体在足够确信时会自行终止。"},
    "reflect_label":      {"en": "Reflect every N steps", "cn": "每 N 步反思一次"},
    "reflect_help":       {"en": "How often the agent pauses to review its findings and self-correct. Smaller = reflects more frequently (more careful, slower); larger = acts longer between reviews.",
                           "cn": "智能体每隔多少步停下来复盘已有发现并自我纠正。值越小反思越频繁（更严谨但更慢）；值越大则连续行动更久才复盘一次。"},
    "graph_limits_label": {"en": "Graph size limits (advanced)", "cn": "图规模上限（高级）"},
    "graph_limits_caption":{"en": "Defaults are usually fine. The agent can also tune these per call on its own.",
                           "cn": "通常用默认即可；智能体在自主调查时也会按需自行调整这些值。"},
    "max_nodes_label":    {"en": "Max graph nodes", "cn": "图最大节点数"},
    "max_nodes_help":     {"en": "Upper bound on nodes in the built graph. The graph keeps the most-connected addresses plus the target; larger = more complete but slower to render/detect.",
                           "cn": "构建图的节点数上限。图会保留关联度最高的地址及目标地址；值越大越完整，但渲染/检测越慢。"},
    "max_txs_label":      {"en": "Max txs pulled (target)", "cn": "目标地址最大拉取交易数"},
    "max_txs_help":       {"en": "How many recent transactions to pull for the target address (paginated). Larger = deeper history on very active addresses but slower.",
                           "cn": "为目标地址分页拉取的最近交易数上限。值越大，活跃地址的历史越完整，但更慢。"},
    "autonomy_loop":      {"en": "Autonomy loop", "cn": "自主循环"},
    "loop_steps":         {"en": "PLAN → ACT → OBSERVE → REFLECT → REPLAN",
                           "cn": "规划 → 行动 → 观察 → 反思 → 重规划"},
    "loop_caption":       {"en": "Every action is chosen by the LLM at runtime.",
                           "cn": "每一步行动都由 LLM 在运行时自主决定。"},
    "about":              {"en": "About", "cn": "关于"},
    "about_caption":      {"en": "LLM agent + GB-TGAD + EAS (Sepolia)",
                           "cn": "LLM 智能体 + GB-TGAD + EAS（Sepolia）"},
    "header_sub":         {"en": "LLM-driven autonomous on-chain investigation",
                           "cn": "由 LLM 驱动的自主链上调查"},
    "address_placeholder": {"en": "0x...", "cn": "0x..."},
    "investigate_btn":    {"en": "Investigate", "cn": "开始调查"},
    "goal_label":         {"en": "Investigation goal (optional)", "cn": "调查目标（可选）"},
    "goal_placeholder":   {"en": "Leave blank for default: detect illicit activity & attest on-chain",
                           "cn": "留空则使用默认目标：检测非法活动并在链上存证"},
    "live_stream":        {"en": "Live investigation stream", "cn": "实时调查流"},
    "spinner":            {"en": "The agent is investigating autonomously...",
                           "cn": "智能体正在自主调查……"},
    "status_starting":    {"en": "Starting - connecting to the LLM and planning (the first step can take 10-30 s)",
                           "cn": "启动中——正在连接 LLM 并规划（第一步通常需 10-30 秒）"},
    "status_running":     {"en": "{node} · step {step} · {secs}s elapsed",
                           "cn": "{node} · 第 {step} 步 · 已用 {secs} 秒"},
    "status_done":        {"en": "Investigation finished - {step} steps in {secs}s",
                           "cn": "调查完成——共 {step} 步，用时 {secs} 秒"},
    "status_error":       {"en": "Investigation failed", "cn": "调查失败"},
    "first_event":        {"en": "The agent is reading the chain and forming a plan…",
                           "cn": "智能体正在读取链上数据并制定计划……"},
    "plan_prefix":        {"en": "Plan", "cn": "规划"},
    "step_word":          {"en": "step", "cn": "步骤"},
    "risk_word":          {"en": "risk", "cn": "风险"},
    "investigate_failed": {"en": "Investigation failed", "cn": "调查失败"},
    "complete":           {"en": "Investigation complete - {steps} steps, {addrs} addresses investigated.",
                           "cn": "调查完成——共 {steps} 步，调查了 {addrs} 个地址。"},
    "results_title":      {"en": "Investigation Results", "cn": "调查结果"},
    "risk_high":          {"en": "HIGH RISK", "cn": "高风险"},
    "risk_med":           {"en": "MEDIUM", "cn": "中风险"},
    "risk_low":           {"en": "LOW RISK", "cn": "低风险"},
    "card_risk":          {"en": "Risk score", "cn": "风险评分"},
    "card_steps":         {"en": "Steps", "cn": "步数"},
    "card_addrs":         {"en": "Addresses", "cn": "地址数"},
    "card_verdict":       {"en": "Verdict", "cn": "结论"},
    "bar_safe":           {"en": "0.0 safe", "cn": "0.0 安全"},
    "bar_threshold":      {"en": "0.6 threshold", "cn": "0.6 阈值"},
    "bar_danger":         {"en": "1.0 danger", "cn": "1.0 危险"},
    "run_record":         {"en": "Long-Horizon Run Record", "cn": "长程运行记录"},
    "run_record_desc":    {"en": "The agent's full long-horizon execution trace. Left: every step plotted by phase (Plan / Act / Observe / Reflect / Replan) over time - proof it ran a multi-step investigation, not a one-shot call. Right: how many times each tool was called.",
                           "cn": "Agent 完整的长程执行轨迹。左图：按阶段（规划/行动/观察/反思/重规划）在时间轴上展开的每一步——证明它进行的是多步长程调查，而非一次性调用。右图：各工具被调用的次数。"},
    "hypotheses":         {"en": "Hypotheses (with self-correction)",
                           "cn": "假设（含自我纠正）"},
    "hypotheses_desc":    {"en": "Hypotheses the agent proposed, then confirmed or refuted as evidence arrived. Columns: status (active / confirmed / refuted), confidence 0-1, the hypothesis, and how many times it was revised.",
                           "cn": "Agent 提出的假设，以及随证据更新被确认或推翻的过程。列含义：status 状态（active 进行中 / confirmed 已确认 / refuted 已推翻）、confidence 置信度 0-1、假设内容、revisions 被修正次数。"},
    "no_hypotheses":      {"en": "No hypotheses recorded.", "cn": "未记录任何假设。"},
    "fund_paths":         {"en": "Traced fund paths", "cn": "追踪到的资金路径"},
    "no_fund_paths":      {"en": "No fund paths traced.", "cn": "未追踪到资金路径。"},
    "graph_evidence":     {"en": "Graph evidence", "cn": "图证据"},
    "final_report":       {"en": "Final Report & Attestation", "cn": "最终报告与存证"},
    "verdict_label":      {"en": "Verdict", "cn": "结论"},
    "no_attestation":     {"en": "No on-chain attestation produced this run.",
                           "cn": "本次运行未产生链上存证。"},
    "blind_label":        {"en": "Blind-label mode", "cn": "标签盲测模式"},
    "blind_help":         {"en": "Seal any matched label during the run and reveal/reconcile it against the blind, behaviour-only verdict at the end. Prevents label leakage so the agent's reasoning is evaluated fairly.",
                           "cn": "调查期间封存命中的标签，结论得出后再揭晓并与“仅凭行为”的盲推结论比对。避免标签泄露/作弊，公平检验智能体的真实推理能力。"},
    "recon_title":        {"en": "Blind verdict vs sealed label", "cn": "盲推结论 vs 封存标签"},
    "recon_blind":        {"en": "Blind verdict (no label seen)", "cn": "盲推结论（未看标签）"},
    "recon_sealed":       {"en": "Sealed label (revealed)", "cn": "封存标签（已揭晓）"},
    "recon_agree":        {"en": "AGREE - the blind verdict matches the held-out label.",
                           "cn": "一致——盲推结论与封存的标签真值吻合。"},
    "recon_conflict":     {"en": "CONFLICT - the blind verdict disagrees with the label. Review needed.",
                           "cn": "矛盾——盲推结论与标签不符，需要复核。"},
    "recon_identity":     {"en": "IDENTITY REVEALED - the label names the entity but carries no risk category; the blind verdict stands on behaviour.",
                           "cn": "身份揭晓——标签给出实体身份但无风险类别，盲推结论以行为为准。"},
    "export_btn":         {"en": "Export run record (JSON + HTML)",
                           "cn": "导出运行记录（JSON + HTML）"},
    "saved":              {"en": "Saved", "cn": "已保存"},
    "case_file":          {"en": "Case file", "cn": "案件档案"},
    "copy_logs_title":    {"en": "Copy All Logs", "cn": "一键复制全部日志"},
    "copy_logs_hint":     {"en": "Copy the whole run to your clipboard, or download it as a text file.",
                           "cn": "一键复制整个调查过程到剪贴板，也可直接下载为文本文件。"},
    "copy_btn":           {"en": "Copy all logs", "cn": "复制全部日志"},
    "copied":             {"en": "Copied", "cn": "已复制"},
    "copy_failed":        {"en": "Copy failed - please select the text below and copy manually.",
                           "cn": "复制失败，请手动选中下方文本复制。"},
    "download_btn":       {"en": "Download logs (.txt)", "cn": "下载日志（.txt）"},
    "no_logs":            {"en": "No logs recorded yet.", "cn": "暂无可复制的日志。"},
    # ── shell / empty state ──
    "side_setup":         {"en": "Setup", "cn": "配置"},
    "side_scope":         {"en": "Scope", "cn": "调查范围"},
    "side_agent":         {"en": "Agent", "cn": "智能体"},
    "side_integrity":     {"en": "Evaluation integrity", "cn": "评测完整性"},
    "chip_blind_on":      {"en": "Blind-label on", "cn": "标签盲测 开"},
    "chip_blind_off":     {"en": "Blind-label off", "cn": "标签盲测 关"},
    "chip_attest":        {"en": "EAS · Sepolia", "cn": "EAS · Sepolia"},
    "target_panel":       {"en": "Target address", "cn": "调查目标地址"},
    "empty_headline":     {"en": "Point it at an Ethereum address.",
                           "cn": "给它一个以太坊地址。"},
    "empty_body":         {"en": "The agent decides every move itself - it reads the chain, forms hypotheses, traces fund flows, and revises its own conclusions as evidence arrives. A full investigation runs for dozens of steps.",
                           "cn": "接下来的每一步都由智能体自己决定——读取链上数据、提出假设、追踪资金流向，并在证据出现时推翻或修正自己的结论。一次完整调查会持续数十步。"},
    "cap1_title":         {"en": "17 tools, chosen at runtime", "cn": "17 个工具，运行时自主选择"},
    "cap1_body":          {"en": "Nothing is a fixed pipeline. The model picks each next action - pull transactions, build the graph, run GB-TGAD anomaly detection, trace a path deeper.",
                           "cn": "没有任何固定流程。模型自行决定下一步做什么——拉取交易、构建图、运行 GB-TGAD 异常检测、沿着某条路径继续深挖。"},
    "cap2_title":         {"en": "Labels are sealed, not looked up", "cn": "标签被封存，而非直接查表"},
    "cap2_body":          {"en": "Any matched label is hidden from the model during the run and revealed only at the end, so the verdict has to be earned from on-chain behaviour.",
                           "cn": "命中的标签在调查期间对模型隐藏，结束后才揭晓比对，因此结论必须完全从链上行为中推出来。"},
    "cap3_title":         {"en": "The verdict goes on-chain", "cn": "结论上链存证"},
    "cap3_body":          {"en": "The final report is pinned to IPFS and an EAS attestation is written on Sepolia, so the finding is timestamped and independently checkable.",
                           "cn": "最终报告固定到 IPFS，并在 Sepolia 上写入 EAS 存证，使结论具备可独立验证的时间戳。"},
    "try_label":          {"en": "Or start from an address this agent has already investigated",
                           "cn": "也可以从这些已经调查过的地址开始"},
    "sample_mixer":       {"en": "Mixer", "cn": "混币器"},
    "sample_cex":         {"en": "Exchange", "cn": "交易所"},
    "sample_dex":         {"en": "DEX router", "cn": "DEX 路由"},
    "section_export":     {"en": "Export", "cn": "导出"},
}

# Public addresses this agent has already run, used as one-click demo targets.
# The name is shown to the *user* only - the agent still runs blind (see BLIND_MODE).
SAMPLE_TARGETS = [
    ("sample_mixer", "0x12D66f87A04A9E220743712cE6d9bB1B5616B8Fc"),
    ("sample_cex",   "0x28C6c06298d514Db089934071355E5743bf21d60"),
    ("sample_dex",   "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"),
]

# node -> (phase index, label). The index is shown as a small mono tag.
NODE_LABEL_I18N = {
    "plan":    {"en": ("01", "Plan"),                  "cn": ("01", "规划")},
    "act":     {"en": ("02", "Act (LLM reasoning)"),   "cn": ("02", "行动（LLM 推理）")},
    "observe": {"en": ("03", "Observe (tool result)"), "cn": ("03", "观察（工具结果）")},
    "reflect": {"en": ("04", "Reflect / self-correct"), "cn": ("04", "反思 / 自我纠正")},
    "replan":  {"en": ("05", "Replan (wrap up)"),      "cn": ("05", "重规划（收尾）")},
}

# Language selector (kept first so the whole UI re-renders in the chosen language).
# The brand block above it is only a reserved slot here: it needs LANG to render,
# but visually belongs at the very top of the sidebar.
LANG_OPTIONS = {"English": "en", "中文": "cn"}
_brand_slot = st.sidebar.container()
with st.sidebar:
    st.markdown("<div class='cs-side-label'>Language · 语言</div>", unsafe_allow_html=True)
    _lang_choice = st.radio(
        "Language / 语言",
        list(LANG_OPTIONS.keys()),
        horizontal=True,
        key="lang_choice",
        label_visibility="collapsed",
    )
LANG = LANG_OPTIONS[_lang_choice]


def t(key: str, **kwargs) -> str:
    """Translate a key into the current language, with optional .format() kwargs."""
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(LANG) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text


def node_label(node_name: str):
    return NODE_LABEL_I18N.get(node_name, {}).get(LANG, ("··", node_name))


_SECTION_NO = [0]


def section(title: str, numbered: bool = False) -> None:
    """Hairline-topped section heading; numbered sections get a running mono index."""
    tag = ""
    if numbered:
        _SECTION_NO[0] += 1
        tag = f"<span class='cs-idx'>{_SECTION_NO[0]:02d}</span>"
    st.markdown(f"<div class='cs-section'>{tag}<div class='cs-title'>{title}</div></div>",
                unsafe_allow_html=True)


# ── Sidebar ──
with _brand_slot:
    st.markdown(f"<div class='cs-side-brand'><b>ChainScope</b><span>{t('tagline')}</span></div>",
                unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<div class='cs-side-label'>{t('side_setup')}</div>", unsafe_allow_html=True)
    model = st.selectbox(t("model_label"), [cs_config.LLM_MODEL], index=0,
                         help=t("model_help"))

    st.markdown(f"<div class='cs-side-label'>{t('side_scope')}</div>", unsafe_allow_html=True)
    st.session_state.setdefault("window_days", 30)
    window_days = st.number_input(t("window_label"), min_value=1, max_value=6000,
                                  step=1, key="window_days", help=t("window_help"))

    st.markdown(f"<div class='cs-side-label'>{t('side_agent')}</div>", unsafe_allow_html=True)
    max_steps = st.slider(t("max_steps_label"), 8, 60, 24,
                          help=t("max_steps_help"))
    reflect_every = st.slider(t("reflect_label"), 2, 8, 4,
                              help=t("reflect_help"))

    st.markdown(f"<div class='cs-side-label'>{t('side_integrity')}</div>", unsafe_allow_html=True)
    blind_mode = st.checkbox(t("blind_label"), value=bool(cs_config.BLIND_MODE),
                             help=t("blind_help"))
    cs_config.BLIND_MODE = bool(blind_mode)

    st.markdown(f"<div class='cs-side-foot'>{t('about_caption')}</div>", unsafe_allow_html=True)

# ── Top bar ──
_blind_chip = t("chip_blind_on") if blind_mode else t("chip_blind_off")
st.markdown(f"""<div class="cs-top">
<div class="cs-wordmark"><b>ChainScope</b><span>{t('tagline')}</span></div>
<div class="cs-status">
<span><i class="cs-dot"></i>{_html.escape(str(cs_config.LLM_MODEL))}</span>
<span><i class="cs-dot{'' if blind_mode else ' off'}"></i>{_blind_chip}</span>
<span><i class="cs-dot"></i>{t('chip_attest')}</span>
</div>
</div>""", unsafe_allow_html=True)

# The hero (headline + lede) is rendered into this slot only while no
# investigation is running, so it sits above the command bar without reordering
# the widgets below it.
hero_slot = st.container()

# ── Command bar ──
st.markdown(f"<div class='cs-eyebrow cs-field-label'>{t('target_panel')}</div>",
            unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1.15], vertical_alignment="center", gap="small")
with col_input:
    address = st.text_input("Ethereum Address", placeholder=t("address_placeholder"),
                            key="address_input", label_visibility="collapsed")
with col_btn:
    investigate = st.button(t("investigate_btn"), type="primary", key="investigate_btn",
                            width="stretch")

custom_goal = st.text_input(t("goal_label"), placeholder=t("goal_placeholder"), key="goal_input")


def _probe_activity():
    """Probe the target's tx activity span and auto-suggest a non-empty window.

    Runs as a button callback (before widgets are re-instantiated), so it can
    safely write the recommended value into st.session_state['window_days'].
    """
    addr = (st.session_state.get("address_input") or "").strip()
    if not addr:
        st.session_state["activity_span"] = {"error": "no_address"}
        return
    span = default_scout.get_activity_span(addr)
    if span.get("has_activity") and span.get("days_since_earliest") is not None:
        de = span["days_since_earliest"]
        buffer = max(5.0, de * 0.1)  # cover full sampled history + 10% (min 5d)
        suggested = int(min(6000, math.ceil(de + buffer)))
        span["suggested_window"] = suggested
        st.session_state["window_days"] = suggested  # auto-apply to avoid empty graph
    st.session_state["activity_span"] = span


def _use_sample(addr: str):
    """Fill the address box from a one-click demo target."""
    st.session_state["address_input"] = addr


with st.container(horizontal=True, vertical_alignment="center", gap="small", key="cs_samples"):
    st.markdown(f"<span class='cs-try'>{t('try_label')}</span>", unsafe_allow_html=True,
                width="content")
    for _key, _addr in SAMPLE_TARGETS:
        st.button(f"{t(_key)} · {_addr[:6]}…{_addr[-4:]}", key=f"sample_{_addr}",
                  type="tertiary", on_click=_use_sample, args=(_addr,), help=_addr)
    st.button(t("activity_probe_btn"), key="probe_btn", on_click=_probe_activity)

_span = st.session_state.get("activity_span")
if _span is not None:
    if _span.get("error") == "no_address":
        st.warning(t("activity_no_address"))
    elif not _span.get("has_activity"):
        st.warning(t("activity_none"))
    else:
        st.info(t("activity_summary",
                  dl=int(round(_span["days_since_latest"])),
                  de=int(round(_span["days_since_earliest"]))))
        if _span.get("suggested_window"):
            st.success(t("activity_suggest", sug=_span["suggested_window"]))


def render_hero():
    """Headline + lede shown above the command bar before an investigation.

    HTML is emitted flush-left with no blank lines: Streamlit runs the string
    through Markdown first, so indentation or blank lines would split the block.
    """
    parts = [p.strip() for p in t("loop_steps").split("→")]
    loop = "<i>→</i>".join(f"<span>{p}</span>" for p in parts)
    st.markdown(f"""<div class="cs-hero">
<div class="cs-eyebrow cs-loopline">{loop}</div>
<div class="cs-h1">{t('empty_headline')}</div>
<p>{t('empty_body')}</p>
</div>""", unsafe_allow_html=True)


def render_capabilities():
    """Three typographic columns under the command bar (empty state only)."""
    st.markdown(f"""<div class="cs-caps">
<div class="cs-cap"><div class="cs-eyebrow">01</div><div class="cs-cap-title">{t('cap1_title')}</div><p>{t('cap1_body')}</p></div>
<div class="cs-cap"><div class="cs-eyebrow">02</div><div class="cs-cap-title">{t('cap2_title')}</div><p>{t('cap2_body')}</p></div>
<div class="cs-cap"><div class="cs-eyebrow">03</div><div class="cs-cap-title">{t('cap3_title')}</div><p>{t('cap3_body')}</p></div>
</div>
<div class="cs-footnote">{t('loop_caption')}</div>""", unsafe_allow_html=True)


def _render_act(update):
    """Render an ACT update: LLM reasoning + any tool calls it decided on."""
    for m in update.get("messages", []):
        content = getattr(m, "content", "")
        if content:
            st.markdown(f"<div class='cs-who'>{_html.escape(str(cs_config.LLM_MODEL))}</div>\n\n{content}",
                        unsafe_allow_html=True)
        for tc in getattr(m, "tool_calls", None) or []:
            args = str(tc.get("args", {}))[:160]
            st.markdown(
                f"<div class='cs-tool'>↳ <b>{_html.escape(str(tc.get('name')))}</b>"
                f"({_html.escape(args)})</div>",
                unsafe_allow_html=True,
            )


def _render_observe(update, captured):
    for m in update.get("messages", []):
        name = getattr(m, "name", "tool")
        content = str(getattr(m, "content", ""))
        captured.setdefault(name, []).append(content)
        with st.expander(f"{name} → result", expanded=False):
            st.code(content[:1500], language="text")


def build_full_log(case, captured) -> str:
    """Assemble a single plain-text blob of the whole run for one-click copy."""
    lines = [
        "=" * 70,
        "ChainScope Investigation Log",
        f"Target : {case.target}",
        f"Goal   : {case.goal}",
        f"Risk   : {case.risk_estimate:.2f}   Steps: {case.step}   "
        f"Addresses: {len(case.visited)}",
        "=" * 70,
        "",
        "--- Iteration Log ---",
    ]
    for s in case.iteration_log:
        lines.append(f"[step {s.step}] [{s.node}] {s.content}")

    if captured:
        lines += ["", "--- Tool Outputs ---"]
        for name, outputs in captured.items():
            for i, out in enumerate(outputs):
                lines.append(f"\n>>> {name} (#{i + 1})")
                lines.append(str(out))

    if case.verdict:
        lines += ["", "--- Verdict ---", str(case.verdict)]

    return "\n".join(lines)


def render_copy_logs(full_log: str):
    """Render an always-visible 'copy to clipboard' button + a download fallback."""
    import json as _json

    # Download button (always works, no clipboard permissions needed).
    st.download_button(
        t("download_btn"),
        data=full_log,
        file_name="chainscope_logs.txt",
        mime="text/plain",
    )

    # One-click copy button via a tiny HTML/JS component.
    payload = _json.dumps(full_log)
    btn_label = t("copy_btn")
    copied = t("copied")
    failed = t("copy_failed")
    html = f"""
    <style>
      body {{ margin: 0; background: transparent; }}
      #cs-copy-btn {{
          font-family: Geist, -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
          background: #E8EAEE; color: #0B0C0F; border: 1px solid #E8EAEE; border-radius: 8px;
          padding: 9px 18px; font-size: 13.5px; font-weight: 600; cursor: pointer; margin-bottom: 10px;
      }}
      #cs-copy-btn:hover {{ background: #FFFFFF; }}
      #cs-copy-msg {{ margin-left: 12px; font-size: 13px; color: #55CFA4;
                      font-family: -apple-system, 'Segoe UI', 'PingFang SC', sans-serif; }}
      #cs-log-text {{
          box-sizing: border-box; width: 100%; height: 240px; margin-top: 4px; resize: vertical;
          font-family: 'Geist Mono', ui-monospace, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.5;
          border: 1px solid #23262E; border-radius: 8px; padding: 12px;
          background: #13151A; color: #D2D6DE; outline: none;
      }}
    </style>
    <div>
      <button id="cs-copy-btn">{_html.escape(btn_label)}</button>
      <span id="cs-copy-msg"></span>
      <textarea id="cs-log-text" readonly>{_html.escape(full_log)}</textarea>
      <script>
        const csLog = {payload};
        const csBtn = document.getElementById("cs-copy-btn");
        const csMsg = document.getElementById("cs-copy-msg");
        const csArea = document.getElementById("cs-log-text");
        csBtn.addEventListener("click", async () => {{
          try {{
            if (navigator.clipboard && navigator.clipboard.writeText) {{
              await navigator.clipboard.writeText(csLog);
            }} else {{
              csArea.select();
              document.execCommand("copy");
            }}
            csMsg.style.color = "#55CFA4";
            csMsg.textContent = {_json.dumps(copied)};
          }} catch (e) {{
            try {{
              csArea.select();
              document.execCommand("copy");
              csMsg.style.color = "#55CFA4";
              csMsg.textContent = {_json.dumps(copied)};
            }} catch (e2) {{
              csMsg.style.color = "#E3AE5C";
              csMsg.textContent = {_json.dumps(failed)};
            }}
          }}
        }});
      </script>
    </div>
    """
    st.components.v1.html(html, height=320)


if not (investigate and address):
    with hero_slot:
        render_hero()
    render_capabilities()

if investigate and address:
    goal = custom_goal.strip() or DEFAULT_GOAL.format(address=address)
    case = new_case(address, goal)
    graph = build_investigation_graph(model=model, max_steps=max_steps,
                                      reflect_every=reflect_every)
    init = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=goal)],
        "step": 0, "max_steps": max_steps, "reflect_every": reflect_every, "nudges": 0,
    }
    config = {"recursion_limit": max_steps * 3 + 30}

    section(t("live_stream"))
    status = st.status(t("status_starting"), expanded=True)
    stream_box = st.container()
    with stream_box:
        first_event_ph = st.empty()
        first_event_ph.caption(t("first_event"))
    captured: dict[str, list] = {}
    t0 = time.time()
    saw_event = False

    try:
        for chunk in graph.stream(init, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                if not saw_event:
                    first_event_ph.empty()  # clear the "forming a plan" placeholder
                    saw_event = True
                idx, label = node_label(node_name)
                elapsed = int(time.time() - t0)
                status.update(label=t("status_running", node=label,
                                      step=case.step, secs=elapsed))
                with stream_box:
                    if node_name == "act":
                        _render_act(update)
                    elif node_name == "observe":
                        _render_observe(update, captured)
                    elif node_name == "plan":
                        for m in update.get("messages", []):
                            c = getattr(m, "content", "")
                            if c and getattr(m, "type", "") == "ai":
                                st.info(f"**{t('plan_prefix')}** — {c}")
                    elif node_name in ("reflect", "replan"):
                        st.markdown(
                            f"<div class='cs-phase'><span class='idx'>{idx}</span>{label} · "
                            f"{t('step_word')} {case.step} · "
                            f"{t('risk_word')} ≈ {case.risk_estimate:.2f}</div>",
                            unsafe_allow_html=True,
                        )
        run_ok = True
        status.update(label=t("status_done", step=case.step,
                              secs=int(time.time() - t0)),
                      state="complete", expanded=False)
    except Exception as e:
        run_ok = False
        status.update(label=t("status_error"), state="error")
        st.error(f"{t('investigate_failed')}: {e}")

    case_path = case.save()

    if run_ok:
        st.success(t("complete", steps=case.step, addrs=len(case.visited)))

        # ═══ Results Dashboard ═══
        section(t("results_title"), numbered=True)
        overall = case.risk_estimate
        risk_class = "risk-high" if overall >= 0.6 else ("risk-med" if overall >= 0.3 else "risk-low")
        risk_label = t("risk_high") if overall >= 0.6 else (t("risk_med") if overall >= 0.3 else t("risk_low"))

        st.markdown(f"""<div class="{risk_class}">
<div class="cs-stats">
<div><div class="cs-eyebrow">{t('card_risk')}</div><div class="cs-stat cs-risk">{overall:.2f}</div></div>
<div><div class="cs-eyebrow">{t('card_verdict')}</div><div class="cs-stat small cs-risk"><i class="cs-dot"></i>{risk_label}</div></div>
<div><div class="cs-eyebrow">{t('card_steps')}</div><div class="cs-stat">{case.step}</div></div>
<div><div class="cs-eyebrow">{t('card_addrs')}</div><div class="cs-stat">{len(case.visited)}</div></div>
</div>
<div class="cs-bar"><div class="fill" style="width:{overall*100:.1f}%"></div><div class="tick"></div></div>
<div class="cs-bar-legend"><span>{t('bar_safe')}</span><span>{t('bar_threshold')}</span><span>{t('bar_danger')}</span></div>
</div>""", unsafe_allow_html=True)

        # ── Long-horizon run record ──
        section(t("run_record"), numbered=True)
        st.markdown(f"<p class='cs-desc'>{t('run_record_desc')}</p>", unsafe_allow_html=True)
        c_tl, c_usage = st.columns([1.35, 1], gap="large")
        with c_tl:
            st.plotly_chart(run_trace.plot_timeline(case, lang=LANG), width="stretch")
        with c_usage:
            st.plotly_chart(run_trace.plot_tool_usage(case, lang=LANG), width="stretch")

        # ── Hypotheses + fund paths ──
        section(t("hypotheses"), numbered=True)
        c_hyp, c_path = st.columns([1.35, 1], gap="large")
        with c_hyp:
            st.markdown(f"<p class='cs-desc'>{t('hypotheses_desc')}</p>", unsafe_allow_html=True)
            rows = run_trace.hypothesis_table(case)
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.info(t("no_hypotheses"))
        with c_path:
            st.markdown(f"<div class='cs-sub'>{t('fund_paths')}</div>", unsafe_allow_html=True)
            if case.suspicious_paths:
                for p in case.suspicious_paths[:8]:
                    arrow = " → ".join(a[:10] for a in p.path)
                    st.markdown(
                        f"<div class='cs-path'><span class='dir'>{_html.escape(str(p.direction))}</span>"
                        f"<span class='meta'>{p.hops} hop · {p.total_value_eth:.3f} ETH</span>"
                        f"<div class='route'>{_html.escape(arrow)}</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info(t("no_fund_paths"))

        # ── Blind-label reconciliation ──
        recon = getattr(case, "reconciliation", None)
        if recon:
            section(t("recon_title"), numbered=True)
            bl = str(recon.get("blind_label", "")).upper()
            tl = recon.get("target_label")
            sealed = f"{tl.get('name', '?')} [{tl.get('category') or '—'}]" if tl else "—"
            st.markdown(f"""<div class="cs-stats" style="grid-template-columns:1fr 1fr">
<div><div class="cs-eyebrow">{t('recon_blind')}</div><div class="cs-stat small">{recon.get('blind_risk', 0.0):.2f} · {_html.escape(bl)}</div></div>
<div><div class="cs-eyebrow">{t('recon_sealed')}</div><div class="cs-stat small">{_html.escape(sealed)}</div></div>
</div>""", unsafe_allow_html=True)
            outcome = recon.get("outcome")
            if outcome == "agree":
                st.success(t("recon_agree"))
            elif outcome == "conflict":
                st.warning(t("recon_conflict"))
            elif outcome == "identity_revealed":
                st.info(t("recon_identity"))

        # ── Report + attestation ──
        section(t("final_report"), numbered=True)
        c_rep, c_att = st.columns([1.5, 1], gap="large")
        with c_rep:
            report_msgs = captured.get("compile_final_report", [])
            if report_msgs:
                st.markdown(report_msgs[-1])
        with c_att:
            att = captured.get("publish_attestation", [])
            if att:
                st.code(att[-1], language="text", wrap_lines=True)
            else:
                st.info(t("no_attestation"))

        # ── Export + one-click copy of the whole run ──
        section(t("section_export"), numbered=True)
        if st.button(t("export_btn")):
            paths = run_trace.export_run_record(case)
            st.success(f"{t('saved')}: {paths['json']}  /  {paths['html']}")
        st.caption(f"{t('case_file')}: {case_path}")

        st.markdown(f"<div class='cs-sub'>{t('copy_logs_title')}</div>", unsafe_allow_html=True)
        full_log = build_full_log(case, captured)
        if full_log.strip():
            st.caption(t("copy_logs_hint"))
            render_copy_logs(full_log)
        else:
            st.info(t("no_logs"))

# ── Footer ──
st.markdown(
    "<div class='cs-footer'><span>ChainScope v2.0 · LangGraph agent · GB-TGAD · EAS Sepolia</span>"
    "<span><a href='https://github.com/SKYEJT/ChainScope'>GitHub ↗</a></span></div>",
    unsafe_allow_html=True,
)
