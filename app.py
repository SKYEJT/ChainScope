"""ChainScope - Autonomous On-Chain Investigation Agent (Streamlit Frontend).

This UI drives the real LangGraph agent: the LLM plans, calls tools, observes
results, reflects/self-corrects across many steps, and concludes with an on-chain
attestation. The investigation is NOT a fixed pipeline — every action is chosen
by the model at runtime, and the long-horizon run is streamed live.
"""
import hmac
import math
import os
import time
from chainscope.tools.chainscout import default_scout

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

from chainscope import config as cs_config
from chainscope.agent.graph import build_investigation_graph
from chainscope.agent.case_file import new_case
from chainscope.agent.prompts import SYSTEM_PROMPT, DEFAULT_GOAL
from chainscope.utils import run_trace

st.set_page_config(page_title="ChainScope", layout="wide", page_icon="🔍",
                   initial_sidebar_state="expanded")


# ─────────────────────────────────────────────────────────────────────────────
# Access gate for public deployments (Streamlit Community Cloud etc.).
# Every investigation burns paid LLM tokens, so a public URL must not be open to
# anyone. The gate is active only when APP_PASSWORD is provided (as an env var
# or in Streamlit secrets); local runs without it behave exactly as before.
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
    st.markdown("# 🔍 ChainScope")
    st.caption("This deployment is password protected · 本站点需要访问密码")
    with st.form("access_gate"):
        supplied = st.text_input("Access password / 访问密码", type="password")
        submitted = st.form_submit_button("Enter / 进入")
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
    "model_label":        {"en": "🤖 LLM Model", "cn": "🤖 LLM 模型"},
    "model_help":         {"en": "Set by LLM_MODEL / LLM_BASE_URL / LLM_API_KEY in .env (any OpenAI-compatible endpoint with tool calling). Change .env and restart to switch.",
                           "cn": "由 .env 中的 LLM_MODEL / LLM_BASE_URL / LLM_API_KEY 决定（任何支持工具调用的 OpenAI 兼容接口）。修改 .env 并重启即可切换。"},
    "window_label":       {"en": "⏱️ Time Window (days)", "cn": "⏱️ 时间窗口（天）"},
    "window_help":        {"en": "How far back (in days) to pull on-chain transactions. No 90-day cap (up to 6000); larger windows pull deeper history and the graph is auto-capped to stay fast. Tip: click 'Probe activity span' first so the window actually captures this address's transactions — otherwise the graph can be empty.",
                           "cn": "向前回溯多少天的链上交易（已取消 90 天上限，最大 6000）；窗口越大回溯的历史越多，图会自动裁剪到上限以保持流畅。建议先点下方“探测交易活跃区间”，确保窗口能覆盖到该地址的交易，否则图可能为空。"},
    "activity_probe_btn": {"en": "🔎 Probe activity span", "cn": "🔎 探测交易活跃区间"},
    "activity_no_address":{"en": "Enter an address above first, then probe.",
                           "cn": "请先在上方输入要调查的地址，再点击探测。"},
    "activity_none":      {"en": "⚠️ No normal transactions found for this address in the available sources — any time window will yield an empty graph. Double-check the address or try another.",
                           "cn": "⚠️ 在可查数据源中没有找到该地址的普通交易——任何时间窗口都会得到空图。请核对地址或更换一个。"},
    "activity_summary":   {"en": "📊 Latest tx ~{dl} days ago · first tx ~{de} days ago (exact span).",
                           "cn": "📊 最近一笔交易约 {dl} 天前 · 首笔交易约 {de} 天前（精确区间）。"},
    "activity_suggest":   {"en": "✅ Auto-set the time window to {sug} days (covers the full history with buffer) to avoid an empty graph.",
                           "cn": "✅ 已自动把时间窗口设为 {sug} 天（覆盖完整历史并留余量），以避免空图。"},
    "graph_capped":       {"en": "ℹ️ Raw graph had {orig} nodes; capped to the {n} most-connected (target always kept) to keep detection & rendering fast.",
                           "cn": "ℹ️ 原始图有 {orig} 个节点，已裁剪为关联度最高的 {n} 个（始终保留目标地址），以保证检测与渲染速度。"},
    "max_steps_label":    {"en": "🧠 Max Agent Steps", "cn": "🧠 最大执行步数"},
    "max_steps_help":     {"en": "Long-horizon budget. The agent self-terminates when confident.",
                           "cn": "长程预算。智能体在足够确信时会自行终止。"},
    "reflect_label":      {"en": "🔁 Reflect Every N Steps", "cn": "🔁 每 N 步反思一次"},
    "reflect_help":       {"en": "How often the agent pauses to review its findings and self-correct. Smaller = reflects more frequently (more careful, slower); larger = acts longer between reviews.",
                           "cn": "智能体每隔多少步停下来复盘已有发现并自我纠正。值越小反思越频繁（更严谨但更慢）；值越大则连续行动更久才复盘一次。"},
    "graph_limits_label": {"en": "🧱 Graph size limits (advanced)", "cn": "🧱 图规模上限（高级）"},
    "graph_limits_caption":{"en": "Defaults are usually fine. The agent can also tune these per call on its own.",
                           "cn": "通常用默认即可；智能体在自主调查时也会按需自行调整这些值。"},
    "max_nodes_label":    {"en": "Max graph nodes", "cn": "图最大节点数"},
    "max_nodes_help":     {"en": "Upper bound on nodes in the built graph. The graph keeps the most-connected addresses plus the target; larger = more complete but slower to render/detect.",
                           "cn": "构建图的节点数上限。图会保留关联度最高的地址及目标地址；值越大越完整，但渲染/检测越慢。"},
    "max_txs_label":      {"en": "Max txs pulled (target)", "cn": "目标地址最大拉取交易数"},
    "max_txs_help":       {"en": "How many recent transactions to pull for the target address (paginated). Larger = deeper history on very active addresses but slower.",
                           "cn": "为目标地址分页拉取的最近交易数上限。值越大，活跃地址的历史越完整，但更慢。"},
    "autonomy_loop":      {"en": "Autonomy Loop", "cn": "自主循环"},
    "loop_steps":         {"en": "PLAN → ACT → OBSERVE → REFLECT → REPLAN",
                           "cn": "规划 → 行动 → 观察 → 反思 → 重规划"},
    "loop_caption":       {"en": "Every action is chosen by the LLM at runtime.",
                           "cn": "每一步行动都由 LLM 在运行时自主决定。"},
    "about":              {"en": "ℹ️ About", "cn": "ℹ️ 关于"},
    "about_caption":      {"en": "LLM agent + GB-TGAD + EAS (Sepolia)",
                           "cn": "LLM 智能体 + GB-TGAD + EAS（Sepolia）"},
    "header_sub":         {"en": "LLM-driven autonomous on-chain investigation",
                           "cn": "由 LLM 驱动的自主链上调查"},
    "address_placeholder": {"en": "0x...", "cn": "0x..."},
    "investigate_btn":    {"en": "🚀 Investigate", "cn": "🚀 开始调查"},
    "goal_label":         {"en": "Investigation goal (optional)", "cn": "调查目标（可选）"},
    "goal_placeholder":   {"en": "Leave blank for default: detect illicit activity & attest on-chain",
                           "cn": "留空则使用默认目标：检测非法活动并在链上存证"},
    "live_stream":        {"en": "🛰️ Live Investigation Stream", "cn": "🛰️ 实时调查流"},
    "spinner":            {"en": "The agent is investigating autonomously...",
                           "cn": "智能体正在自主调查……"},
    "status_starting":    {"en": "🚀 Starting… connecting to the LLM and planning (first step can take 10-30s)",
                           "cn": "🚀 启动中……正在连接 LLM 并规划（第一步通常需 10-30 秒）"},
    "status_running":     {"en": "Working: {node} · step {step} · {secs}s elapsed",
                           "cn": "调查中：{node} · 第 {step} 步 · 已用 {secs} 秒"},
    "status_done":        {"en": "✅ Investigation finished — {step} steps in {secs}s",
                           "cn": "✅ 调查完成 —— 共 {step} 步，用时 {secs} 秒"},
    "status_error":       {"en": "❌ Investigation failed", "cn": "❌ 调查失败"},
    "first_event":        {"en": "⏳ The agent is reading the chain and forming a plan…",
                           "cn": "⏳ 智能体正在读取链上数据并制定计划……"},
    "plan_prefix":        {"en": "Plan", "cn": "规划"},
    "step_word":          {"en": "step", "cn": "步骤"},
    "risk_word":          {"en": "risk", "cn": "风险"},
    "investigate_failed": {"en": "Investigation failed", "cn": "调查失败"},
    "complete":           {"en": "✅ Investigation complete — {steps} steps, {addrs} addresses investigated.",
                           "cn": "✅ 调查完成 —— 共 {steps} 步，调查了 {addrs} 个地址。"},
    "results_title":      {"en": "📊 Investigation Results", "cn": "📊 调查结果"},
    "risk_high":          {"en": "HIGH RISK", "cn": "高风险"},
    "risk_med":           {"en": "MEDIUM", "cn": "中风险"},
    "risk_low":           {"en": "LOW RISK", "cn": "低风险"},
    "card_risk":          {"en": "Risk Score", "cn": "风险评分"},
    "card_steps":         {"en": "Steps", "cn": "步数"},
    "card_addrs":         {"en": "Addresses", "cn": "地址数"},
    "card_verdict":       {"en": "Verdict", "cn": "结论"},
    "bar_safe":           {"en": "0.0 (Safe)", "cn": "0.0（安全）"},
    "bar_threshold":      {"en": "0.6 (Threshold)", "cn": "0.6（阈值）"},
    "bar_danger":         {"en": "1.0 (Danger)", "cn": "1.0（危险）"},
    "run_record":         {"en": "🧠 Long-Horizon Run Record", "cn": "🧠 长程运行记录"},
    "run_record_desc":    {"en": "The agent's full long-horizon execution trace. Left: every step plotted by phase (Plan/Act/Observe/Reflect/Replan) over time — proof it ran a multi-step investigation, not a one-shot call. Right: how many times each tool was called.",
                           "cn": "Agent 完整的长程执行轨迹。左图：按阶段（规划/行动/观察/反思/重规划）在时间轴上展开的每一步——证明它进行的是多步长程调查，而非一次性调用。右图：各工具被调用的次数。"},
    "hypotheses":         {"en": "🔬 Hypotheses (with self-correction)",
                           "cn": "🔬 假设（含自我纠正）"},
    "hypotheses_desc":    {"en": "Hypotheses the agent proposed, then confirmed or refuted as evidence arrived (self-correction). Columns: status (active/confirmed/refuted), confidence 0-1, the hypothesis, and how many times it was revised.",
                           "cn": "Agent 提出的假设，以及随证据更新被确认或推翻的过程（自我纠正）。列含义：status 状态（active 进行中 / confirmed 已确认 / refuted 已推翻）、confidence 置信度 0-1、假设内容、revisions 被修正次数。"},
    "no_hypotheses":      {"en": "No hypotheses recorded.", "cn": "未记录任何假设。"},
    "fund_paths":         {"en": "💸 Traced Fund Paths", "cn": "💸 追踪到的资金路径"},
    "no_fund_paths":      {"en": "No fund paths traced.", "cn": "未追踪到资金路径。"},
    "graph_evidence":     {"en": "🕸️ Graph Evidence", "cn": "🕸️ 图证据"},
    "final_report":       {"en": "📝 Final Report & Attestation", "cn": "📝 最终报告与存证"},
    "verdict_label":      {"en": "Verdict", "cn": "结论"},
    "no_attestation":     {"en": "No on-chain attestation produced this run.",
                           "cn": "本次运行未产生链上存证。"},
    "blind_label":        {"en": "🕶️ Blind-label mode", "cn": "🕶️ 标签盲测模式"},
    "blind_help":         {"en": "Seal any matched label during the run and reveal/reconcile it against the blind, behaviour-only verdict at the end. Prevents label leakage so the agent's reasoning is evaluated fairly.",
                           "cn": "调查期间封存命中的标签，结论得出后再揭晓并与“仅凭行为”的盲推结论比对。避免标签泄露/作弊，公平检验智能体的真实推理能力。"},
    "recon_title":        {"en": "🔒 Blind Verdict vs Sealed Label", "cn": "🔒 盲推结论 vs 封存标签"},
    "recon_blind":        {"en": "Blind verdict (no label seen)", "cn": "盲推结论（未看标签）"},
    "recon_sealed":       {"en": "Sealed label (revealed)", "cn": "封存标签（已揭晓）"},
    "recon_agree":        {"en": "✅ AGREE — the blind verdict matches the held-out label.",
                           "cn": "✅ 一致 —— 盲推结论与封存的标签真值吻合。"},
    "recon_conflict":     {"en": "⚠️ CONFLICT — the blind verdict disagrees with the label. Review needed.",
                           "cn": "⚠️ 矛盾 —— 盲推结论与标签不符，需要复核。"},
    "recon_identity":     {"en": "ℹ️ IDENTITY REVEALED — the label names the entity but carries no risk category; the blind verdict stands on behaviour.",
                           "cn": "ℹ️ 身份揭晓 —— 标签给出实体身份但无风险类别，盲推结论以行为为准。"},
    "export_btn":         {"en": "📦 Export run record (JSON + HTML)",
                           "cn": "📦 导出运行记录（JSON + HTML）"},
    "saved":              {"en": "Saved", "cn": "已保存"},
    "case_file":          {"en": "📁 Case file", "cn": "📁 案件档案"},
    "copy_logs_title":    {"en": "📋 Copy All Logs", "cn": "📋 一键复制全部日志"},
    "copy_logs_hint":     {"en": "Click the button below to copy all logs to your clipboard, or download them as a file.",
                           "cn": "点击下方按钮即可一键复制全部日志到剪贴板，也可直接下载为文件。"},
    "copy_btn":           {"en": "📋 Copy all logs", "cn": "📋 复制全部日志"},
    "copied":             {"en": "✅ Copied!", "cn": "✅ 已复制！"},
    "copy_failed":        {"en": "⚠️ Copy failed — please select the text below and copy manually.",
                           "cn": "⚠️ 复制失败，请手动选中下方文本复制。"},
    "download_btn":       {"en": "⬇️ Download logs (.txt)", "cn": "⬇️ 下载日志（.txt）"},
    "no_logs":            {"en": "No logs recorded yet.", "cn": "暂无可复制的日志。"},
    # ── shell / empty state ──
    "side_setup":         {"en": "Setup", "cn": "配置"},
    "side_scope":         {"en": "Scope", "cn": "调查范围"},
    "side_agent":         {"en": "Agent", "cn": "智能体"},
    "side_integrity":     {"en": "Evaluation integrity", "cn": "评测完整性"},
    "chip_blind_on":      {"en": "Blind-label ON", "cn": "标签盲测 开"},
    "chip_blind_off":     {"en": "Blind-label OFF", "cn": "标签盲测 关"},
    "chip_attest":        {"en": "EAS · Sepolia", "cn": "EAS · Sepolia"},
    "target_panel":       {"en": "Target address", "cn": "调查目标地址"},
    "empty_headline":     {"en": "Point it at an Ethereum address.",
                           "cn": "给它一个以太坊地址。"},
    "empty_body":         {"en": "The agent decides every move itself — it reads the chain, forms hypotheses, traces fund flows, and revises its own conclusions as evidence arrives. A full investigation runs for dozens of steps.",
                           "cn": "接下来的每一步都由智能体自己决定——读取链上数据、提出假设、追踪资金流向，并在证据出现时推翻或修正自己的结论。一次完整调查会持续数十步。"},
    "cap1_title":         {"en": "17 tools, chosen at runtime", "cn": "17 个工具，运行时自主选择"},
    "cap1_body":          {"en": "Nothing is a fixed pipeline. The model picks each next action — pull transactions, build the graph, run GB-TGAD anomaly detection, trace a path deeper.",
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
}

# Public addresses this agent has already run, used as one-click demo targets.
# The name is shown to the *user* only — the agent still runs blind (see BLIND_MODE).
SAMPLE_TARGETS = [
    ("sample_mixer", "0x12D66f87A04A9E220743712cE6d9bB1B5616B8Fc"),
    ("sample_cex",   "0x28C6c06298d514Db089934071355E5743bf21d60"),
    ("sample_dex",   "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"),
]

NODE_LABEL_I18N = {
    "plan":    {"en": ("🧭", "Plan"),                 "cn": ("🧭", "规划")},
    "act":     {"en": ("🤖", "Act (LLM reasoning)"),  "cn": ("🤖", "行动（LLM 推理）")},
    "observe": {"en": ("🔧", "Observe (tool result)"), "cn": ("🔧", "观察（工具结果）")},
    "reflect": {"en": ("🔁", "Reflect / self-correct"), "cn": ("🔁", "反思 / 自我纠正")},
    "replan":  {"en": ("⏰", "Replan (wrap up)"),      "cn": ("⏰", "重规划（收尾）")},
}

# Language selector (kept first so the whole UI re-renders in the chosen language).
# The brand block above it is only a reserved slot here: it needs LANG to render,
# but visually belongs at the very top of the sidebar.
LANG_OPTIONS = {"English": "en", "中文": "cn"}
_brand_slot = st.sidebar.container()
with st.sidebar:
    _lang_choice = st.radio(
        "🌐 Language / 语言",
        list(LANG_OPTIONS.keys()),
        horizontal=True,
        key="lang_choice",
    )
LANG = LANG_OPTIONS[_lang_choice]


def t(key: str, **kwargs) -> str:
    """Translate a key into the current language, with optional .format() kwargs."""
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(LANG) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text


def node_label(node_name: str):
    return NODE_LABEL_I18N.get(node_name, {}).get(LANG, ("•", node_name))


st.markdown("""
<style>
:root {
    --cs-bg:        #0A0C12;
    --cs-surface:   #141A24;
    --cs-surface-2: #1A2230;
    --cs-border:    #232D3D;
    --cs-border-hi: #35435C;
    --cs-text:      #E8EDF5;
    --cs-dim:       #93A1B5;
    --cs-faint:     #64748B;
    --cs-accent:    #7C6BFF;
    --cs-accent-2:  #22D3EE;
    --cs-ok:        #2ECC8F;
    --cs-warn:      #F5B942;
    --cs-bad:       #FF5C6C;
    --cs-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}

/* ── Page shell ───────────────────────────────────────────────────────── */
.stApp { background:
    radial-gradient(1100px 520px at 12% -8%, rgba(124,107,255,.13), transparent 60%),
    radial-gradient(900px 460px at 92% -4%, rgba(34,211,238,.09), transparent 60%),
    var(--cs-bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
hr { border-color: var(--cs-border) !important; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E131C 0%, #0B0F17 100%);
    border-right: 1px solid var(--cs-border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.cs-side-title {
    font-size: 11px; font-weight: 700; letter-spacing: .13em; text-transform: uppercase;
    color: var(--cs-faint); margin: 16px 0 6px 0; padding-left: 9px;
    border-left: 2px solid var(--cs-accent);
}

/* ── Brand / hero ─────────────────────────────────────────────────────── */
.cs-brand {
    font-size: 40px; font-weight: 800; letter-spacing: -.025em; line-height: 1.1; margin: 0;
    background: linear-gradient(92deg, #FFFFFF 8%, var(--cs-accent) 55%, var(--cs-accent-2) 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.cs-tagline { color: var(--cs-dim); font-size: 15px; margin: 6px 0 0 0; }
.cs-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 4px 0; }
.cs-chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--cs-surface); border: 1px solid var(--cs-border);
    border-radius: 999px; padding: 5px 13px; font-size: 12.5px; color: var(--cs-dim);
    transition: border-color .15s ease, color .15s ease;
}
.cs-chip:hover { border-color: var(--cs-border-hi); color: var(--cs-text); }
.cs-chip b { color: var(--cs-text); font-weight: 600; font-family: var(--cs-mono); font-size: 12px; }
.cs-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--cs-ok);
          box-shadow: 0 0 0 3px rgba(46,204,143,.16); }
.cs-dot.off { background: var(--cs-faint); box-shadow: 0 0 0 3px rgba(100,116,139,.14); }

/* ── Panels ───────────────────────────────────────────────────────────── */
.cs-panel {
    background: var(--cs-surface); border: 1px solid var(--cs-border);
    border-radius: 16px; padding: 22px 24px; margin: 6px 0 18px 0;
}
.cs-panel-label {
    font-size: 11px; font-weight: 700; letter-spacing: .13em; text-transform: uppercase;
    color: var(--cs-faint); margin-bottom: 12px;
}

/* ── Loop / capability strip ──────────────────────────────────────────── */
.cs-loop { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.cs-loop span {
    font-family: var(--cs-mono); font-size: 12px; color: var(--cs-text);
    background: var(--cs-surface-2); border: 1px solid var(--cs-border-hi);
    border-radius: 8px; padding: 7px 13px;
}
.cs-loop i { color: var(--cs-accent); font-style: normal; font-size: 15px; }
.cs-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 6px; }
.cs-card {
    position: relative; background: var(--cs-surface); border: 1px solid var(--cs-border);
    border-radius: 14px; padding: 18px 20px;
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.cs-card:hover {
    transform: translateY(-3px); border-color: var(--cs-border-hi);
    box-shadow: 0 12px 30px rgba(0,0,0,.35);
}
.cs-card h4 { margin: 0 0 6px 0; font-size: 14.5px; font-weight: 650; color: var(--cs-text); }
.cs-card p  { margin: 0; font-size: 13px; line-height: 1.55; color: var(--cs-dim); }

/* ── Result metric cards ──────────────────────────────────────────────── */
.metric-card {
    position: relative; overflow: hidden;
    background: var(--cs-surface); border: 1px solid var(--cs-border);
    border-radius: 16px; padding: 20px 18px; text-align: center;
    transition: transform .18s ease, box-shadow .18s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0,0,0,.35); }
.metric-card::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
    background: linear-gradient(90deg, var(--cs-accent), var(--cs-accent-2));
}
.metric-card.risk-high::before { background: linear-gradient(90deg, var(--cs-bad), #FF9A6B); }
.metric-card.risk-med::before  { background: linear-gradient(90deg, var(--cs-warn), #FFD98A); }
.metric-card.risk-low::before  { background: linear-gradient(90deg, var(--cs-ok), var(--cs-accent-2)); }
.metric-card h3 {
    margin: 0; font-size: 11px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--cs-faint);
}
.metric-card h1 {
    margin: 10px 0 2px 0; font-size: 34px; font-weight: 750; line-height: 1.1;
    font-family: var(--cs-mono); color: var(--cs-text);
}
.metric-card.risk-high h1 { color: var(--cs-bad); }
.metric-card.risk-low  h1 { color: var(--cs-ok); }
.metric-card.risk-med  h1 { color: var(--cs-warn); }

/* ── Risk bar ─────────────────────────────────────────────────────────── */
.score-bar-bg {
    position: relative; background: var(--cs-surface-2);
    border: 1px solid var(--cs-border); border-radius: 999px; height: 12px;
}
.score-bar-fill { border-radius: 999px; height: 100%; transition: width .6s ease; }
.score-bar-bg::after {
    content: ""; position: absolute; left: 60%; top: -4px; bottom: -4px;
    width: 2px; background: var(--cs-border-hi);
}
.cs-bar-legend {
    display: flex; justify-content: space-between; margin-top: 8px;
    color: var(--cs-faint); font-size: 11.5px; font-family: var(--cs-mono);
}

/* ── Streamlit widgets ────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--cs-surface-2) !important; border: 1px solid var(--cs-border) !important;
    border-radius: 10px !important; color: var(--cs-text) !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--cs-accent) !important; box-shadow: 0 0 0 3px rgba(124,107,255,.16) !important;
}
#cs-address input { font-family: var(--cs-mono); font-size: 15px; letter-spacing: .01em; }
.stButton > button {
    border-radius: 10px; border: 1px solid var(--cs-border-hi);
    background: var(--cs-surface-2); color: var(--cs-text); font-weight: 550;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover {
    border-color: var(--cs-accent); color: var(--cs-text);
    transform: translateY(-1px); box-shadow: 0 6px 18px rgba(124,107,255,.14);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(92deg, var(--cs-accent), #5A73FF);
    border: none; color: #fff; font-weight: 650;
    box-shadow: 0 6px 20px rgba(124,107,255,.28);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px); color: #fff;
    box-shadow: 0 10px 28px rgba(124,107,255,.42);
}
[data-testid="stExpander"] {
    background: var(--cs-surface); border: 1px solid var(--cs-border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { font-family: var(--cs-mono); font-size: 12.5px; }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--cs-border); }
[data-testid="stMetricValue"] { font-family: var(--cs-mono); }
code { background: var(--cs-surface-2) !important; color: var(--cs-accent-2) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--cs-border); border-radius: 12px; }
h2, h3 { letter-spacing: -.015em; }

/* ── Live stream rows ─────────────────────────────────────────────────── */
.cs-stream-tool {
    font-family: var(--cs-mono); font-size: 12.5px; color: var(--cs-dim);
    border-left: 2px solid var(--cs-accent); padding: 2px 0 2px 12px; margin: 2px 0 2px 14px;
}
.cs-phase {
    display: flex; align-items: center; gap: 10px; margin: 18px 0 10px 0;
    color: var(--cs-dim); font-size: 12.5px; font-family: var(--cs-mono);
}
.cs-phase::after { content: ""; flex: 1; height: 1px; background: var(--cs-border); }

/* ── Footer ───────────────────────────────────────────────────────────── */
.cs-footer {
    text-align: center; color: var(--cs-faint); font-size: 12.5px; padding: 8px 0 4px 0;
}
.cs-footer a { color: var(--cs-accent-2); text-decoration: none; }
.cs-footer a:hover { text-decoration: underline; }

/* ── Scrollbar + text selection (premium polish) ──────────────────────── */
::selection { background: rgba(124,107,255,.35); color: #fff; }
* { scrollbar-width: thin; scrollbar-color: var(--cs-border-hi) transparent; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
    background: var(--cs-border-hi); border-radius: 8px; border: 2px solid var(--cs-bg);
}
::-webkit-scrollbar-thumb:hover { background: #46577A; }
::-webkit-scrollbar-track { background: transparent; }

@media (max-width: 900px) { .cs-cards { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with _brand_slot:
    st.markdown(
        f"<div style='font-size:22px;font-weight:750;letter-spacing:-.02em'>🔍 ChainScope</div>"
        f"<div style='color:#93A1B5;font-size:12.5px;margin:2px 0 10px 0'>{t('tagline')}</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown(f"<div class='cs-side-title'>{t('side_setup')}</div>", unsafe_allow_html=True)
    model = st.selectbox(t("model_label"), [cs_config.LLM_MODEL], index=0,
                         help=t("model_help"))

    st.markdown(f"<div class='cs-side-title'>{t('side_scope')}</div>", unsafe_allow_html=True)
    st.session_state.setdefault("window_days", 30)
    window_days = st.number_input(t("window_label"), min_value=1, max_value=6000,
                                  step=1, key="window_days", help=t("window_help"))

    st.markdown(f"<div class='cs-side-title'>{t('side_agent')}</div>", unsafe_allow_html=True)
    max_steps = st.slider(t("max_steps_label"), 8, 60, 24,
                          help=t("max_steps_help"))
    reflect_every = st.slider(t("reflect_label"), 2, 8, 4,
                              help=t("reflect_help"))

    st.markdown(f"<div class='cs-side-title'>{t('side_integrity')}</div>", unsafe_allow_html=True)
    blind_mode = st.checkbox(t("blind_label"), value=bool(cs_config.BLIND_MODE),
                             help=t("blind_help"))
    cs_config.BLIND_MODE = bool(blind_mode)

    st.divider()
    st.caption(t("about_caption"))

# ── Header ──
_blind_chip = t("chip_blind_on") if blind_mode else t("chip_blind_off")
_model_chip = t("model_label").replace("🤖 ", "")
st.markdown(f"""
<h1 class="cs-brand">ChainScope</h1>
<p class="cs-tagline">{t('header_sub')}</p>
<div class="cs-chips">
<span class="cs-chip"><span class="cs-dot"></span>{_model_chip} <b>{cs_config.LLM_MODEL}</b></span>
<span class="cs-chip"><span class="cs-dot{'' if blind_mode else ' off'}"></span>{_blind_chip}</span>
<span class="cs-chip"><span class="cs-dot"></span>{t('chip_attest')}</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"<div class='cs-panel-label' style='margin:22px 0 8px 0'>{t('target_panel')}</div>",
            unsafe_allow_html=True)
col_input, col_btn = st.columns([4, 1])
with col_input:
    st.markdown("<div id='cs-address'>", unsafe_allow_html=True)
    address = st.text_input("Ethereum Address", placeholder=t("address_placeholder"),
                            key="address_input", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
with col_btn:
    investigate = st.button(t("investigate_btn"), type="primary", use_container_width=True)

col_goal, _ = st.columns([4, 1])  # same split as the address row, so they line up
with col_goal:
    custom_goal = st.text_input(
        t("goal_label"),
        placeholder=t("goal_placeholder"),
    )


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


_probe_col, *_sample_cols = st.columns([1.7, 1.1, 1.1, 1.1])
with _probe_col:
    st.button(t("activity_probe_btn"), on_click=_probe_activity, use_container_width=True)
for _col, (_key, _addr) in zip(_sample_cols, SAMPLE_TARGETS):
    with _col:
        st.button(f"{t(_key)} · {_addr[:6]}…", key=f"sample_{_addr}",
                  on_click=_use_sample, args=(_addr,), use_container_width=True,
                  help=_addr)

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


def render_empty_state():
    """What the page shows before an investigation is launched.

    The HTML is emitted flush-left: Streamlit runs the string through Markdown
    first, so indented lines would turn into code blocks.
    """
    parts = [p.strip() for p in t("loop_steps").split("→")]
    phases = "<i>→</i>".join(f"<span>{p}</span>" for p in parts)
    st.markdown(
        f"""
<div class="cs-panel">
<div class="cs-panel-label">{t('autonomy_loop')}</div>
<div class="cs-loop">{phases}</div>
<p style="color:#93A1B5;font-size:13px;margin:14px 0 0 0">{t('loop_caption')}</p>
</div>
<h3 style="margin:26px 0 6px 0">{t('empty_headline')}</h3>
<p style="color:#93A1B5;max-width:74ch;line-height:1.6;margin:0 0 20px 0">{t('empty_body')}</p>
<div class="cs-cards">
<div class="cs-card"><h4>{t('cap1_title')}</h4><p>{t('cap1_body')}</p></div>
<div class="cs-card"><h4>{t('cap2_title')}</h4><p>{t('cap2_body')}</p></div>
<div class="cs-card"><h4>{t('cap3_title')}</h4><p>{t('cap3_body')}</p></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_act(update):
    """Render an ACT update: LLM reasoning + any tool calls it decided on."""
    for m in update.get("messages", []):
        content = getattr(m, "content", "")
        if content:
            st.markdown(f"🤖 **{cs_config.LLM_MODEL}:** {content}")
        for tc in getattr(m, "tool_calls", None) or []:
            args = str(tc.get("args", {}))[:160]
            st.markdown(
                f"<div class='cs-stream-tool'>↳ <b style='color:#7C6BFF'>"
                f"{tc.get('name')}</b>({__import__('html').escape(args)})</div>",
                unsafe_allow_html=True,
            )


def _render_observe(update, captured):
    for m in update.get("messages", []):
        name = getattr(m, "name", "tool")
        content = str(getattr(m, "content", ""))
        captured.setdefault(name, []).append(content)
        with st.expander(f"🔧 {name} → result", expanded=False):
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
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif;">
      <button id="cs-copy-btn" style="
          background:linear-gradient(92deg,#7C6BFF,#5A73FF); color:#fff; border:none;
          border-radius:10px; padding:9px 18px; font-size:14px; font-weight:600;
          cursor:pointer; margin-bottom:8px;">
        {btn_label}
      </button>
      <span id="cs-copy-msg" style="margin-left:10px; font-size:13px; color:#2ECC8F;"></span>
      <textarea id="cs-log-text" style="
          width:100%; height:240px; margin-top:6px; font-family:ui-monospace,Menlo,Consolas,monospace;
          font-size:12px; border:1px solid #232D3D; border-radius:10px;
          padding:12px; background:#141A24; color:#E8EDF5;"
          readonly>{__import__('html').escape(full_log)}</textarea>
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
            csMsg.style.color = "#2ECC8F";
            csMsg.textContent = {_json.dumps(copied)};
          }} catch (e) {{
            try {{
              csArea.select();
              document.execCommand("copy");
              csMsg.style.color = "#2ECC8F";
              csMsg.textContent = {_json.dumps(copied)};
            }} catch (e2) {{
              csMsg.style.color = "#F5B942";
              csMsg.textContent = {_json.dumps(failed)};
            }}
          }}
        }});
      </script>
    </div>
    """
    st.components.v1.html(html, height=340)


if not (investigate and address):
    render_empty_state()

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

    st.markdown(f"### {t('live_stream')}")
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
                emoji, label = node_label(node_name)
                elapsed = int(time.time() - t0)
                status.update(label=t("status_running",
                                      node=f"{emoji} {label}",
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
                                st.info(f"🧭 **{t('plan_prefix')}:**\n\n{c}")
                    elif node_name in ("reflect", "replan"):
                        st.markdown(
                            f"<div class='cs-phase'>{emoji} {label} · "
                            f"{t('step_word')} {case.step} · "
                            f"{t('risk_word')}≈{case.risk_estimate:.2f}</div>",
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
        st.markdown("---")
        st.markdown(f"## {t('results_title')}")
        overall = case.risk_estimate
        risk_class = "risk-high" if overall >= 0.6 else ("risk-med" if overall >= 0.3 else "risk-low")
        risk_label = t("risk_high") if overall >= 0.6 else (t("risk_med") if overall >= 0.3 else t("risk_low"))

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card {risk_class}"><h3>{t("card_risk")}</h3>'
                        f'<h1>{overall:.2f}</h1></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h3>{t("card_steps")}</h3>'
                        f'<h1 style="color:#7C6BFF">{case.step}</h1></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h3>{t("card_addrs")}</h3>'
                        f'<h1 style="color:#22D3EE">{len(case.visited)}</h1></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card {risk_class}"><h3>{t("card_verdict")}</h3>'
                        f'<h1>{risk_label}</h1></div>', unsafe_allow_html=True)

        bar_color = ("linear-gradient(90deg,#FF9A6B,#FF5C6C)" if overall >= 0.6 else
                     "linear-gradient(90deg,#FFD98A,#F5B942)" if overall >= 0.3 else
                     "linear-gradient(90deg,#22D3EE,#2ECC8F)")
        st.markdown(f"""
<div style="margin:18px 0 24px 0">
<div class="score-bar-bg"><div class="score-bar-fill" style="width:{overall*100:.1f}%; background:{bar_color}"></div></div>
<div class="cs-bar-legend"><span>{t("bar_safe")}</span><span>{t("bar_threshold")}</span><span>{t("bar_danger")}</span></div>
</div>
""", unsafe_allow_html=True)

        # ── Long-horizon run record ──
        st.markdown(f"### {t('run_record')}")
        st.caption(t("run_record_desc"))
        c_tl, c_usage = st.columns([1.6, 1])
        with c_tl:
            st.plotly_chart(run_trace.plot_timeline(case, lang=LANG), use_container_width=True)
        with c_usage:
            st.plotly_chart(run_trace.plot_tool_usage(case, lang=LANG), use_container_width=True)

        # ── Hypotheses + fund paths ──
        c_hyp, c_path = st.columns(2)
        with c_hyp:
            st.subheader(t("hypotheses"))
            st.caption(t("hypotheses_desc"))
            rows = run_trace.hypothesis_table(case)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info(t("no_hypotheses"))
        with c_path:
            st.subheader(t("fund_paths"))
            if case.suspicious_paths:
                for p in case.suspicious_paths[:8]:
                    arrow = " → ".join(a[:10] for a in p.path)
                    st.markdown(f"- `{p.direction}` {p.hops}hop ~{p.total_value_eth:.3f} ETH: {arrow}")
            else:
                st.info(t("no_fund_paths"))

        # ── Blind-label reconciliation ──
        recon = getattr(case, "reconciliation", None)
        if recon:
            st.markdown(f"### {t('recon_title')}")
            rc1, rc2 = st.columns(2)
            with rc1:
                bl = str(recon.get("blind_label", "")).upper()
                st.metric(t("recon_blind"), f"{recon.get('blind_risk', 0.0):.2f} ({bl})")
            with rc2:
                tl = recon.get("target_label")
                if tl:
                    st.metric(t("recon_sealed"),
                              f"{tl.get('name', '?')} [{tl.get('category') or '—'}]")
                else:
                    st.metric(t("recon_sealed"), "—")
            outcome = recon.get("outcome")
            if outcome == "agree":
                st.success(t("recon_agree"))
            elif outcome == "conflict":
                st.warning(t("recon_conflict"))
            elif outcome == "identity_revealed":
                st.info(t("recon_identity"))

        # ── Report + attestation ──
        st.markdown(f"### {t('final_report')}")
        c_rep, c_att = st.columns([1.5, 1])
        with c_rep:
            report_msgs = captured.get("compile_final_report", [])
            if report_msgs:
                st.markdown(report_msgs[-1])
        with c_att:
            att = captured.get("publish_attestation", [])
            if att:
                st.code(att[-1], language="text")
            else:
                st.info(t("no_attestation"))

        # ── Export run record ──
        st.markdown("---")
        if st.button(t("export_btn")):
            paths = run_trace.export_run_record(case)
            st.success(f"{t('saved')}: {paths['json']}  /  {paths['html']}")
        st.caption(f"{t('case_file')}: {case_path}")

        # ── One-click copy all logs ──
        st.markdown("---")
        st.markdown(f"### {t('copy_logs_title')}")
        full_log = build_full_log(case, captured)
        if full_log.strip():
            st.caption(t("copy_logs_hint"))
            render_copy_logs(full_log)
        else:
            st.info(t("no_logs"))

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div class='cs-footer'>ChainScope v2.0 &nbsp;·&nbsp; LangGraph agent + GB-TGAD + EAS Sepolia "
    "&nbsp;·&nbsp; <a href='https://github.com/SKYEJT/ChainScope'>GitHub</a></div>",
    unsafe_allow_html=True,
)
