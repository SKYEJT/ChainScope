"""Long-horizon run-record visualization.

Turns a CaseFile's iteration log into the artifacts the hackathon track asks for:
a task-decomposition view (the agent's plan), a tool-call timeline (proving the
run is long and tool-driven), and a tool-usage breakdown. Also exports a
self-contained HTML/JSON run record.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import plotly.graph_objects as go

# node -> (label, color)
NODE_STYLE = {
    "plan": ("Plan", "#3182ce"),
    "act": ("Act", "#805ad5"),
    "observe": ("Observe", "#38a169"),
    "reflect": ("Reflect", "#d69e2e"),
    "replan": ("Replan", "#e53e3e"),
    "report": ("Report", "#319795"),
}


def tool_call_rows(case) -> list[dict]:
    """Flatten the iteration log into rows for tables/timelines."""
    rows = []
    for s in case.iteration_log:
        rows.append({
            "step": s.step,
            "node": s.node,
            "content": s.content,
        })
    return rows


# Localized chart strings: key -> {lang: text}
_CHART_TEXT = {
    "timeline_title": {"en": "Long-Horizon Run Timeline", "cn": "长程运行时间线"},
    "step": {"en": "Step", "cn": "步骤"},
    "tool_usage_title": {"en": "Tool Usage", "cn": "工具调用统计"},
    "calls": {"en": "calls", "cn": "调用次数"},
}


def _ct(key: str, lang: str = "en") -> str:
    return _CHART_TEXT.get(key, {}).get(lang, _CHART_TEXT.get(key, {}).get("en", key))


def plot_timeline(case, lang: str = "en") -> go.Figure:
    """Scatter timeline of agent activity: x=step, y=node lane, hover=content."""
    lanes = list(NODE_STYLE.keys())
    fig = go.Figure()
    for node in lanes:
        pts = [s for s in case.iteration_log if s.node == node]
        if not pts:
            continue
        label, color = NODE_STYLE[node]
        fig.add_trace(go.Scatter(
            x=[s.step for s in pts],
            y=[label] * len(pts),
            mode="markers",
            marker=dict(size=12, color=color, line=dict(width=1, color="white")),
            name=label,
            text=[s.content[:160] for s in pts],
            hovertemplate="step %{x}<br>%{text}<extra></extra>",
        ))
    fig.update_layout(
        title=_ct("timeline_title", lang),
        xaxis_title=_ct("step", lang),
        yaxis=dict(categoryorder="array", categoryarray=[NODE_STYLE[n][0] for n in lanes]),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        plot_bgcolor="#f8f9fa",
    )
    return fig


def plot_tool_usage(case, lang: str = "en") -> go.Figure:
    """Bar chart of how often each tool was invoked (from observe entries)."""
    counts: dict[str, int] = {}
    for s in case.iteration_log:
        if s.node != "observe":
            continue
        tool = s.content.split(":", 1)[0].strip()
        counts[tool] = counts.get(tool, 0) + 1
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    fig = go.Figure(go.Bar(
        x=list(counts.values()),
        y=list(counts.keys()),
        orientation="h",
        marker_color="#805ad5",
    ))
    fig.update_layout(
        title=_ct("tool_usage_title", lang),
        xaxis_title=_ct("calls", lang),
        height=max(220, 26 * len(counts) + 60),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#f8f9fa",
    )
    return fig


def plan_text(case) -> str:
    """Return the agent's initial plan (first plan-node log), if any."""
    for s in case.iteration_log:
        if s.node == "plan":
            return s.content
    return "(no explicit plan recorded)"


def hypothesis_table(case) -> list[dict]:
    rows = []
    for h in case.hypotheses:
        rows.append({
            "id": h.id,
            "status": h.status,
            "confidence": round(h.confidence, 2),
            "statement": h.statement,
            "revisions": len(h.history) - 1,
        })
    return rows


def export_run_record(case, directory: Path | str = "data/run_records") -> dict:
    """Write JSON + a simple standalone HTML run record. Returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{case.target[:10]}_{int(time.time())}"

    json_path = directory / f"{base}.json"
    json_path.write_text(json.dumps(case.to_dict(), indent=2, ensure_ascii=False))

    timeline = plot_timeline(case).to_html(full_html=False, include_plotlyjs="cdn")
    usage = plot_tool_usage(case).to_html(full_html=False, include_plotlyjs=False)
    rows = "".join(
        f"<tr><td>{r['step']}</td><td>{r['node']}</td><td>{_esc(r['content'])}</td></tr>"
        for r in tool_call_rows(case)
    )
    html = f"""<!doctype html><meta charset="utf-8">
<title>ChainScope Run Record — {case.target}</title>
<body style="font-family:system-ui;max-width:1000px;margin:24px auto;color:#1a202c">
<h1>ChainScope Run Record</h1>
<p><b>Target:</b> {case.target}<br>
<b>Final risk:</b> {case.risk_estimate:.2f} &nbsp; <b>Steps:</b> {case.step} &nbsp;
<b>Addresses investigated:</b> {len(case.visited)}</p>
<h2>Plan</h2><pre style="white-space:pre-wrap;background:#f7fafc;padding:12px;border-radius:8px">{_esc(plan_text(case))}</pre>
<h2>Timeline</h2>{timeline}
<h2>Tool Usage</h2>{usage}
<h2>Verdict</h2><pre style="white-space:pre-wrap;background:#f7fafc;padding:12px;border-radius:8px">{_esc(case.verdict)}</pre>
<h2>Iteration Log</h2>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
<tr><th>Step</th><th>Node</th><th>Content</th></tr>{rows}</table>
</body>"""
    html_path = directory / f"{base}.html"
    html_path.write_text(html)
    return {"json": str(json_path), "html": str(html_path)}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
