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

# node -> (label, color). Colors match the dark UI palette in app.py.
NODE_STYLE = {
    "plan": ("Plan", "#5B8DEF"),
    "act": ("Act", "#7C6BFF"),
    "observe": ("Observe", "#2ECC8F"),
    "reflect": ("Reflect", "#F5B942"),
    "replan": ("Replan", "#FF5C6C"),
    "report": ("Report", "#22D3EE"),
}

_AXIS = dict(gridcolor="#232D3D", zerolinecolor="#232D3D", linecolor="#232D3D")


def _style_dark(fig: go.Figure) -> go.Figure:
    """Make a chart sit on the dark UI instead of glaring off it.

    Applied after the per-chart update_layout: plotly rejects a container and
    its magic-underscore child (title / title_font) in the same call.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#93A1B5", size=12),
        title_font=dict(color="#E8EDF5", size=15),
        hoverlabel=dict(bgcolor="#1A2230", bordercolor="#35435C",
                        font=dict(color="#E8EDF5", size=12)),
    )
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


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
            marker=dict(size=12, color=color, line=dict(width=1, color="#0A0C12")),
            name=label,
            text=[s.content[:160] for s in pts],
            hovertemplate="step %{x}<br>%{text}<extra></extra>",
        ))
    fig.update_layout(
        title=_ct("timeline_title", lang),
        xaxis_title=_ct("step", lang),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    _style_dark(fig)
    fig.update_yaxes(categoryorder="array",
                     categoryarray=[NODE_STYLE[n][0] for n in lanes])
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
        marker_color="#7C6BFF",
    ))
    fig.update_layout(
        title=_ct("tool_usage_title", lang),
        xaxis_title=_ct("calls", lang),
        height=max(220, 26 * len(counts) + 60),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return _style_dark(fig)


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
<style>
body {{ font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC", sans-serif;
       max-width: 1000px; margin: 24px auto; padding: 0 20px;
       background: #0A0C12; color: #E8EDF5; }}
h1, h2 {{ letter-spacing: -.02em; }}
h2 {{ margin-top: 34px; border-bottom: 1px solid #232D3D; padding-bottom: 8px; }}
pre {{ white-space: pre-wrap; background: #141A24; border: 1px solid #232D3D;
      padding: 14px; border-radius: 10px; color: #E8EDF5; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #232D3D; padding: 7px 9px; text-align: left;
         vertical-align: top; }}
th {{ background: #1A2230; }}
</style>
<body>
<h1>ChainScope Run Record</h1>
<p><b>Target:</b> {case.target}<br>
<b>Final risk:</b> {case.risk_estimate:.2f} &nbsp; <b>Steps:</b> {case.step} &nbsp;
<b>Addresses investigated:</b> {len(case.visited)}</p>
<h2>Plan</h2><pre>{_esc(plan_text(case))}</pre>
<h2>Timeline</h2>{timeline}
<h2>Tool Usage</h2>{usage}
<h2>Verdict</h2><pre>{_esc(case.verdict)}</pre>
<h2>Iteration Log</h2>
<table><tr><th>Step</th><th>Node</th><th>Content</th></tr>{rows}</table>
</body>"""
    html_path = directory / f"{base}.html"
    html_path.write_text(html)
    return {"json": str(json_path), "html": str(html_path)}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
