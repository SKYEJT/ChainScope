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

# node -> (label, color). A restrained palette that matches the monochrome UI:
# the model's own moves are light, data is blue, corrections are warm.
NODE_STYLE = {
    "plan": ("Plan", "#8A93A6"),
    "act": ("Act", "#E8EAEE"),
    "observe": ("Observe", "#7FB6FF"),
    "reflect": ("Reflect", "#E3AE5C"),
    "replan": ("Replan", "#F0716B"),
    "report": ("Report", "#55CFA4"),
}

_BG = "#0B0C0F"
_TEXT = "#E8EAEE"
_MUTED = "#8A93A6"
_FAINT = "#6A7280"
_LINE = "#1C1F26"
_FONT = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
_AXIS = dict(gridcolor=_LINE, zerolinecolor=_LINE, linecolor=_LINE, showline=False,
             tickfont=dict(color=_FAINT, size=11, family=_FONT), title_font=dict(color=_FAINT, size=11))


def _style_dark(fig: go.Figure) -> go.Figure:
    """Make a chart sit quietly on the dark UI: transparent paper, hairline grid,
    small mono type, a muted left-aligned title.

    Applied after the per-chart update_layout: plotly rejects a container and
    its magic-underscore child (title / title_font) in the same call.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_MUTED, size=11, family=_FONT),
        title_font=dict(color=_MUTED, size=12, family=_FONT),
        title_x=0,
        title_xanchor="left",
        hoverlabel=dict(bgcolor="#191C22", bordercolor="#30343E",
                        font=dict(color=_TEXT, size=12, family=_FONT)),
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
    "timeline_title": {"en": "RUN TIMELINE", "cn": "运行时间线"},
    "step": {"en": "step", "cn": "步骤"},
    "tool_usage_title": {"en": "TOOL USAGE", "cn": "工具调用统计"},
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
            marker=dict(size=10, color=color, line=dict(width=0)),
            name=label,
            text=[s.content[:160] for s in pts],
            hovertemplate="step %{x}<br>%{text}<extra></extra>",
        ))
    fig.update_layout(
        title=_ct("timeline_title", lang),
        xaxis_title=_ct("step", lang),
        height=300,
        margin=dict(l=10, r=10, t=44, b=10),
        showlegend=False,
    )
    _style_dark(fig)
    fig.update_yaxes(categoryorder="array",
                     categoryarray=[NODE_STYLE[n][0] for n in lanes],
                     showgrid=False, tickfont=dict(color=_MUTED, size=12, family=_FONT))
    fig.update_xaxes(dtick=1 if case.step <= 20 else None)
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
        marker_color="#C9CDD4",
        width=0.5,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=_ct("tool_usage_title", lang),
        xaxis_title=_ct("calls", lang),
        height=max(220, 26 * len(counts) + 60),
        margin=dict(l=10, r=10, t=44, b=10),
        bargap=0.35,
    )
    _style_dark(fig)
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     tickfont=dict(color=_MUTED, size=12, family=_FONT))
    fig.update_xaxes(dtick=1)
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
<style>
body {{ font-family: Geist, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 1000px; margin: 40px auto; padding: 0 24px;
       background: {_BG}; color: {_TEXT}; line-height: 1.55; }}
h1, h2 {{ letter-spacing: -.02em; font-weight: 600; }}
h1 {{ font-size: 30px; margin: 0 0 6px 0; }}
h2 {{ font-size: 13px; letter-spacing: .14em; text-transform: uppercase; color: {_FAINT};
      margin-top: 40px; border-top: 1px solid #23262E; padding-top: 18px; }}
.meta {{ color: {_MUTED}; font-family: {_FONT}; font-size: 13px; }}
pre {{ white-space: pre-wrap; background: #13151A; border: 1px solid #23262E;
      padding: 14px 16px; border-radius: 8px; color: {_TEXT}; font-family: {_FONT}; font-size: 12.5px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #23262E; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ color: {_FAINT}; font-weight: 500; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }}
td:first-child, td:nth-child(2) {{ font-family: {_FONT}; color: {_MUTED}; white-space: nowrap; }}
</style>
<body>
<h1>ChainScope Run Record</h1>
<p class="meta">Target {case.target}<br>
Final risk {case.risk_estimate:.2f} &nbsp;·&nbsp; Steps {case.step} &nbsp;·&nbsp;
Addresses investigated {len(case.visited)}</p>
<h2>Plan</h2><pre>{_esc(plan_text(case))}</pre>
<h2>Timeline</h2>{timeline}
<h2>Tool usage</h2>{usage}
<h2>Verdict</h2><pre>{_esc(case.verdict)}</pre>
<h2>Iteration log</h2>
<table><tr><th>Step</th><th>Node</th><th>Content</th></tr>{rows}</table>
</body>"""
    html_path = directory / f"{base}.html"
    html_path.write_text(html)
    return {"json": str(json_path), "html": str(html_path)}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
