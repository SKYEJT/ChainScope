"""Frontend (Streamlit) unit + visualization tests — NO API key / network needed.

This exercises the pure-logic and rendering layer that app.py depends on, so it
can run in CI / offline:

  1. i18n completeness  — every UI string has en+cn; every t("key") referenced
                          in app.py actually exists in TRANSLATIONS (catches the
                          classic "untranslated key silently falls back" bug).
  2. CaseFile model     — the object the results dashboard renders from.
  3. build_full_log()   — the "copy all logs" helper inside app.py.
  4. visualization.py   — radar / grain-ball / pyvis network figures.
  5. run_trace.py       — timeline / tool-usage / hypothesis table / export.

Run:
    python tests/test_frontend_units.py
"""
import os
import re
import sys
import tempfile
import warnings
from types import SimpleNamespace

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch
import plotly.graph_objects as go

PASS: list = []
FAIL: list = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append((name, detail))
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))


def section(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# ── 0. import app.py in bare mode (st.* become no-ops + warnings) ──────────────
section("0. Import app.py (bare mode, no Streamlit runtime)")
try:
    import app  # noqa: E402
    check("import app.py succeeds (no syntax / dep / module-init error)", True)
except Exception as e:  # pragma: no cover
    check("import app.py succeeds", False, repr(e))
    print(f"\nResults: {len(PASS)} passed, {len(FAIL)} failed (fatal)")
    sys.exit(1)

from chainscope.utils import run_trace  # noqa: E402
from chainscope.utils.visualization import (  # noqa: E402
    plot_radar_plotly, build_network_html, plot_grain_balls_plotly,
)
from chainscope.agent.case_file import new_case  # noqa: E402
from chainscope.agent.prompts import DEFAULT_GOAL, SYSTEM_PROMPT  # noqa: E402


# ── 1. i18n completeness ──────────────────────────────────────────────────────
section("1. i18n completeness")
T = app.TRANSLATIONS
incomplete = [k for k, v in T.items() if not v.get("en") or not v.get("cn")]
check(f"every TRANSLATIONS key ({len(T)}) has non-empty en + cn",
      not incomplete, f"incomplete: {incomplete}")

bad_nodes = [k for k, v in app.NODE_LABEL_I18N.items()
             if "en" not in v or "cn" not in v
             or len(v["en"]) != 2 or len(v["cn"]) != 2]
check("every NODE_LABEL_I18N entry has (emoji,label) for en+cn",
      not bad_nodes, f"bad: {bad_nodes}")

src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
used_keys = set(re.findall(r"\bt\(\s*[\"']([^\"']+)[\"']", src))
undefined = sorted(used_keys - set(T.keys()))
check(f"all {len(used_keys)} t(\"...\") keys used in app.py are defined",
      not undefined, f"undefined: {undefined}")

ct = run_trace._CHART_TEXT
bad_ct = [k for k, v in ct.items() if not v.get("en") or not v.get("cn")]
check("run_trace._CHART_TEXT keys have en+cn", not bad_ct, f"bad: {bad_ct}")

check("t() returns a non-empty string", bool(app.t("results_title")))
check("t() applies .format() kwargs", "5" in app.t("complete", steps=5, addrs=3))
check("DEFAULT_GOAL exposes {address} placeholder", "{address}" in DEFAULT_GOAL)
check("SYSTEM_PROMPT is non-empty", bool(SYSTEM_PROMPT.strip()))


# ── 2. CaseFile data model ────────────────────────────────────────────────────
section("2. CaseFile data model (drives the results dashboard)")
case = new_case("0xABCdef0000000000000000000000000000000001", "test goal")
check("new_case sets target + goal", case.target.startswith("0xABC") and case.goal == "test goal")
check("initial step == 0", case.step == 0)
check("next_step() increments", case.next_step() == 1 and case.step == 1)
case.log_step("plan", "first plan: collect txs then build graph")
case.log_step("observe", "get_transactions: 12 txs")
check("log_step appends to iteration_log", len(case.iteration_log) == 2)
h = case.add_hypothesis("address is a mixer", confidence=0.4)
check("add_hypothesis -> active", h.status == "active" and len(case.hypotheses) == 1)
case.revise_hypothesis(h.id, "confirmed", "fan-in pattern", confidence=0.8)
check("revise_hypothesis updates status + history",
      case.hypotheses[0].status == "confirmed" and len(case.hypotheses[0].history) == 2)
case.add_path("outflow", ["0xaaa", "0xbbb", "0xccc"], total_value_eth=1.5)
check("add_path computes hops", case.suspicious_paths[0].hops == 2)
case.set_risk(1.7)
check("set_risk clamps into [0,1]", case.risk_estimate == 1.0)
d = case.to_dict()
check("to_dict has required keys",
      all(k in d for k in ["target", "goal", "risk_estimate", "hypotheses",
                           "iteration_log", "suspicious_paths"]))
with tempfile.TemporaryDirectory() as tmp:
    check("save() writes a json file", os.path.exists(case.save(directory=tmp)))


# ── 3. build_full_log() (copy-all-logs helper) ───────────────────────────────
section("3. build_full_log()")
case.close("HIGH RISK: likely mixer")
log = app.build_full_log(case, {"get_transactions": ["12 txs found"],
                                 "detect_anomaly": ["score=0.81"]})
check("log includes target address", case.target in log)
check("log includes iteration steps", "first plan" in log)
check("log includes captured tool outputs", "12 txs found" in log and "score=0.81" in log)
check("log includes the verdict", "HIGH RISK" in log)


# ── 4. visualization.py figures ──────────────────────────────────────────────
section("4. visualization.py (radar / grain-balls / network)")
snap = SimpleNamespace(
    node_id=["0x1111111111aa", "0x2222222222bb", "0x3333333333cc"],
    x=torch.randn(3, 38),
    edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    n_nodes=3, capped=False, orig_n_nodes=3,
)
scores = np.array([0.10, 0.72, 0.45])

fig = plot_radar_plotly({"s_attr": 0.2, "s_struct": 0.6, "s_flow": 0.4})
check("plot_radar_plotly -> Figure with traces",
      isinstance(fig, go.Figure) and len(fig.data) >= 1)

fig2 = plot_grain_balls_plotly(snap, scores)
check("plot_grain_balls_plotly -> Figure", isinstance(fig2, go.Figure))

empty = plot_grain_balls_plotly(snap, np.array([]))
check("plot_grain_balls_plotly handles empty input gracefully",
      isinstance(empty, go.Figure) and "No nodes" in (empty.layout.title.text or ""))

try:
    html = build_network_html(snap, scores)
    check("build_network_html -> non-empty html string",
          isinstance(html, str) and len(html) > 200)
except Exception as e:
    check("build_network_html -> non-empty html string", False, repr(e))


# ── 5. run_trace.py figures + export ─────────────────────────────────────────
section("5. run_trace.py (timeline / usage / table / export)")
check("plot_timeline -> Figure", isinstance(run_trace.plot_timeline(case, lang="cn"), go.Figure))
check("plot_tool_usage -> Figure", isinstance(run_trace.plot_tool_usage(case, lang="cn"), go.Figure))
rows = run_trace.hypothesis_table(case)
check("hypothesis_table -> rows with status",
      isinstance(rows, list) and len(rows) == 1 and "status" in rows[0])
with tempfile.TemporaryDirectory() as tmp:
    paths = run_trace.export_run_record(case, directory=tmp)
    check("export_run_record writes json + html",
          os.path.exists(paths["json"]) and os.path.exists(paths["html"]))


# ── summary ──────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"  PASSED: {len(PASS)}")
print(f"  FAILED: {len(FAIL)}")
for name, detail in FAIL:
    print(f"    - {name}: {detail}")
sys.exit(0 if not FAIL else 1)
