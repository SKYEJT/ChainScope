"""Render every frontend visualization to tests/_artifacts/ — NO API key / network.

The unit tests only assert "a Figure/HTML object came back"; they can't tell you
whether a chart actually *looks* right. This script builds a synthetic CaseFile +
graph snapshot and writes each artifact the dashboard produces to disk, so you can
open them in a browser to eyeball the visuals and reuse them as demo assets:

    radar.html         — 4-component anomaly radar (Plotly)
    grain_balls.html   — grain-ball membership scatter (Plotly)
    network.html       — interactive transaction graph (pyvis)
    timeline.html      — long-horizon run timeline (Plotly)
    tool_usage.html    — tool-call usage bar chart (Plotly)
    full_log.txt       — the one-click "copy all logs" blob
    0x..._<ts>.json    — exported run record (JSON)
    0x..._<ts>.html    — exported run record (standalone HTML)

Run:
    python tests/export_frontend_artifacts.py
"""
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

import app  # noqa: E402  (also validates app.py imports cleanly)
from chainscope.agent.case_file import new_case  # noqa: E402
from chainscope.utils import run_trace  # noqa: E402
from chainscope.utils.visualization import (  # noqa: E402
    plot_radar, plot_radar_plotly, plot_grain_balls_plotly, build_network_html,
)

OUT = os.path.join(ROOT, "tests", "_artifacts")


def build_demo_case():
    """A synthetic but realistic closed investigation (drives the run-record charts)."""
    case = new_case("0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe",
                    "Detect illicit activity & attest on-chain")
    case.log_step("plan", "Plan: pull txs, build graph, detect anomaly, trace funds, attest.")
    h = case.add_hypothesis("Target is a mixer intermediary (fan-in / fan-out)", confidence=0.4)
    timeline = [
        ("get_transactions",     "12 transactions found (fan-in from 9 addrs)."),
        ("detect_anomaly",       "overall score = 0.81 (s_struct = 0.72 dominant)."),
        ("trace_fund_flow",      "outflow 2 hops ~12.3 ETH into a mixer deposit."),
        ("lookup_address_label", "destination = Tornado Cash router (mixer)."),
        ("detect_anomaly",       "re-check on neighbor: score = 0.66."),
        ("compile_final_report", "HIGH RISK report compiled."),
        ("publish_attestation",  "EAS uid=0xATTEST_MOCK cid=bafy_mock."),
    ]
    for tool, out in timeline:
        case.next_step(); case.log_step("act", f"calling {tool}")
        case.next_step(); case.log_step("observe", f"{tool}: {out}")
    case.log_step("reflect", "reflection checkpoint")
    case.revise_hypothesis(h.id, "confirmed",
                           "fan-in/out + mixer destination confirm mixing", confidence=0.85)
    case.add_path("outflow",
                  ["0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe",
                   "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                   "0xcccccccccccccccccccccccccccccccccccccccc"],
                  total_value_eth=12.3)
    for a in ("0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe",
              "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "0xcccccccccccccccccccccccccccccccccccccccc"):
        case.mark_visited(a)
    case.set_risk(0.82)
    case.close("HIGH RISK: mixer intermediary — fan-in/out + confirmed Tornado Cash outflow.")
    return case


def build_demo_snapshot(n=18):
    """A synthetic PyG-like snapshot with 38-dim features and a ring + hub topology."""
    node_id = [f"0x{(i * 0x1111):040x}" for i in range(n)]
    src = list(range(n)) + [0] * (n - 1)          # ring + a hub at node 0
    dst = [(i + 1) % n for i in range(n)] + list(range(1, n))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return SimpleNamespace(node_id=node_id, x=torch.randn(n, 38),
                           edge_index=edge_index, n_nodes=n, capped=False, orig_n_nodes=n)


def write(name, content):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] {name:<22} ({len(content):>8,} bytes)  ->  {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    case = build_demo_case()
    snap = build_demo_snapshot()
    rng = np.random.default_rng(7)
    scores = np.clip(np.abs(rng.standard_normal(snap.n_nodes)) * 0.45, 0, 1)
    scores[0] = 0.88  # make the hub clearly anomalous

    print("=" * 70)
    print("Exporting frontend artifacts")
    print("=" * 70)

    write("radar.html", plot_radar_plotly(
        {"s_attr": 0.30, "s_struct": 0.72, "s_flow": 0.45, "s_temp": 0.20}
    ).to_html(include_plotlyjs="cdn"))
    write("grain_balls.html",
          plot_grain_balls_plotly(snap, scores).to_html(include_plotlyjs="cdn"))
    write("network.html", build_network_html(snap, scores))
    write("timeline.html",
          run_trace.plot_timeline(case, lang="cn").to_html(include_plotlyjs="cdn"))
    write("tool_usage.html",
          run_trace.plot_tool_usage(case, lang="cn").to_html(include_plotlyjs="cdn"))
    write("full_log.txt", app.build_full_log(
        case, {"get_transactions": ["12 txs found"], "detect_anomaly": ["score=0.81"]}))

    paths = run_trace.export_run_record(case, directory=OUT)
    print(f"  [OK] run record (JSON)       ->  {paths['json']}")
    print(f"  [OK] run record (HTML)       ->  {paths['html']}")

    # ── static PNG snapshots (for chat/slides; English titles avoid missing CJK fonts) ──
    print("\n  PNG snapshots:")
    plot_radar({"s_attr": 0.30, "s_struct": 0.72, "s_flow": 0.45, "s_temp": 0.20}).savefig(
        os.path.join(OUT, "radar_mpl.png"), dpi=110, bbox_inches="tight")
    print(f"  [OK] radar_mpl.png           ->  {os.path.join(OUT, 'radar_mpl.png')}")
    try:
        import plotly.io as pio
        png_specs = [
            ("radar.png", plot_radar_plotly(
                {"s_attr": 0.30, "s_struct": 0.72, "s_flow": 0.45, "s_temp": 0.20}), 520, 420),
            ("grain_balls.png", plot_grain_balls_plotly(snap, scores), 760, 470),
            ("timeline.png", run_trace.plot_timeline(case, lang="en"), 860, 320),
            ("tool_usage.png", run_trace.plot_tool_usage(case, lang="en"), 640, 360),
        ]
        for name, fig, w, h in png_specs:
            pio.write_image(fig, os.path.join(OUT, name), format="png", width=w, height=h, scale=2)
            print(f"  [OK] {name:<22} ->  {os.path.join(OUT, name)}")
    except Exception as e:
        print(f"  [..] plotly PNG export skipped (pip install kaleido to enable): {repr(e)[:80]}")

    print("\nDone. Open the .html files in a browser to eyeball the visuals "
          "(or attach the PNGs to the demo/slides).")


if __name__ == "__main__":
    main()
