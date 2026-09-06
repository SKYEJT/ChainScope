"""Frontend END-TO-END UI test with a MOCKED agent — NO API key / network.

The existing AppTest suite (test_frontend_apptest.py) deliberately stops *before*
clicking Investigate, so the entire results dashboard — metric cards, run-record
timeline, hypotheses table, traced fund paths, final report, attestation and the
copy-all-logs panel (app.py lines ~384-567) — was never exercised by any test.

This test fills that gap. It swaps the real LangGraph agent for a deterministic
fake that streams a realistic PLAN -> ACT -> OBSERVE -> REFLECT -> REPLAN run and
populates the active CaseFile exactly like the memory tools would, then drives
Streamlit's AppTest harness to actually click "Investigate" and asserts the whole
results UI renders with no uncaught exception.

The only networked call left on the results path (build_snapshot for the graph
evidence block) is patched to fail, so app.py's try/except skips it offline.

Run:
    python tests/test_frontend_e2e_mock.py
"""
import os
import sys
import warnings
from unittest.mock import patch

warnings.filterwarnings("ignore")
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
APP = ROOT + "/app.py"
# Keep the optional access gate out of the way even when the local .env sets
# APP_PASSWORD (load_dotenv never overrides a variable that already exists).
os.environ.setdefault("APP_PASSWORD", "")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import chainscope.agent.graph as graph_mod  # noqa: E402
import chainscope.tools.graph_builder_tool as gbt  # noqa: E402
from chainscope.agent.case_file import get_active_case  # noqa: E402

ADDR = "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"
DST = "0xcccccccccccccccccccccccccccccccccccccccc"
MID = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

PASS: list = []
FAIL: list = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if (detail and not cond) else ""))


class _FakeGraph:
    """Stand-in for the compiled LangGraph; streams a scripted long-horizon run
    and mutates the active CaseFile the way the real memory tools would."""

    def stream(self, init, config=None, stream_mode=None):
        case = get_active_case()

        def act(text, tool, args):
            if case is not None:
                case.log_step("act", text)
            return {"act": {"messages": [AIMessage(
                content=text, tool_calls=[{"name": tool, "args": args, "id": "call_" + tool}])]}}

        def obs(name, content):
            if case is not None:
                case.next_step()
                case.log_step("observe", f"{name}: {content}")
            return {"observe": {"messages": [
                ToolMessage(content=content, name=name, tool_call_id="call_" + name)]}}

        # ── PLAN ──
        h = None
        if case is not None:
            case.log_step("plan", "Plan: 1) pull txs 2) build graph 3) detect 4) trace funds 5) attest")
            h = case.add_hypothesis("Target is a mixer intermediary (fan-in / fan-out)", confidence=0.4)
        yield {"plan": {"messages": [AIMessage(
            content="My plan: pull txs, build the graph, run detection, follow the money, then attest.")]}}

        # ── ACT / OBSERVE cycles ──
        yield act("Pulling recent transactions for the target.", "get_transactions", {"address": ADDR})
        if case is not None:
            case.mark_visited(ADDR)
        yield obs("get_transactions", "12 transactions found (fan-in from 9 addresses, fan-out to 2).")

        yield act("Building the address graph and scoring anomalies.", "detect_anomaly", {"address": ADDR})
        if case is not None:
            case.mark_visited(MID)
        yield obs("detect_anomaly", "overall anomaly score = 0.81 (s_struct = 0.72 dominant).")

        # ── REFLECT ──
        if case is not None:
            case.log_step("reflect", "reflection checkpoint")
        yield {"reflect": {"messages": [AIMessage(
            content="Structural score is high — trace the largest outflow next.")]}}

        yield act("Tracing the largest outflow across hops.", "trace_fund_flow", {"address": ADDR})
        if case is not None:
            case.mark_visited(DST)
            case.add_path("outflow", [ADDR, MID, DST], total_value_eth=12.3)
        yield obs("trace_fund_flow", "outflow 2 hops ~12.3 ETH into a known mixer deposit address.")

        yield act("Labeling the destination address.", "lookup_address_label", {"address": DST})
        yield obs("lookup_address_label", f"{DST[:10]}... = Tornado Cash router (mixer).")

        # ── self-correct + risk estimate ──
        if case is not None and h is not None:
            case.revise_hypothesis(h.id, "confirmed",
                                   "fan-in/out + mixer destination confirm mixing", confidence=0.85)
            case.set_risk(0.82)

        # ── REPLAN / wrap-up ──
        if case is not None:
            case.log_step("replan", "wrap-up forced (step budget low)")
        yield {"replan": {"messages": [AIMessage(content="Concluding: compile report and attest.")]}}

        yield act("Compiling the final report.", "compile_final_report", {})
        yield obs("compile_final_report",
                  "## Verdict: HIGH RISK\nFan-in/out structure with a confirmed Tornado Cash outflow (~12.3 ETH).")

        yield act("Publishing the attestation on-chain.", "publish_attestation", {})
        yield obs("publish_attestation",
                  "EAS attestation uid=0xATTEST_MOCK  IPFS cid=bafy_mock_cid  (Sepolia).")

        if case is not None:
            case.close("HIGH RISK: mixer intermediary — fan-in/out + confirmed Tornado Cash outflow.")


def _fake_build_graph(*args, **kwargs):
    return _FakeGraph()


def alltext(at) -> str:
    """Collect text from every element type the dashboard writes to."""
    parts: list = []
    for attr in ("markdown", "title", "header", "subheader", "caption",
                 "text", "code", "info", "success", "warning", "error"):
        coll = getattr(at, attr, None)
        if coll is None:
            continue
        try:
            parts += [getattr(e, "value", "") or "" for e in coll]
        except Exception:
            pass
    return "\n".join(parts)


print("=" * 64)
print("Streamlit AppTest — full Investigate flow (mocked agent, offline)")
print("=" * 64)

at = AppTest.from_file(APP, default_timeout=120)

with patch.object(graph_mod, "build_investigation_graph", _fake_build_graph), \
     patch.object(gbt._builder, "build_snapshot",
                  side_effect=RuntimeError("offline test: snapshot skipped")):
    at.run()                  # initial render -> populates the element tree
    at.text_input[0].set_value(ADDR)
    at.button[0].click()      # 0 = Investigate, 1 = Probe activity span
    at.run()                  # re-run with Investigate clicked -> drives the fake agent

txt = alltext(at)

check("Investigate flow raises no exception", not at.exception, str(at.exception))
check("results dashboard renders", "Investigation Results" in txt, txt[:200])
check("success banner shows step + address count", "Investigation complete" in txt)
check("risk score 0.82 shown on a metric card", "0.82" in txt)
check("HIGH RISK verdict shown", "HIGH RISK" in txt)
check("run-record section rendered", "Long-Horizon Run Record" in txt)
check("hypotheses table rendered", len(at.dataframe) >= 1, f"dataframes={len(at.dataframe)}")
check("traced fund path shown", "outflow" in txt.lower())
check("final report markdown shown", "Tornado" in txt or "Verdict" in txt)
check("attestation block rendered", "Final Report" in txt or "attestation" in txt.lower())
check("copy-all-logs panel rendered", "copy all logs" in txt.lower())

print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
print(f"  PASSED: {len(PASS)}")
print(f"  FAILED: {len(FAIL)}")
for n, d in FAIL:
    print(f"    - {n}: {d}")
sys.exit(0 if not FAIL else 1)
