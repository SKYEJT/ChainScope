"""Frontend UI tests using Streamlit's official AppTest harness — NO API key.

AppTest runs app.py headless (a real ScriptRunContext, but no browser) and lets
us assert what the UI renders and how it reacts to input. We verify:

  - the app boots with no uncaught exception,
  - the core controls exist (model / window / steps / address / buttons),
  - the EN<->中文 language toggle actually re-renders localized text.

We deliberately DO NOT click "Investigate" or "Probe activity span": those hit
Alchemy/Etherscan/Z.AI and the live agent. That belongs to the manual E2E layer.

Run:
    python tests/test_frontend_apptest.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
APP = ROOT + "/app.py"

from streamlit.testing.v1 import AppTest  # noqa: E402

PASS: list = []
FAIL: list = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if (detail and not cond) else ""))


def all_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


# ── initial render ───────────────────────────────────────────────────────────
print("=" * 64); print("Streamlit AppTest — initial headless render"); print("=" * 64)
at = AppTest.from_file(APP, default_timeout=180)
at.run()
check("initial run raises no exception", not at.exception, str(at.exception))

md = all_markdown(at)
check("renders 'ChainScope' branding", "ChainScope" in md)
check("model selectbox present w/ glm-4.7 default",
      len(at.selectbox) >= 1 and "glm-4.7" in list(at.selectbox[0].options))
check("time-window number_input present", len(at.number_input) >= 1)
check("max-steps + reflect sliders present", len(at.slider) >= 2)
check("address text_input present", len(at.text_input) >= 1)
check("investigate + probe buttons present", len(at.button) >= 2)


# ── language toggle EN -> 中文 ────────────────────────────────────────────────
print("\n" + "=" * 64); print("Language toggle  EN -> 中文"); print("=" * 64)
at.radio[0].set_value("中文").run()
check("language switch raises no exception", not at.exception, str(at.exception))
md_cn = all_markdown(at)
check("localized Chinese string appears after switch",
      ("自主链上调查" in md_cn) or ("由 GLM 驱动" in md_cn), md_cn[:160])


# ── enter an address (without launching the agent) ───────────────────────────
print("\n" + "=" * 64); print("Type an address (no Investigate click)"); print("=" * 64)
at.text_input[0].set_value("0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe").run()
check("entering an address raises no exception", not at.exception, str(at.exception))


# ── summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64); print("SUMMARY"); print("=" * 64)
print(f"  PASSED: {len(PASS)}")
print(f"  FAILED: {len(FAIL)}")
for n, d in FAIL:
    print(f"    - {n}: {d}")
sys.exit(0 if not FAIL else 1)
