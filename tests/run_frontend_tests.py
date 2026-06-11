"""One-shot frontend test runner — runs all automated suites + prints a summary.

NO API key / network needed. This is the single command to run before a demo or
commit to confirm the whole frontend (logic, UI, and the full investigate flow)
is green.

    python tests/run_frontend_tests.py

It runs, in order:
    1. test_frontend_units.py       — pure logic + i18n + visualization objects
    2. test_frontend_apptest.py     — Streamlit AppTest, initial headless render
    3. test_frontend_e2e_mock.py    — Streamlit AppTest, full Investigate flow (mock agent)

Exit code is 0 only if every suite passes.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SUITES = [
    ("Unit / logic + visualization", "test_frontend_units.py"),
    ("AppTest headless UI (initial render)", "test_frontend_apptest.py"),
    ("AppTest end-to-end (mocked agent)", "test_frontend_e2e_mock.py"),
]

# Streamlit prints a lot of harmless noise in bare/headless mode; hide it.
_NOISE = (
    "ScriptRunContext", "bare mode", "Session state does not", "streamlit run",
    "to view this", "Please replace", "will be removed", "use_container_width",
    "components.v1.html", "st.iframe", "For `use_container", "Warning:",
)


def run(title, fname) -> bool:
    print("\n" + "#" * 70)
    print(f"# {title}")
    print(f"#   {fname}")
    print("#" * 70)
    r = subprocess.run([PY, os.path.join(HERE, fname)],
                       capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if any(s in line for s in _NOISE):
            continue
        print(line)
    return r.returncode == 0


def main():
    results = [(title, run(title, fname)) for title, fname in SUITES]

    print("\n" + "=" * 70)
    print("FRONTEND TEST RUNNER — OVERALL")
    print("=" * 70)
    for title, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {title}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n  {n_ok}/{len(results)} suites passed.")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
