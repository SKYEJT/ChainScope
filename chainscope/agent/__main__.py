"""CLI entry: python -m chainscope.agent [address]

Runs a full autonomous investigation and prints the resulting case summary.
"""
import sys

from chainscope.agent.graph import run_investigation


def main():
    address = sys.argv[1] if len(sys.argv) > 1 else "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    print(f"=== ChainScope autonomous investigation: {address} ===\n")
    out = run_investigation(address)
    case = out["case"]

    print("\n" + "=" * 60)
    print("CASE SUMMARY")
    print("=" * 60)
    print(case.summary())
    print(f"\nFinal risk: {case.risk_estimate:.2f} | verdict: {case.verdict[:200]}")
    print(f"Case saved to: {out['case_path']}")

    print("\n" + "=" * 60)
    print("FINAL AGENT MESSAGE")
    print("=" * 60)
    for msg in reversed(out["messages"]):
        content = getattr(msg, "content", "")
        if content and getattr(msg, "type", "") == "ai":
            print(content[:1500])
            break


if __name__ == "__main__":
    main()
