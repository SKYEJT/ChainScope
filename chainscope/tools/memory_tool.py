"""Memory & reflection tools - let GLM maintain case state and self-correct.

These are the tools that turn a one-shot pipeline into a long-horizon, self-
correcting investigation. The agent proposes hypotheses, records evidence for or
against them, revises them when contradicted (self-correction), reviews the case
to decide what to do next, and finally compiles a report.
"""
from langchain_core.tools import tool

# NOTE: case_file is imported lazily inside each tool function. Importing it at
# module load would pull in chainscope.agent.__init__ -> graph -> these tools,
# a circular import that breaks `import chainscope.tools.memory_tool` on its own.


@tool
def propose_hypothesis(statement: str, confidence: float = 0.5) -> str:
    """Propose a new investigative hypothesis to test.

    Example: "This address is a layering intermediary for laundered funds."
    Returns the hypothesis id you will use to record findings and revise it.

    Args:
        statement: the hypothesis in one clear sentence
        confidence: your initial confidence 0.0-1.0
    """
    from chainscope.agent.case_file import get_active_case
    case = get_active_case()
    if case is None:
        return "[ERROR] No active case."
    h = case.add_hypothesis(statement, confidence)
    return f"Hypothesis registered: {h.id} (confidence {h.confidence:.2f}). Use this id in record_finding/revise_hypothesis."


@tool
def record_finding(summary: str, hypothesis_id: str = "", supports: bool = True,
                   confidence: float = 0.5) -> str:
    """Record a piece of evidence and how it bears on a hypothesis.

    Args:
        summary: what you found (one sentence)
        hypothesis_id: which hypothesis this relates to (optional)
        supports: True if it supports the hypothesis, False if it contradicts it
        confidence: strength of this evidence 0.0-1.0
    """
    from chainscope.agent.case_file import (
        get_active_case, STANCE_SUPPORT, STANCE_REFUTE, STANCE_NEUTRAL,
    )
    case = get_active_case()
    if case is None:
        return "[ERROR] No active case."
    hid = hypothesis_id.strip() or None
    if hid and case.get_hypothesis(hid) is None:
        return f"[ERROR] Unknown hypothesis id '{hid}'. Call propose_hypothesis first or omit it."
    stance = STANCE_SUPPORT if supports else STANCE_REFUTE
    if hid is None:
        stance = STANCE_NEUTRAL
    e = case.add_evidence(tool="agent_finding", summary=summary,
                          hypothesis_id=hid, stance=stance, confidence=confidence)
    # nudge hypothesis confidence
    if hid:
        h = case.get_hypothesis(hid)
        delta = confidence * (0.2 if supports else -0.2)
        h.confidence = max(0.0, min(1.0, h.confidence + delta))
        h.updated_step = case.step
    return f"Evidence {e.id} recorded ({stance})."


@tool
def revise_hypothesis(hypothesis_id: str, new_status: str, reason: str) -> str:
    """Revise a hypothesis when evidence confirms or contradicts it (self-correction).

    Args:
        hypothesis_id: the hypothesis to update
        new_status: one of "confirmed", "refuted", "active"
        reason: why you are changing it
    """
    from chainscope.agent.case_file import (
        get_active_case, STATUS_CONFIRMED, STATUS_REFUTED, STATUS_ACTIVE,
    )
    case = get_active_case()
    if case is None:
        return "[ERROR] No active case."
    status_map = {"confirmed": STATUS_CONFIRMED, "refuted": STATUS_REFUTED, "active": STATUS_ACTIVE}
    status = status_map.get(new_status.strip().lower())
    if status is None:
        return "[ERROR] new_status must be confirmed | refuted | active."
    h = case.revise_hypothesis(hypothesis_id, status, reason)
    if h is None:
        return f"[ERROR] Unknown hypothesis id '{hypothesis_id}'."
    return f"Hypothesis {h.id} -> {status.upper()} (confidence {h.confidence:.2f}). Reason logged."


@tool
def set_risk_estimate(score: float, rationale: str) -> str:
    """Update the overall risk estimate for the target as the case evolves.

    Args:
        score: current overall risk 0.0 (safe) - 1.0 (clearly illicit)
        rationale: brief justification
    """
    from chainscope.agent.case_file import get_active_case
    case = get_active_case()
    if case is None:
        return "[ERROR] No active case."
    case.set_risk(score)
    case.log_step("reflect", f"risk={case.risk_estimate:.2f}: {rationale}")
    return f"Risk estimate set to {case.risk_estimate:.2f}."


@tool
def review_case() -> str:
    """Review the current case file: hypotheses, evidence, paths, visited, risk.

    Use this to reflect before deciding the next action or before concluding.
    """
    from chainscope.agent.case_file import get_active_case
    case = get_active_case()
    if case is None:
        return "[ERROR] No active case."
    return case.summary()


def _risk_floor_from_case(case) -> float:
    """Evidence-driven risk floor for the final verdict.

    A clearly-supported illicit conclusion must not be under-scored into a LOW
    verdict (which would flip the held-out label reconciliation to CONFLICT at
    the 0.6 boundary). This floor triggers ONLY on a high-risk hypothesis the
    agent has NOT refuted, so it never inflates the risk of addresses whose
    mixer/fraud hypotheses were refuted (e.g. vitalik, exchanges). It inspects
    hypotheses only — never path hop counts, which do not separate legitimate
    high-throughput hubs from launderers.
    """
    from chainscope.agent.case_file import STATUS_CONFIRMED, STATUS_REFUTED
    HIGH_RISK_KW = ("mixer", "launder", "layering", "tumbl",
                    "sanction", "fraud", "phish", "scam")
    floor = 0.0
    for h in getattr(case, "hypotheses", []):
        if h.status == STATUS_REFUTED:
            continue
        if any(k in (h.statement or "").lower() for k in HIGH_RISK_KW):
            if h.status == STATUS_CONFIRMED:
                floor = max(floor, 0.75)
            elif h.confidence >= 0.8:  # high-confidence but still-open high-risk
                floor = max(floor, 0.65)
    return floor


@tool
def compile_final_report(verdict: str, risk_score: float, recommendation: str) -> str:
    """Compile the final investigation report from the case file and conclude.

    Call this once you have gathered sufficient evidence. It assembles a
    structured report (used for on-chain attestation) and closes the case.

    Args:
        verdict: your final risk verdict narrative
        risk_score: final overall risk 0.0-1.0
        recommendation: recommended action (e.g. flag, monitor, clear)
    """
    from chainscope.agent.case_file import (
        get_active_case, STATUS_CONFIRMED, STATUS_REFUTED, STATUS_ACTIVE,
    )
    case = get_active_case()
    if case is None:
        return "[ERROR] No active case."

    floor = _risk_floor_from_case(case)
    if risk_score < floor:
        case.log_step("reflect",
                      f"risk floor applied: {risk_score:.2f} -> {floor:.2f} "
                      f"(non-refuted high-risk hypothesis)")
        risk_score = floor
    case.set_risk(risk_score)

    lines = [f"# Investigation Report: {case.target}", ""]
    lines.append(f"**Final Risk Score:** {case.risk_estimate:.2f}  ")
    is_anom = case.risk_estimate >= 0.6
    lines.append(f"**Status:** {'ANOMALOUS / HIGH RISK' if is_anom else 'NORMAL / LOW RISK'}  ")
    lines.append(f"**Steps taken:** {case.step} | **Addresses investigated:** {len(case.visited)}")
    lines.append("")
    lines.append(f"## Verdict\n{verdict}")
    lines.append("")

    confirmed = [h for h in case.hypotheses if h.status == STATUS_CONFIRMED]
    refuted = [h for h in case.hypotheses if h.status == STATUS_REFUTED]
    active = [h for h in case.hypotheses if h.status == STATUS_ACTIVE]
    if case.hypotheses:
        lines.append("## Hypotheses")
        for h in confirmed:
            lines.append(f"- ✅ CONFIRMED ({h.confidence:.2f}): {h.statement}")
        for h in refuted:
            lines.append(f"- ❌ REFUTED ({h.confidence:.2f}): {h.statement}")
        for h in active:
            lines.append(f"- ◻️ INCONCLUSIVE ({h.confidence:.2f}): {h.statement}")
        lines.append("")

    if case.suspicious_paths:
        lines.append("## Traced Fund Paths")
        for p in case.suspicious_paths[:8]:
            arrow = " -> ".join(a[:10] for a in p.path)
            lines.append(f"- [{p.direction}, {p.hops} hops, ~{p.total_value_eth:.3f} ETH] {arrow}")
        lines.append("")

    if case.evidence:
        lines.append("## Key Evidence")
        for e in case.evidence[-10:]:
            lines.append(f"- ({e.tool}) {e.summary}")
        lines.append("")

    lines.append(f"## Recommendation\n{recommendation}")

    # ── Reveal & reconcile sealed (held-out) labels against the blind verdict ──
    if case.has_held_labels():
        recon = case.reconcile_labels(case.risk_estimate)
        bl = recon["blind_label"].upper()
        lines.append("")
        lines.append("## Label Reconciliation (held-out verification)")
        lines.append(f"Blind verdict (reached WITHOUT seeing any label): "
                     f"risk={case.risk_estimate:.2f} ({bl}).")
        tl = recon.get("target_label")
        if tl:
            lines.append(f"Sealed label for target: \"{tl['name']}\" "
                         f"[{tl['category'] or 'no risk category'}] (source: {tl['source']}).")
        outcome = recon["outcome"]
        if outcome == "agree":
            lines.append("Result: AGREE ✅ — the label-free verdict matches the held-out label.")
        elif outcome == "conflict":
            lines.append(f"Result: CONFLICT ⚠️ — blind verdict is {bl}, but the label implies "
                         f"{recon['expected_from_label'].upper()} risk. Needs review.")
        elif outcome == "identity_revealed":
            lines.append("Result: IDENTITY REVEALED ℹ️ — the label names the entity but carries "
                         "no risk category; the blind verdict stands on behaviour.")
        if recon.get("other_sealed"):
            lines.append(f"({len(recon['other_sealed'])} other labelled address(es) were also "
                         f"sealed during this run.)")

    report = "\n".join(lines)

    case.close(verdict=verdict)
    case.log_step("report", f"Final report compiled. Risk={case.risk_estimate:.2f}")
    return report


MEMORY_TOOLS = [
    propose_hypothesis,
    record_finding,
    revise_hypothesis,
    set_risk_estimate,
    review_case,
    compile_final_report,
]
