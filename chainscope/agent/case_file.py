"""CaseFile - the persistent investigation notebook for ChainScope's agent.

This is the heart of the long-horizon behavior. Instead of running a fixed
pipeline, the agent maintains a CaseFile across many steps: hypotheses it is
testing, evidence it has gathered, addresses it has visited, suspicious fund
paths, a running risk estimate, and a full iteration log. The agent reads and
updates this object via the memory/reflection tools, and the LangGraph nodes
route based on its contents.

The CaseFile is serialized to data/cases/<addr>_<ts>.json and doubles as the
"long-horizon run record" deliverable required by the hackathon track.

Tools are stateless functions, so we expose a tiny module-level registry
(set_active_case / get_active_case) that the graph wires up at the start of a
run. For the single-user demo this is sufficient and keeps the tool signatures
clean.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from chainscope.config import DATA_DIR

CASES_DIR = DATA_DIR / "cases"

# Hypothesis lifecycle
STATUS_ACTIVE = "active"
STATUS_CONFIRMED = "confirmed"
STATUS_REFUTED = "refuted"

# Evidence stance toward a hypothesis
STANCE_SUPPORT = "support"
STANCE_REFUTE = "refute"
STANCE_NEUTRAL = "neutral"

# Expected risk direction for a known label category (used to reconcile a sealed
# held-out label against the agent's blind, behaviour-only verdict at report time).
_CATEGORY_RISK = {
    "sanctioned": "high",
    "mixer": "high",
    "exchange": "low",
    "defi": "low",
    "bridge": "low",
}


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


@dataclass
class Hypothesis:
    id: str
    statement: str
    status: str = STATUS_ACTIVE
    confidence: float = 0.5
    created_step: int = 0
    updated_step: int = 0
    history: list = field(default_factory=list)  # list of {step, status, confidence, reason}


@dataclass
class Evidence:
    id: str
    step: int
    tool: str
    summary: str
    hypothesis_id: Optional[str] = None
    stance: str = STANCE_NEUTRAL
    confidence: float = 0.5


@dataclass
class FundPath:
    id: str
    direction: str            # "inflow" | "outflow"
    path: list = field(default_factory=list)   # [addr0, addr1, ...]
    total_value_eth: float = 0.0
    hops: int = 0
    note: str = ""


@dataclass
class IterationStep:
    step: int
    timestamp: str
    node: str                 # plan | act | observe | reflect | replan | report
    content: str


@dataclass
class CaseFile:
    """A single investigation's full state."""

    target: str
    goal: str = ""
    status: str = "open"      # open | closed
    verdict: str = ""         # final risk verdict text
    risk_estimate: float = 0.0
    created_at: str = field(default_factory=_now)
    closed_at: str = ""

    hypotheses: list = field(default_factory=list)        # list[Hypothesis]
    evidence: list = field(default_factory=list)          # list[Evidence]
    visited: list = field(default_factory=list)           # list[str] (lowercased)
    suspicious_paths: list = field(default_factory=list)  # list[FundPath]
    iteration_log: list = field(default_factory=list)     # list[IterationStep]

    # Blind mode: labels matched during the run are SEALED here instead of
    # being shown to the LLM, then revealed/reconciled against the blind verdict.
    held_labels: dict = field(default_factory=dict)       # addr(lower) -> {name, category, source}
    reconciliation: dict = field(default_factory=dict)    # filled at report time

    _step: int = 0

    # ── step counter ──
    def next_step(self) -> int:
        self._step += 1
        return self._step

    @property
    def step(self) -> int:
        return self._step

    # ── hypotheses ──
    def add_hypothesis(self, statement: str, confidence: float = 0.5) -> Hypothesis:
        h = Hypothesis(
            id=_short_id("H"),
            statement=statement,
            confidence=float(confidence),
            created_step=self._step,
            updated_step=self._step,
            history=[{"step": self._step, "status": STATUS_ACTIVE,
                      "confidence": float(confidence), "reason": "created"}],
        )
        self.hypotheses.append(h)
        return h

    def get_hypothesis(self, hid: str) -> Optional[Hypothesis]:
        for h in self.hypotheses:
            if h.id == hid:
                return h
        return None

    def revise_hypothesis(self, hid: str, status: str, reason: str,
                          confidence: Optional[float] = None) -> Optional[Hypothesis]:
        h = self.get_hypothesis(hid)
        if h is None:
            return None
        h.status = status
        if confidence is not None:
            h.confidence = float(confidence)
        h.updated_step = self._step
        h.history.append({"step": self._step, "status": status,
                          "confidence": h.confidence, "reason": reason})
        return h

    # ── evidence ──
    def add_evidence(self, tool: str, summary: str, hypothesis_id: Optional[str] = None,
                     stance: str = STANCE_NEUTRAL, confidence: float = 0.5) -> Evidence:
        e = Evidence(
            id=_short_id("E"),
            step=self._step,
            tool=tool,
            summary=summary,
            hypothesis_id=hypothesis_id,
            stance=stance,
            confidence=float(confidence),
        )
        self.evidence.append(e)
        return e

    # ── visited addresses ──
    def mark_visited(self, address: str) -> None:
        a = address.lower()
        if a not in self.visited:
            self.visited.append(a)

    def is_visited(self, address: str) -> bool:
        return address.lower() in self.visited

    # ── fund paths ──
    def add_path(self, direction: str, path: list, total_value_eth: float = 0.0,
                 note: str = "") -> FundPath:
        p = FundPath(
            id=_short_id("P"),
            direction=direction,
            path=[a.lower() for a in path],
            total_value_eth=float(total_value_eth),
            hops=max(len(path) - 1, 0),
            note=note,
        )
        self.suspicious_paths.append(p)
        return p

    # ── iteration log ──
    def log_step(self, node: str, content: str) -> IterationStep:
        s = IterationStep(step=self._step, timestamp=_now(), node=node, content=content)
        self.iteration_log.append(s)
        return s

    # ── risk ──
    def set_risk(self, value: float) -> None:
        self.risk_estimate = max(0.0, min(1.0, float(value)))

    # ── held-out labels (blind mode) ──
    def hold_label(self, address: str, name: str, category: str = "",
                   source: str = "local") -> None:
        """Seal a matched label so it is NOT shown to the LLM during the run.

        Revealed and reconciled against the blind verdict in compile_final_report.
        """
        a = (address or "").lower()
        if a and a not in self.held_labels:
            self.held_labels[a] = {
                "name": name or "",
                "category": (category or "").lower(),
                "source": source,
            }

    def get_held_label(self, address: str) -> Optional[dict]:
        return self.held_labels.get((address or "").lower())

    def has_held_labels(self) -> bool:
        return bool(self.held_labels)

    def reconcile_labels(self, blind_risk: float) -> dict:
        """Reveal sealed labels and reconcile them with the blind, behaviour-only verdict.

        Stores and returns a reconciliation dict describing whether the agent's
        label-free verdict agrees with the held-out ground-truth label.
        """
        blind_label = "high" if float(blind_risk) >= 0.6 else "low"
        target_sealed = self.get_held_label(self.target)

        expected = "unknown"
        outcome = "no_label"
        if target_sealed is not None:
            cat = (target_sealed.get("category") or "").lower()
            expected = _CATEGORY_RISK.get(cat, "unknown")
            if expected == "unknown":
                # We learned who the entity is, but the label carries no risk
                # direction (e.g. a plain Blockscout name) — no ground truth to grade.
                outcome = "identity_revealed"
            elif expected == blind_label:
                outcome = "agree"
            else:
                outcome = "conflict"

        recon = {
            "blind_risk": round(float(blind_risk), 4),
            "blind_label": blind_label,
            "target": self.target,
            "target_label": target_sealed,            # {name, category, source} or None
            "expected_from_label": expected,          # high | low | unknown
            "outcome": outcome,                       # agree | conflict | identity_revealed | no_label
            "other_sealed": {a: v for a, v in self.held_labels.items()
                             if a != (self.target or "").lower()},
        }
        self.reconciliation = recon
        return recon

    # ── summary for the LLM to reflect on ──
    def summary(self) -> str:
        lines = [f"CASE SUMMARY for {self.target}"]
        lines.append(f"Goal: {self.goal}")
        lines.append(f"Step: {self._step} | Risk estimate: {self.risk_estimate:.2f} | Status: {self.status}")
        lines.append(f"Visited addresses: {len(self.visited)}")

        if self.hypotheses:
            lines.append("\nHypotheses:")
            for h in self.hypotheses:
                lines.append(f"  [{h.status.upper()} c={h.confidence:.2f}] ({h.id}) {h.statement}")
        else:
            lines.append("\nHypotheses: (none yet)")

        if self.evidence:
            lines.append("\nRecent evidence:")
            for e in self.evidence[-6:]:
                tgt = f" -> {e.hypothesis_id}({e.stance})" if e.hypothesis_id else ""
                lines.append(f"  ({e.tool}){tgt}: {e.summary[:120]}")

        if self.suspicious_paths:
            lines.append("\nSuspicious fund paths:")
            for p in self.suspicious_paths[-5:]:
                lines.append(f"  [{p.direction} {p.hops}hop ~{p.total_value_eth:.3f}ETH] "
                             f"{' -> '.join(a[:10] for a in p.path)} {('('+p.note+')') if p.note else ''}")
        return "\n".join(lines)

    # ── serialization ──
    def to_dict(self) -> dict:
        d = {
            "target": self.target,
            "goal": self.goal,
            "status": self.status,
            "verdict": self.verdict,
            "risk_estimate": self.risk_estimate,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "total_steps": self._step,
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "evidence": [asdict(e) for e in self.evidence],
            "visited": self.visited,
            "suspicious_paths": [asdict(p) for p in self.suspicious_paths],
            "iteration_log": [asdict(s) for s in self.iteration_log],
            "held_labels": self.held_labels,
            "reconciliation": self.reconciliation,
        }
        return d

    def close(self, verdict: str) -> None:
        self.status = "closed"
        self.verdict = verdict
        self.closed_at = _now()

    def save(self, directory: Path | str = CASES_DIR) -> str:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fname = f"{self.target[:10]}_{int(time.time())}.json"
        path = directory / fname
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return str(path)


# ── Module-level active-case registry (wired up by the graph) ──
_active_case: Optional[CaseFile] = None


def set_active_case(case: CaseFile) -> None:
    global _active_case
    _active_case = case


def get_active_case() -> Optional[CaseFile]:
    return _active_case


def new_case(target: str, goal: str = "") -> CaseFile:
    case = CaseFile(target=target, goal=goal)
    set_active_case(case)
    return case
