"""LangGraph state graph for ChainScope's long-horizon investigation agent.

Explicit nodes: PLAN -> ACT -> OBSERVE -> (REFLECT | REPLAN) -> ACT ...
This makes the autonomous loop legible to judges: the agent plans, takes an
action, observes the tool result, periodically reflects/self-corrects, and is
forced to conclude when the step budget runs low.

    START -> plan -> act
    act    -> observe            (if the model requested tools)
    act    -> END                (if the model gave a final answer)
    observe-> reflect            (every `reflect_every` steps)
    observe-> replan             (when step budget is nearly exhausted)
    observe-> act                (otherwise)
    reflect-> act
    replan -> act
"""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from chainscope.llm import get_llm
from chainscope.agent.case_file import new_case, get_active_case
from chainscope.agent.prompts import (
    SYSTEM_PROMPT,
    DEFAULT_GOAL,
    REFLECTION_TEMPLATE,
    WRAPUP_TEMPLATE,
    BEGIN_TEMPLATE,
    NUDGE_TEMPLATE,
)

# ── Tool registry (all 17 tools) ──
from chainscope.tools.chainscout_tool import CHAIN_SCOUT_TOOLS
from chainscope.tools.graph_builder_tool import GRAPH_BUILDER_TOOLS
from chainscope.tools.detector_tool import DETECTOR_TOOLS
from chainscope.tools.attest_tool import ATTEST_TOOLS
from chainscope.tools.trace_tool import TRACE_TOOLS
from chainscope.tools.label_tool import LABEL_TOOLS
from chainscope.tools.memory_tool import MEMORY_TOOLS

ALL_TOOLS = (
    CHAIN_SCOUT_TOOLS
    + GRAPH_BUILDER_TOOLS
    + DETECTOR_TOOLS
    + TRACE_TOOLS
    + LABEL_TOOLS
    + MEMORY_TOOLS
    + ATTEST_TOOLS
)

DEFAULT_MAX_STEPS = 40
DEFAULT_REFLECT_EVERY = 4
HARD_STOP_BUFFER = 8  # extra steps allowed after budget to let the agent conclude


class InvState(TypedDict):
    messages: Annotated[list, add_messages]
    step: int
    max_steps: int
    reflect_every: int
    nudges: int


def build_investigation_graph(model: Optional[str] = None,
                              max_steps: int = DEFAULT_MAX_STEPS,
                              reflect_every: int = DEFAULT_REFLECT_EVERY):
    """Compile and return the investigation state graph.

    `max_steps` / `reflect_every` here are defaults baked into the nodes; they can
    also be overridden per-run via the initial state.
    """
    plain_llm = get_llm(model)
    tool_llm = get_llm(model).bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    # ── nodes ──
    def plan_node(state: InvState) -> dict:
        case = get_active_case()
        prompt = HumanMessage(content=(
            "Before acting, outline your investigation plan in 3-6 concise steps "
            "for this target, and name 1-3 initial hypotheses you will test. "
            "Then you will execute one action at a time."
        ))
        try:
            resp = plain_llm.invoke(state["messages"] + [prompt])
            plan_text = resp.content
        except Exception as e:
            resp = None
            plan_text = f"(planning step skipped: {e})"
        if case is not None:
            case.log_step("plan", str(plan_text)[:600])
        target = case.target if case is not None else "the target address"
        out_msgs = [prompt]
        if resp is not None:
            out_msgs.append(resp)
        out_msgs.append(HumanMessage(content=BEGIN_TEMPLATE.format(address=target)))
        return {"messages": out_msgs}

    def act_node(state: InvState) -> dict:
        resp = tool_llm.invoke(state["messages"])
        case = get_active_case()
        if case is not None and getattr(resp, "content", None):
            case.log_step("act", str(resp.content)[:400])
        return {"messages": [resp]}

    def observe_node(state: InvState) -> dict:
        out = tool_node.invoke(state)
        case = get_active_case()
        if case is not None:
            case.next_step()
            for m in out.get("messages", []):
                name = getattr(m, "name", "tool")
                case.log_step("observe", f"{name}: {str(getattr(m, 'content', ''))[:200]}")
        out["step"] = state.get("step", 0) + 1
        return out

    def reflect_node(state: InvState) -> dict:
        case = get_active_case()
        summary = case.summary() if case is not None else "(no active case)"
        if case is not None:
            case.log_step("reflect", "reflection checkpoint")
        return {"messages": [HumanMessage(content=REFLECTION_TEMPLATE.format(summary=summary))]}

    def replan_node(state: InvState) -> dict:
        case = get_active_case()
        if case is not None:
            case.log_step("replan", "wrap-up forced (step budget low)")
        return {"messages": [HumanMessage(content=WRAPUP_TEMPLATE)]}

    def nudge_node(state: InvState) -> dict:
        case = get_active_case()
        target = case.target if case is not None else "the target address"
        if case is not None:
            case.log_step("replan", "nudge: agent stalled before acting")
        return {
            "messages": [HumanMessage(content=NUDGE_TEMPLATE.format(address=target))],
            "nudges": state.get("nudges", 0) + 1,
        }

    # ── routing ──
    def after_act(state: InvState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "observe"
        # The model answered without calling a tool. If it stalled before doing
        # any real work (e.g. asked the user for input), nudge it back on track.
        case = get_active_case()
        if (case is not None and case.status != "closed"
                and state.get("step", 0) == 0 and state.get("nudges", 0) < 2):
            return "nudge"
        return END

    def after_observe(state: InvState):
        step = state.get("step", 0)
        ms = state.get("max_steps", max_steps)
        re_every = state.get("reflect_every", reflect_every)
        if step >= ms + HARD_STOP_BUFFER:
            return END
        if step >= ms:
            return "replan"
        if re_every > 0 and step % re_every == 0:
            return "reflect"
        return "act"

    # ── assemble ──
    g = StateGraph(InvState)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("observe", observe_node)
    g.add_node("reflect", reflect_node)
    g.add_node("replan", replan_node)
    g.add_node("nudge", nudge_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", after_act,
                            {"observe": "observe", "nudge": "nudge", END: END})
    g.add_conditional_edges(
        "observe", after_observe,
        {"act": "act", "reflect": "reflect", "replan": "replan", END: END},
    )
    g.add_edge("reflect", "act")
    g.add_edge("replan", "act")
    g.add_edge("nudge", "act")

    return g.compile()


def run_investigation(address: str, goal: Optional[str] = None,
                      model: Optional[str] = None,
                      max_steps: int = DEFAULT_MAX_STEPS,
                      reflect_every: int = DEFAULT_REFLECT_EVERY,
                      callbacks: Optional[list] = None,
                      recursion_limit: Optional[int] = None) -> dict:
    """Run a full autonomous investigation on an address.

    Returns dict with: case (CaseFile), messages (final), case_path (saved json).
    """
    goal = goal or DEFAULT_GOAL.format(address=address)
    case = new_case(address, goal)

    graph = build_investigation_graph(model=model, max_steps=max_steps,
                                      reflect_every=reflect_every)
    init: InvState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=goal)],
        "step": 0,
        "max_steps": max_steps,
        "reflect_every": reflect_every,
        "nudges": 0,
    }
    config: dict = {"recursion_limit": recursion_limit or (max_steps * 3 + 30)}
    if callbacks:
        config["callbacks"] = callbacks

    result = graph.invoke(init, config=config)
    case_path = case.save()
    return {"case": case, "messages": result.get("messages", []), "case_path": case_path}
