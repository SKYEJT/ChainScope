"""ChainScope autonomous investigation agent (LangGraph state machine).

Public API:
    run_investigation(address, ...)        -> run a full autonomous investigation
    build_investigation_graph(model, ...)  -> compiled LangGraph for streaming
    ALL_TOOLS                              -> the 18-tool toolbox
    CaseFile / get_active_case / new_case  -> case memory

Backward compatibility:
    create_agent_with_tools(...)           -> compiled graph (legacy shim)
"""
from chainscope.agent.graph import (
    ALL_TOOLS,
    build_investigation_graph,
    run_investigation,
    DEFAULT_MAX_STEPS,
    DEFAULT_REFLECT_EVERY,
)
from chainscope.agent.case_file import (
    CaseFile,
    new_case,
    get_active_case,
    set_active_case,
)
from chainscope.agent.prompts import SYSTEM_PROMPT, DEFAULT_GOAL


def create_agent_with_tools(model: str | None = None, recursion_limit: int = 40,
                            logger=None):
    """Legacy shim: build and return the compiled investigation graph.

    Kept so older demo scripts that did
        agent = create_agent_with_tools()
        agent.stream(...)
    keep working. New code should prefer run_investigation() or
    build_investigation_graph(). `recursion_limit` / `logger` are accepted for
    signature compatibility but ignored — the agent now records everything in
    the CaseFile (see case_file.py).
    """
    return build_investigation_graph(model=model)


__all__ = [
    "ALL_TOOLS",
    "build_investigation_graph",
    "run_investigation",
    "create_agent_with_tools",
    "CaseFile",
    "new_case",
    "get_active_case",
    "set_active_case",
    "SYSTEM_PROMPT",
    "DEFAULT_GOAL",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_REFLECT_EVERY",
]
