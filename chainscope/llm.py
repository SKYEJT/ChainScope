"""LLM configuration for ChainScope Agent.

The project is locked to GLM-5.1 (Z.AI track rule): get_llm enforces it via
config.require_glm_5_1 and raises if ZAI_MODEL is anything else.

Relevant env vars (see config.py):
  ZAI_MODEL      must be glm-5.1 (default)
  ZAI_API_KEY    API key
  ZAI_BASE_URL   OpenAI-compatible base url
"""
from langchain_openai import ChatOpenAI

from chainscope.config import ZAI_API_KEY, ZAI_BASE_URL, ZAI_MODEL, require_glm_5_1


def get_llm(model: str | None = None, temperature: float = 0.1, **kwargs) -> ChatOpenAI:
    """Get a configured GLM chat model (OpenAI-compatible endpoint).

    Args:
        model: explicit model name; falls back to ZAI_MODEL env. MUST resolve to
               glm-5.1 — anything else raises RuntimeError (Z.AI track requirement).
        temperature: sampling temperature.
        **kwargs: forwarded to ChatOpenAI (e.g. max_tokens, timeout).
    """
    model = require_glm_5_1(model if model is not None else ZAI_MODEL)
    return ChatOpenAI(
        model=model,
        api_key=ZAI_API_KEY,
        base_url=ZAI_BASE_URL,
        temperature=temperature,
        **kwargs,
    )
