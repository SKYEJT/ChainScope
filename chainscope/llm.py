"""LLM configuration for ChainScope Agent.

Single entry point for every model call. Any OpenAI-compatible chat endpoint
that supports tool calling works; the choice lives in .env (see config.py):

  LLM_MODEL      e.g. glm-5.1 (default), deepseek-v4-flash, deepseek-v4-pro
  LLM_API_KEY    API key for that endpoint
  LLM_BASE_URL   OpenAI-compatible base url, e.g. https://api.deepseek.com

The legacy ZAI_* names are still accepted as fallbacks.
"""
from langchain_openai import ChatOpenAI

from chainscope.config import LLM_API_KEY, LLM_BASE_URL, resolve_model


def get_llm(model: str | None = None, temperature: float = 0.1, **kwargs) -> ChatOpenAI:
    """Get a configured chat model (OpenAI-compatible endpoint).

    Args:
        model: explicit model name; falls back to LLM_MODEL from the environment.
        temperature: sampling temperature.
        **kwargs: forwarded to ChatOpenAI (e.g. max_tokens, timeout).
    """
    return ChatOpenAI(
        model=resolve_model(model),
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        **kwargs,
    )
