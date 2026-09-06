"""ChainScope configuration center."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# ── API Keys ──
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")

# ── LLM (any OpenAI-compatible chat endpoint with tool calling) ──
# ChainScope was built for the Z.AI hackathon track on GLM-5.1 and used to hard-
# reject every other model. Post-hackathon the model is a deployment choice:
#   LLM_API_KEY / LLM_BASE_URL / LLM_MODEL   canonical names
#   ZAI_API_KEY / ZAI_BASE_URL / ZAI_MODEL   legacy names, still honoured as fallbacks
# Defaults reproduce the original GLM-5.1 setup, so an untouched .env behaves as before.
# The agent relies on function calling, so the model must support OpenAI-style tools
# (GLM-5.x, deepseek-v4-flash / deepseek-v4-pro, gpt-4.x, qwen-plus, ...).
DEFAULT_MODEL = "glm-5.1"
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("ZAI_API_KEY", "")
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("ZAI_BASE_URL")
                or "https://open.bigmodel.cn/api/paas/v4")
LLM_MODEL = (os.getenv("LLM_MODEL") or os.getenv("ZAI_MODEL") or DEFAULT_MODEL).strip()

# Backwards-compatible aliases for code that still imports the old names.
ZAI_API_KEY, ZAI_BASE_URL, ZAI_MODEL = LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def resolve_model(model: str | None = None) -> str:
    """Return the model name to use: an explicit request wins, else LLM_MODEL.
    Fail fast on an empty name so a misconfigured deployment does not start."""
    chosen = (model if model is not None else LLM_MODEL or "").strip()
    if not chosen:
        raise RuntimeError(
            "No LLM model configured. Set LLM_MODEL (and LLM_API_KEY / LLM_BASE_URL) in .env."
        )
    return chosen

# ── Ethereum ──
SEPOLIA_RPC_URL = os.getenv(
    "SEPOLIA_RPC_URL",
    f"https://eth-sepolia.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
)
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")

# ── IPFS ──
IPFS_TOKEN = os.getenv("IPFS_TOKEN", "")

# ── GB-TGAD ──
GBTGAD_FEAT_DIM = 38  # Ethereum address features (vs 165 for Elliptic)

# ── Blind-label mode ──
# When ON, lookup_address_label SEALS any matched label into the CaseFile instead
# of revealing it to the LLM, forcing a behaviour-only "blind" verdict. The sealed
# label is then revealed and reconciled against the blind verdict at report time.
# This turns the label from an "answer shortcut" (information leakage) into a
# held-out ground truth, so the agent's reasoning ability can be evaluated fairly.
# Set CHAINSCOPE_BLIND_MODE=0 to restore the old label-aware behaviour (for A/B).
BLIND_MODE = os.getenv("CHAINSCOPE_BLIND_MODE", "1").strip().lower() not in (
    "0", "false", "no", "off", "",
)
