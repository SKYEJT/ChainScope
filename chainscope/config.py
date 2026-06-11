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
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
# ChainScope REQUIRES GLM-5.1 (Z.AI track rule: the agent's core long-horizon task
# must be driven by GLM-5.1). Any other model is rejected at startup — see below.
REQUIRED_MODEL = "glm-5.1"
ZAI_MODEL = os.getenv("ZAI_MODEL", REQUIRED_MODEL).strip()


def require_glm_5_1(model: str | None = None) -> str:
    """Enforce GLM-5.1. Raise RuntimeError if the configured/requested model is not
    glm-5.1, so the app fails fast instead of silently running on a weaker model."""
    chosen = model if model is not None else ZAI_MODEL
    if (chosen or "").strip().lower() != REQUIRED_MODEL:
        raise RuntimeError(
            f"ChainScope requires model '{REQUIRED_MODEL}', but got '{chosen}'. "
            f"Set ZAI_MODEL={REQUIRED_MODEL} in .env (Z.AI track rule: the agent "
            f"must be driven by {REQUIRED_MODEL})."
        )
    return REQUIRED_MODEL

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
