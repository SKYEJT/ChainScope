#!/usr/bin/env python3
"""Fine-tune GB-TGAD on Ethereum snapshots."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from chainscope.tools.detector import GBTGADDetector

SNAPSHOTS_PATH = str(REPO_ROOT / "data" / "eth_snapshots.pt")
OUTPUT_PATH = str(REPO_ROOT / "data" / "finetuned_model.pt")

def main():
    snaps = torch.load(SNAPSHOTS_PATH, weights_only=False)
    print(f"Loaded {len(snaps)} snapshots")

    detector = GBTGADDetector(device="cpu")
    detector.finetune(snaps, epochs=20, lr=1e-4)

    torch.save(detector.model.state_dict(), OUTPUT_PATH)
    print(f"Finetuned model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
