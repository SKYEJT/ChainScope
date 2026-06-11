"""Train GB-TGAD on Elliptic - full unsupervised training."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BEST_MODEL_PATH = str(REPO_ROOT / "data" / "best_model.pt")

import torch
from chainscope.models.data_loader import EllipticDataset
from chainscope.models.model import GBTGAD, GBTGADConfig
from chainscope.models.contrast_loss import total_unsupervised_loss


def train_elliptic(epochs=100, lr=1e-3, device="cuda"):
    # ── Data ──
    dataset = EllipticDataset()
    train_snaps = dataset.train_snapshots
    print(f"[Data] {len(train_snaps)} train snapshots")

    # ── Model ──
    config = GBTGADConfig(feat_dim=165)
    model = GBTGAD(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # ── Training loop ──
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Build grain balls every gb_refresh epochs (offline, no grad)
        if epoch == 1 or epoch % config.gb_refresh == 0:
            model.eval()
            model.build_grain_balls(train_snaps, device)
            model.train()

        # Get the full assignment vector from the latest grain ball build
        assignment_all = model._traj_gbs.assignment  # [N_all_train]

        # Per-snapshot training step
        offset = 0
        for snap in train_snaps:
            snap_dev = _to_device(snap, device)
            N_t = snap.n_nodes

            # Extract per-snapshot assignment
            assign_t = assignment_all[offset: offset + N_t].to(device)
            offset += N_t

            # Encode
            z_out, z_in = model.bigin(snap_dev)
            # Get temporal context for this snapshot
            # (simplified: use encode_all_snapshots for full context)
            # For per-step training, we use a simple mean context
            ctx = (z_out.mean(dim=0) + z_in.mean(dim=0)) / 2.0  # [hidden_dim]
            h = model.fusion(z_out, z_in, ctx)

            # Loss
            loss, loss_dict = total_unsupervised_loss(
                h=h,
                z_out=z_out,
                z_in=z_in,
                x_orig=snap_dev.x,
                assignment=assign_t,
                recon_proj=model.recon_proj,
                lambda_contrast=config.lambda_contrast,
                lambda_flow=config.lambda_flow,
                lambda_recon=config.lambda_recon,
                tau=config.tau,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(train_snaps)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg:.4f} best={best_loss:.4f}")

    # ── Calibrate score weights ──
    print("[Calibrate] Optimizing anomaly score weights...")
    model.calibrate_score_weights(train_snaps, device)

    # Save final model (with calibrated weights)
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"Done. Best loss={best_loss:.4f}, saved to data/best_model.pt")


def _to_device(data, device):
    """Move PyG Data tensors to device."""
    from torch_geometric.data import Data
    kwargs = {}
    for key in ["x", "edge_index", "edge_index_rev", "y"]:
        if hasattr(data, key) and isinstance(getattr(data, key), torch.Tensor):
            kwargs[key] = getattr(data, key).to(device)
    new_data = Data(**kwargs)
    new_data.timestep = data.timestep
    new_data.n_nodes = data.n_nodes
    return new_data


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    train_elliptic(device=device)
