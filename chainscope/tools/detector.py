"""GB-TGAD Detector - anomaly detection for Ethereum address graphs.

Loads Elliptic pretrained weights (165-dim) and adapts to Ethereum (38-dim).
Provides single-snapshot and multi-snapshot inference.
Fine-tuning on Ethereum snapshots with layer freezing.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from chainscope.config import GBTGAD_FEAT_DIM, DATA_DIR
from chainscope.models.model import GBTGAD, GBTGADConfig
from chainscope.models.contrast_loss import total_unsupervised_loss

PRETRAINED_PATH = DATA_DIR / "best_model.pt"


class GBTGADDetector:
    """Anomaly detector using GB-TGAD with pretrained weight adaptation."""

    def __init__(self, pretrained_path=None, device="cpu"):
        self.device = torch.device(device)
        self.config = GBTGADConfig(feat_dim=GBTGAD_FEAT_DIM)  # 38
        self.model = GBTGAD(self.config).to(self.device)
        self.pretrained_path = Path(pretrained_path or PRETRAINED_PATH)
        self._load_and_adapt()

    def _load_and_adapt(self):
        """Load Elliptic pretrained weights (165-dim) and adapt to 38-dim.

        Strategy:
          - Layers with matching shapes: direct copy (most GIN/GRU/fusion layers)
          - Input projection (input_proj.weight): truncate columns 165→38
          - Input projection (input_proj.bias): same shape, direct copy
          - Recon projector last layer: truncate rows 165→38
          - log_score_w: direct copy (4-dim, shape-independent)
        """
        if not self.pretrained_path.exists():
            print(f"[WARN] No pretrained weights at {self.pretrained_path}, using random init.")
            return

        ell_config = GBTGADConfig(feat_dim=165)
        ell_model = GBTGAD(ell_config)
        state = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)
        ell_model.load_state_dict(state)

        model_state = self.model.state_dict()
        pretrained = ell_model.state_dict()
        adapted = {}
        adapted_keys = []
        skipped_keys = []

        for k, v in pretrained.items():
            if k not in model_state:
                skipped_keys.append(k)
                continue
            target_shape = model_state[k].shape

            if v.shape == target_shape:
                # Direct copy — most layers have same shape
                adapted[k] = v
            elif len(v.shape) == 2:
                # 2D adaptation: truncate each dim independently
                # input_proj.weight: [hidden, 165+hidden] → [hidden, 38+hidden] (cols differ)
                # recon_proj.2.weight: [165, hidden] → [38, hidden] (rows differ)
                adapted[k] = v[:target_shape[0], :target_shape[1]].clone()
                adapted_keys.append(f"{k}: {list(v.shape)}→{list(target_shape)}")
            elif len(v.shape) == 1 and v.shape[0] > target_shape[0]:
                # 1D bias: truncate
                adapted[k] = v[:target_shape[0]].clone()
                adapted_keys.append(f"{k}: {list(v.shape)}→{list(target_shape)}")
            else:
                skipped_keys.append(f"{k}: shape mismatch {list(v.shape)} vs {list(target_shape)}")

        model_state.update(adapted)
        self.model.load_state_dict(model_state)

        n_adapted = len(adapted)
        n_total = len(model_state)
        print(f"[OK] Loaded & adapted pretrained weights: {n_adapted}/{n_total} params")
        if adapted_keys:
            print(f"  Adapted layers:")
            for s in adapted_keys:
                print(f"    {s}")
        if skipped_keys:
            print(f"  Skipped: {skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}")

    def _heuristic_scores(self, snap):
        """Compute heuristic anomaly scores from graph structure.

        Supplements weak model embeddings (Bitcoin→Ethereum domain gap).

        Key insight: Etherscan API only returns target-related txs,
        so neighbors always appear as degree-1 leaves. Therefore
        one-time-interactor ratio is useless. Instead we use:

        Flow heuristic: degree imbalance + pure direction penalty.
          - Balanced in≈out → low (normal economic behavior)
          - Pure drain (out only) or source (in only) → high

        Structure heuristic: pure direction + high out-degree spreading.
          - Pure out with many recipients → spreading (suspicious)
          - Pure in with many sources → collecting (suspicious)
          - Balanced → low (normal)

        Returns:
            (s_flow_heur, s_struct_heur) as torch tensors [N]
        """
        N = snap.x.size(0)
        ei = snap.edge_index  # [2, E]

        # In-degree and out-degree per node (vectorized scatter; was a per-edge loop)
        in_deg = torch.zeros(N, device=self.device)
        out_deg = torch.zeros(N, device=self.device)
        if ei.numel() > 0:
            src, dst = ei[0], ei[1]
            src = src[src < N]
            dst = dst[dst < N]
            out_deg.scatter_add_(0, src, torch.ones_like(src, dtype=out_deg.dtype))
            in_deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=in_deg.dtype))

        total_deg = in_deg + out_deg + 1e-8

        # Flow heuristic: degree imbalance
        is_pure_direction = ((in_deg == 0) | (out_deg == 0)).float()
        imbalance = (out_deg - in_deg).abs() / (total_deg + 1.0)
        s_flow_heur = torch.maximum(imbalance, is_pure_direction * 0.8)
        s_flow_heur = s_flow_heur.clamp(0, 1)

        # Structure heuristic: pure direction + spreading
        # Pure out-drain with many recipients = cash-out pattern
        # Pure in-source with many sources = collection pattern
        # Both are structurally suspicious
        out_spread = is_pure_direction * (out_deg / (total_deg + 1.0))  # 0 for balanced
        in_collect = is_pure_direction * (in_deg / (total_deg + 1.0))   # 0 for balanced
        # Scale by log of degree to avoid very high scores for exchanges
        spread_factor = torch.log1p(torch.maximum(out_deg, in_deg)) / 10.0
        s_struct_heur = torch.maximum(out_spread, in_collect) * spread_factor
        # Pure direction with any degree gets at least 0.3
        s_struct_heur = torch.maximum(s_struct_heur, is_pure_direction * 0.3)
        s_struct_heur = s_struct_heur.clamp(0, 1)

        return s_flow_heur, s_struct_heur

    def detect(self, snapshot):
        """Run anomaly detection on a single PyG Data snapshot.

        For a single snapshot without pre-built grain balls, we compute
        a lightweight anomaly score based on:
          - s_attr: reconstruction error (high → anomalous)
          - s_flow: out/in flow embedding cosine similarity (low → anomalous)
          - Combined score using calibrated weights

        Args:
            snapshot: PyG Data with x, edge_index, edge_index_rev

        Returns:
            dict with keys: scores, s_attr, s_flow, n_nodes
        """
        self.model.eval()
        with torch.no_grad():
            snap = snapshot.to(self.device)
            N = snap.x.size(0) if hasattr(snap, 'x') and snap.x is not None else 0

            if N == 0:
                return {"scores": torch.tensor([]), "s_attr": torch.tensor([]),
                        "s_flow": torch.tensor([]), "n_nodes": 0}

            # Encode: get z_out, z_in
            z_out, z_in = self.model.bigin(snap)

            # Simple context: mean of z_out and z_in
            ctx = (z_out.mean(dim=0) + z_in.mean(dim=0)) / 2.0

            # Fuse into final embedding
            h = self.model.fusion(z_out, z_in, ctx)

            # ── s_attr: reconstruction error ──
            x_hat = self.model.recon_proj(h)  # [N, feat_dim]
            recon_err = (x_hat - snap.x).pow(2).mean(dim=1)  # [N]
            # Z-score normalization across nodes. The previous median-exp form
            # saturated to ~1.0 for almost every target (the centre node is
            # nearly always the largest reconstruction error), so s_attr had no
            # discriminative power. A z-score keeps the component informative.
            mu = recon_err.mean()
            sigma = recon_err.std().clamp(min=1e-8)
            s_attr = torch.sigmoid((recon_err - mu) / sigma)

            # ── s_flow: out/in flow cosine similarity ──
            cos_sim = F.cosine_similarity(z_out, z_in, dim=-1)  # [N]
            s_flow_model = (1.0 - cos_sim) / 2.0  # [0, 1], higher = more anomalous

            # ── Heuristic scores from graph structure ──
            # (supplement weak model embeddings due to Bitcoin→Ethereum domain gap)
            s_flow_heur, s_struct_heur = self._heuristic_scores(snap)

            # Blend model + heuristic (heuristic dominant due to domain gap)
            s_flow = 0.3 * s_flow_model + 0.7 * s_flow_heur
            s_struct = s_struct_heur

            # ── Combined score ──
            # s_flow already blends model(0.3)+heuristic(0.7); the old formula
            # then ADDED s_flow_model again (0.25), double-counting the model
            # term. On single snapshots the neighbours are degree-1 leaves, so
            # s_struct carries little signal — flow is the main discriminator
            # between mixers (in/out of different economic types) and normal
            # hubs. Drop the duplicate model term and lean weight onto flow.
            scores = 0.15 * s_attr + 0.60 * s_flow + 0.25 * s_struct

            return {
                "scores": scores.cpu(),
                "s_attr": s_attr.cpu(),
                "s_struct": s_struct.cpu(),
                "s_flow": s_flow.cpu(),
                "n_nodes": N,
            }

    def finetune(self, snapshots, epochs=20, lr=1e-4):
        """Fine-tune last layers on Ethereum snapshots with layer freezing.

        Strategy:
          - Freeze all BiGIN layers except the last GIN conv
          - Keep fusion and recon_proj unfrozen (adapt to Ethereum feature space)
          - Train with unsupervised loss (contrast + flow + recon)
          - Rebuild grain balls on new embeddings after training

        Args:
            snapshots: list of PyG Data snapshots from Ethereum
            epochs: number of fine-tuning epochs
            lr: learning rate (lower than pretraining)
        """
        # ── Freeze all parameters first ──
        for param in self.model.parameters():
            param.requires_grad = False

        # ── Unfreeze specific layers for fine-tuning ──
        unfrozen_names = []
        for name, param in self.model.named_parameters():
            # Last GIN conv in both directions
            if "bigin" in name and (
                "gin_out_conv.1" in name or "gin_in_conv.1" in name
                or "gin_out.1" in name or "gin_in.1" in name
            ):
                param.requires_grad = True
                unfrozen_names.append(name)
            # Fusion layer (adapts to new embedding distribution)
            if "fusion" in name:
                param.requires_grad = True
                unfrozen_names.append(name)
            # Recon projector (adapts to 38-dim Ethereum features)
            if "recon_proj" in name:
                param.requires_grad = True
                unfrozen_names.append(name)
            # Score weights (recalibrate for Ethereum)
            if "log_score_w" in name:
                param.requires_grad = True
                unfrozen_names.append(name)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[Finetune] Trainable: {trainable}/{total} params")
        print(f"[Finetune] Unfrozen layers: {unfrozen_names[:8]}...")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

        # ── Fine-tuning loop ──
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_snaps = 0
            for snap in snapshots:
                snap = snap.to(self.device)
                N = snap.x.size(0)
                if N < 2:
                    continue

                optimizer.zero_grad()

                # Encode
                z_out, z_in = self.model.bigin(snap)
                ctx = (z_out.mean(dim=0) + z_in.mean(dim=0)) / 2.0
                h = self.model.fusion(z_out, z_in, ctx)

                # Dummy assignment (all nodes in same ball) for contrast loss
                assignment = torch.zeros(N, dtype=torch.long, device=self.device)

                loss, _ = total_unsupervised_loss(
                    h=h, z_out=z_out, z_in=z_in,
                    x_orig=snap.x, assignment=assignment,
                    recon_proj=self.model.recon_proj,
                    lambda_contrast=self.config.lambda_contrast,
                    lambda_flow=self.config.lambda_flow,
                    lambda_recon=self.config.lambda_recon,
                    tau=self.config.tau,
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_snaps += 1

            avg_loss = total_loss / max(n_snaps, 1)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

        # ── Rebuild grain balls on new embeddings ──
        if len(snapshots) >= 2:
            try:
                self.model.build_grain_balls(snapshots, self.device)
                print("[Finetune] Grain balls rebuilt on Ethereum snapshots")
            except Exception as e:
                print(f"[Finetune] Grain ball rebuild skipped: {e}")
        else:
            print("[Finetune] Too few snapshots for grain ball rebuild (need >=2)")

        # ── Re-freeze all for inference ──
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[Finetune] Done. Final loss={avg_loss:.4f}")
