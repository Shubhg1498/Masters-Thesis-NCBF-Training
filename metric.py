#!/usr/bin/env python3


import argparse
import torch
import numpy as np
from typing import Optional, Dict, Any

from config import (
    BEAM_COUNT,
    D_SAFE,
    STATE_DIM,
    CONTROL_DIM,
    LOG_SCALAR_CBF  # not strictly needed but imported for consistency
)

from neuralcbfdataset import NeuralCBFDataset
from model import NeuralCBF_SDRE
from loss import alpha_K, system_matrices  # reuse original helpers

# ------------------------------------------------------------
@torch.no_grad()
def collect_cbf_metrics(net,
                        loader,
                        device: torch.device,
                        max_batches: Optional[int] = None,
                        compute_K: bool = False,
                        compute_P_cond: bool = False) -> Dict[str, Any]:
    """
    Collect diagnostic metrics over a data loader.

    Args:
        net: model (NeuralCBF_SDRE)
        loader: DataLoader producing dict batches
        device: torch.device
        max_batches: limit number of batches for speed
        compute_K: include raw largest eigenvalue of K statistics
        compute_P_cond: include condition number stats for P

    Returns:
        dict of scalar metrics
    """
    net.eval()
    safe_h_sum = unsafe_h_sum = 0.0
    safe_count = unsafe_count = 0
    boundary_count = 0
    total_count = 0

    h_list = []
    d_list = []
    safe_h_list = []
    unsafe_h_list = []

    max_eigs_raw = []
    p_conds = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        # Move tensors
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # enforce dropped odom coordinates
        batch['robot_state'][:, 0:2] = 0.0

        h, u, P, R = net(batch['robot_state'], batch['lidar'])
        label = batch['label'].squeeze(1)
        safe_mask = label > 0.5
        unsafe_mask = ~safe_mask

        if 'min_dist' in batch:
            d = batch['min_dist'].squeeze(1)
        elif 'h_gt' in batch:
            d = (batch['h_gt'].squeeze(1) + D_SAFE).clamp(min=0)
        else:
            d = torch.zeros_like(h)

        if safe_mask.any():
            safe_h_sum += h[safe_mask].sum().item()
            safe_count += safe_mask.sum().item()
            safe_h_list.append(h[safe_mask])
        if unsafe_mask.any():
            unsafe_h_sum += h[unsafe_mask].sum().item()
            unsafe_count += unsafe_mask.sum().item()
            unsafe_h_list.append(h[unsafe_mask])

        h_list.append(h)
        d_list.append(d)

        if 'is_boundary' in batch:
            bmask = batch['is_boundary'].squeeze(1) > 0.5
            boundary_count += bmask.sum().item()

        total_count += h.size(0)

        if compute_K:
            # raw K eigenvalues
            state = batch['robot_state']
            A, Bmat = system_matrices(state)
            S = P
            Bsz, dev = state.size(0), state.device
            Rreg = R + 1e-4 * torch.eye(CONTROL_DIM, device=dev).expand(Bsz, -1, -1)
            I2 = torch.eye(CONTROL_DIM, device=dev).expand_as(Rreg)
            Rinv = torch.linalg.solve(Rreg, I2)
            BRB = Bmat @ Rinv @ Bmat.transpose(1, 2)
            term1 = A.transpose(1, 2) @ S + S @ A
            term2 = 2.0 * (S @ BRB @ S)
            alpha_over_h = alpha_K(h) / (h.abs() + 1e-3)
            alpha_over_h = alpha_over_h.clamp(0, 50).view(-1, 1, 1)
            I = torch.eye(STATE_DIM, device=dev).expand(Bsz, -1, -1)
            norm2 = state.norm(dim=1, keepdim=True).pow(2).view(-1, 1, 1) + 1e-6
            term3 = alpha_over_h * (I / norm2 - P)
            K = term1 - term2 - term3
            K = 0.5 * (K + K.transpose(1, 2))
            eigvals = torch.linalg.eigvals(K).real
            raw_max = eigvals.max(dim=1).values
            max_eigs_raw.append(raw_max)

        if compute_P_cond:
            evals = torch.linalg.eigvals(P).real
            evals = torch.clamp(evals, min=1e-9)
            cond = (evals.max(dim=1).values / evals.min(dim=1).values)
            p_conds.append(cond)

    # Concatenate
    h_all = torch.cat(h_list) if h_list else torch.tensor([], device=device)
    d_all = torch.cat(d_list) if d_list else torch.tensor([], device=device)
    safe_h_all = torch.cat(safe_h_list) if safe_h_list else torch.tensor([], device=device)
    unsafe_h_all = torch.cat(unsafe_h_list) if unsafe_h_list else torch.tensor([], device=device)

    mean_h_safe = safe_h_all.mean().item() if safe_h_all.numel()>0 else float('nan')
    mean_h_unsafe = unsafe_h_all.mean().item() if unsafe_h_all.numel()>0 else float('nan')

    if h_all.numel()>1 and torch.std(h_all)>1e-8 and torch.std(d_all)>1e-8:
        corr = torch.corrcoef(torch.stack([h_all, d_all]))[0, 1].item()
    else:
        corr = float('nan')

    if safe_h_all.numel() > 0:
        pct_safe_over_095 = (safe_h_all > 0.95).float().mean().item()
        pct_safe_grad_band = ((safe_h_all > 0.0) & (safe_h_all < 0.4)).float().mean().item()
    else:
        pct_safe_over_095 = float('nan')
        pct_safe_grad_band = float('nan')

    boundary_frac = boundary_count / total_count if total_count>0 else float('nan')
    unsafe_frac = unsafe_count / total_count if total_count>0 else float('nan')

    metrics = {
        "mean_h_safe": mean_h_safe,
        "mean_h_unsafe": mean_h_unsafe,
        "delta_mean_h": (mean_h_safe - mean_h_unsafe)
                        if (safe_h_all.numel()>0 and unsafe_h_all.numel()>0) else float('nan'),
        "corr_h_dist": corr,
        "pct_safe_h_gt_0.95": pct_safe_over_095,
        "pct_safe_h_in_0_0.4": pct_safe_grad_band,
        "boundary_fraction": boundary_frac,
        "unsafe_fraction": unsafe_frac,
        "num_samples": total_count
    }

    if compute_K and max_eigs_raw:
        all_raw = torch.cat(max_eigs_raw)
        metrics["raw_max_eigK_median"] = all_raw.median().item()
        metrics["raw_max_eigK_p90"] = all_raw.quantile(0.90).item()

    if compute_P_cond and p_conds:
        all_cond = torch.cat(p_conds)
        metrics["P_cond_median"] = all_cond.median().item()
        metrics["P_cond_p90"] = all_cond.quantile(0.90).item()

    return metrics

# ------------------------------------------------------------
def _build_loader(csv_path, batch, device, weighted=False):
    ds = NeuralCBFDataset(csv_path, max_beams=BEAM_COUNT)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch, shuffle=not weighted, pin_memory=True)

# ------------------------------------------------------------
def eval_checkpoint(args):
    device = torch.device(args.device)
    net = NeuralCBF_SDRE()
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.to(device)

    loader = _build_loader(args.csv, args.batch, device, weighted=False)
    metrics = collect_cbf_metrics(net, loader, device,
                                  max_batches=args.max_batches,
                                  compute_K=bool(args.compute_K),
                                  compute_P_cond=bool(args.compute_P_cond))
    print("=== Metrics ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (int,float)) else f"{k}: {v}")
    return metrics

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV dataset file")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pth file")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="Limit batches for speed (int).")
    ap.add_argument("--compute_K", type=int, default=0)
    ap.add_argument("--compute_P_cond", type=int, default=0)
    args = ap.parse_args()
    eval_checkpoint(args)

if __name__ == "__main__":
    main()
