import torch
import torch.nn.functional as F

from config import (
    MAX_ACC, MAX_ALPHA,
    LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4, LAMBDA5,
    LAMBDA_RANK, LAMBDA_CAL,
    USE_BOUNDARY_LOSS, EPS1, EPS2,
    STATE_DIM, CONTROL_DIM
)

# ---------------- Helpers ----------------
def project_u(u: torch.Tensor) -> torch.Tensor:
    av = torch.clamp(u[:, 0], -MAX_ACC, MAX_ACC)
    aom = torch.clamp(u[:, 1], -MAX_ALPHA, MAX_ALPHA)
    return torch.stack([av, aom], dim=1)

def alpha_K(h: torch.Tensor, k: float = 2.0) -> torch.Tensor:
    pos = k * F.relu(h)
    neg = (1.0 / (0.5 + h.abs())) * (h < 0).float()
    return pos + neg

def distance_ranking_loss(h, d, margin_d=0.05, margin_h=0.10, max_pairs=2048):
    """
    Pairwise ranking: if d_i > d_j + margin_d expect h_i >= h_j + margin_h.
    d: distance or proxy (min_dist). h,d shape (B,)
    """
    if h.numel() < 2:
        return h.new_tensor(0.)
    B = h.size(0)
    # random pairing by permutation
    perm = torch.randperm(B, device=h.device)
    hi, di = h[perm], d[perm]
    hj, dj = h, d
    mask = di > dj + margin_d
    if not mask.any():
        return h.new_tensor(0.)
    if mask.sum() > max_pairs:
        idx = mask.nonzero(as_tuple=False).squeeze()
        idx = idx[torch.randperm(idx.numel(), device=h.device)[:max_pairs]]
        m2 = torch.zeros_like(mask)
        m2[idx] = True
        mask = m2
    loss = F.relu((hj + margin_h) - hi)[mask].mean()
    return loss

# ---------------- Main Loss ----------------
def compute_loss(net,
                 batch,
                 disable_j4: bool = False,
                 lambda4_eff: float = None,
                 use_ranking: bool = True,
                 use_calibration: bool = True):
    """
    lambda4_eff: externally ramped weight (if None, uses LAMBDA4).
    """
    state = batch['robot_state']          # (B,5)
    lidar = batch['lidar']
    label = batch['label'].squeeze(1)     # (B,)

    h, u_raw, P, R = net(state, lidar)

    unsafe = label < 0.5
    safe = ~unsafe

    # J1 / J2 classification hinges
    J1 = F.relu(h + EPS1)[unsafe].mean() if unsafe.any() else h.new_tensor(0.)
    J2 = F.relu(1.0 - h)[safe].mean() if safe.any() else h.new_tensor(0.)

    # J3 projection (only meaningful if u_raw not bounded already)
    diff = u_raw - project_u(u_raw)
    J3 = diff.norm(dim=1).mean()

    # J4 matrix inequality surrogate
    if disable_j4:
        J4 = h.new_tensor(0.)
    else:
        A, Bmat = system_matrices(state)
        S = P  # approximation ignoring dP/dx
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
        K = 0.5 * (K + K.transpose(1, 2))  # enforce symmetry
        K = torch.nan_to_num(K, nan=0.0, posinf=1e5, neginf=-1e5).clamp(-1e4, 1e4)

        eigvals = torch.linalg.eigvals(K).real
        max_eig = eigvals.max(dim=1).values
        J4 = F.relu(max_eig + EPS2).mean()

    # J5 boundary
    if USE_BOUNDARY_LOSS and 'is_boundary' in batch:
        bmask = batch['is_boundary'].squeeze(1) > 0.5
        J5 = F.relu(h[bmask] + EPS1).mean() if bmask.any() else h.new_tensor(0.)
    else:
        J5 = h.new_tensor(0.)

    # --- Optional ranking & calibration ---
    if use_ranking and LAMBDA_RANK > 0:
        if 'min_dist' in batch:
            min_dist = batch['min_dist'].squeeze(1)
        elif 'h_gt' in batch:
            # Recover distance proxy: d = h_gt + D_SAFE (import D_SAFE if needed)
            from config import D_SAFE
            min_dist = (batch['h_gt'].squeeze(1) + D_SAFE).clamp(min=0)
        else:
            min_dist = h.detach()
        J_rank = distance_ranking_loss(h, min_dist)
    else:
        J_rank = h.new_tensor(0.)

    if use_calibration and LAMBDA_CAL > 0:
        safe_h = h[safe]
        unsafe_h = h[unsafe]
        if safe_h.numel() > 0 and unsafe_h.numel() > 0:
            J_cal = F.relu((unsafe_h.mean() + 0.1) - safe_h.mean())
        else:
            J_cal = h.new_tensor(0.)
    else:
        J_cal = h.new_tensor(0.)

    if lambda4_eff is None:
        lambda4_eff = LAMBDA4

    total = (LAMBDA1 * J1 +
             LAMBDA2 * J2 +
             LAMBDA3 * J3 +
             lambda4_eff * J4 +
             LAMBDA5 * J5 +
             LAMBDA_RANK * J_rank +
             LAMBDA_CAL * J_cal)

    stats = {
        "loss_total": total.item(),
        "J1": J1.item(),
        "J2": J2.item(),
        "J3": J3.item(),
        "J4": J4.item(),
        "J5": J5.item(),
        "J_rank": J_rank.item(),
        "J_cal": J_cal.item()
    }
    return total, stats, h.detach()

# -------------- Dynamics Helpers (same as original) --------------
def system_matrices(state: torch.Tensor):
    """
    Unicycle-style extended dynamics:
        state = [x, y, θ, v, ω]  (x,y assumed zeroed but keep dimension)
        control = [a_v, a_ω]
    """
    B, dev = state.size(0), state.device
    theta = state[:, 2]
    v = state[:, 3]

    A = torch.zeros(B, STATE_DIM, STATE_DIM, device=dev)
    A[:, 0, 2] = -v * torch.sin(theta)
    A[:, 0, 3] = torch.cos(theta)
    A[:, 1, 2] = v * torch.cos(theta)
    A[:, 1, 3] = torch.sin(theta)
    A[:, 2, 4] = 1.0

    Bmat = torch.zeros(B, STATE_DIM, CONTROL_DIM, device=dev)
    Bmat[:, 3, 0] = 1.0
    Bmat[:, 4, 1] = 1.0
    return A, Bmat
