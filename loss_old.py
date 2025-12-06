import torch, torch.nn.functional as F
from config import (
    V_MAX, OMEGA_MAX,
    LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4, LAMBDA5,
    USE_BOUNDARY_LOSS, EPS1, EPS2, STATE_DIM
)
from model import NeuralCBF_SDRE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_u(u: torch.Tensor) -> torch.Tensor:
    """Clip the network’s raw accelerations to the admissible box ‖v,ω‖."""
    v  = torch.clamp(u[:, 0], -V_MAX,   V_MAX)
    om = torch.clamp(u[:, 1], -OMEGA_MAX, OMEGA_MAX)
    return torch.stack([v, om], dim=1)


def alpha(h: torch.Tensor, k: float = 2.0) -> torch.Tensor:
    """
    Class-K∞ function (Eq. 17 in the paper)
       α(h) = k·max(0,h)   if h≥0
             = 1/(0.5+|h|) if h<0
    """
    pos = k * F.relu(h)                       # k⋅h   for h>0
    neg = 1.0 / (0.5 + h.abs())               # 1/(0.5+|h|)
    neg = neg * (h < 0).float()               # mask to h<0 region
    return pos + neg
# ---------------------------------------------------------------------------


def compute_loss(model: NeuralCBF_SDRE, batch):
    state  = batch['robot_state']             # (B,5)
    lidar  = batch['lidar']
    label  = batch['label'].squeeze(1)

    h, u_raw, P, R = model(state, lidar)      # network forward

    # ───────────────────────── J4  (matrix eigenvalue term) ─────────────────
    BATCH, dev = state.size(0), state.device

    A = torch.tensor(
        [[0,0,0,1,0],
         [0,0,0,0,1],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]], dtype=torch.float32, device=dev)

    Bmat = torch.tensor(
        [[0,0],[0,0],[0,0],[1,0],[0,1]], dtype=torch.float32, device=dev
    ).unsqueeze(0).expand(BATCH, -1, -1)                          # (B,5,2)

    Rinv = torch.linalg.inv(R)                                    # (B,2,2)

    # α(h)/h  — guard   h≈0   to avoid division by zero
    eps_h = 1e-5
    safe_h  = torch.where(h.abs() < eps_h,
                      h.sign() * eps_h,          # keep sign, avoid 0-div
                      h) 
    alpha_over_h = (alpha(h) / safe_h).view(-1, 1, 1)

    norm2      = state.pow(2).sum(dim=1, keepdim=True) 
    inv_norm2  = (1.0 / (norm2 + 1e-4)).view(-1, 1, 1)
    #inv_norm2  = 1.0 / (state.norm(dim=1, keepdim=True).pow(2) + 1e-6)
    #inv_norm2  = inv_norm2.view(-1,1,1)

    I = torch.eye(STATE_DIM, device=dev).expand(BATCH, -1, -1)    # (B,5,5)

    K = (A.T @ P + P @ A
         - 2 * P @ Bmat @ Rinv @ Bmat.transpose(1,2) @ P
         - alpha_over_h * (inv_norm2 * I - P))

    K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)
    # largest (real part of) eigen-value
    eig_max = torch.linalg.eigvals(K).real.max(dim=1).values
    J4 = F.relu(eig_max + EPS2).mean()

    # ────────────────── J1 / J2 : safe vs unsafe margin ───────────────────
    unsafe = label < 0.5
    safe   = ~unsafe

    J1 = F.relu(h + EPS1)[unsafe].mean() if unsafe.any() else h.new_tensor(0.)
    J2 = F.relu(1. - h)[safe  ].mean() if safe.any()   else h.new_tensor(0.)

    # ────────────────── J3 : control projection penalty  ──────────────────
    diff = u_raw - _project_u(u_raw)         # (B,2)
    J3   = torch.norm(diff, p=2, dim=1).mean()
    #J3 = F.mse_loss(u_raw, _project_u(u_raw))

    # ────────────────── J5 : optional boundary alignment  ─────────────────
    if USE_BOUNDARY_LOSS:
        bmask = batch['is_boundary'].to(h.device).squeeze(1) > 0.5
        J5 = h[bmask].abs().mean() if bmask.any() else h.new_tensor(0.)
    else:
        J5 = h.new_tensor(0.)

    # total
    total = (LAMBDA1*J1 + LAMBDA2*J2 + LAMBDA3*J3 +
             LAMBDA4*J4 + LAMBDA5*J5)

    return total, {
        "loss_total": total.detach().item(),
        "loss_J1J2" : (LAMBDA1*J1 + LAMBDA2*J2).detach().item(),
        "loss_ctrl" : J3.detach().item(),
        "loss_cbf"  : J4.detach().item(),
        "loss_J5"   : J5.detach().item(),
    }
