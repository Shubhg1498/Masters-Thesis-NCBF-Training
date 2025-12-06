import torch
import torch.nn as nn
from config import INPUT_DIM, STATE_DIM, BEAM_COUNT, CONTROL_DIM

class NeuralCBF_SDRE(nn.Module):
    """
    Neural CBF + SDRE parameterization with translation + residual.

    Inputs:
        state : (B, STATE_DIM) = [0, 0, yaw, v, ω]  (odom_x, odom_y zeroed upstream)
        lidar : (B, BEAM_COUNT)

    Outputs:
        h  : (B,)    CBF value  (h >= 0 => safe set)
        u  : (B,2)   raw control proposal (either tanh-bounded or unbounded)
        P  : (B,STATE_DIM,STATE_DIM)  positive definite
        R  : (B,2,2) positive definite
    """
    def __init__(self,
                 latent_dim: int = 128,
                 eps_pd: float = 1e-3,
                 min_diag: float = 1e-4,
                 dropout: float = 0.10,
                 use_unbounded_u: bool = False):
        super().__init__()
        self.state_dim = STATE_DIM
        self.ctrl_dim = CONTROL_DIM
        self.use_unbounded_u = use_unbounded_u

        self.register_buffer("eps_pd", torch.tensor(eps_pd))
        self.register_buffer("min_diag", torch.tensor(min_diag))

        self.latent = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU()
        )

        p_tril = self.state_dim * (self.state_dim + 1) // 2
        r_tril = self.ctrl_dim * (self.ctrl_dim + 1) // 2
        self.p_head = nn.Linear(latent_dim, p_tril)
        self.r_head_mat = nn.Linear(latent_dim, r_tril)
        self.u_head = nn.Linear(latent_dim, self.ctrl_dim)

        # Translation & residual heads
        self.c_head = nn.Linear(latent_dim, self.state_dim)
        self.r_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    # ---------- helpers ----------
    def _vec_to_tril(self, vec: torch.Tensor, dim: int) -> torch.Tensor:
        B = vec.size(0)
        L = torch.zeros(B, dim, dim, device=vec.device, dtype=vec.dtype)
        idx = torch.tril_indices(dim, dim, device=vec.device)
        L[:, idx[0], idx[1]] = vec
        return L

    def _tril_to_pd(self, tril_vec: torch.Tensor, dim: int) -> torch.Tensor:
        L = self._vec_to_tril(tril_vec, dim)
        diag = torch.diagonal(L, 0, -2, -1)
        diag = torch.where(diag.abs() < self.min_diag,
                           self.min_diag.expand_as(diag), diag)
        for i in range(dim):
            L[:, i, i] = diag[:, i]
        eye = torch.eye(dim, device=L.device)
        M = L @ L.transpose(-1, -2) + self.eps_pd * eye
        return M.clamp(-1e3, 1e3)

    # ---------- forward ----------
    def forward(self, state: torch.Tensor, lidar: torch.Tensor):
        if lidar.size(1) != BEAM_COUNT:
            raise ValueError(f"Expected lidar with {BEAM_COUNT} beams, got {lidar.size(1)}")

        z = self.latent(torch.cat([state, lidar], dim=1))

        # Scale heads initially -> smaller PD matrices
        Lp_vec = 0.1 * self.p_head(z)
        Lr_vec = 0.1 * self.r_head_mat(z)
        P = self._tril_to_pd(Lp_vec, self.state_dim)
        R = self._tril_to_pd(Lr_vec, self.ctrl_dim)

        if self.use_unbounded_u:
            u = self.u_head(z)
        else:
            u = torch.tanh(self.u_head(z))

        c = self.c_head(z)                        # translation
        x_shift = state - c
        xcol = x_shift.unsqueeze(-1)
        quad = (xcol.transpose(-2, -1) @ (P @ xcol)).squeeze(-1).squeeze(-1)
        r = 0.05 * self.r_head(z).squeeze(-1)     # small residual scale
        h = 1.0 - quad + r
        return h, u, P, R
