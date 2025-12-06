#!/usr/bin/env python3
"""
sanity_check.py

 • Export TorchScript if needed
 • Compute boundary offset h0  (|d_min − D_SAFE| < 0.05 m)
 • Print calibrated h values for a set of wall distances
"""

import argparse, pathlib, math
import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------- project modules ----------------
from config import BEAM_COUNT, SENSOR_MAX_RANGE, D_SAFE
from neuralcbfdataset import NeuralCBFDataset
from model import NeuralCBF_SDRE
from train import export_torchscript              # helper already in train.py
# -------------------------------------------------

# ---------- helpers ----------
def make_state(v=0.6, omega=0.0, yaw=0.0):
    """Return (1,5) state tensor with positions zeroed (as in training)."""
    return torch.tensor([[0.0, 0.0, yaw, v, omega]], dtype=torch.float32)

def make_lidar(front_dist):
    """Synthetic LiDAR with a flat wall at distance front_dist."""
    r = np.full(BEAM_COUNT, SENSOR_MAX_RANGE, dtype=np.float32)
    r[:5]  = front_dist
    r[-5:] = front_dist
    return torch.from_numpy(r).unsqueeze(0)

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True,  help="Path to .pth checkpoint")
ap.add_argument("--csv",  required=True,  help="Train CSV to estimate h0")
ap.add_argument("--ts",   default="cbf_final.ts", help="Output TorchScript path")
ap.add_argument("--batch", type=int, default=256)
args = ap.parse_args()

# ---------- load model ----------
net = NeuralCBF_SDRE()
ckpt = torch.load(args.ckpt, map_location="cpu")
net.load_state_dict(ckpt["model"])
net.eval()

# export TorchScript (only once)
ts_path = pathlib.Path(args.ts)
if not ts_path.is_file():
    export_torchscript(net, ts_path)
    print(f"✓ TorchScript saved → {ts_path}")

# ---------- compute boundary offset h0 ----------
ds = NeuralCBFDataset(args.csv, max_beams=BEAM_COUNT)
loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

hs = []
with torch.no_grad():
    for batch in loader:
        d_min = batch["h_gt"].squeeze(1) + D_SAFE            # (B,)
        mask  = (d_min > D_SAFE - 0.05) & (d_min < D_SAFE + 0.05)
        if mask.any():
            h, _, _, _ = net(batch["robot_state"], batch["lidar"])
            hs.append(h[mask])

if not hs:
    raise RuntimeError("No boundary samples found – widen the ±0.05 m band.")
h0 = torch.cat(hs).mean().item()
print(f"\nBoundary offset  h0 = {h0:.3f}")

# ---------- quick wall‑distance sanity check ----------
ts = torch.jit.load(str(ts_path)).eval()

print("\nCalibrated h ( h_eff = h_raw - h0 ) for a stationary wall:")
for d in [1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.20]:
    state = make_state(v=0.6)              # typical forward speed
    lidar = make_lidar(front_dist=d)
    with torch.no_grad():
        h_raw, _, _, _ = ts(state, lidar)
    h_eff = h_raw.item() - h0
    label = "safe  " if d > D_SAFE else "unsafe"
    print(f"d_min = {d:4.2f} m  ({label})  →  h_raw={h_raw.item():+.3f}   h_eff={h_eff:+.3f}")
