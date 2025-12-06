#!/usr/bin/env python3
"""Quantitative & visual check of learned CBF values ĥ vs ground-truth h*."""

import argparse, pathlib, math, torch
from torch.utils.data import DataLoader
import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from neuralcbfdataset import NeuralCBFDataset
from model           import NeuralCBF_SDRE      # for state-dict loading
from config          import D_SAFE

# --------- helpers -----------------------------------------
@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    all_pred, all_gt, all_lbl = [], [], []
    for batch in tqdm(loader, desc="[collect]"):
        for k,v in batch.items():
            batch[k] = v.to(device)
        h_pred, *_ = model(batch["robot_state"], batch["lidar"])   # (B,)
        all_pred.append(h_pred.cpu())
        all_gt  .append(batch["h_gt"].squeeze(1).cpu())
        all_lbl .append(batch["label"].squeeze(1).cpu())
    return torch.cat(all_pred), torch.cat(all_gt), torch.cat(all_lbl)

# --------- main --------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",   required=True)
    p.add_argument("--ts",    default=None, help="TorchScript file")
    p.add_argument("--model", default=None, help="state-dict checkpoint")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--negate", action="store_true",
                   help="multiply network output ĥ by −1 before evaluation")
    args = p.parse_args()

    # data ---------------------------------------------------
    ds      = NeuralCBFDataset(args.csv)
    loader  = DataLoader(ds, batch_size=args.batch,
                         shuffle=False, num_workers=0, pin_memory=False)

    dev = torch.device(args.device)

    # network -----------------------------------------------
    if args.ts:
        net = torch.jit.load(str(args.ts)).to(dev)
        print(f"Loaded TorchScript: {args.ts}")
    elif args.model:
        net = NeuralCBF_SDRE().to(dev)
        ck  = torch.load(args.model, map_location=dev)
        net.load_state_dict(ck["model"] if isinstance(ck,dict) else ck)
        print(f"Loaded state-dict : {args.model}")
    else:
        raise ValueError("provide --ts or --model")

    # inference ---------------------------------------------
    h_pred, h_gt, lbl = collect(net, loader, dev)
    if args.negate:
        h_pred.mul_(-1.0)

    # metrics ------------------------------------------------
    mae   = torch.mean(torch.abs(h_pred - h_gt)).item()
    corr  = torch.corrcoef(torch.stack([h_pred, h_gt]))[0,1].item()
    # classification by sign:
    pred_safe = (h_pred > 0).float()
    acc   = (pred_safe == lbl).float().mean().item()

    print("\n=== h-value audit on", pathlib.Path(args.csv).name, "===")
    print(f"MAE(ĥ, h*)        : {mae:8.4f}")
    print(f"corr(ĥ, h*)       : {corr:8.4f}")
    print(f"sign-accuracy      : {acc*100:6.2f}%  (sign(ĥ) vs safe/unsafe label)")
    print(f"safe ratio in CSV  : {lbl.mean().item()*100:6.2f}%")

    # -------------------- plots ----------------------------
    h_pred_np, h_gt_np, lbl_np = map(lambda t: t.numpy(), (h_pred, h_gt, lbl))

    # 1) scatter
    plt.figure(figsize=(6,5))
    plt.scatter(h_gt_np, h_pred_np, s=6, alpha=0.25)
    lim = [-max(abs(h_gt_np).max(), abs(h_pred_np).max())*1.05,
            max(abs(h_gt_np).max(), abs(h_pred_np).max())*1.05]
    plt.plot(lim, lim, 'k--', lw=1)
    plt.xlabel("ground-truth  h*  (d_min − D_SAFE)")
    plt.ylabel("network prediction  ĥ")
    plt.title("CBF value: prediction vs ground-truth")
    plt.grid(True)
    plt.tight_layout(); plt.savefig("scatter_h_test.png", dpi=180)

    # 2) histogram
    plt.figure(figsize=(6,4))
    plt.hist(h_pred_np[lbl_np>0.5], bins=60, alpha=0.6, label="safe")
    plt.hist(h_pred_np[lbl_np<0.5], bins=60, alpha=0.6, label="unsafe")
    plt.axvline(0, ls='--', c='k'); plt.legend()
    plt.xlabel("predicted ĥ"); plt.ylabel("count")
    plt.title("Distribution of predicted ĥ on test set")
    plt.tight_layout(); plt.savefig("hist_h_test.png", dpi=180)

    print("Scatter plot  → scatter_h_test.png")
    print("Histogram     → hist_h_test.png")

if __name__ == "__main__":
    main()
