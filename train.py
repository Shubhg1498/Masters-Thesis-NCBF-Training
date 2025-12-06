#!/usr/bin/env python3
"""
SDRE-Neural-CBF training script (enhanced with AMP compatibility, λ4 ramp,
scalar CBF violation metric, optional ranking/calibration losses).

Usage Example:
  python train.py \
      --csv_train data/train.csv \
      --csv_val   data/val.csv   \
      --epochs 60 --batch 64 \
      --device cuda --j4_warmup_epochs 3
"""

import argparse, time, pathlib, random, numpy as np, torch, inspect
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import wandb

from config import (
    BEAM_COUNT,
    LAMBDA4,
    J4_RAMP_EPOCHS,
    LOG_SCALAR_CBF
)

from neuralcbfdataset import NeuralCBFDataset
from model import NeuralCBF_SDRE
from loss import compute_loss, alpha_K, project_u

# ---------------- Utility: seeding ----------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------- AMP Compatibility Shim ----------------
def _get_grad_scaler(device, use_amp: bool):
    if not use_amp:
        class _NullScaler:
            def scale(self, x): return x
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
            def is_enabled(self): return False
        from contextlib import contextmanager
        @contextmanager
        def _null_autocast(**kw):
            yield
        return _NullScaler(), _null_autocast

    # Try new torch.amp
    try:
        from torch import amp
        GradScalerNew = amp.GradScaler
        sig = inspect.signature(GradScalerNew.__init__)
        if 'device_type' in sig.parameters:
            scaler = GradScalerNew(device_type='cuda' if device.type == 'cuda' else 'cpu')
            def autocast_wrapper(**kw):
                return amp.autocast(device_type=device.type, **kw)
        else:
            scaler = GradScalerNew()
            def autocast_wrapper(**kw):
                return amp.autocast(device_type=device.type, **kw)
        return scaler, autocast_wrapper
    except (ImportError, AttributeError):
        pass
    # Fallback legacy
    from torch.cuda import amp as cuda_amp
    scaler = cuda_amp.GradScaler()
    def autocast_wrapper(**kw):
        return cuda_amp.autocast(**kw)
    return scaler, autocast_wrapper

# ---------------- DataLoader builder ----------------
def make_loader(csv_path: str, batch_size: int, weighted: bool, num_workers: int = 0):
    ds = NeuralCBFDataset(csv_path, max_beams=BEAM_COUNT)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset {csv_path} is empty.")

    if not weighted:
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

    # Weighted oversampling (if dataset did not already duplicate)
    labels = torch.tensor([int(r["label"]) for r in ds.rows])
    n_safe = (labels == 1).sum().item()
    n_un = (labels == 0).sum().item()
    w_safe = 1.0 / max(n_safe, 1)
    w_un = 1.0 / max(n_un, 1)
    weights = torch.where(labels == 1,
                          torch.full_like(labels, w_safe),
                          torch.full_like(labels, w_un))
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True)

# ---------------- Scalar CBF Violation Metric ----------------
def cbf_scalar_violation_rate(net, loader, device, max_batches=5):
    """
    Empirical rate of violations of: ∇h · (f + g u) + α(h) >= 0
    (Using current learned u; simplified drift / input matrices.)
    """
    if not LOG_SCALAR_CBF:
        return -1.0
    net.eval()
    vio, total = 0, 0
    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches:
            break
        state = batch['robot_state'].to(device).clone().detach().requires_grad_(True)
        lidar = batch['lidar'].to(device)

        h, u, _, _ = net(state, lidar)

        grad_h = torch.autograd.grad(h.sum(), state, retain_graph=False, create_graph=False)[0]

        # Simple unicycle drift f (matching system_matrices structure)
        f = torch.zeros_like(state)
        theta = state[:, 2]
        v = state[:, 3]
        omega = state[:, 4]
        f[:, 0] = v * torch.cos(theta)
        f[:, 1] = v * torch.sin(theta)
        f[:, 2] = omega
        # Control effect approximate: treat u components as accelerations on v, ω
        g_u = torch.zeros_like(state)
        g_u[:, 3] = u[:, 0]
        g_u[:, 4] = u[:, 1]

        lhs = (grad_h * (f + g_u)).sum(dim=1) + alpha_K(h)
        vio += (lhs < 0).sum().item()
        total += h.size(0)

    return vio / max(1, total)

# ---------------- TorchScript Export ----------------
def export_torchscript(model: torch.nn.Module, out_path: pathlib.Path):
    model.eval()
    dummy_state = torch.zeros(1, 5)
    dummy_lidar = torch.zeros(1, BEAM_COUNT)
    try:
        traced = torch.jit.trace(model.cpu(), (dummy_state, dummy_lidar))
        traced.save(str(out_path))
        print(f"✓ TorchScript exported → {out_path}")
    except Exception as e:
        print(f"⚠ TorchScript export failed: {e}")

# ---------------- Training ----------------
def train(cfg):
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    net = NeuralCBF_SDRE(use_unbounded_u=bool(cfg.unbounded_u)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    # Scheduler
    if cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.lr_factor,
            patience=cfg.lr_patience, min_lr=cfg.lr_min)
    elif cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs - cfg.j4_warmup_epochs, eta_min=cfg.lr_min)
    else:
        sched = None

    # Resume
    start_ep, best_val = 0, float('inf')
    if cfg.resume and pathlib.Path(cfg.resume).is_file():
        ckpt = torch.load(cfg.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if sched and ckpt.get("sched") is not None:
            sched.load_state_dict(ckpt["sched"])
        start_ep = ckpt["epoch"] + 1
        best_val = ckpt["best"]
        if getattr(cfg, "reset_lr", 0):
            for g in opt.param_groups:
                g["lr"] = cfg.lr
        print(f"✓ Resumed from {cfg.resume} at epoch {start_ep}")

    # Data
    train_loader = make_loader(cfg.csv_train, cfg.batch, cfg.weighted_sampler)
    val_loader = make_loader(cfg.csv_val, cfg.batch, weighted=False)

    wandb.init(project="neural-cbf",
               name=cfg.run_name or None,
               config=vars(cfg),
               save_code=False)

    out_dir = pathlib.Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # AMP
    use_amp = (device.type == 'cuda') and bool(cfg.amp)
    scaler, autocast = _get_grad_scaler(device, use_amp)

    def save_ckpt(tag: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict() if sched else None,
            "best": best_val
        }, out_dir / f"{tag}.pth")

    # Training loop
    for ep in range(start_ep, cfg.epochs):
        net.train()
        t0 = time.time()
        running = 0.0
        accum = {}

        # λ4 ramp
        if ep < cfg.j4_warmup_epochs:
            lambda4_eff = 0.0
            disable_j4 = True
        elif ep < cfg.j4_warmup_epochs + cfg.freeze_expansion_epochs:
            lambda4_eff = 0.0                  # still freeze λ4
            disable_j4 = False
        else:
            disable_j4 = False
            if J4_RAMP_EPOCHS > 0:
                frac = min(1.0, (ep - cfg.j4_warmup_epochs) / J4_RAMP_EPOCHS)
            else:
                frac = 1.0
            lambda4_eff = LAMBDA4 * frac

        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg.epochs} [train]", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            # Zero out absolute position dims if dataset not already changed
            batch['robot_state'][:, 0:2] = 0.0

            with autocast(enabled=use_amp):
                loss, stats, _ = compute_loss(
                    net, batch,
                    disable_j4=disable_j4,
                    lambda4_eff=lambda4_eff
                )

            if (not torch.isfinite(loss)) or (loss.item() > cfg.loss_clip):
                print(f"[Skip] loss={loss.item():.3f}")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()

            running += loss.item()
            for k, v in stats.items():
                accum[k] = accum.get(k, 0.0) + float(v)

        train_loss = running / max(1, len(train_loader))
        mean_stats = {f"train/{k}": v / max(1, len(train_loader))
                      for k, v in accum.items()}

        # Validation
        net.eval()
        val_running = 0.0
        val_accum = {}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {ep+1}/{cfg.epochs} [val]", leave=False):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)
                batch['robot_state'][:, 0:2] = 0.0
                vloss, vstats, _ = compute_loss(net, batch, disable_j4=False)
                if torch.isfinite(vloss):
                    val_running += vloss.item()
                    for k, v in vstats.items():
                        val_accum[k] = val_accum.get(k, 0.0) + float(v)

        val_loss = val_running / max(1, len(val_loader))
        val_stats = {f"val/{k}": v / max(1, len(val_loader)) for k, v in val_accum.items()}

        # Scheduler step
        if sched:
            if cfg.scheduler == "plateau":
                sched.step(val_loss)
            else:  # cosine
                if ep >= cfg.j4_warmup_epochs:
                    sched.step()
            lr_now = sched.optimizer.param_groups[0]["lr"]
        else:
            lr_now = opt.param_groups[0]["lr"]

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_ckpt("best", ep)
            export_torchscript(net, out_dir / "best_model.ts")

        if (ep % cfg.ckpt_every == 0) or (ep == cfg.epochs - 1):
            save_ckpt(f"epoch_{ep:04d}", ep)

        # Scalar CBF violation (subset for speed)
        vio_rate = cbf_scalar_violation_rate(net, val_loader, device) if LOG_SCALAR_CBF else -1.0

        dt = time.time() - t0
        print(f"Ep {ep+1:03}/{cfg.epochs}  train {train_loss:.4f}  val {val_loss:.4f} "
              f"vio {vio_rate:.3f}  lr {lr_now:.2e}  [{dt:5.1f}s]{' *' if improved else ''}")

        wandb.log({
            "epoch": ep + 1,
            "lr": lr_now,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "scalar_cbf_violation": vio_rate,
            "lambda4_eff": lambda4_eff,
            **mean_stats,
            **val_stats
        })

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_train", required=True)
    ap.add_argument("--csv_val", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    ap.add_argument("--lr_patience", type=int, default=8)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--weighted_sampler", type=int, default=0)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--j4_warmup_epochs", type=int, default=3)
    ap.add_argument("--loss_clip", type=float, default=50.0)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--unbounded_u", type=int, default=0,
                    help="If 1, remove tanh on control head so J3 becomes meaningful.")
    ap.add_argument('--freeze_expansion_epochs', type=int, default=3,
                    help='extra epochs after warm‑up with λ4=0')
    ap.add_argument("--reset_lr", type=int, default=0,
                help="1 ⇒ after resume, overwrite optimizer LR with --lr")
    cfg = ap.parse_args()
    train(cfg)
