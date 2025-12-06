# calibrate_h.py
import argparse, torch, numpy as np, csv
from model import NeuralCBF_SDRE  # or torch.jit.load if you only have TS
from neuralcbfdataset import NeuralCBFDataset
from torch.utils.data import DataLoader

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--tgt_safe', type=float, default=0.30)   # desired mean for safe
    ap.add_argument('--tgt_unsafe', type=float, default=-0.60)# desired mean for unsafe
    ap.add_argument('--batch', type=int, default=256)
    return ap.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)

    # Load model (adapt if you only have TorchScript)
    if args.ckpt.endswith('.ts'):
        net = torch.jit.load(args.ckpt, map_location=device).eval()
    else:
        net = NeuralCBF_SDRE().to(device)
        ck = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(ck['model']); net.eval()

    ds = NeuralCBFDataset(args.csv)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    h_all, y_all = [], []
    for batch in dl:
        state = batch['robot_state'].to(device)
        lidar = batch['lidar'].to(device)
        label = batch['label'].cpu().numpy().squeeze()
        h_raw, _, _, _ = net(state, lidar)
        h_all.append(h_raw.cpu().numpy())
        y_all.append(label)
    h_all = np.concatenate(h_all)
    y_all = np.concatenate(y_all)

    mu_s = h_all[y_all==1].mean()
    mu_u = h_all[y_all==0].mean()

    # Solve for a,b so that safe→tgt_safe and unsafe→tgt_unsafe
    if abs(mu_s - mu_u) < 1e-6:
        a = 1.0
        b = args.tgt_safe - mu_s          # pure bias
    else:
        a = (args.tgt_safe - args.tgt_unsafe) / (mu_s - mu_u)
        b = args.tgt_safe - a * mu_s

    print("=== Calibration ===")
    print(f"mu_safe={mu_s:.4f}  mu_unsafe={mu_u:.4f}")
    print(f"a={a:.6f}  b={b:.6f}")
    # Optional: report new means
    h_cal = a*h_all + b
    print(f"new_mu_safe={h_cal[y_all==1].mean():.4f}  new_mu_unsafe={h_cal[y_all==0].mean():.4f}")

    # Save to yaml/json snippet you can paste
    import json
    print("\nPaste into your node params:")
    print(json.dumps({"H_SCALE": float(a), "H_BIAS": float(b)}, indent=2))

if __name__ == '__main__':
    main()
