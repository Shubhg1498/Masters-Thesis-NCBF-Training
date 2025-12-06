#!/usr/bin/env python3
import argparse, torch, pathlib
from model import NeuralCBF_SDRE
from train import export_torchscript          # reuse the helper
from config import BEAM_COUNT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  default="cbf_final.ts")
    args = ap.parse_args()

    net = NeuralCBF_SDRE()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ckpt["model"])
    export_torchscript(net, pathlib.Path(args.out))

if __name__ == "__main__":
    main()
