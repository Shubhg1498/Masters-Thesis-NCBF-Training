import torch, numpy as np, math, pathlib

BEAM_COUNT        = 120
SENSOR_MAX_RANGE  = 10.0
FORWARD_SECTOR    = 7
STATE_DIM         = 5

def make_scan(dist):
    scan = np.full(BEAM_COUNT, SENSOR_MAX_RANGE, np.float32)
    scan[:FORWARD_SECTOR]  = dist
    scan[-FORWARD_SECTOR:] = dist
    return torch.from_numpy(scan).unsqueeze(0)        # (1,120)

net   = torch.jit.load("neural_cbf.ts").eval()
dev   = next(net.parameters()).device
print("device:", dev)

for d in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    lidar  = make_scan(d).to(dev)
    state  = torch.tensor([[d, 0., 0., 0., 0.]], device=dev)  # <-- x = d
    with torch.no_grad():
        h, *_ = net(state, lidar)
    print(f"d = {d:.1f} m   ->   h = {h.item(): .4f}")
