from typing import List, Dict, Any
import csv, math
import torch
from torch.utils.data import Dataset
from config import BEAM_COUNT, SENSOR_MAX_RANGE, D_SAFE

# Configurable boundary band (was EPS_BDR); adjust to achieve ~8-15% boundary samples
BOUNDARY_BAND = 0.08   # meters

class NeuralCBFDataset(Dataset):
    """CSV → PyTorch mini-batch (no physical duplication; uses sample weights)."""
    def __init__(self, csv_file: str, max_beams: int = BEAM_COUNT):
        self.max_beams = max_beams
        self.rows: List[dict] = []
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                parsed = self._parse(raw)
                if parsed is not None:
                    self.rows.append(parsed)
        self.rows.sort(key=lambda r: r['t'])
        # Precompute weights (oversample unsafe & boundary)
        self.sample_weights = torch.tensor([r['weight'] for r in self.rows], dtype=torch.float32)
        if self.sample_weights.sum() <= 0:
            self.sample_weights = torch.ones(len(self.rows))

    # ------------------------------------------------------------
    def _parse(self, row: Dict[str, str]):
        try:
            t = float(row['time'])
            # State: [odom_x, odom_y, yaw, v, omega] -> drop absolute x,y by zeroing
            yaw = float(row['odom_yaw'])
            v   = float(row['odom_lin_vel'])
            omg = float(row['odom_ang_vel'])
            state = [0.0, 0.0, yaw, v, omg]   # odom_x, odom_y discarded (translation invariance)

            # Command (keep for potential future use / diagnostics)
            cmd = [float(row['cmd_lin_x']), float(row['cmd_ang_z'])]

            d_min = float(row['min_lidar_dist'])
            label = 1.0 if d_min > D_SAFE else 0.0      # 1 = safe, 0 = unsafe
            is_bdr = 1.0 if abs(d_min - D_SAFE) <= BOUNDARY_BAND else 0.0
            h_gt = d_min - D_SAFE                       # distance margin

            # Parse lidar string
            full_scan = row['compressed_lidar'].split()
            # Convert tokens; replace inf with SENSOR_MAX_RANGE
            ranges = []
            for tok in full_scan:
                if tok.lower() in ('inf', 'nan'):
                    ranges.append(SENSOR_MAX_RANGE)
                else:
                    try:
                        r = float(tok)
                    except ValueError:
                        r = SENSOR_MAX_RANGE
                    if math.isinf(r) or math.isnan(r):
                        r = SENSOR_MAX_RANGE
                    ranges.append(max(0.0, min(r, SENSOR_MAX_RANGE)))

            n_full = len(ranges)
            if n_full == 0:
                return None
            centre = n_full // 2
            half = self.max_beams // 2
            start = max(0, centre - half)
            end   = min(n_full, centre + half)
            slice_vals = ranges[start:end]
            # Pad if needed
            if len(slice_vals) < self.max_beams:
                slice_vals += [SENSOR_MAX_RANGE] * (self.max_beams - len(slice_vals))
            lidar_vals = slice_vals[:self.max_beams]

            # Weight (soft oversampling)
            if label == 0.0:
                w = 1.0  # unsafe
            elif is_bdr > 0.5:
                w = 1.3  # boundary safe
            else:
                w = 3.5

            return dict(t=t, state=state, cmd=cmd, lidar=lidar_vals,
                        label=label, is_boundary=is_bdr, h_gt=h_gt,
                        min_dist=d_min, weight=w)
        except KeyError:
            return None

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.rows)

    # ------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cur = self.rows[idx]
        # Find a future state with different time for finite difference (if needed)
        j = idx + 1
        while j < len(self.rows) and abs(self.rows[j]['t'] - cur['t']) < 1e-4:
            j += 1
        nxt = self.rows[min(j, len(self.rows)-1)]

        to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
        return {
            'robot_state'      : to_tensor(cur['state']),
            'robot_state_next' : to_tensor(nxt['state']),
            'lidar'            : to_tensor(cur['lidar']),
            'cmd'              : to_tensor(cur['cmd']),
            'label'            : torch.tensor([cur['label']], dtype=torch.float32),
            'is_boundary'      : torch.tensor([cur['is_boundary']], dtype=torch.float32),
            'h_gt'             : torch.tensor([cur['h_gt']], dtype=torch.float32),
            'min_dist'         : torch.tensor([cur['min_dist']], dtype=torch.float32),
            'weight'           : torch.tensor([cur['weight']], dtype=torch.float32),
            't_curr'           : torch.tensor([cur['t']], dtype=torch.float32),
            't_next'           : torch.tensor([nxt['t']], dtype=torch.float32),
        }
