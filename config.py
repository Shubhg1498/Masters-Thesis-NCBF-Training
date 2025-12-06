# config.py  ──────────────────────────────────────────────
# Global constants so all modules share the same numbers.
# ---------------------------------------------------------

# LiDAR & state size
BEAM_COUNT  = 120
STATE_DIM   = 5               # x, y, yaw, v, ω
INPUT_DIM   = BEAM_COUNT + STATE_DIM
CONTROL_DIM = 2               # a_v, a_ω
# Sensor range handling
SENSOR_MAX_RANGE = 8.0       # clamp 'inf' beams to this value (m)

# Safety margin
ROBOT_RADIUS = 0.30           # m
BUFFER       = 0.5           # m
D_SAFE       = ROBOT_RADIUS + BUFFER   # target distance for h = 0

# Action limits  (projection Π_U)
MAX_ACC      = 2.0            # m s⁻¹
MAX_ALPHA    = 1.0            # rad s⁻¹

# Loss-term switches / weights
USE_BOUNDARY_LOSS = True
LAMBDA5           = 1.0       # weight for J5

# Core loss weights (from the paper, can tune)
LAMBDA1 = 2.5
LAMBDA2 = 7.0
LAMBDA3 = 0.2
LAMBDA4 = 0.05
LAMBDA5 = 0.7
EPS1    = 0.08  # slack for obstacle / boundary margin
EPS2    = 0.01  # slack for invariance eigenvalues
USE_BOUNDARY_LOSS = True      # toggle J5

LAMBDA_RANK = 3.0      # weight for distance ranking loss
LAMBDA_CAL  = 0.0      # weight for calibration (mean separation) loss
USE_UNBOUNDED_U = 1   # set to 1 to remove tanh and make J3 active
J4_RAMP_EPOCHS = 15   # epochs to ramp lambda4 after warmup
RANK_MARGIN_D  = 0.05  # meters difference to form a ranking pair
RANK_MARGIN_H  = 0.10  # desired h margin between ranked pairs
CAL_MARGIN     = 0.10  # margin for mean separation (unsafe_mean + CAL_MARGIN <= safe_mean)
POWER_ITER_STEPS = 5   # steps for power iteration (largest eigenvalue)
USE_POWER_ITER = 1     # 1 to use power iteration instead of full eigvals
LOG_SCALAR_CBF = 1     # 1 to compute scalar inequality violation metric (slower)