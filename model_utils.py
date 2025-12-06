# model_utils.py
import math, numpy as np, torch
from   config import BEAM_COUNT, SENSOR_MAX_RANGE

# ------------------------------------------------------------
def state_from_odom(msg):
    px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
    qx,qy,qz,qw = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, \
                  msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    v   = msg.twist.twist.linear.x
    om  = msg.twist.twist.angular.z
    return [px, py, yaw, v, om]

# ------------------------------------------------------------
def preprocess_scan(scan):

    # -------- detect type ---------
    if hasattr(scan, "ranges"):            # LaserScan message
        rng = np.asarray(scan.ranges, dtype=np.float32)
    else:                                  # already list / ndarray
        rng = np.asarray(scan, dtype=np.float32)

    # -------- sanitise invalid ----
    rng[~np.isfinite(rng)] = SENSOR_MAX_RANGE

    # -------- fix length ----------
    if rng.size > BEAM_COUNT:
        rng = rng[:BEAM_COUNT]
    elif rng.size < BEAM_COUNT:
        rng = np.pad(rng, (0, BEAM_COUNT - rng.size),
                     constant_values=SENSOR_MAX_RANGE)

    return rng.astype(np.float32)
