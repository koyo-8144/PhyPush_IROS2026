"""Is the sim's push_dir_b the direction the EE TRAVELS, or the bearing it
approaches FROM?

On the real robot this is now settled: PUSH_ANGLE_DEG is the approach bearing
(object -> EE), 180 deg from the direction the arm actually moves, confirmed on
five trials by forward kinematics to within 2 deg.

If the sim's push_dir_b is the TRAVEL direction instead, the two sides are 180
deg apart -- and on the +/-54..126 grid a 180 deg offset maps 54<->126 and
72<->108, which REVERSES the |angle| ordering and would flip the sign of every
rho_sym in the agreement table.

This compares push_dir_b against the realised EE velocity in the same file.
"""
import re
import numpy as np, pandas as pd

CSV = ("/home/psxkf4/IsaacLab/source/collected_data/"
       "data_cube_closed_gripper_multi_angle_sb3/"
       "data_cube_closed_gripper_multi_angle_sb3.csv")

cols = list(pd.read_csv(CSV, nrows=0).columns)
ts = sorted({int(m.group(1)) for c in cols
             if (m := re.fullmatch(r"raw_ee_vel_t(\d+)_ax0", c))})
if not ts:
    raise SystemExit("no raw_ee_vel_t*_ax0 columns found")
print(f"EE-velocity time indices found: t{ts[0]}..t{ts[-1]} ({len(ts)})")

vel = [f"raw_ee_vel_t{t}_ax{a}" for t in ts for a in (0, 1)]
df = pd.read_csv(CSV, usecols=["push_dir_b_x", "push_dir_b_y"] + vel)
print(f"rows: {len(df)}")

def wrap(a): return -((-a + 180.0) % 360.0 - 180.0)

# Mean EE velocity over the window, per row, in the file's own x/y axes.
vx = df[[f"raw_ee_vel_t{t}_ax0" for t in ts]].to_numpy(float).mean(1)
vy = df[[f"raw_ee_vel_t{t}_ax1" for t in ts]].to_numpy(float).mean(1)
speed = np.hypot(vx, vy)
ok = speed > np.percentile(speed, 10)        # drop near-stationary rows
print(f"usable rows (moving): {ok.sum()}")

push = np.degrees(np.arctan2(df["push_dir_b_y"], df["push_dir_b_x"]))[ok]
trav = np.degrees(np.arctan2(vy, vx))[ok]
diff = wrap(trav - push)

print(f"\npush_dir_b angle : {push.min():+7.1f} .. {push.max():+7.1f} deg")
print(f"EE travel angle  : {trav.min():+7.1f} .. {trav.max():+7.1f} deg")
print(f"\n(EE travel) - (push_dir_b):")
print(f"  median {np.median(diff):+7.2f} deg   "
      f"mean|.| {np.abs(diff).mean():6.2f}   "
      f"mean|180-|.|| {np.abs(180 - np.abs(diff)).mean():6.2f}")
near0 = (np.abs(diff) < 30).mean()
near180 = (np.abs(np.abs(diff) - 180) < 30).mean()
print(f"  within 30 deg of    0 : {near0:6.1%}")
print(f"  within 30 deg of +-180: {near180:6.1%}")

if near0 > 0.8:
    print("\n--> push_dir_b is the TRAVEL direction."
          "\n    It is 180 deg from the real PUSH_ANGLE_DEG convention.")
elif near180 > 0.8:
    print("\n--> push_dir_b is the APPROACH BEARING."
          "\n    Same convention as the real PUSH_ANGLE_DEG. Nothing to fix.")
else:
    print("\n--> neither: the two are not in the same frame."
          "\n    Send this output and the sim collection code.")