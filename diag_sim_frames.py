"""Why is (EE travel) - (push_dir_b) sitting at ~90 deg?

Three explanations all give median ~+90 and mean|diff| ~90, so the summary
cannot separate them. The SLOPE of diff against the push_dir angle can:

    slope  0  -> raw_ee_vel is in a frame rotated 90 deg from base
    slope -1  -> raw_ee_vel is in the EE/local frame (push along a fixed local
                 axis), so its direction says nothing about the base frame
    slope -2  -> ax0/ax1 are not the x/y I assumed

Reads a sample of rows, not the whole sweep.
"""
import re
import numpy as np, pandas as pd

CSV = ("/home/psxkf4/IsaacLab/source/collected_data/"
       "data_cube_closed_gripper_multi_angle_sb3/"
       "data_cube_closed_gripper_multi_angle_sb3.csv")
NROWS = 20000

cols = list(pd.read_csv(CSV, nrows=0).columns)
ts = sorted({int(m.group(1)) for c in cols
             if (m := re.fullmatch(r"raw_ee_vel_t(\d+)_ax0", c))})
axes = sorted({int(m.group(1)) for c in cols
               if (m := re.fullmatch(r"raw_ee_vel_t0_ax(\d+)", c))})
print(f"timesteps t{ts[0]}..t{ts[-1]}   axes ax{axes[0]}..ax{axes[-1]}")

use = ["push_dir_b_x", "push_dir_b_y"] + \
      [f"raw_ee_vel_t{t}_ax{a}" for t in ts for a in axes]
df = pd.read_csv(CSV, usecols=use, nrows=NROWS)
print(f"rows read: {len(df)}")

def wrap(a): return -((-a + 180.0) % 360.0 - 180.0)

V = np.stack([df[[f"raw_ee_vel_t{t}_ax{a}" for t in ts]].to_numpy(float)
              for a in axes], axis=-1)          # (rows, time, axis)

print("\nRMS of each axis over all rows/timesteps:")
for a in axes:
    print(f"  ax{a}: {np.sqrt((V[:, :, a] ** 2).mean()):.5f}")
print("  (for a horizontal push in a BASE frame, exactly one of the first "
      "three\n   should be near zero -- the vertical one)")

push = np.degrees(np.arctan2(df["push_dir_b_y"], df["push_dir_b_x"])).to_numpy()
print(f"\npush_dir_b angle: {len(np.unique(np.round(push,3)))} distinct values; "
      f"first few {np.unique(np.round(push,1))[:6]}")

# Direction per timestep, within a row: steady or wandering?
ang_t = np.degrees(np.arctan2(V[:, :, 1], V[:, :, 0]))
spread = np.array([np.std(wrap(r - np.median(r))) for r in ang_t[:2000]])
print(f"\nwithin-row spread of the travel angle: median {np.median(spread):.1f} deg")
print("  (the real trials sit near 1 deg; a large value means the window "
      "covers\n   more than one phase of motion and a mean direction is "
      "meaningless)")

mid = slice(len(ts) // 3, 2 * len(ts) // 3)     # middle third of the window
vx, vy = V[:, mid, 0].mean(1), V[:, mid, 1].mean(1)
trav = np.degrees(np.arctan2(vy, vx))
diff = wrap(trav - push)

print(f"\ndiff = travel - push_dir_b (middle third of the window):")
h, edges = np.histogram(diff, bins=12, range=(-180, 180))
for c, lo in zip(h, edges[:-1]):
    print(f"  {lo:+7.0f}..{lo+30:+4.0f} {'#' * int(60 * c / max(h.max(),1)):<60} {c}")

# The discriminating number.
for name, x in (("push angle", push),):
    A = np.vstack([x, np.ones_like(x)]).T
    # unwrap diff against the model before fitting, so wrapping does not
    # destroy a real linear relation
    best = None
    for slope in (0.0, -1.0, -2.0, 1.0, 2.0):
        resid = wrap(diff - slope * x)
        s = np.std(wrap(resid - np.median(resid)))
        print(f"  model diff = {slope:+.0f}*{name} + c  ->  "
              f"c = {np.median(resid):+7.2f}, residual sd = {s:6.2f}")
        if best is None or s < best[1]:
            best = (slope, s)
    print(f"\n  BEST FIT: slope {best[0]:+.0f} (residual sd {best[1]:.2f} deg)")