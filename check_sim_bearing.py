"""Independent check: is the sim's push_dir_b the TRAVEL direction?

The same geometry test that settled the real side, run on the sim sweep. No
code reading, no frame assumptions beyond "arm_ee_pos_* is in the base frame",
which session_const documents ("3 EE pos (base frame)").

    bearing = atan2(ee_y - obj_y, ee_x - obj_x)      # object -> EE
    push    = atan2(push_dir_b_y, push_dir_b_x)

    push - bearing ~ 180  ->  push_dir_b is the TRAVEL direction (EE drives
                              inward). 180 deg from the real PUSH_ANGLE_DEG,
                              which IS the bearing. Confirms SIM_ANGLE_OFFSET=180.

    push - bearing ~ 0    ->  push_dir_b is the bearing, same as the real side,
                              and the offset is WRONG.
"""
import os
import numpy as np, pandas as pd

CSV = ("/home/psxkf4/IsaacLab/source/collected_data/"
       "data_cube_closed_gripper_multi_angle_sb3/"
       "data_cube_closed_gripper_multi_angle_sb3.csv")
# session_const: OBJ_INIT_X = (TABLE_OFFSETS[0,0] + TABLE_OFFSETS[1,0]) / 2
OBJ_X = float(os.environ.get("OBJ_X", (0.10 + 0.65) / 2.0))
OBJ_Y = float(os.environ.get("OBJ_Y", 0.0))

df = pd.read_csv(CSV, usecols=["push_dir_b_x", "push_dir_b_y",
                               "arm_ee_pos_x", "arm_ee_pos_y"])
print(f"rows: {len(df)}   object assumed at ({OBJ_X:.3f}, {OBJ_Y:.3f})")

def wrap(a): return -((-np.asarray(a, float) + 180.0) % 360.0 - 180.0)

dx = df["arm_ee_pos_x"].to_numpy(float) - OBJ_X
dy = df["arm_ee_pos_y"].to_numpy(float) - OBJ_Y
r = np.hypot(dx, dy)
print(f"EE-to-object distance: median {np.median(r):.3f} m, "
      f"10-90% {np.percentile(r,10):.3f}..{np.percentile(r,90):.3f} m")
print("  (the real trials sat at 0.19 m; a near-zero distance makes the "
      "bearing\n   undefined and those rows are dropped)")

ok = r > 0.02
bearing = np.degrees(np.arctan2(dy[ok], dx[ok]))
push = np.degrees(np.arctan2(df["push_dir_b_y"].to_numpy(float)[ok],
                             df["push_dir_b_x"].to_numpy(float)[ok]))
diff = wrap(push - bearing)

print(f"\npush_dir_b - bearing   ({ok.sum()} rows):")
h, e = np.histogram(diff, bins=12, range=(-180, 180))
for c, lo in zip(h, e[:-1]):
    print(f"  {lo:+7.0f}..{lo+30:+4.0f} {'#' * int(60*c/max(h.max(),1)):<60} {c}")

near180 = (np.abs(np.abs(diff) - 180) < 45).mean()
near0 = (np.abs(diff) < 45).mean()
print(f"\n  within 45 deg of +-180: {near180:6.1%}")
print(f"  within 45 deg of    0 : {near0:6.1%}")
if near180 > 0.8:
    print("\n--> push_dir_b is the TRAVEL direction. SIM_ANGLE_OFFSET=180 is correct.")
elif near0 > 0.8:
    print("\n--> push_dir_b is the APPROACH BEARING. The offset is WRONG -- stop "
          "and tell me.")
else:
    print("\n--> inconclusive; send this output.")