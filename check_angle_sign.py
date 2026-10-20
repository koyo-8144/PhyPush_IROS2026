"""Does the sim's realised push direction follow +a0*180 or -a0*180?

Reads only the columns it needs, so it is fast on the full sweep.
"""
import numpy as np, pandas as pd

CSV = ("/home/psxkf4/IsaacLab/source/collected_data/"
       "data_cube_closed_gripper_multi_angle_sb3/"
       "data_cube_closed_gripper_multi_angle_sb3.csv")

head = pd.read_csv(CSV, nrows=0)
cols = list(head.columns)
plain = [c for c in cols if not c.startswith(("input_vel", "arm_"))]
print("non-window columns:")
print(" ", plain)

need = [c for c in ("push_dir_b_x", "push_dir_b_y") if c in cols]
if len(need) < 2:
    raise SystemExit("\npush_dir_b_x / push_dir_b_y not found -- send the list above.")

# Anything that could be the action / a0 that the angle is built from.
cand = [c for c in plain if any(k in c.lower() for k in
                                ("a0", "action", "push_angle", "angle"))]
print("\ncandidate action/angle columns:", cand or "NONE FOUND")

df = pd.read_csv(CSV, usecols=need + cand)
ang = np.degrees(np.arctan2(df["push_dir_b_y"], df["push_dir_b_x"]))
ang = -((-ang + 180.0) % 360.0 - 180.0)          # wrap to (-180, 180]
print(f"\nrealised angle: {len(ang)} rows, "
      f"{ang.min():.1f} to {ang.max():.1f} deg, "
      f"{(ang < 0).sum()} negative / {(ang >= 0).sum()} positive")

def circ_err(a, b):
    return np.abs(-((-(a - b) + 180.0) % 360.0 - 180.0))

for c in cand:
    v = df[c].astype(float)
    for name, pred in (("a0*180", v * 180.0), ("a0*180 (already deg)", v)):
        same, flip = circ_err(ang, pred).mean(), circ_err(ang, -pred).mean()
        if min(same, flip) > 45:        # this column is not the angle source
            continue
        verdict = "SAME sign" if same < flip else "OPPOSITE sign"
        print(f"\n  {c}  as {name}")
        print(f"    mean |realised - (+pred)| = {same:6.2f} deg")
        print(f"    mean |realised - (-pred)| = {flip:6.2f} deg")
        print(f"    --> {verdict}")