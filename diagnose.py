#!/usr/bin/env python
"""
diagnose.py -- two diagnostics for the multi-angle estimator, standalone so you
do not have to touch evaluate.py to run them.

    1. report_robot_force_coverage(df)
       How many rows have an all-zero robot wrench, i.e. the gripper contact
       filter never fired. On those rows Newton's law cannot close and mass is
       unidentifiable regardless of the model -- a hard precision ceiling. Broken
       out by yaw / push side / seed to show whether the failure is structured.

    2. evaluate_by_orientation(model, df, ...)
       Estimation error per (object yaw x push side), plus correlations between
       the arm features and the error. Answers whether the orientation confound
       is real -- if error is flat across yaw, conditioning buys nothing and
       tuning it is wasted effort.

Run:
    python diagnose.py

It reads CSV_PATH, the checkpoint under CHECKPOINT_DIR (edit the two constants
below to match evaluate.py), applies the SAVED training arm stats, and prints
both reports.
"""
import os
import json
import numpy as np
import pandas as pd
import torch

from configs import (CSV_PATH, MULTI_ANGLE, USE_ARM_STATE, ARM_FEATURE_MODE,
                     M_SEEN_MIN, M_SEEN_MAX, MU_SEEN_MIN, MU_SEEN_MAX,
                     FRAME_MODE, GLOBAL_M_RANGE, GLOBAL_MU_RANGE)
from dataset import (load_dataset_csv, get_arm_feature_cols, load_arm_stats,
                     ARM_DIM, ARM_FEATURE_COLS)
from models import PhysicsTransformerEstimator

# ---- EDIT THESE TO MATCH evaluate.py ----
TIME = "20260811_184657"
TIME = "20260811_221243"
MODEL_STR = "pinn_pcri-L1_p5c10.0_multiangle"
CHECKPOINT_DIR = f"./results/checkpoints/from_20260811/{TIME}/{MODEL_STR}"
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "transformer_epoch1000.pth")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")
# -----------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 1. ROBOT FORCE COVERAGE
# =====================================================================
def report_robot_force_coverage(df, seq_len=60):
    col_ax = 5 if FRAME_MODE == "local" else 3
    cols = [f"pinn_robot_wrench_t{t}_ax{col_ax}" for t in range(100)]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("[ROBOT FORCE] wrench columns not found; skipping.")
        return

    print("\n" + "=" * 70)
    print("ROBOT CONTACT FORCE COVERAGE")
    print("=" * 70)

    w = df[cols].values
    max_abs = np.abs(w).max(axis=1)
    dead = max_abs < 1e-9

    print(f"\n  rows with all-zero robot wrench: {dead.sum()} / {len(df)} "
          f"({100 * dead.mean():.1f}%)")
    print(f"  max|F_robot| over push: median {np.median(max_abs):.4f} N, "
          f"p10 {np.percentile(max_abs, 10):.4f} N, "
          f"p90 {np.percentile(max_abs, 90):.4f} N")

    if dead.sum() == 0:
        print("  All rows carry a non-zero robot wrench -- physics can close.")
        print("=" * 70 + "\n")
        return

    print("\n  [WARNING] On these rows F_robot - F_fric cannot reconstruct F_net,")
    print("            so mass is unidentifiable no matter the model. This is a")
    print("            hard precision ceiling, not a tuning problem.")

    d = df.copy()
    d["_dead"] = dead
    for col in ["obj_yaw_base", "push_face_index", "seed"]:
        if col in d.columns:
            rate = d.groupby(col)["_dead"].mean().round(4) * 100
            print(f"\n  zero-force rate (%) by {col}:")
            print(rate.to_string())
            if rate.max() - rate.min() > 10:
                print(f"    [NOTE] varies by {rate.max() - rate.min():.1f} points "
                      f"across {col} -- structured failure, not random noise.")

    if {"obj_yaw_base", "push_face_index"}.issubset(d.columns):
        piv = d.pivot_table(index="obj_yaw_base", columns="push_face_index",
                            values="_dead", aggfunc="mean").round(4) * 100
        print("\n  zero-force rate (%) by (yaw x push side):")
        print(piv.to_string())

    print("=" * 70 + "\n")


# =====================================================================
# 2. ERROR BY ORIENTATION
# =====================================================================
def evaluate_by_orientation(model, df, vel_cols, seq_len, cond_cols, arm_mean,
                            arm_std):
    if not {"obj_yaw_base", "push_face_index"}.issubset(df.columns):
        print("[ORIENTATION] obj_yaw_base / push_face_index missing; skipping.")
        return

    d = df[(df["gt_mass"].between(M_SEEN_MIN, M_SEEN_MAX)) &
           (df["gt_mu"].between(MU_SEEN_MIN, MU_SEEN_MAX)) &
           (df["start_t"] + seq_len <= 100)].copy()
    if len(d) == 0:
        print("[ORIENTATION] no in-domain rows; skipping.")
        return

    X_vel = torch.tensor(
        d[vel_cols].values.reshape(-1, seq_len, 1)).float().to(DEVICE)

    cond = None
    if cond_cols:
        cond_arr = (d[cond_cols].values.astype(np.float32) - arm_mean) / arm_std
        cond = torch.tensor(cond_arr).float().to(DEVICE)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_vel), 2048):
            c = cond[i:i + 2048] if cond is not None else None
            p, _, _ = model(X_vel[i:i + 2048], cond=c)
            preds.append(p.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    d["pred_mass"] = preds[:, 0]
    d["pred_mu"] = preds[:, 1]
    d["ae_mass"] = np.abs(d["pred_mass"] - d["gt_mass"])
    d["ae_mu"] = np.abs(d["pred_mu"] - d["gt_mu"])

    print("\n" + "=" * 70)
    print("ESTIMATION ERROR BY ORIENTATION AND PUSH SIDE")
    print("=" * 70)

    by_cell = d.groupby(["obj_yaw_base", "push_face_index"]).agg(
        n=("ae_mass", "size"),
        mass_mae=("ae_mass", "mean"),
        mu_mae=("ae_mu", "mean"),
    ).round(4)
    print(by_cell.to_string())

    overall_m = d["ae_mass"].mean()
    overall_mu = d["ae_mu"].mean()
    print(f"\n  OVERALL in-domain: mass MAE {overall_m:.4f} kg "
          f"({100 * overall_m / GLOBAL_M_RANGE:.1f}% of range), "
          f"mu MAE {overall_mu:.4f} ({100 * overall_mu / GLOBAL_MU_RANGE:.1f}%)")

    by_yaw = d.groupby("obj_yaw_base")[["ae_mass", "ae_mu"]].mean()
    sm = by_yaw["ae_mass"].max() - by_yaw["ae_mass"].min()
    su = by_yaw["ae_mu"].max() - by_yaw["ae_mu"].min()
    print(f"  mass MAE spread across yaws: {sm:.4f} kg "
          f"({100 * sm / max(by_yaw['ae_mass'].mean(), 1e-9):.0f}% of mean)")
    print(f"  mu   MAE spread across yaws: {su:.4f} "
          f"({100 * su / max(by_yaw['ae_mu'].mean(), 1e-9):.0f}% of mean)")
    print("  Spread << mean  -> orientation does NOT matter; conditioning is")
    print("                     not the bottleneck, look at data/capacity.")
    print("  Spread ~ mean   -> orientation matters; conditioning is relevant.")

    arm_present = [c for c in ARM_FEATURE_COLS if c in d.columns]
    if arm_present:
        corr = d[arm_present + ["ae_mass", "ae_mu"]].corr().loc[
            arm_present, ["ae_mass", "ae_mu"]]
        corr["max_abs"] = corr.abs().max(axis=1)
        corr = corr.sort_values("max_abs", ascending=False).round(4)
        print("\n  ARM FEATURE vs ERROR (Pearson, top 8):")
        print(corr.head(8).to_string())
        print("  |r| < 0.1 everywhere -> arm features do not track error;")
        print("                          conditioning cannot help much.")

    print("=" * 70 + "\n")
    by_cell.to_csv(os.path.join(CHECKPOINT_DIR, "diag_error_by_orientation.csv"))


def main():
    print(f"Loading {CSV_PATH} ...")
    df = load_dataset_csv()

    acc_cols = sorted([c for c in df.columns if "input_acc_" in c],
                      key=lambda x: int(x.split("_")[-1]))
    vel_cols = sorted([c for c in df.columns if "input_vel_" in c],
                      key=lambda x: int(x.split("_")[-1]))
    seq_len = len(acc_cols)

    # ---- Diagnostic 1 needs no model ----
    if MULTI_ANGLE:
        report_robot_force_coverage(df, seq_len=seq_len)

    # ---- Build model from the saved config, load weights ----
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    cond_cols, arm_mean, arm_std = [], None, None
    cond_dim = 0
    if USE_ARM_STATE:
        cond_cols, arm_mean, arm_std = load_arm_stats()   # SAVED training stats
        cond_dim = ARM_DIM
        print(f"[ARM] loaded stats for {cond_cols} (cond_dim={cond_dim})")

    model = PhysicsTransformerEstimator(
        input_dim=1,
        d_model=config["d_model"],
        nhead=4,
        num_encoder_layers=config["num_enc"],
        seq_len=config.get("seq_len", seq_len),
        dropout=config["dropout"],
        sharpness=config["sharpness"],
        cross_sharpness=config["cross_sharpness"],
        m_sharpness=config["m_sharpness"],
        mu_sharpness=config["mu_sharpness"],
        version=config["transformer_ver"],
        max_mass_scale=config["last_layer_ms"],
        max_mu_scale=config["last_layer_mus"],
        cond_dim=cond_dim,
    ).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[FATAL] weights not found: {WEIGHTS_PATH}")
        return
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("Weights loaded.")

    # ---- Diagnostic 2 ----
    evaluate_by_orientation(model, df, vel_cols, seq_len,
                            cond_cols, arm_mean, arm_std)


if __name__ == "__main__":
    main()