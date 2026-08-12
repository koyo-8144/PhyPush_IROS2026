#!/usr/bin/env python
"""
ablate_token.py -- diagnose the arm-conditioned regression WITHOUT retraining.

The conditioned model is 8x worse on in-distribution mass (0.022 -> 0.185 kg).
That size of in-distribution drop is not "the feature is unhelpful" -- it is a
mechanical fault. This script separates the two possible causes on the EXISTING
checkpoint, in a couple of minutes:

  A) EVAL-TIME MIS-SCALING. The token is fed at a different scale than training,
     so a model that leaned on it produces garbage. If so, the trunk is fine and
     the fix is the stats sidecar (already added to dataset.py / evaluate.py).

  B) CORRUPTED TRUNK. Conditioning hijacked the encoder during training, so the
     velocity signal was down-weighted and the model cannot estimate without the
     token. If so, the checkpoint is not salvageable -- retrain.

Method: run the conditioned checkpoint three ways on the in-domain split and
compare mass/mu MAE:
  1. cond = SAVED-stats standardized features   (correct conditioning)
  2. cond = zeros                                (= standardized mean posture)
  3. cond = recomputed-on-eval features         (reproduces the old buggy path)

Reading:
  - (1) good, (3) bad   -> cause A. The sidecar fix already solves it; re-run
                            evaluate.py.
  - (1) ~ (2) ~ good    -> the token is being ignored; conditioning adds nothing
                            but does no harm. Honest result: use unconditioned.
  - all three bad, incl (1) with correct stats -> cause B. Retrain; this
                            checkpoint's trunk is corrupted.

Edit the CHECKPOINT constants to match evaluate.py, then: python ablate_token.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from models import PhysicsTransformerEstimator
from dataset import load_dataset_csv, get_arm_feature_cols, ARM_DIM
from configs import (CSV_PATH, M_SEEN_MIN, M_SEEN_MAX, MU_SEEN_MIN, MU_SEEN_MAX,
                     USE_ARM_STATE, ARM_FEATURE_MODE)

# ---- match evaluate.py ----
TIME = "20260811_221243"
MODEL_STR = "pinn_pcri-L1_p5c10.0_multiangle"
CHECKPOINT_DIR = f"./results/checkpoints/from_20260811/{TIME}/{MODEL_STR}"
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "transformer_epoch1000.pth")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")
# ---------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))


def run(model, X_vel, cond):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_vel), 2048):
            c = cond[i:i + 2048] if cond is not None else None
            p, _, _ = model(X_vel[i:i + 2048], cond=c)
            out.append(p.cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    if not USE_ARM_STATE:
        print("USE_ARM_STATE is False -- nothing to ablate.")
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    df = load_dataset_csv()

    vel_cols = sorted([c for c in df.columns if "input_vel_" in c],
                      key=lambda x: int(x.split('_')[-1]))
    seq_len = len([c for c in df.columns if "input_acc_" in c])

    # In-domain split (the split matters only for choosing rows; we evaluate on
    # the in-domain test set the same way evaluate.py's m_seen_mu_seen does).
    d = df[(df['gt_mass'].between(M_SEEN_MIN, M_SEEN_MAX)) &
           (df['gt_mu'].between(MU_SEEN_MIN, MU_SEEN_MAX)) &
           (df['start_t'] + seq_len <= 100)].copy()
    print(f"In-domain rows: {len(d)}")

    X_vel = torch.tensor(
        d[vel_cols].values.reshape(-1, seq_len, 1)).float().to(DEVICE)
    gt_mass = d['gt_mass'].values
    gt_mu = d['gt_mu'].values

    feat_cols = get_arm_feature_cols()
    raw = d[feat_cols].values.astype(np.float32)

    # --- stats variant 1: SAVED training stats ---
    saved_ok = True
    try:
        from dataset import load_arm_stats
        _, mean_saved, std_saved = load_arm_stats()
        mean_saved = np.asarray(mean_saved, np.float32).reshape(1, -1)
        std_saved = np.asarray(std_saved, np.float32).reshape(1, -1)
    except Exception as e:
        saved_ok = False
        print(f"[warn] no saved stats ({e}); retrain to generate them. "
              f"Variant 1 will be skipped.")

    # --- stats variant 3: recomputed on eval (the OLD buggy path) ---
    groups = (df['seed'].astype(int) * 100000 + df['env_id'].astype(int)).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, _ = next(gss.split(np.arange(len(df)), groups=groups))
    raw_full = df[feat_cols].values.astype(np.float32)
    mean_recomp = raw_full[tr_idx].mean(axis=0, keepdims=True)
    std_recomp = raw_full[tr_idx].std(axis=0, keepdims=True) + 1e-8

    if saved_ok:
        print("\n[stats] saved vs recomputed:")
        print(f"  mean saved    = {np.round(mean_saved.flatten(), 5)}")
        print(f"  mean recomp   = {np.round(mean_recomp.flatten(), 5)}")
        print(f"  std  saved    = {np.round(std_saved.flatten(), 5)}")
        print(f"  std  recomp   = {np.round(std_recomp.flatten(), 5)}")
        dm = np.abs(mean_saved - mean_recomp).max()
        ds = np.abs(std_saved - std_recomp).max()
        print(f"  max|Δmean|={dm:.2e}  max|Δstd|={ds:.2e}  "
              f"{'MATCH' if dm < 1e-5 and ds < 1e-5 else 'MISMATCH <-- this is the bug'}")

    model = PhysicsTransformerEstimator(
        input_dim=1, d_model=config['d_model'], nhead=4,
        num_encoder_layers=config['num_enc'],
        seq_len=config.get('seq_len', seq_len),
        dropout=config['dropout'], sharpness=config['sharpness'],
        cross_sharpness=config['cross_sharpness'],
        m_sharpness=config['m_sharpness'], mu_sharpness=config['mu_sharpness'],
        version=config['transformer_ver'],
        max_mass_scale=config['last_layer_ms'],
        max_mu_scale=config['last_layer_mus'],
        cond_dim=ARM_DIM,
    ).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))

    print("\n" + "=" * 62)
    print(f"ABLATION on in-domain test set  (mode='{ARM_FEATURE_MODE}', "
          f"cond_dim={ARM_DIM})")
    print("=" * 62)
    print(f"{'variant':<34}{'mass MAE':>12}{'mu MAE':>12}")
    print("-" * 62)

    results = {}

    if saved_ok:
        cond1 = torch.tensor((raw - mean_saved) / std_saved).float().to(DEVICE)
        p1 = run(model, X_vel, cond1)
        results['1_saved_stats'] = (mae(p1[:, 0], gt_mass), mae(p1[:, 1], gt_mu))
        print(f"{'1. cond = SAVED-stats (correct)':<34}"
              f"{results['1_saved_stats'][0]:>12.4f}{results['1_saved_stats'][1]:>12.4f}")

    cond2 = torch.zeros((len(X_vel), ARM_DIM)).float().to(DEVICE)
    p2 = run(model, X_vel, cond2)
    results['2_zeros'] = (mae(p2[:, 0], gt_mass), mae(p2[:, 1], gt_mu))
    print(f"{'2. cond = zeros (mean posture)':<34}"
          f"{results['2_zeros'][0]:>12.4f}{results['2_zeros'][1]:>12.4f}")

    cond3 = torch.tensor((raw - mean_recomp) / std_recomp).float().to(DEVICE)
    p3 = run(model, X_vel, cond3)
    results['3_recomputed'] = (mae(p3[:, 0], gt_mass), mae(p3[:, 1], gt_mu))
    print(f"{'3. cond = recomputed (old path)':<34}"
          f"{results['3_recomputed'][0]:>12.4f}{results['3_recomputed'][1]:>12.4f}")

    print("=" * 62)
    print("\nInterpretation:")
    m = {k: v[0] for k, v in results.items()}
    good = 0.06   # rough in-domain 'healthy' mass MAE from the unconditioned run
    if saved_ok and m.get('1_saved_stats', 9) < good < m.get('3_recomputed', 0):
        print("  Variant 1 good, variant 3 bad  ->  CAUSE A (eval mis-scaling).")
        print("  The sidecar fix already applied to evaluate.py solves it.")
        print("  Just re-run evaluate.py -- no retraining needed.")
    elif saved_ok and m.get('1_saved_stats', 9) > good:
        print("  Even variant 1 (correct stats) is bad  ->  CAUSE B (trunk")
        print("  corrupted during training). Retrain; this checkpoint is lost.")
    elif abs(results.get('2_zeros', (9, 9))[0]
             - results.get('1_saved_stats', (9, 9))[0]) < 0.01:
        print("  Variant 1 ~ variant 2  ->  the token is essentially ignored.")
        print("  Conditioning neither helps nor hurts here; the honest result")
        print("  is to report the UNCONDITIONED model.")
    else:
        print("  Mixed signal -- compare the three rows against the 0.022 kg")
        print("  in-domain mass MAE of the unconditioned PhyPush model.")
    print()


if __name__ == "__main__":
    main()