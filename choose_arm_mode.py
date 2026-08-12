#!/usr/bin/env python
"""
choose_arm_mode.py -- decide which ARM_FEATURE_MODE to use, with minimal output.

Prints only what the decision needs, for every column in every preset:

  1. VARIES?      relative std. A near-constant feature carries no signal and
                  just adds a noisy input dimension.
  2. TRACKS ERROR? |Pearson r| between the feature and the model's mass / mu
                  error. If a feature does not correlate with the error, the
                  model cannot use it to correct that error -- conditioning on
                  it cannot help.
  3. REDUNDANT?   correlation between features within a preset. Two highly
                  correlated features carry one feature's worth of information.

Then a one-line verdict per preset.

Needs a trained checkpoint to compute the error columns. Point the CHECKPOINT
constants at ANY trained model (conditioned or not) -- the error it makes on the
in-domain set is what we correlate against. Edit the constants, then run:

    python choose_arm_mode.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch

from configs import (CSV_PATH, M_SEEN_MIN, M_SEEN_MAX, MU_SEEN_MIN, MU_SEEN_MAX)
from dataset import (load_dataset_csv, ARM_FEATURE_PRESETS,
                     ARM_STATIC_COLS, ARM_STATE_COLS)
from models import PhysicsTransformerEstimator

# ---- point at any trained checkpoint (the UNCONDITIONED one is ideal: its
#      error reflects what conditioning would need to fix) ----
TIME = "20260812_012057"
MODEL_STR = "pinn_pcri-L1_p5c10.0_multiangle"
CHECKPOINT_DIR = f"./results/checkpoints/from_20260811/{TIME}/{MODEL_STR}"
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "transformer_epoch1000.pth")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")
# ---------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All columns referenced by any preset, deduped in a stable order.
ALL_COLS = []
for _cols in ARM_FEATURE_PRESETS.values():
    for c in _cols:
        if c not in ALL_COLS:
            ALL_COLS.append(c)


def get_errors(df, seq_len):
    """Run the checkpoint on the in-domain set, return per-row |mass err|, |mu err|.

    The model may be conditioned; if so we feed zeros (mean posture) so the error
    reflects the velocity-only competence, which is what we want to correlate the
    features against. If cond_dim is 0 we pass None.
    """
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    vel_cols = sorted([c for c in df.columns if "input_vel_" in c],
                      key=lambda x: int(x.split('_')[-1]))
    X_vel = torch.tensor(
        df[vel_cols].values.reshape(-1, seq_len, 1)).float().to(DEVICE)

    cond_dim = config.get('use_arm_state', False)
    from dataset import ARM_DIM
    cond_dim = ARM_DIM if config.get('use_arm_state', False) else 0

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
        cond_dim=cond_dim,
    ).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_vel), 2048):
            xb = X_vel[i:i + 2048]
            cb = torch.zeros((len(xb), cond_dim)).float().to(DEVICE) if cond_dim else None
            p, _, _ = model(xb, cond=cb)
            preds.append(p.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    ae_mass = np.abs(preds[:, 0] - df['gt_mass'].values)
    ae_mu = np.abs(preds[:, 1] - df['gt_mu'].values)
    return ae_mass, ae_mu


def main():
    df = load_dataset_csv()

    seq_len = len([c for c in df.columns if "input_acc_" in c])
    d = df[(df['gt_mass'].between(M_SEEN_MIN, M_SEEN_MAX)) &
           (df['gt_mu'].between(MU_SEEN_MIN, MU_SEEN_MAX)) &
           (df['start_t'] + seq_len <= 100)].copy()

    missing = [c for c in ALL_COLS if c not in d.columns]
    if missing:
        print(f"[WARNING] Columns missing from CSV: {missing}")
        ALL_COLS[:] = [c for c in ALL_COLS if c in d.columns]

    print(f"In-domain rows: {len(d)}")

    have_model = os.path.exists(WEIGHTS_PATH)
    if have_model:
        ae_mass, ae_mu = get_errors(d, seq_len)
        d = d.assign(_ae_mass=ae_mass, _ae_mu=ae_mu)
        print(f"Model in-domain mass MAE {ae_mass.mean():.4f}, "
              f"mu MAE {ae_mu.mean():.4f}  (feeding mean posture)\n")
    else:
        print(f"[NOTE] No checkpoint at {WEIGHTS_PATH}; skipping error correlation.\n")

    # ---- 1 & 2: per-feature variance and error correlation ----
    print("=" * 74)
    print("PER-FEATURE DIAGNOSTIC")
    print("=" * 74)
    print(f"{'feature':<26}{'mean':>9}{'std':>9}{'rel_std':>9}"
          f"{'|r| mass':>10}{'|r| mu':>9}")
    print("-" * 74)

    stats = {}
    for c in ALL_COLS:
        x = d[c].values.astype(np.float64)
        mean, std = x.mean(), x.std()
        rel = std / max(abs(mean), 1e-12)
        rM = rU = np.nan
        if have_model and std > 1e-12:
            rM = abs(np.corrcoef(x, d['_ae_mass'].values)[0, 1])
            rU = abs(np.corrcoef(x, d['_ae_mu'].values)[0, 1])
        stats[c] = dict(mean=mean, std=std, rel=rel, rM=rM, rU=rU)

        flag = "  <- ~const" if rel < 1e-3 else ""
        rM_s = f"{rM:>10.3f}" if np.isfinite(rM) else f"{'--':>10}"
        rU_s = f"{rU:>9.3f}" if np.isfinite(rU) else f"{'--':>9}"
        print(f"{c:<26}{mean:>9.4f}{std:>9.4f}{rel:>9.2e}{rM_s}{rU_s}{flag}")

    # ---- 3: redundancy within presets ----
    print("\n" + "=" * 74)
    print("REDUNDANCY (max |corr| to another feature in the same preset)")
    print("=" * 74)
    for mode, cols in ARM_FEATURE_PRESETS.items():
        cols = [c for c in cols if c in d.columns]
        if len(cols) < 2:
            print(f"  {mode:<16}: single feature, no redundancy")
            continue
        sub = d[cols].astype(np.float64)
        cm = sub.corr().abs().values
        np.fill_diagonal(cm, 0.0)
        worst = cm.max()
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        print(f"  {mode:<16}: max |corr| {worst:.2f} "
              f"between {cols[i]} & {cols[j]}")

    # ---- verdict per preset ----
    print("\n" + "=" * 74)
    print("VERDICT PER PRESET")
    print("=" * 74)
    for mode, cols in ARM_FEATURE_PRESETS.items():
        cols = [c for c in cols if c in d.columns]
        n = len(cols)
        n_const = sum(stats[c]['rel'] < 1e-3 for c in cols)
        if have_model:
            best_rM = max((stats[c]['rM'] for c in cols
                           if np.isfinite(stats[c]['rM'])), default=np.nan)
            best_rU = max((stats[c]['rU'] for c in cols
                           if np.isfinite(stats[c]['rU'])), default=np.nan)
            corr_txt = (f"best |r|: mass {best_rM:.2f}, mu {best_rU:.2f}"
                        if np.isfinite(best_rM) else "no error corr")
        else:
            corr_txt = "no model -> corr unknown"
        dead = f", {n_const} near-constant" if n_const else ""
        print(f"  {mode:<16} ({n} feat{dead}): {corr_txt}")

    print("\nHow to read this:")
    print("  - A feature with |r| < 0.1 against BOTH errors cannot help the model")
    print("    correct anything; including it only adds a noisy input.")
    print("  - Prefer the SMALLEST preset whose best |r| is meaningful (>~0.2).")
    print("  - If mu correlates but mass does not, that matches the finding that")
    print("    conditioning should feed the FRICTION head only, not the trunk.")
    print("  - High within-preset redundancy means the extra columns are")
    print("    duplicates; the smaller preset loses little.")


if __name__ == "__main__":
    main()