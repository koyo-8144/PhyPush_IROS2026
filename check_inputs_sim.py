"""
Input ranges of every INPUT_VARIANT, on the SIMULATION sweep (configs.CSV_PATH).

Answers one question: what values does each variant's model actually receive, and
how much do they vary? A channel that barely varies carries no information, and
standardizing it divides by a near-zero std, so any sim-to-real offset in that
channel is amplified by 1/std.

Run this first, then check_inputs_real.py on the robot machine, and compare the
two summary CSVs -- they share a schema on purpose.

WRITES  results/input_ranges/
            sim_input_ranges.csv        one row per (variant, channel)
            sim_channel_distributions.png
            sim_channel_vs_step.png
            sim_channel_vs_yaw.png

USAGE
    python check_inputs_sim.py
    MAX_ROWS=50000 python check_inputs_sim.py      # subsample for a quick look
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import CSV_PATH, VARIANT_TABLE, MULTI_ANGLE
from dataset import window_cols, arm_stats_path

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "input_ranges")
MAX_ROWS = int(os.environ.get("MAX_ROWS", 0))          # 0 = every row
CHUNKSIZE = int(os.environ.get("CHUNKSIZE", 20000))

VEL_PREFIX = "input_vel_"

# Human-readable names, and whether a bigger value is the adverse case.
CHANNEL_INFO = {
    "arm_manip_w":     ("manipulability $w$", "min"),
    "arm_dir_manip_w": ("directional manipulability $w_{dir}$", "min"),
    "arm_lam_w":       (r"mean translational $\Lambda_{ii}$ [kg]", "max"),
    "arm_meff_w":      ("effective mass $m_{eff}$ [kg]", "max"),
}


# =============================================================================
# LOAD -- only the columns this script needs
# =============================================================================
def load_channels():
    """(df, prefixes) with the window channels, velocity and the metadata columns."""
    header = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    prefixes = [p for p in CHANNEL_INFO if window_cols(header, p)]
    keep = []
    for p in prefixes + [VEL_PREFIX]:
        keep += window_cols(header, p)
    meta = [c for c in ("gt_mass", "gt_mu", "obj_yaw_base", "push_face_index",
                        "collect_idx", "seed", "env_id", "start_t", "domain")
            if c in header]
    keep += meta

    print(f"[load] {CSV_PATH}")
    print(f"  channels found: {prefixes or 'NONE'}")
    print(f"  loading {len(keep)} of {len(header)} columns")

    parts, n = [], 0
    for chunk in pd.read_csv(CSV_PATH, usecols=keep, chunksize=CHUNKSIZE):
        parts.append(chunk.astype({c: np.float32 for c in chunk.columns
                                   if chunk[c].dtype == np.float64}))
        n += len(chunk)
        if MAX_ROWS and n >= MAX_ROWS:
            break
        print(f"  ... {n} rows", end="\r")
    df = pd.concat(parts, ignore_index=True)
    if MAX_ROWS:
        df = df.iloc[:MAX_ROWS]
    print(f"\n  {len(df)} rows loaded")
    return df, prefixes


def sidecar_stats(variant):
    """(mean, std, log_transform) from the training sidecar, or None if absent.

    arm_stats_path() names the file for the CURRENT configs.INPUT_VARIANT, so the
    variant part is swapped in here to read every variant's file from one run.
    """
    import re
    path = re.sub(r"\.arm_stats\.[^.]+\.npz$", f".arm_stats.{variant}.npz",
                  arm_stats_path())
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return (float(np.asarray(d["mean"]).flatten()[0]),
            float(np.asarray(d["std"]).flatten()[0]),
            bool(d["log_transform"]) if "log_transform" in d else False)


# =============================================================================
# STATS
# =============================================================================
def channel_matrix(df, prefix, log_transform):
    """(W, valid) in MODEL SPACE: log(x) where the variant logs it."""
    cols = window_cols(df, prefix)
    W = df[cols].values.astype(np.float64)
    if log_transform:
        valid = np.isfinite(W).all(axis=1) & (W > 0).all(axis=1)
        W = np.where(valid[:, None], np.log(np.clip(W, 1e-12, None)), np.nan)
    else:
        valid = np.isfinite(W).all(axis=1)
    return W[valid], valid


def summarize(df, prefixes):
    """One row per variant; velocity is reported once, since every variant uses it."""
    rows = []

    vel = df[window_cols(df, VEL_PREFIX)].values.astype(np.float64)
    rows.append(dict(
        variant="(all)", channel=VEL_PREFIX + "*", role="velocity input",
        log=False, n_pushes=len(vel), n_steps=vel.shape[1],
        mean=vel.mean(), std=vel.std(),
        p1=np.percentile(vel, 1), p50=np.percentile(vel, 50),
        p99=np.percentile(vel, 99), vmin=vel.min(), vmax=vel.max(),
        rel_variation=vel.std() / max(abs(vel.mean()), 1e-12),
        within_push=np.mean(vel.std(axis=1)), between_push=vel.mean(axis=1).std()))

    for variant, (prefix, cond_dim, input_dim, log_tf) in VARIANT_TABLE.items():
        if prefix is None:
            rows.append(dict(variant=variant, channel="(none)", role="baseline",
                             log=False, n_pushes=len(df), n_steps=0))
            continue
        if prefix not in prefixes:
            print(f"  [skip] {variant}: no '{prefix}*' columns in this CSV")
            continue

        W, valid = channel_matrix(df, prefix, log_tf)
        if len(W) == 0:
            print(f"  [skip] {variant}: no valid rows")
            continue
        role = "cond token" if cond_dim else "input channel"
        # The cond variants feed ONE number per push: the window minimum.
        X = W.min(axis=1, keepdims=True) if cond_dim else W

        mean, std = float(X.mean()), float(X.std())
        rel = (np.exp(std) - 1.0) if log_tf else std / max(abs(mean), 1e-12)
        rec = dict(
            variant=variant, channel=f"{prefix}*", role=role, log=bool(log_tf),
            n_pushes=len(W), n_steps=W.shape[1],
            mean=mean, std=std,
            p1=float(np.percentile(X, 1)), p50=float(np.percentile(X, 50)),
            p99=float(np.percentile(X, 99)), vmin=float(X.min()), vmax=float(X.max()),
            rel_variation=rel,
            within_push=float(np.mean(W.std(axis=1))),
            between_push=float(W.mean(axis=1).std()),
            amplification=1.0 / max(std, 1e-12))
        if log_tf:
            rec["geom_mean_kg"] = float(np.exp(mean))

        sc = sidecar_stats(variant)
        if sc is not None:
            s_mean, s_std, s_log = sc
            rec.update(sidecar_mean=s_mean, sidecar_std=s_std, sidecar_log=s_log)
            z = (X - s_mean) / s_std
            rec.update(z_mean=float(z.mean()), z_std=float(z.std()),
                       z_min=float(z.min()), z_max=float(z.max()))
            if s_log != bool(log_tf):
                print(f"  [WARNING] {variant}: sidecar log_transform={s_log} but "
                      f"configs says {log_tf}.")

        for tgt in ("gt_mass", "gt_mu", "obj_yaw_base"):
            if tgt in df.columns:
                t = df[tgt].values.astype(np.float64)[valid]
                m = W.mean(axis=1)
                if t.std() > 0 and m.std() > 0:
                    rec[f"corr_{tgt}"] = float(np.corrcoef(m, t)[0, 1])
        rows.append(rec)
    return pd.DataFrame(rows)


def print_summary(tab):
    print("\n" + "=" * 96)
    print(" SIMULATION INPUT RANGES")
    print("=" * 96)
    print(f"  {'variant':<20}{'channel':<18}{'role':<15}{'mean':>10}{'std':>10}"
          f"{'var %':>9}{'min':>9}{'max':>9}")
    for _, r in tab.iterrows():
        if r.get("n_steps", 0) == 0 and r["role"] == "baseline":
            print(f"  {r['variant']:<20}{'(none)':<18}{'baseline':<15}"
                  f"{'velocity only':>47}")
            continue
        print(f"  {r['variant']:<20}{r['channel']:<18}{r['role']:<15}"
              f"{r['mean']:>10.4f}{r['std']:>10.4f}"
              f"{100 * r['rel_variation']:>8.2f}%{r['vmin']:>9.3f}{r['vmax']:>9.3f}")

    flat = tab[(tab.get("rel_variation", pd.Series(dtype=float)) < 0.02)
               & (tab["role"] != "baseline")]
    for _, r in flat.iterrows():
        print(f"\n  [WARNING] {r['variant']}: '{r['channel']}' varies only "
              f"{100 * r['rel_variation']:.2f}% across the sweep.")
        print(f"            It cannot describe the object, and standardizing divides "
              f"by {r['std']:.5f},")
        print(f"            so a 10% sim-to-real offset arrives at z="
              f"{(np.log(1.1) if r['log'] else 0.1 * abs(r['mean'])) / max(r['std'], 1e-12):+.1f}.")

    if "z_mean" in tab.columns:
        print("\n  Against the training sidecar (should be ~0 +/- 1 on this data):")
        for _, r in tab.dropna(subset=["z_mean"]).iterrows():
            print(f"    {r['variant']:<20} z mean {r['z_mean']:+7.2f}  "
                  f"std {r['z_std']:5.2f}  range [{r['z_min']:+.2f}, {r['z_max']:+.2f}]")
    print("=" * 96)


# =============================================================================
# PLOTS
# =============================================================================
def plot_distributions(df, prefixes, path):
    fig, axes = plt.subplots(2, len(prefixes), figsize=(5.5 * len(prefixes), 9),
                             squeeze=False)
    for j, p in enumerate(prefixes):
        label, adverse = CHANNEL_INFO[p]
        log_tf = any(spec[0] == p and spec[3] for spec in VARIANT_TABLE.values())
        raw = df[window_cols(df, p)].values.astype(np.float64)

        ax = axes[0][j]
        ax.hist(raw.ravel(), bins=80, color="tab:blue", alpha=0.75)
        ax.set_title(f"{p}*  all steps")
        ax.set_xlabel(label)
        ax.set_ylabel("count")

        ax = axes[1][j]
        red = raw.max(axis=1) if adverse == "max" else raw.min(axis=1)
        ax.hist(red, bins=60, color="tab:red", alpha=0.75)
        ax.set_title(f"per-push window {adverse}")
        ax.set_xlabel(label)
        ax.set_ylabel("pushes")
        spread = 100 * red.std() / max(abs(red.mean()), 1e-12)
        ax.text(0.02, 0.95, f"spread {spread:.2f}% of mean"
                + ("\nlog-transformed for the model" if log_tf else ""),
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.suptitle("Simulation: channel distributions (raw units)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  {path}")


def plot_vs_step(df, prefixes, path):
    fig, axes = plt.subplots(1, len(prefixes) + 1,
                             figsize=(5.0 * (len(prefixes) + 1), 4.4), squeeze=False)
    for j, p in enumerate(["velocity"] + prefixes):
        ax = axes[0][j]
        if p == "velocity":
            M = df[window_cols(df, VEL_PREFIX)].values.astype(np.float64)
            label, color = "v [m/s]", "tab:blue"
        else:
            M = df[window_cols(df, p)].values.astype(np.float64)
            label, color = CHANNEL_INFO[p][0], "tab:brown"
        steps = np.arange(M.shape[1])
        mu, sd = M.mean(axis=0), M.std(axis=0)
        ax.plot(steps, mu, color=color, linewidth=2)
        ax.fill_between(steps, mu - sd, mu + sd, color=color, alpha=0.25)
        ax.plot(steps, np.percentile(M, 1, axis=0), color=color, ls=":", lw=1)
        ax.plot(steps, np.percentile(M, 99, axis=0), color=color, ls=":", lw=1)
        ax.set_title(p if p != "velocity" else "input_vel_*")
        ax.set_xlabel("window step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle("Simulation: value against window step (mean +/- std, 1/99 pct)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  {path}")


def plot_vs_yaw(df, prefixes, path):
    if "obj_yaw_base" not in df.columns:
        return
    yaw = np.degrees(df["obj_yaw_base"].values.astype(float))
    bins = np.arange(np.floor(yaw.min() / 9) * 9, yaw.max() + 9, 9)
    idx = np.clip(np.digitize(yaw, bins) - 1, 0, len(bins) - 2)
    centres = bins[:-1] + 4.5

    fig, axes = plt.subplots(1, len(prefixes), figsize=(5.5 * len(prefixes), 4.4),
                             squeeze=False)
    for j, p in enumerate(prefixes):
        ax = axes[0][j]
        M = df[window_cols(df, p)].values.astype(np.float64).mean(axis=1)
        mu = np.array([M[idx == b].mean() if (idx == b).any() else np.nan
                       for b in range(len(centres))])
        sd = np.array([M[idx == b].std() if (idx == b).any() else np.nan
                       for b in range(len(centres))])
        ax.errorbar(centres, mu, yerr=sd, marker="o", markersize=4, capsize=2)
        ax.set_title(f"{p}* vs object yaw")
        ax.set_xlabel("object yaw [deg, 9 deg bins]")
        ax.set_ylabel(CHANNEL_INFO[p][0])
        ax.grid(alpha=0.3)
        # How much of the channel's variation the yaw explains.
        ok = ~np.isnan(mu)
        frac = (np.nanstd(mu) / max(M.std(), 1e-12)) if ok.any() else np.nan
        ax.text(0.02, 0.95, f"bin-mean spread / total spread = {frac:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.suptitle("Simulation: is the channel a function of the push angle?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  {path}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)

    df, prefixes = load_channels()
    if not prefixes:
        raise SystemExit("No arm window channels in this CSV.")

    tab = summarize(df, prefixes)
    print_summary(tab)

    csv_path = os.path.join(OUT_DIR, "sim_input_ranges.csv")
    tab.to_csv(csv_path, index=False)
    print(f"\n  written:\n  {csv_path}")

    plot_distributions(df, prefixes, os.path.join(OUT_DIR, "sim_channel_distributions.png"))
    plot_vs_step(df, prefixes, os.path.join(OUT_DIR, "sim_channel_vs_step.png"))
    plot_vs_yaw(df, prefixes, os.path.join(OUT_DIR, "sim_channel_vs_yaw.png"))

    print("\n  Next: copy this CSV to the robot machine and run check_inputs_real.py,")
    print("  which compares the real channels against these ranges.")


if __name__ == "__main__":
    main()