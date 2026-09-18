"""
Push-angle analysis for the simulation estimators.

Mirrors the two real-robot scripts:
    make_angle_tables.py              worst-angle tables + per-estimator histograms
    inspect_angle_manipulability.py   manipulability sequences grouped by angle

THREE ADAPTATIONS ARE FORCED BY THE SIMULATION DATA

1. NO OBJECT-SURFACE PAIR. The real robot has 15 physical pairs; sim has hundreds of
   held-out property groups, too many to plot and too thin individually. PROPERTY
   BUCKETS -- a quantile grid over mass and mu -- stand in: ~9 domains of ~1400 rows,
   comparable in count to the 15 real pairs and well populated per angle bin.

2. ANGLES ARE CONTINUOUS. The real robot repeats a fixed set, so "worst angle" is a
   value; here it is a BIN, and the histograms count bins.

3. SEQUENCES CANNOT BE DRAWN PER ROW. Thousands of traces would be unreadable, so the
   mean trace per angle bin is drawn with a +/-1 std band.

EVALUATION IS RESTRICTED TO HELD-OUT PROPERTY GROUPS. Each (mass, mu) pair appears
~40 times in the multi-angle sweep, so scoring every row would mostly measure
memorization. The GroupShuffleSplit here replays the one training used.

USAGE
    python analyze_push_angle.py
    VARIANTS="vel_only vel_manip_cond" python analyze_push_angle.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from sklearn.model_selection import GroupShuffleSplit
from configs import GRIPPER_CLOSED

# ==========================================
# 1. CONFIGURATION
# ==========================================
if GRIPPER_CLOSED:
    csv_folder_name = "data_cube_closed_gripper_multi_angle_sb3"
else:
    csv_folder_name = "data_cube_opened_gripper_multi_angle_sb3"

CSV_PATH = os.environ.get(
    "CSV_PATH",
    f"/home/psxkf4/IsaacLab/source/collected_data/{csv_folder_name}/{csv_folder_name}.csv")

TRANS_BASE = os.environ.get("TRANS_BASE", "/home/psxkf4/PhyPush/results/checkpoints/from_20260913")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "angle_analysis")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
MANIP_DIR = os.path.join(OUT_DIR, "manip_by_angle")

# Normalize by the range the model was TRAINED to cover, not the held-out set's own
# span, so these numbers sit on the same scale as the real-world tables and neither
# can be moved by the choice of test objects.
M_RANGE = 2.0 - 0.2
MU_RANGE = 0.5 - 0.15

# Seen bounds, for the in-domain row filter.
M_SEEN_MIN, M_SEEN_MAX = 0.2, 2.0
MU_SEEN_MIN, MU_SEEN_MAX = 0.15, 0.5

GROUP_SPLIT_SEED = 42
SEQ_LEN = 60

N_ANGLE_BINS = int(os.environ.get("N_ANGLE_BINS", 12))
N_M_BUCKETS = int(os.environ.get("N_M_BUCKETS", 3))
N_MU_BUCKETS = int(os.environ.get("N_MU_BUCKETS", 3))
TOP_ANGLES = int(os.environ.get("TOP_ANGLES", 3))
MIN_ROWS_PER_BIN = int(os.environ.get("MIN_ROWS_PER_BIN", 20))

BATCH = 512

# Push-quality filter, matching dataset.py. Set to None to disable.
MAX_PUSH_LATERAL_OFFSET = 0.005
MAX_PUSH_COM_OFFSET = 0.05

#   variant -> (window prefix, cond_dim, input_dim, run_time, model_string)
if GRIPPER_CLOSED:
    VARIANT_SPEC = {
        "vel_only": (
            None, 0, 1, "20260913_134633", "pinn_pcri-L1_p5c10.0_multiangle"),
        "vel_manip_cond": (
            "arm_manip_w", 1, 1, "20260913_102210",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_manip_cond"),
        "vel_dirmanip_cond": (
            "arm_dir_manip_w", 1, 1, "20260913_110510",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_dirmanip_cond"),
        "vel_manip_seq": (
            "arm_manip_w", 0, 2, "20260913_120624",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_manip_seq"),
        "vel_dirmanip_seq": (
            "arm_dir_manip_w", 0, 2, "20260913_124812",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_dirmanip_seq"),
        "vel_osim_seq": (
            "arm_lam_w", 0, 2, "20260916_195532",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_osim_seq"),
        "vel_eff_seq": (
            "arm_dir_manip_w", 0, 2, "20260916_231604",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_eff_seq"),
    }
else:
    VARIANT_SPEC = {
        # "vel_only": (
        #     None, 0, 1, "20260913_134633", "pinn_pcri-L1_p5c10.0_multiangle"),
        # "vel_manip_cond": (
        #     "arm_manip_w", 1, 1, "20260913_102210",
        #     "pinn_pcri-L1_p5c10.0_multiangle_vel_manip_cond"),
        # "vel_dirmanip_cond": (
        #     "arm_dir_manip_w", 1, 1, "20260913_110510",
        #     "pinn_pcri-L1_p5c10.0_multiangle_vel_dirmanip_cond"),
        # "vel_manip_seq": (
        #     "arm_manip_w", 0, 2, "20260913_120624",
        #     "pinn_pcri-L1_p5c10.0_multiangle_vel_manip_seq"),
        # "vel_dirmanip_seq": (
        #     "arm_dir_manip_w", 0, 2, "20260913_124812",
        #     "pinn_pcri-L1_p5c10.0_multiangle_vel_dirmanip_seq"),
        "vel_osim_seq": (
            "arm_lam_w", 0, 2, "20260917_144436",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_osim_seq"),
        "vel_eff_seq": (
            "arm_meff_w", 0, 2, "20260917_153423",
            "pinn_pcri-L1_p5c10.0_multiangle_vel_eff_seq"),
    }

VARIANT_LABEL = {
    "vel_only":          r"vel only",
    "vel_manip_cond":    r"$w$ cond",
    "vel_dirmanip_cond": r"$w_{dir}$ cond",
    "vel_manip_seq":     r"$w$ seq",
    "vel_dirmanip_seq":  r"$w_{dir}$ seq",
}

VARIANTS = os.environ.get("VARIANTS", " ".join(VARIANT_SPEC)).split()


def window_cols(header, prefix):
    """prefix + pure digits, ordered numerically. Strict digit match so a future
    column sharing the prefix cannot join the sequence and shift every timestep."""
    cols = [c for c in header if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


def transformer_paths(variant):
    _, _, _, run_time, model_string = VARIANT_SPEC[variant]
    folder = os.path.join(TRANS_BASE, run_time, model_string)
    return (os.path.join(folder, "transformer_epoch1000.pth"),
            os.path.join(folder, "config.json"))


def load_variant_stats(variant):
    """Standardization from the sidecar PhyPush wrote during training. Never
    recomputed here: a different split gives a different mean/std and silently
    mis-scales the input to a frozen model trained to depend on it."""
    prefix = VARIANT_SPEC[variant][0]
    if prefix is None:
        return None, None
    path = f"{CSV_PATH}.arm_stats.{variant}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Variant stats not found at {path}")
    d = np.load(path, allow_pickle=True)
    saved = str(d["variant"]) if "variant" in d else (
        str(d["mode"]) if "mode" in d else None)
    if saved is not None and saved != variant:
        raise ValueError(f"stats at {path} are for '{saved}', expected '{variant}'")
    return float(np.asarray(d["mean"]).flatten()[0]), float(np.asarray(d["std"]).flatten()[0])


# ==========================================
# 2. DATA
# ==========================================
def load_heldout():
    """Held-out rows, with push angle and bin attached."""
    header = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()

    vel_cols = sorted([c for c in header if "input_vel_" in c],
                      key=lambda x: int(x.split("_")[-1]))
    if len(vel_cols) != SEQ_LEN:
        print(f"[WARNING] {len(vel_cols)} input_vel_* columns, expected {SEQ_LEN}")

    need = vel_cols + ["gt_mass", "gt_mu", "start_t", "push_dir_b_x", "push_dir_b_y"]
    for c in ("seed", "env_id", "push_start_lateral_offset", "push_com_offset",
              "worst_manipulability", "worst_dir_manipulability"):
        if c in header:
            need.append(c)
    for pfx in ("arm_manip_w", "arm_dir_manip_w"):
        need += window_cols(header, pfx)
    need = list(dict.fromkeys(c for c in need if c in header))

    missing = [c for c in ("push_dir_b_x", "push_dir_b_y") if c not in header]
    if missing:
        raise SystemExit(f"CSV has no {missing}; the push angle cannot be derived.")

    print(f"  reading {len(need)} of {len(header)} columns")
    df = pd.read_csv(CSV_PATH, usecols=need)
    n0 = len(df)

    # Same filter order as dataset.create_dataloaders.
    df = df[(df["gt_mass"] >= M_SEEN_MIN) & (df["gt_mass"] <= M_SEEN_MAX)
            & (df["gt_mu"] >= MU_SEEN_MIN) & (df["gt_mu"] <= MU_SEEN_MAX)].copy()
    df = df[(df["start_t"] + SEQ_LEN) <= 100].copy()
    if MAX_PUSH_LATERAL_OFFSET is not None and "push_start_lateral_offset" in df.columns:
        df = df[df["push_start_lateral_offset"] < MAX_PUSH_LATERAL_OFFSET].copy()
    if MAX_PUSH_COM_OFFSET is not None and "push_com_offset" in df.columns:
        df = df[df["push_com_offset"] < MAX_PUSH_COM_OFFSET].copy()

    # Held-out property groups only: each (mass, mu) pair appears ~40 times, so
    # scoring every row would mostly measure memorization.
    if "seed" in df.columns and "env_id" in df.columns:
        groups = (df["seed"].astype(int) * 100000 + df["env_id"].astype(int)).values
    else:
        groups = (df["gt_mass"].round(6).astype(str) + "_"
                  + df["gt_mu"].round(6).astype(str)).values
    idx = np.arange(len(df))
    _, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=GROUP_SPLIT_SEED).split(idx, groups=groups))
    df = df.iloc[test_idx].reset_index(drop=True)

    print(f"  {n0} rows -> {len(df)} held-out "
          f"({len(np.unique(groups[test_idx]))} property groups)")

    # Push direction in the base frame -> angle.
    df["push_angle_deg"] = np.degrees(np.arctan2(df["push_dir_b_y"], df["push_dir_b_x"]))
    edges = np.linspace(-180.0, 180.0, N_ANGLE_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    df["angle_bin"] = np.clip(np.digitize(df["push_angle_deg"], edges) - 1,
                              0, N_ANGLE_BINS - 1)

    return df, vel_cols, centers


def add_property_buckets(df):
    """Mass x mu buckets, the stand-in for the real robot's object-surface pairs.
    Quantile edges so every bucket holds a similar number of rows."""
    df = df.copy()
    mq = pd.qcut(df["gt_mass"], N_M_BUCKETS, labels=False, duplicates="drop")
    uq = pd.qcut(df["gt_mu"], N_MU_BUCKETS, labels=False, duplicates="drop")
    df["domain"] = [f"m{int(a)}_mu{int(b)}" for a, b in zip(mq, uq)]
    return df


def build_input(df, vel_cols, variant, mean, std, device):
    prefix, cond_dim, input_dim, _, _ = VARIANT_SPEC[variant]
    X = df[vel_cols].values.astype(np.float32)[:, :, None]
    COND = None
    if prefix is not None:
        wc = window_cols(df.columns.tolist(), prefix)
        if len(wc) != X.shape[1]:
            raise ValueError(f"'{prefix}*' has {len(wc)} steps, velocity has {X.shape[1]}")
        W = df[wc].values.astype(np.float32)
        if cond_dim > 0:
            # MINIMUM over the window -- the same reduction dataset.py applies.
            COND = ((W.min(axis=1, keepdims=True) - mean) / std).astype(np.float32)
        else:
            X = np.concatenate([X, ((W - mean) / std)[:, :, None]], axis=-1)
    if X.shape[-1] != input_dim:
        raise ValueError(f"built X with {X.shape[-1]} channel(s), expected {input_dim}")
    return (torch.tensor(X).float().to(device),
            torch.tensor(COND).float().to(device) if COND is not None else None)


def predict(df, vel_cols, variant, device):
    from models import PhysicsTransformerEstimator

    mean, std = load_variant_stats(variant)
    w_path, c_path = transformer_paths(variant)
    if not (os.path.exists(w_path) and os.path.exists(c_path)):
        raise FileNotFoundError(f"transformer missing under {os.path.dirname(w_path)}")
    cfg = json.load(open(c_path))

    ck_v = cfg.get("input_variant")
    if ck_v is not None and ck_v != variant:
        raise ValueError(f"VARIANT_SPEC points at a checkpoint whose config.json "
                         f"says input_variant='{ck_v}'")

    _, cond_dim, input_dim, _, _ = VARIANT_SPEC[variant]
    model = PhysicsTransformerEstimator(
        input_dim=input_dim, d_model=cfg["d_model"], nhead=4,
        num_encoder_layers=cfg["num_enc"], seq_len=int(cfg.get("seq_len", SEQ_LEN)),
        dropout=0.0, sharpness=cfg["sharpness"], cross_sharpness=cfg["cross_sharpness"],
        m_sharpness=cfg["m_sharpness"], mu_sharpness=cfg["mu_sharpness"],
        version=cfg["transformer_ver"], max_mass_scale=cfg["last_layer_ms"],
        max_mu_scale=cfg["last_layer_mus"], cond_dim=cond_dim).to(device)
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()

    X, COND = build_input(df, vel_cols, variant, mean, std, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            c = COND[i:i + BATCH] if COND is not None else None
            out.append(model(X[i:i + BATCH], cond=c)[0].cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


# ==========================================
# 3. ANGLE vs ERROR
# ==========================================
def _spearman(a, b):
    """Rank correlation: robust to the heavy tail in absolute error and to a
    monotone but non-linear angle dependence that Pearson would understate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan, n
    ra, rb = np.empty(n), np.empty(n)
    ra[np.argsort(a, kind="mergesort")] = np.arange(n)
    rb[np.argsort(b, kind="mergesort")] = np.arange(n)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return (float((ra * rb).sum() / d) if d > 0 else np.nan), n


def variant_summary(df, preds, variant):
    m_err = np.abs(preds[:, 0] - df["gt_mass"].values)
    u_err = np.abs(preds[:, 1] - df["gt_mu"].values)
    ang = df["push_angle_deg"].values
    rec = {"variant": variant, "n": len(df),
           "mass_mae": round(float(m_err.mean()), 5),
           "mu_mae": round(float(u_err.mean()), 5),
           "mass_nmae_pct": round(100 * float(m_err.mean()) / M_RANGE, 2),
           "mu_nmae_pct": round(100 * float(u_err.mean()) / MU_RANGE, 2),
           "mass_nrmse_pct": round(100 * float(np.sqrt((m_err ** 2).mean())) / M_RANGE, 2),
           "mu_nrmse_pct": round(100 * float(np.sqrt((u_err ** 2).mean())) / MU_RANGE, 2)}
    for name, x in (("signed", ang), ("abs", np.abs(ang))):
        for mname, e in (("mass", m_err), ("mu", u_err)):
            rho, _ = _spearman(x, e)
            rec[f"{mname}_{name}_spearman"] = round(rho, 4) if not np.isnan(rho) else np.nan
    return rec


def plot_error_vs_angle(df, preds_by_variant, centers, out_dir):
    """Mean absolute error per angle bin, one line per variant."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, (col, gt, label, unit) in zip(axes, [
            (0, "gt_mass", "Mass", "kg"), (1, "gt_mu", "Friction Coeff", "")]):
        for v, pr in preds_by_variant.items():
            e = np.abs(pr[:, col] - df[gt].values)
            means = [e[(df["angle_bin"] == b).values].mean()
                     if (df["angle_bin"] == b).sum() >= MIN_ROWS_PER_BIN else np.nan
                     for b in range(len(centers))]
            ax.plot(centers, means, marker="o", markersize=4, linewidth=1.8,
                    alpha=0.85, label=VARIANT_LABEL.get(v, v))
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Push angle bin centre [deg]")
        ax.set_ylabel(f"Mean absolute {label} error" + (f" [{unit}]" if unit else ""))
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Simulation: absolute error vs push angle (held-out groups)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, "sim_error_vs_angle.png")
    plt.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")


# ==========================================
# 4. WORST ANGLE BIN PER PROPERTY BUCKET
# ==========================================
def worst_angle_by_bucket(df, preds_by_variant, centers):
    """For each (bucket, variant), which angle bin has the highest mean error.

    Mirrors the real analysis, where 'worst' is defined within an object. A bin that
    is worst for many buckets AND many variants is a geometry effect rather than one
    model's or one bucket's weakness.
    """
    rows = []
    for v, pr in preds_by_variant.items():
        d = df.copy()
        d["abs_mass_err"] = np.abs(pr[:, 0] - d["gt_mass"].values)
        d["abs_mu_err"] = np.abs(pr[:, 1] - d["gt_mu"].values)
        for dom, g in d.groupby("domain"):
            for metric, col in (("mass", "abs_mass_err"), ("friction", "abs_mu_err")):
                pb = g.groupby("angle_bin")[col].agg(["mean", "size"])
                # A bin with a handful of rows can top the table by chance.
                pb = pb[pb["size"] >= MIN_ROWS_PER_BIN]
                if pb.empty:
                    continue
                wb, bb = int(pb["mean"].idxmax()), int(pb["mean"].idxmin())
                rows.append({
                    "domain": dom, "tag": v, "metric": metric,
                    "n_bins": len(pb), "n_rows": int(len(g)),
                    "mean_abs_err": round(float(g[col].mean()), 5),
                    "max_abs_err": round(float(pb["mean"].max()), 5),
                    "worst_bin": wb, "worst_angle_deg": round(float(centers[wb]), 2),
                    "best_bin": bb, "best_angle_deg": round(float(centers[bb]), 2),
                    # Whether the frequency means anything: a worst bin only 1.05x the
                    # best is not a weakness however often it wins the argmax.
                    "worst_over_best": round(float(
                        pb["mean"].max() / max(pb["mean"].min(), 1e-12)), 3),
                })
    return pd.DataFrame(rows)


def plot_worst_angle_hist(wa, centers, out_dir):
    for metric in ("mass", "friction"):
        sub = wa[wa["metric"] == metric]
        if sub.empty:
            continue
        variants = [v for v in VARIANT_SPEC if v in sub["tag"].unique()]
        xpos = np.arange(len(centers))
        ymax = max([1] + [int(sub[sub["tag"] == v]["worst_bin"].value_counts().max())
                          for v in variants if len(sub[sub["tag"] == v])])

        fig, axes = plt.subplots(1, len(variants), figsize=(3.7 * len(variants), 3.9),
                                 squeeze=False, sharey=True)
        for ax, v in zip(axes[0], variants):
            t = sub[sub["tag"] == v]
            counts = np.zeros(len(centers), dtype=int)
            for b, n in t["worst_bin"].value_counts().items():
                counts[int(b)] = int(n)
            peak = counts.max()
            ax.bar(xpos, counts, alpha=0.9, edgecolor="black", linewidth=0.4,
                   color=["#d62728" if n == peak and n > 0 else "#4c72b0" for n in counts])
            for x, n in zip(xpos, counts):
                if n:
                    ax.text(x, n, str(n), ha="center", va="bottom", fontsize=7)
            ax.set_title(f"{VARIANT_LABEL.get(v, v)}\nmean |err| "
                         f"{t['mean_abs_err'].mean():.4f} (n={len(t)})",
                         fontsize=9, fontweight="bold")
            ax.set_xticks(xpos)
            ax.set_xticklabels([f"{c:+.0f}" for c in centers], rotation=90, fontsize=7)
            ax.set_ylim(0, ymax * 1.2)
            ax.set_xlabel("Angle bin [deg]")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0][0].set_ylabel("Buckets where worst")
        fig.suptitle(f"Simulation: worst angle bin per variant -- {metric}\n"
                     f"red = that variant's most frequent worst bin",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        p = os.path.join(out_dir, f"sim_worst_angle_hist_{metric}.png")
        plt.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")

        fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(centers) + 3), 4.4))
        tot = np.zeros(len(centers), dtype=int)
        nv = np.zeros(len(centers), dtype=int)
        for b, n in sub["worst_bin"].value_counts().items():
            tot[int(b)] = int(n)
        for b in range(len(centers)):
            nv[b] = sub[sub["worst_bin"] == b]["tag"].nunique()
        ax.bar(xpos, tot, color="#4c72b0", alpha=0.9, edgecolor="black", linewidth=0.5)
        for x, (n, m) in enumerate(zip(tot, nv)):
            if n:
                ax.text(x, n, f"{n}\n({m}v)", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xpos); ax.set_xticklabels([f"{c:+.0f}" for c in centers], rotation=45)
        ax.set_xlabel("Angle bin [deg]"); ax.set_ylabel("Times worst (all variants)")
        ax.set_title(f"Pooled worst angle bin -- {metric}\n"
                     f"(Nv) = distinct variants, out of {sub['tag'].nunique()}",
                     fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, f"sim_worst_angle_hist_{metric}_pooled.png")
        plt.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")


# ==========================================
# 5. LATEX
# ==========================================
def latex_worst_angle(wa, path):
    L = [r"\begin{table}[t]", r"    \centering",
         r"    \caption{Simulation angle robustness: mean absolute error and the "
         r"angle bins at which each variant is worst, counted over property buckets.}",
         r"    \label{tab:sim_worst_angle}", r"    \vspace{2mm}",
         r"    \begin{tabular}{l c c " + " ".join(["c"] * TOP_ANGLES) + "}",
         r"        \toprule",
         r"        \textbf{Variant} & \textbf{Mean $|$err$|$} & \textbf{nMAE} & " +
         " & ".join(f"\\textbf{{Worst {i+1}}}" for i in range(TOP_ANGLES)) + r" \\",
         r"        \midrule"]
    for metric, rng in (("mass", M_RANGE), ("friction", MU_RANGE)):
        sub = wa[wa["metric"] == metric]
        if sub.empty:
            continue
        L.append(f"        \\multicolumn{{{3 + TOP_ANGLES}}}{{l}}"
                 f"{{\\textbf{{{metric.title()}}}}} \\\\")
        L.append(r"        \midrule")
        agg = sub.groupby("tag")["mean_abs_err"].mean()
        best = agg.min()
        for v in [x for x in VARIANT_SPEC if x in sub["tag"].unique()]:
            t = sub[sub["tag"] == v]
            me = float(agg.loc[v])
            err, nm = f"{me:.4f}", f"{100 * me / rng:.1f}\\%"
            if np.isclose(me, best):
                err, nm = f"\\textbf{{{err}}}", f"\\textbf{{{nm}}}"
            cells = [VARIANT_LABEL.get(v, v), err, nm]
            vc = t["worst_angle_deg"].value_counts()
            for i in range(TOP_ANGLES):
                cells.append("--" if i >= len(vc) else
                             f"${vc.index[i]:+.0f}^\\circ$ ($\\times${int(vc.iloc[i])})")
            L.append("        " + " & ".join(cells) + r" \\")
        L.append(r"        \midrule")
    if L[-1] == r"        \midrule":
        L.pop()
    L += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"  {path}")


def latex_nmae(corr, path, value_col="nmae_pct", metric_name="nMAE"):
    L = [r"\begin{table}[t]", r"    \centering",
         r"    \caption{Simulation " + metric_name + r" (\%) on held-out property "
         r"groups. Normalized by the training range: $m \in [0.2, 2.0]$~kg, "
         r"$\mu \in [0.15, 0.5]$.}",
         r"    \label{tab:sim_" + metric_name.lower() + "}", r"    \vspace{2mm}",
         r"    \begin{tabular}{l c c}", r"        \toprule",
         r"        \textbf{Variant} & \textbf{Mass " + metric_name + r" (\%)} & "
         r"\textbf{$\mu$ " + metric_name + r" (\%)} \\", r"        \midrule"]
    mcol, ucol = f"mass_{value_col}", f"mu_{value_col}"
    bm, bu = corr[mcol].min(), corr[ucol].min()
    for _, r in corr.iterrows():
        cells = [VARIANT_LABEL.get(r["variant"], r["variant"])]
        for val, best in ((r[mcol], bm), (r[ucol], bu)):
            s = f"{val:.2f}"
            cells.append(f"\\textbf{{{s}}}" if np.isclose(val, best) else s)
        L.append("        " + " & ".join(cells) + r" \\")
    L += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"  {path}")


# ==========================================
# 6. MANIPULABILITY BY ANGLE
# ==========================================
def manipulability_by_angle(df, centers, out_dir):
    """Both channels as a function of push angle.

    Individual traces are not drawn -- thousands would be unreadable -- so the mean
    per angle bin is shown with a +/-1 std band, carrying the same message as the real
    per-pair figure where each angle appears once.

    The consistency panel asks what matters for interpreting the *_cond variants: is
    the conditioning scalar a function of the push angle, or of the object? If
    between-bin spread dominates within-bin spread, the scalar mostly encodes which
    angle the push was.
    """
    os.makedirs(out_dir, exist_ok=True)
    channels = [("arm_manip_w", "$w$", "Manipulability"),
                ("arm_dir_manip_w", "$w_{dir}$", "Directional manipulability")]
    have = [(p, s, n) for p, s, n in channels if window_cols(df.columns.tolist(), p)]
    if not have:
        print("  [skip] no arm_manip_w* / arm_dir_manip_w* columns")
        return None

    cmap = cm.coolwarm
    lim = max(abs(centers.min()), abs(centers.max()))
    norm = mcolors.Normalize(vmin=-lim, vmax=lim)

    fig, axes = plt.subplots(1, len(have), figsize=(8 * len(have), 5), squeeze=False)
    store = {}
    for ax, (pfx, sym, name) in zip(axes[0], have):
        W = df[window_cols(df.columns.tolist(), pfx)].values.astype(np.float32)
        store[pfx] = W
        for b in range(len(centers)):
            sel = (df["angle_bin"] == b).values
            if sel.sum() < 5:
                continue
            m, s = W[sel].mean(axis=0), W[sel].std(axis=0)
            c = cmap(norm(centers[b]))
            ax.plot(m, color=c, linewidth=1.8)
            ax.fill_between(np.arange(len(m)), m - s, m + s, color=c, alpha=0.12)
        ax.set_xlabel("Window step (0..59)"); ax.set_ylabel(sym)
        ax.set_title(f"{name}: mean per angle bin ($\\pm$1 std)")
        ax.grid(True, alpha=0.3)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes[0].tolist(), label="Push angle [deg]", pad=0.02)
    fig.suptitle("Simulation: manipulability over the inference window, by push angle",
                 fontsize=13, fontweight="bold")
    p = os.path.join(out_dir, "sim_manip_seq_by_angle.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  {p}")

    fig, axes = plt.subplots(2, len(have), figsize=(8 * len(have), 9), squeeze=False)
    summary = []
    for j, (pfx, sym, _) in enumerate(have):
        wmin = store[pfx].min(axis=1)

        ax = axes[0][j]
        ax.scatter(df["push_angle_deg"], wmin, s=3, alpha=0.15, color="tab:blue")
        bm = [wmin[(df["angle_bin"] == b).values].mean()
              if (df["angle_bin"] == b).sum() else np.nan for b in range(len(centers))]
        ax.plot(centers, bm, color="tab:red", marker="o", linewidth=2.2, label="bin mean")
        ax.set_xlabel("Push angle [deg]"); ax.set_ylabel(f"min {sym} over the window")
        ax.set_title(f"{sym}: conditioning scalar vs angle")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

        ax = axes[1][j]
        data = [wmin[(df["angle_bin"] == b).values] for b in range(len(centers))]
        data = [d for d in data if len(d) > 1]
        ax.boxplot(data, positions=np.arange(len(data)), widths=0.6, showfliers=False)
        within = float(np.mean([np.std(d) for d in data]))
        between = float(np.std([np.mean(d) for d in data]))
        ratio = between / max(within, 1e-12)
        summary.append((pfx, within, between, ratio))
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels([f"{c:+.0f}" for c in centers[:len(data)]], rotation=45)
        ax.set_xlabel("Angle bin [deg]"); ax.set_ylabel(f"min {sym}")
        ax.set_title(f"{sym}: between-bin / within-bin spread = {ratio:.1f}x")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Is the conditioning scalar a function of angle, or of the object?\n"
                 "top: every row with the bin mean. bottom: spread within each bin",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, "sim_manip_vs_angle.png")
    plt.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")

    rows = []
    for b in range(len(centers)):
        sel = (df["angle_bin"] == b).values
        if sel.sum() == 0:
            continue
        rec = {"bin": b, "angle_center": round(float(centers[b]), 2), "n": int(sel.sum())}
        for pfx, _, _ in have:
            w = store[pfx][sel]
            rec[f"{pfx}_min_mean"] = round(float(w.min(axis=1).mean()), 6)
            rec[f"{pfx}_min_std"] = round(float(w.min(axis=1).std()), 6)
        rows.append(rec)
    p = os.path.join(out_dir, "sim_manip_by_angle.csv")
    pd.DataFrame(rows).to_csv(p, index=False); print(f"  {p}")
    return summary


# ==========================================
# 7. MAIN
# ==========================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(" SIMULATION PUSH-ANGLE ANALYSIS")
    print("=" * 78)
    print(f"  device   : {device}")
    print(f"  csv      : {CSV_PATH}")
    print(f"  variants : {VARIANTS}")
    print(f"  norm     : m /{M_RANGE:.2f}  mu /{MU_RANGE:.3f}  (training range)")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    df, vel_cols, centers = load_heldout()
    df = add_property_buckets(df)
    print(f"  buckets  : {df['domain'].nunique()} "
          f"({N_M_BUCKETS} mass x {N_MU_BUCKETS} mu), "
          f"~{len(df) // max(df['domain'].nunique(), 1)} rows each")
    print(f"  angle    : {df['push_angle_deg'].min():.1f} to "
          f"{df['push_angle_deg'].max():.1f} deg, {N_ANGLE_BINS} bins")

    preds_by_variant, corr_rows = {}, []
    for v in VARIANTS:
        if v not in VARIANT_SPEC:
            print(f"  [skip] unknown variant {v!r}")
            continue
        print(f"\n--- {v} ---")
        try:
            pr = predict(df, vel_cols, v, device)
        except Exception as exc:
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            continue
        preds_by_variant[v] = pr
        rec = variant_summary(df, pr, v)
        corr_rows.append(rec)
        print(f"  mass nMAE {rec['mass_nmae_pct']:6.2f}%   "
              f"mu nMAE {rec['mu_nmae_pct']:6.2f}%   "
              f"rho(|angle|, mass err) {rec['mass_abs_spearman']:+.3f}")

    if not preds_by_variant:
        raise SystemExit("\nNo variant produced predictions.")

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(TABLE_DIR, "sim_variant_summary.csv"), index=False)
    print(f"\n  {os.path.join(TABLE_DIR, 'sim_variant_summary.csv')}")

    plot_error_vs_angle(df, preds_by_variant, centers, OUT_DIR)

    wa = worst_angle_by_bucket(df, preds_by_variant, centers)
    wa.to_csv(os.path.join(TABLE_DIR, "sim_worst_angle.csv"), index=False)
    print(f"  {os.path.join(TABLE_DIR, 'sim_worst_angle.csv')}")
    latex_worst_angle(wa, os.path.join(TABLE_DIR, "sim_worst_angle.tex"))
    plot_worst_angle_hist(wa, centers, OUT_DIR)

    latex_nmae(corr, os.path.join(TABLE_DIR, "sim_nmae.tex"), "nmae_pct", "nMAE")
    latex_nmae(corr, os.path.join(TABLE_DIR, "sim_nrmse.tex"), "nrmse_pct", "nRMSE")

    summary = manipulability_by_angle(df, centers, MANIP_DIR)

    print("\n" + "=" * 78)
    print(" MASS nMAE % / MU nMAE %")
    print("=" * 78)
    print(corr[["variant", "mass_nmae_pct", "mu_nmae_pct",
                "mass_nrmse_pct", "mu_nrmse_pct"]].to_string(index=False))

    print("\n" + "=" * 78)
    print(" ANGLE BINS WORST ACROSS VARIANTS")
    print("=" * 78)
    for metric in ("mass", "friction"):
        sub = wa[wa["metric"] == metric]
        if sub.empty:
            continue
        tot = sub.groupby("worst_angle_deg").agg(
            buckets=("worst_angle_deg", "size"), variants=("tag", "nunique"),
            margin=("worst_over_best", "mean")).sort_values("buckets", ascending=False)
        print(f"\n  {metric}:   {'angle':>8} {'buckets':>8} {'variants':>9} {'margin':>8}")
        for a, r in tot.head(5).iterrows():
            print(f"            {a:>7.0f}° {int(r['buckets']):>8} "
                  f"{int(r['variants']):>9} {r['margin']:>7.2f}x")
        print("     margin = worst bin mean / best bin mean. Near 1.0 means the bin")
        print("     tops the argmax without being meaningfully worse.")

    if summary:
        print("\n" + "=" * 78)
        print(" HOW MUCH OF THE CONDITIONING SCALAR IS ANGLE?")
        print("=" * 78)
        print(f"  {'channel':<20} {'within-bin':>12} {'between-bin':>13} {'ratio':>8}")
        for pfx, within, between, ratio in summary:
            print(f"  {pfx:<20} {within:>12.6f} {between:>13.6f} {ratio:>7.1f}x")
        print("\n  On hardware the object pose is fixed, so the arm trajectory is set by")
        print("  the push angle alone and this ratio should be high. In simulation the")
        print("  object yaw varies, so the same angle reaches different configurations")
        print("  and the scalar carries pose information beyond the angle. A much lower")
        print("  ratio here than on the real robot is a real difference between the two")
        print("  settings, and bears on whether a conditioning gain should transfer.")


if __name__ == "__main__":
    main()