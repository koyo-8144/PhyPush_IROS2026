import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configs import (G, M_SEEN_MAX, M_SEEN_MIN, MU_SEEN_MAX, MU_SEEN_MIN,
                     CSV_PATH, FRAME_MODE, MULTI_ANGLE, USE_ARM_STATE,
                     INPUT_VARIANT, VARIANT_WINDOW_PREFIX, VARIANT_TABLE)

from dataset import (create_dataloaders, MULTI_ANGLE_COLS, ARM_STATIC_COLS,
                     ARM_STATE_COLS, ARM_FEATURE_COLS, window_cols,
                     variant_window)
from utils import clean_force_col, add_min_max_text

# Derived from configs.VARIANT_TABLE so this script can never disagree with
# what train.py builds.
ALL_VARIANTS = {name: spec[0] for name, spec in VARIANT_TABLE.items()}

# How each variant reaches the model: (mechanism, input_dim, cond_dim).
VARIANT_SHAPE = {
    name: ("baseline" if pfx is None else
           "cond token" if cdim > 0 else "input channel", idim, cdim)
    for name, (pfx, cdim, idim, _) in VARIANT_TABLE.items()
}

# Channels that are log-transformed before standardization (inertial ones).
LOG_PREFIXES = {spec[0] for spec in VARIANT_TABLE.values() if spec[3]}


def visualize_all_variants(df, num_samples=3, standardize=True):
    """Show what EVERY variant actually hands the model, for the same rows.

    One row of panels per variant, one column per sample. The velocity sequence
    is drawn in every panel because it is always the input; what changes is what
    sits beside it:

      vel_only        nothing.
      *_cond          a single scalar, the MINIMUM over the window, drawn as a
                      horizontal line. The model gets it as one extra token
                      prepended to the encoder sequence -- it has no time axis,
                      which is why it is flat here.
      *_seq           the full 60-step channel on a second y-axis, concatenated
                      to velocity so input_dim becomes 2. vel_osim_seq (Lambda)
                      and vel_eff_seq (m_eff) are log-transformed first, so with
                      standardize=True their panels show standardized log values.

    Because every panel shows the same three rows, the comparison is like for
    like: the question each variant poses is visible side by side.

    standardize=True plots the conditioning/sequence values after the same
    (x - mean) / std the dataloader applies, so the panels show the scale the
    model actually sees rather than raw physical units. Velocity is left raw --
    the dataloader does not standardize it either.
    """
    print("\n" + "=" * 78)
    print("ALL INPUT VARIANTS -- what each one feeds the model")
    print("=" * 78)

    vel_cols = sorted([c for c in df.columns if "input_vel_" in c],
                      key=lambda x: int(x.split('_')[-1]))
    if not vel_cols:
        print("  [ERROR] No 'input_vel_*' columns; nothing to show.")
        return
    T = len(vel_cols)

    # Resolve every channel once, and report which variants this CSV supports.
    channels = {}       # raw values, for raw display and scalar reductions
    model_space = {}    # after the variant transform (log for inertial channels)
    valid_rows = {}
    for pfx in sorted(set(v for v in ALL_VARIANTS.values() if v is not None)):
        cols = window_cols(df, pfx)
        channels[pfx] = df[cols].values.astype(np.float32) if cols else None
        if cols:
            Wt, ok, _ = variant_window(df, prefix=pfx,
                                       log_transform=pfx in LOG_PREFIXES)
            model_space[pfx], valid_rows[pfx] = Wt, ok
            if not ok.all():
                print(f"  [NOTE] '{pfx}*': {int((~ok).sum())} rows with a zero / "
                      f"non-finite step; training drops them for this variant.")
        if cols and len(cols) != T:
            print(f"  [ERROR] '{pfx}*' has {len(cols)} steps but velocity has {T}. "
                  f"Not time-aligned.")

    print(f"\n  {'variant':<20} {'mechanism':<15} {'input_dim':>9} "
          f"{'cond_dim':>8}  available")
    print("  " + "-" * 74)
    usable = []
    for name, pfx in ALL_VARIANTS.items():
        mech, idim, cdim = VARIANT_SHAPE[name]
        ok = pfx is None or channels.get(pfx) is not None
        print(f"  {name:<20} {mech:<15} {idim:>9} {cdim:>8}  "
              f"{'yes' if ok else 'NO -- column missing'}")
        if ok:
            usable.append(name)

    # Pick rows valid for EVERY available channel so all panels show the same rows.
    all_ok = np.ones(len(df), dtype=bool)
    for ok in valid_rows.values():
        all_ok &= ok
    cand = np.flatnonzero(all_ok) if all_ok.any() else np.arange(len(df))
    idx = cand[np.linspace(0, len(cand) - 1, min(num_samples, len(cand))).astype(int)]
    vel = df[vel_cols].values.astype(np.float32)

    # Standardization constants, computed over the whole subset. The real
    # dataloader uses TRAIN-SPLIT statistics; these are close enough to show the
    # scale but are NOT the values training will use.
    stats = {}
    for pfx, W in channels.items():
        if W is None:
            continue
        ok = valid_rows[pfx]
        mn = W[ok].min(axis=1)
        Wt = model_space[pfx][ok]          # sequence stats in model space
        stats[pfx] = {
            "cond_mean": float(mn.mean()), "cond_std": float(mn.std()) + 1e-8,
            "seq_mean": float(Wt.mean()),  "seq_std": float(Wt.std()) + 1e-8,
        }

    print(f"\n  showing rows {list(idx)}")
    if standardize:
        print("  conditioning / sequence values are STANDARDIZED (subset stats, "
              "not train-split)")

    sns.set_theme(style="whitegrid")
    n_rows = len(usable)
    fig, axes = plt.subplots(n_rows, len(idx),
                             figsize=(5.2 * len(idx), 2.9 * n_rows),
                             squeeze=False)
    steps = np.arange(T)

    for r, name in enumerate(usable):
        pfx = ALL_VARIANTS[name]
        mech, idim, cdim = VARIANT_SHAPE[name]

        for c, i in enumerate(idx):
            ax = axes[r][c]
            ax.plot(steps, vel[i], color="tab:blue", linewidth=1.8,
                    label="velocity (always)")
            ax.set_ylabel("v [m/s]", color="tab:blue", fontsize=8)
            ax.tick_params(axis='y', labelcolor="tab:blue", labelsize=7)
            ax.tick_params(axis='x', labelsize=7)

            if pfx is None:
                ax.text(0.5, 0.08, "input_dim=1, cond_dim=0",
                        transform=ax.transAxes, ha="center", fontsize=8,
                        color="gray")

            elif cdim > 0:
                W = channels[pfx]
                raw = float(W[i].min())
                val = ((raw - stats[pfx]["cond_mean"]) / stats[pfx]["cond_std"]
                       if standardize else raw)
                ax2 = ax.twinx()
                ax2.axhline(val, color="tab:red", linestyle="--", linewidth=2.0,
                            label="cond scalar = min(window)")
                # Mark WHERE the minimum occurs -- the scalar discards this.
                ax2.plot([int(W[i].argmin())], [val], marker='v', markersize=9,
                         color="tab:red")
                ax2.set_ylabel("cond (std)" if standardize else "cond",
                               color="tab:red", fontsize=8)
                ax2.tick_params(axis='y', labelcolor="tab:red", labelsize=7)
                ax2.grid(False)
                ax.text(0.02, 0.06, f"raw min = {raw:.5f}\n1 token, no time axis",
                        transform=ax.transAxes, fontsize=7, color="tab:red")

            else:
                is_log = pfx in LOG_PREFIXES
                if standardize:
                    seq = ((model_space[pfx][i] - stats[pfx]["seq_mean"])
                           / stats[pfx]["seq_std"])
                    ylab = "log channel (std)" if is_log else "channel (std)"
                else:
                    seq = channels[pfx][i]
                    ylab = f"{pfx}* [kg]" if is_log else f"{pfx}*"
                color = "tab:brown" if is_log else "tab:purple"
                ax2 = ax.twinx()
                ax2.plot(steps, seq, color=color, linewidth=1.6,
                         label="2nd input channel")
                if is_log and not standardize:
                    ax2.set_yscale("log")
                ax2.set_ylabel(ylab, color=color, fontsize=8)
                ax2.tick_params(axis='y', labelcolor=color, labelsize=7)
                ax2.grid(False)
                ax.text(0.02, 0.06,
                        f"{pfx}*, {T} steps{', log' if is_log else ''}\n"
                        f"concatenated, input_dim=2",
                        transform=ax.transAxes, fontsize=7, color=color)

            if c == 0:
                ax.text(-0.28, 0.5, f"{name}\n({mech})", transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=9, fontweight="bold")
            if r == 0:
                gm = df['gt_mass'].iloc[i] if 'gt_mass' in df.columns else float('nan')
                gu = df['gt_mu'].iloc[i] if 'gt_mu' in df.columns else float('nan')
                ax.set_title(f"row {i}   m={gm:.3f} kg   mu={gu:.3f}", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("Window step", fontsize=8)

    fig.suptitle(f"What each INPUT_VARIANT feeds the model (same rows throughout, "
                 f"{'standardized' if standardize else 'raw'})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    plt.show()

    # ------------------------------------------------------------------
    # What the two reductions cost, in one number per channel.
    # ------------------------------------------------------------------
    print("\n  Information the SCALAR reduction discards:")
    for pfx, W in channels.items():
        # Only the manipulability channels have a scalar (cond) variant.
        if W is None or pfx in LOG_PREFIXES:
            continue
        mn, mx = W.min(axis=1), W.max(axis=1)
        within = (mx - mn)                  # per-push variation over the window
        across = mn.std()                   # variation of the scalar across pushes
        print(f"    {pfx:<18} within-push range: mean {within.mean():.6f}  "
              f"| across-push std of the min: {across:.6f}")
        ratio = within.mean() / max(across, 1e-12)
        if ratio > 1.0:
            print(f"       within/across = {ratio:.2f} > 1: each push varies MORE "
                  f"internally than\n       the scalar varies between pushes. The "
                  f"sequence variant has real\n       structure the scalar throws "
                  f"away.")
        else:
            print(f"       within/across = {ratio:.2f} < 1: the channel is nearly "
                  f"flat within a\n       push, so the sequence variant should "
                  f"behave much like the scalar.")

    print("=" * 78 + "\n")


def report_multi_angle_coverage(df):
    """Verify the orientation / push-side sweep produced what it should.

    Expect ~10 distinct obj_yaw_base values x 2 push_face_index values with
    roughly equal counts. A thin cell means IK failed more often at that
    orientation -- itself evidence that arm configuration matters.
    """
    print("\n" + "=" * 70)
    print("MULTI-ANGLE COVERAGE REPORT")
    print("=" * 70)

    missing = [c for c in MULTI_ANGLE_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing provenance columns: {missing}")
        print("          Check _save_offline_data_csv in phypush_distillation.py.")

    for col in ['obj_yaw_base', 'push_face_index', 'collect_idx', 'seed']:
        if col in df.columns:
            vc = df[col].value_counts().sort_index()
            print(f"\n  {col}  ({vc.size} distinct values)")
            print(vc.to_string())

    report_object_yaw(df)

    if {'obj_yaw_base', 'push_face_index'}.issubset(df.columns):
        pivot = (df.assign(yaw_deg=_yaw_deg(df['obj_yaw_base']))
            .pivot_table(index='yaw_deg', columns='push_face_index',
                        values='gt_mass', aggfunc='count'))
        print("\n  rows per (object yaw [deg] x push side):")
        print(pivot.to_string())

        counts = pivot.values.flatten()
        counts = counts[~np.isnan(counts)]
        if counts.size:
            print(f"\n  cell counts: min={counts.min():.0f}  max={counts.max():.0f}  "
                  f"mean={counts.mean():.1f}")
            if counts.min() < 0.7 * counts.mean():
                print("  [NOTE] Some cells are far thinner than average. Likely IK "
                      "failures at those orientations, since _save_offline_data_csv "
                      "drops rows where push_start/end were not reached.")


    if {'seed', 'env_id'}.issubset(df.columns):
        groups = df['seed'].astype(int) * 100000 + df['env_id'].astype(int)
        n_groups = groups.nunique()
        print(f"\n  property groups (seed x env_id): {n_groups}")
        print(f"  trajectories per property group : {len(df) / max(n_groups, 1):.1f}")
        print("  (expect ~20 if every COLLECT_IDX cell survived the reached-flag filter)")

    print("=" * 70 + "\n")


def report_arm_columns(df):
    """Summarize the recorded arm state and check it varies as expected."""
    print("\n" + "=" * 70)
    print("ARM STATE COLUMN REPORT")
    print("=" * 70)

    missing_static = [c for c in ARM_STATIC_COLS if c not in df.columns]
    missing_state = [c for c in ARM_STATE_COLS if c not in df.columns]

    if missing_static or missing_state:
        print(f"[WARNING] Missing {len(missing_static)} static and "
              f"{len(missing_state)} state columns.")
        if missing_static:
            print(f"          static: {missing_static}")
        if missing_state:
            print(f"          state (first 5): {missing_state[:5]}")
        print("          Re-collect with the arm_state observation group enabled.")
        print("=" * 70 + "\n")
        return

    present = [c for c in ARM_STATIC_COLS + ARM_STATE_COLS if c in df.columns]
    stats = df[present].describe().T[['mean', 'std', 'min', 'max']].round(5)
    print(f"\n  {len(present)} arm columns present:\n")
    print(stats.to_string())

    const_cols = stats.index[stats['std'] < 1e-8].tolist()
    if const_cols:
        print(f"\n  [NOTE] Exactly constant columns: {const_cols}")
        if 'push_start_reached_flag' in const_cols:
            print("         push_start_reached_flag is expected to be constant 1.0 -- "
                  "the CSV writer drops any row where it was 0.")

    rel = (stats['std'] / stats['mean'].abs().clip(lower=1e-12)).sort_values()
    dead = rel[rel < 1e-3]
    if len(dead):
        print(f"\n  [NOTE] Effectively constant (relative std < 1e-3): "
              f"{list(dead.index)}")
        print("         These contribute only numerical noise once standardized.")
        print("         Another argument for ARM_FEATURE_MODE='push_dir' over 'full'.")

    print(f"\n  ARM_FEATURE_COLS in use: {len(ARM_FEATURE_COLS)}")

    if {'obj_yaw_base', 'push_face_index'}.issubset(df.columns):
        by_cell = df.groupby(['obj_yaw_base', 'push_face_index'])[
            ['worst_manipulability', 'ee_position_error', 'arm_q0', 'push_dir_b_y']
        ].mean().round(5)
        print("\n  arm state vs (object yaw x push side):")
        print(by_cell.to_string())

        for col in ['worst_manipulability', 'arm_q0', 'push_dir_b_y']:
            if col not in df.columns:
                continue
            cell_spread = by_cell[col].max() - by_cell[col].min()
            raw_spread = df[col].max() - df[col].min()
            print(f"\n  {col}: cell-mean spread {cell_spread:.5f}, "
                  f"full range {raw_spread:.5f}, std {df[col].std():.5f}")

        manip_rel = 100 * (by_cell['worst_manipulability'].max()
                           - by_cell['worst_manipulability'].min()) \
                    / max(by_cell['worst_manipulability'].mean(), 1e-12)
        print(f"\n  manipulability spread across cells: {manip_rel:.1f}% of mean")
        if manip_rel < 5.0:
            print("  [NOTE] The arm barely changes configuration across cells, so arm "
                  "conditioning is unlikely to explain estimation differences.")

    print("=" * 70 + "\n")

    if 'obj_yaw_base' in df.columns and 'push_face_index' in df.columns:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.boxplot(data=df, x='obj_yaw_base', y='worst_manipulability',
                    hue='push_face_index', ax=axes[0])
        axes[0].set_title("Worst manipulability by orientation")
        axes[0].tick_params(axis='x', rotation=45)

        sns.boxplot(data=df, x='obj_yaw_base', y='ee_position_error',
                    hue='push_face_index', ax=axes[1])
        axes[1].set_title("EE tracking error by orientation")
        axes[1].tick_params(axis='x', rotation=45)

        sns.scatterplot(data=df, x='push_dir_b_x', y='push_dir_b_y',
                        hue='obj_yaw_base', palette='viridis', s=14,
                        alpha=0.7, ax=axes[2])
        axes[2].set_title("Push heading in robot base frame")
        axes[2].set_aspect('equal')

        plt.tight_layout()
        plt.show()


def plot_kinematic_distributions(df):
    """
    Plots histograms for object yaw and push directions to verify continuous distributions.
    """
    cols_to_plot = ['obj_yaw_base', 'push_dir_b_x', 'push_dir_b_y']
    available_cols = [c for c in cols_to_plot if c in df.columns]
    
    if not available_cols:
        return
        
    print("\nPlotting Kinematic Distributions (Yaw & Push Directions)...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(available_cols), figsize=(5 * len(available_cols), 5))
    
    if len(available_cols) == 1:
        axes = [axes]
        
    for ax, col in zip(axes, available_cols):
        valid_data = df[col].dropna()
        
        if col == 'obj_yaw_base':
            # Convert to degrees for better interpretability
            data = np.degrees(valid_data)
            xlabel = "Object Yaw [deg]"
            color = "#2ca02c"
        else:
            data = valid_data
            xlabel = col
            color = "#1f77b4"
            
        sns.histplot(data, kde=True, color=color, ax=ax, bins=50)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        
    plt.tight_layout()
    plt.show()


def report_variant_window(df, num_samples=6):
    """Visualize the per-step window channel the active INPUT_VARIANT feeds in.

    Four panels:
      1. A handful of raw sequences, so the within-push shape is visible at all.
      2. Mean +/- 1 std envelope across the subset, with the two scalar
         reductions marked: the window minimum (what the *_cond variants use)
         and the whole-push worst_* column (what they deliberately do NOT use).
      3. Distribution of the window minimum vs the whole-push minimum. These are
         different quantities and the gap between them is the reason the scalar
         is recomputed here rather than read from the CSV.
      4. Window minimum against object yaw, split by push side -- the question
         the whole multi-angle sweep exists to answer: does arm configuration
         actually move this quantity?
    """
    print("\n" + "=" * 70)
    print(f"INPUT VARIANT WINDOW CHANNEL: {INPUT_VARIANT}")
    print("=" * 70)

    if VARIANT_WINDOW_PREFIX is None:
        print("  INPUT_VARIANT='vel_only': no window channel is used.")
        print("  Set INPUT_VARIANT to one of the manip/dirmanip variants to "
              "inspect it.")
        print("=" * 70 + "\n")
        return

    if VARIANT_WINDOW_PREFIX in LOG_PREFIXES:
        # Inertial channels have their own report (max is the adverse reduction,
        # and there is no whole-push column to compare against).
        print(f"  '{VARIANT_WINDOW_PREFIX}*' is an inertial channel; see "
              f"report_inertial_window below.")
        print("=" * 70 + "\n")
        return

    wcols = window_cols(df, VARIANT_WINDOW_PREFIX)
    if not wcols:
        print(f"  [WARNING] No '{VARIANT_WINDOW_PREFIX}*' columns in this CSV.")
        print(f"            The variant '{INPUT_VARIANT}' cannot be trained on it.")
        print("=" * 70 + "\n")
        return

    W = df[wcols].values.astype(np.float32)          # (N, T)
    T = W.shape[1]
    win_min = W.min(axis=1)
    win_mean = W.mean(axis=1)

    # The whole-push counterpart, for the comparison in panels 2 and 3.
    whole_col = ('worst_manipulability' if VARIANT_WINDOW_PREFIX == 'arm_manip_w'
                 else 'worst_dir_manipulability')
    whole = df[whole_col].values.astype(np.float32) if whole_col in df.columns else None

    vel_cols = sorted([c for c in df.columns if "input_vel_" in c],
                      key=lambda x: int(x.split('_')[-1]))

    print(f"  channel        : {VARIANT_WINDOW_PREFIX}0..{T - 1}  ({T} steps)")
    print(f"  velocity input : input_vel_0..{len(vel_cols) - 1}  "
          f"({len(vel_cols)} steps)")
    if len(vel_cols) != T:
        print(f"  [ERROR] Length mismatch: the two channels are NOT time-aligned. "
              f"create_dataloaders will refuse this.")
    print(f"  rows           : {len(df)}")
    print(f"  per-step value : mean {W.mean():.6f}  "
          f"range [{W.min():.6f}, {W.max():.6f}]")
    print(f"  window minimum : mean {win_min.mean():.6f}  "
          f"range [{win_min.min():.6f}, {win_min.max():.6f}]  "
          f"std {win_min.std():.6f}")

    if whole is not None:
        print(f"  whole-push min : mean {whole.mean():.6f}  "
              f"range [{whole.min():.6f}, {whole.max():.6f}]  "
              f"({whole_col})")
        gap = win_min - whole
        print(f"  window - whole : mean {gap.mean():+.6f}, "
              f"{(gap >= -1e-6).mean() * 100:.1f}% of rows >= 0")
        print("     The window minimum should be >= the whole-push minimum: it "
              "minimizes over\n     60 steps rather than ~300. Equality on every "
              "row would mean the push's\n     worst configuration always falls "
              "inside the inference window.")

    # Relative spread, the same test create_dataloaders applies before training.
    rel = win_min.std() / max(abs(win_min.mean()), 1e-12)
    print(f"  relative std   : {rel:.2e}"
          + ("   [WARNING] near-constant; standardizing amplifies noise"
             if rel < 1e-3 else "   (usable signal)"))

    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    steps = np.arange(T)
    label = ("Manipulability w" if VARIANT_WINDOW_PREFIX == 'arm_manip_w'
             else "Directional manipulability $w_{dir}$")

    # --- Panel 1: raw sequences ---
    ax = axes[0, 0]
    idx = np.linspace(0, len(W) - 1, min(num_samples, len(W))).astype(int)
    for i in idx:
        ax.plot(steps, W[i], linewidth=1.4, alpha=0.85,
                label=f"row {i} (min {W[i].min():.4f})")
    ax.set_xlabel("Window step (aligned to input_vel_*)")
    ax.set_ylabel(label)
    ax.set_title(f"{num_samples} raw sequences")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: envelope + the two scalar reductions ---
    ax = axes[0, 1]
    mu_t, sd_t = W.mean(axis=0), W.std(axis=0)
    ax.plot(steps, mu_t, color="tab:blue", linewidth=2.2, label="mean over rows")
    ax.fill_between(steps, mu_t - sd_t, mu_t + sd_t, color="tab:blue", alpha=0.2,
                    label="+/- 1 std")
    ax.axhline(win_min.mean(), color="tab:red", linestyle="--", linewidth=1.6,
               label=f"mean window min ({win_min.mean():.4f})  <- cond variants")
    if whole is not None:
        ax.axhline(whole.mean(), color="tab:gray", linestyle=":", linewidth=1.6,
                   label=f"mean whole-push min ({whole.mean():.4f})  <- NOT used")
    ax.set_xlabel("Window step")
    ax.set_ylabel(label)
    ax.set_title("Across-row envelope and the scalar reductions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: window min vs whole-push min ---
    ax = axes[1, 0]
    sns.histplot(win_min, kde=True, color="tab:red", ax=ax, label="window min",
                 stat="density", alpha=0.55)
    if whole is not None:
        sns.histplot(whole, kde=True, color="tab:gray", ax=ax,
                     label=f"whole push ({whole_col})", stat="density", alpha=0.45)
    ax.set_xlabel(label)
    ax.set_title("What the conditioning scalar actually is")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: does arm configuration move it? ---
    ax = axes[1, 1]
    if {'obj_yaw_base', 'push_face_index'}.issubset(df.columns):
        plot_df = pd.DataFrame({
            "yaw_deg": np.degrees(df['obj_yaw_base'].values),
            "push_face": df['push_face_index'].values,
            "win_min": win_min,
        })
        for face, sub in plot_df.groupby("push_face"):
            agg = sub.groupby("yaw_deg")["win_min"].agg(['mean', 'std']).reset_index()
            ax.errorbar(agg["yaw_deg"], agg["mean"], yerr=agg["std"].fillna(0),
                        marker='o', markersize=4, capsize=2, linewidth=1.3,
                        label=f"push_face={int(face)}")
        ax.set_xlabel("Object yaw [deg]")
        ax.set_ylabel(f"Window min {label}")
        ax.set_title("Does arm configuration move it?")
        ax.legend(fontsize=8)

        spread = plot_df.groupby("yaw_deg")["win_min"].mean()
        if len(spread) > 1:
            rng = spread.max() - spread.min()
            print(f"  across-yaw spread of the cell means: {rng:.6f} "
                  f"({100.0 * rng / max(abs(win_min.mean()), 1e-12):.1f}% of the mean)")
            print("     A flat line here means the conditioning scalar carries no "
                  "orientation\n     information, and the *_cond variants cannot "
                  "beat vel_only.")
    else:
        ax.text(0.5, 0.5, "obj_yaw_base / push_face_index not in this CSV",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Does arm configuration move it?")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"INPUT_VARIANT = '{INPUT_VARIANT}'   channel "
                 f"'{VARIANT_WINDOW_PREFIX}*'   (N={len(df)})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("=" * 70 + "\n")


def compare_variant_channels(df):
    """Both window channels side by side, whichever variant is active.

    The scalar correlation between them decides whether w_dir is worth a
    separate variant at all: near 1.0 and it is a relabelling of w.
    """
    cols_w = window_cols(df, 'arm_manip_w')
    cols_d = window_cols(df, 'arm_dir_manip_w')
    if not cols_w or not cols_d:
        print("[VARIANT COMPARE] Need both 'arm_manip_w*' and 'arm_dir_manip_w*'; "
              "skipping.")
        return

    Ww = df[cols_w].values.astype(np.float32)
    Wd = df[cols_d].values.astype(np.float32)
    min_w, min_d = Ww.min(axis=1), Wd.min(axis=1)

    print("\n" + "=" * 70)
    print("WINDOW CHANNEL COMPARISON  (w vs w_dir)")
    print("=" * 70)
    print(f"  w      window min: mean {min_w.mean():.6f}  std {min_w.std():.6f}")
    print(f"  w_dir  window min: mean {min_d.mean():.6f}  std {min_d.std():.6f}")
    corr = float(np.corrcoef(min_w, min_d)[0, 1])
    print(f"  correlation: {corr:+.4f}")
    print("     w is a 6-D ellipsoid VOLUME, w_dir a single RADIUS along the push")
    print("     direction, so the magnitudes are not comparable. What matters is")
    print("     the correlation: near 1.0 and w_dir adds nothing beyond w.")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps = np.arange(Ww.shape[1])

    axes[0].plot(steps, Ww.mean(axis=0), color="tab:green", linewidth=2, label="w")
    axes[0].fill_between(steps, Ww.mean(0) - Ww.std(0), Ww.mean(0) + Ww.std(0),
                         color="tab:green", alpha=0.2)
    axes[0].set_title("Manipulability w over the window")
    axes[0].set_xlabel("Window step"); axes[0].set_ylabel("w")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, Wd.mean(axis=0), color="tab:purple", linewidth=2,
                 label="$w_{dir}$")
    axes[1].fill_between(steps, Wd.mean(0) - Wd.std(0), Wd.mean(0) + Wd.std(0),
                         color="tab:purple", alpha=0.2)
    axes[1].set_title("Directional manipulability over the window")
    axes[1].set_xlabel("Window step"); axes[1].set_ylabel("$w_{dir}$")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(min_w, min_d, s=12, alpha=0.4, color="tab:blue")
    axes[2].set_xlabel("window min w")
    axes[2].set_ylabel("window min $w_{dir}$")
    axes[2].set_title(f"Are they redundant?   corr = {corr:+.4f}")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Manipulability vs directional manipulability",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("=" * 70 + "\n")


# =============================================================================
# INERTIAL WINDOW CHANNELS: m_eff and Lambda
# =============================================================================
# Both are computed per physics step in task_space_all_continuous_actions_phypush.py
# (_run_physics_step -> _operational_space_inertia) and stored in arm_state at
#   index 25: m_eff    = 1 / (u^T Lambda^-1 u)          effective mass along push dir u
#   index 26: lam_mean = mean(diag(Lambda[0:3, 0:3]))   direction-free translational mean
# with Lambda = (J M^-1 J^T)^-1, J in the raw Isaac [linear, angular] row order.
# off_policy_algorithm.get_arm_window slices them on the SAME window as
# input_vel_*, so arm_meff_w{t} / arm_lam_w{t} line up with input_vel_{t}.
INERTIAL_CHANNELS = {
    # prefix          (label,                                   at-impact column,     color)
    "arm_meff_w": (r"Effective mass $m_{eff}(u)$ [kg]",          "arm_meff_at_impact", "tab:green"),
    "arm_lam_w":  (r"Mean translational $\Lambda_{ii}$ [kg]",   "arm_lam_at_impact",  "tab:brown"),
}


def report_inertial_window(df, num_samples=6, log_scale=False):
    """Visualize the 60-step operational-space inertia channels.

    One row of panels per channel (m_eff, Lambda):
      1. raw sequences for evenly spaced rows
      2. mean +/- 1 std envelope across rows, with the window MAX marked --
         max is the adverse reduction for inertial channels (the arm's own
         inertia swamps the object's), mirroring min for manipulability
      3. distribution of the window max, with the at-impact value overlaid
      4. window max vs object yaw, split by push side

    A final figure relates the two channels to each other, to the object mass,
    and to directional manipulability, and overlays both on velocity for one row.

    log_scale=True uses a log y-axis. Lambda is the inverse of a matrix that goes
    singular near kinematic singularities, so a few rows can be orders of
    magnitude above the rest.
    """
    print("\n" + "=" * 70)
    print("INERTIAL WINDOW CHANNELS  (m_eff, Lambda)")
    print("=" * 70)

    vel_cols = window_cols(df, "input_vel_")
    T_vel = len(vel_cols)

    data = {}
    for pfx, (label, impact_col, _) in INERTIAL_CHANNELS.items():
        cols = window_cols(df, pfx)
        if not cols:
            print(f"  [WARNING] No '{pfx}*' columns. Re-collect with the 4-channel "
                  f"get_arm_window (channels 23..26).")
            continue
        W = df[cols].values.astype(np.float64)                  # (N, T)
        T = W.shape[1]
        if T != T_vel:
            print(f"  [ERROR] '{pfx}*' has {T} steps but input_vel_* has {T_vel}. "
                  f"Not time-aligned.")

        bad = ~np.isfinite(W) | (W <= 0.0)
        # _operational_space_inertia maps NaN/inf m_eff to 0.0, so a zero here
        # is a failed computation, not a real mass.
        n_bad_rows = int(bad.any(axis=1).sum())

        print(f"\n  {pfx}0..{T - 1}   ({label})")
        print(f"    rows                 : {len(W)}")
        print(f"    per-step value       : median {np.nanmedian(W):.4f}  "
              f"range [{np.nanmin(W):.4f}, {np.nanmax(W):.4f}] kg")
        if n_bad_rows:
            print(f"    [WARNING] {n_bad_rows} rows contain zero / non-finite values "
                  f"(failed solve or singular Lambda). Excluded from the stats below.")
        good = ~bad.any(axis=1)
        Wg = W[good]
        if len(Wg) == 0:
            print("    no valid rows; skipping.")
            continue

        win_max = Wg.max(axis=1)
        win_min = Wg.min(axis=1)
        within = (win_max - win_min) / np.maximum(Wg.mean(axis=1), 1e-12)
        print(f"    window max           : mean {win_max.mean():.4f}  "
              f"std {win_max.std():.4f}  range [{win_max.min():.4f}, {win_max.max():.4f}]")
        print(f"    within-push change   : mean {100 * within.mean():.2f}% of the "
              f"row mean  (max {100 * within.max():.1f}%)")
        print(f"    across-push rel. std : {win_max.std() / max(win_max.mean(), 1e-12):.3f}")

        # Anchor check: w0 is the impact sample by construction.
        if impact_col in df.columns:
            imp = df[impact_col].values.astype(np.float64)
            err = np.abs(imp - W[:, 0])[good]
            rel = err / np.maximum(np.abs(imp[good]), 1e-12)
            print(f"    anchor |w0 - {impact_col}| : max {err.max():.2e} "
                  f"(rel {rel.max():.2e})  "
                  f"{'OK' if rel.max() < 1e-4 else '*** window not anchored at start_t ***'}")

        # Heavy tail: a sign of near-singular configurations inside the window.
        p50, p99 = np.percentile(win_max, [50, 99])
        if p99 > 10 * p50:
            print(f"    [NOTE] heavy tail: p99 {p99:.3f} > 10x median {p50:.3f}. "
                  f"Consider log_scale=True, and a log transform before "
                  f"standardizing if this becomes an input channel.")

        data[pfx] = dict(W=W, good=good, win_max_all=np.where(good, W.max(axis=1), np.nan))

    if not data:
        print("=" * 70 + "\n")
        return

    # ------------------------------------------------------------------
    # Figure 1: per-channel panels
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    n_rows = len(data)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.8 * n_rows), squeeze=False)
    sample_idx = np.linspace(0, len(df) - 1, min(num_samples, len(df))).astype(int)

    for r, (pfx, d) in enumerate(data.items()):
        label, impact_col, color = INERTIAL_CHANNELS[pfx]
        W, good = d["W"], d["good"]
        Wg = W[good]
        steps = np.arange(W.shape[1])
        win_max = Wg.max(axis=1)

        # 1. raw sequences
        ax = axes[r][0]
        for i in sample_idx:
            gm = df['gt_mass'].iloc[i] if 'gt_mass' in df.columns else float('nan')
            ax.plot(steps, W[i], linewidth=1.4, alpha=0.85,
                    label=f"row {i}  m_obj={gm:.2f}")
        ax.set_title(f"{pfx}*: {len(sample_idx)} raw sequences")
        ax.set_xlabel("Window step (aligned to input_vel_*)")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)

        # 2. envelope + adverse reduction
        ax = axes[r][1]
        mu_t, sd_t = Wg.mean(axis=0), Wg.std(axis=0)
        med_t = np.median(Wg, axis=0)
        ax.plot(steps, mu_t, color=color, linewidth=2.2, label="mean")
        ax.plot(steps, med_t, color=color, linestyle=":", linewidth=1.6, label="median")
        ax.fill_between(steps, np.maximum(mu_t - sd_t, 1e-9), mu_t + sd_t,
                        color=color, alpha=0.2, label="+/- 1 std")
        ax.axhline(win_max.mean(), color="tab:red", linestyle="--", linewidth=1.5,
                   label=f"mean window max ({win_max.mean():.3f})")
        argmax_t = Wg.argmax(axis=1)
        ax2 = ax.twinx()
        ax2.hist(argmax_t, bins=np.arange(W.shape[1] + 1) - 0.5, color="gray",
                 alpha=0.25)
        ax2.set_ylabel("count of argmax step", color="gray", fontsize=8)
        ax2.grid(False)
        ax.set_title("Across-row envelope (gray: where the max occurs)")
        ax.set_xlabel("Window step")
        ax.set_ylabel(label)
        ax.legend(fontsize=7, loc="upper left")

        # 3. distribution of window max vs at-impact
        ax = axes[r][2]
        sns.histplot(win_max, kde=True, color="tab:red", stat="density", alpha=0.5,
                     ax=ax, label="window max", log_scale=log_scale)
        if impact_col in df.columns:
            imp = df[impact_col].values.astype(np.float64)[good]
            sns.histplot(imp, kde=True, color=color, stat="density", alpha=0.4,
                         ax=ax, label="at impact (w0)", log_scale=log_scale)
        ax.set_title("Window max vs value at impact")
        ax.set_xlabel(label)
        ax.legend(fontsize=8)

        # 4. vs yaw / push side
        ax = axes[r][3]
        if {'obj_yaw_base', 'push_face_index'}.issubset(df.columns):
            pdf = pd.DataFrame({
                "yaw_deg": np.degrees(df['obj_yaw_base'].values),
                "push_face": df['push_face_index'].values,
                "v": d["win_max_all"],
            }).dropna()
            for face, sub in pdf.groupby("push_face"):
                agg = sub.groupby("yaw_deg")["v"].agg(['mean', 'std']).reset_index()
                ax.errorbar(agg["yaw_deg"], agg["mean"], yerr=agg["std"].fillna(0),
                            marker='o', markersize=4, capsize=2, linewidth=1.3,
                            label=f"push_face={int(face)}")
            cell = pdf.groupby(["yaw_deg", "push_face"])["v"].mean()
            if len(cell) > 1:
                spread = 100 * (cell.max() - cell.min()) / max(cell.mean(), 1e-12)
                print(f"  {pfx}: window-max spread across (yaw x side) cells = "
                      f"{spread:.1f}% of mean")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "obj_yaw_base / push_face_index missing",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Window max vs object yaw")
        ax.set_xlabel("Object yaw [deg]")
        ax.set_ylabel(f"window max {label}")

        if log_scale:
            for k in (0, 1, 3):
                axes[r][k].set_yscale("log")

    fig.suptitle(f"Operational-space inertia over the {T_vel}-step inference window "
                 f"(N={len(df)})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Figure 2: relationships
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    have_both = "arm_meff_w" in data and "arm_lam_w" in data

    # a) m_eff vs Lambda (window max)
    ax = axes[0]
    if have_both:
        a = data["arm_meff_w"]["win_max_all"]
        b = data["arm_lam_w"]["win_max_all"]
        ok = np.isfinite(a) & np.isfinite(b)
        corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        ax.scatter(b[ok], a[ok], s=10, alpha=0.4)
        ax.set_xlabel("window max mean $\\Lambda_{ii}$ [kg]")
        ax.set_ylabel("window max $m_{eff}$ [kg]")
        ax.set_title(f"m_eff vs Lambda   corr = {corr:+.3f}")
        print(f"\n  corr(window max m_eff, window max Lambda) = {corr:+.4f}")
        print("     m_eff depends on the push direction; lam_mean does not. Near 1.0 "
              "means\n     the direction adds little beyond the overall inertia level.")
        if log_scale:
            ax.set_xscale("log"); ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "need both channels", ha="center", transform=ax.transAxes)

    # b) arm inertia vs object mass
    ax = axes[1]
    if "arm_meff_w" in data and "gt_mass" in df.columns:
        mm = data["arm_meff_w"]["win_max_all"]
        gm = df["gt_mass"].values.astype(np.float64)
        ok = np.isfinite(mm)
        ratio = mm[ok] / np.maximum(gm[ok], 1e-9)
        sns.histplot(ratio, kde=True, ax=ax, color="tab:green", log_scale=True)
        ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("m_eff (window max) / gt_mass")
        ax.set_title("Arm inertia relative to object mass")
        print(f"  m_eff / gt_mass: median {np.median(ratio):.2f}, "
              f"{100 * (ratio > 1).mean():.1f}% of rows above 1")
        print("     Above 1, the arm's own inertia along the push exceeds the object's "
              "mass,\n     so the velocity dip at impact carries a diluted mass signal.")

    # c) m_eff vs directional manipulability
    ax = axes[2]
    wd_cols = window_cols(df, "arm_dir_manip_w")
    if "arm_meff_w" in data and wd_cols:
        mm = data["arm_meff_w"]["win_max_all"]
        wd_min = df[wd_cols].values.astype(np.float64).min(axis=1)
        ok = np.isfinite(mm) & np.isfinite(wd_min)
        corr = float(np.corrcoef(mm[ok], wd_min[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        ax.scatter(wd_min[ok], mm[ok], s=10, alpha=0.4, color="tab:purple")
        ax.set_xlabel("window min $w_{dir}$")
        ax.set_ylabel("window max $m_{eff}$ [kg]")
        ax.set_title(f"Inertial vs kinematic   corr = {corr:+.3f}")
        print(f"  corr(window max m_eff, window min w_dir) = {corr:+.4f}")
        print("     Near +/-1 means m_eff is largely a relabelling of w_dir; low |corr| "
              "means\n     the inertial channel carries information the kinematic one "
              "does not.")
        if log_scale:
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "arm_dir_manip_w* missing", ha="center",
                transform=ax.transAxes)

    # d) one row: velocity with both inertial channels
    ax = axes[3]
    good_all = np.logical_and.reduce([d["good"] for d in data.values()])
    if good_all.any() and vel_cols:
        i = int(np.flatnonzero(good_all)[len(np.flatnonzero(good_all)) // 2])
        steps = np.arange(T_vel)
        ax.plot(steps, df[vel_cols].iloc[i].values.astype(float), color="tab:blue",
                linewidth=2, label="velocity")
        ax.set_ylabel("v [m/s]", color="tab:blue")
        ax2 = ax.twinx()
        for pfx, d in data.items():
            _, _, color = INERTIAL_CHANNELS[pfx]
            ax2.plot(np.arange(d["W"].shape[1]), d["W"][i], color=color,
                     linewidth=1.6, label=pfx[:-2])
        ax2.set_ylabel("inertia [kg]")
        ax2.grid(False)
        if log_scale:
            ax2.set_yscale("log")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
        gm = df['gt_mass'].iloc[i] if 'gt_mass' in df.columns else float('nan')
        ax.set_title(f"row {i}  (m_obj={gm:.3f} kg)")
        ax.set_xlabel("Window step")

    fig.suptitle("How the inertial channels relate to each other, the object, "
                 "and w_dir", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("=" * 70 + "\n")


# =============================================================================
# CHANNEL INFORMATIVENESS  (why a variant can collapse)
#
# A variant's second input channel is standardized with ONE mean/std computed
# over the whole sweep: z = (x - mean) / std, on log(x) for the inertial
# channels. Two things follow, and both are checked here:
#
#   1. If the channel barely varies in training, std is tiny and it carries no
#      information about the object -- every push looks the same to the model.
#   2. A tiny std also AMPLIFIES any sim-to-real offset: a real value d in log
#      units lands at z = d / std. With std = 0.0097, a 10% higher real value
#      (d = log 1.1 = 0.0953) lands at z = +9.8, far outside anything seen in
#      training, and the model saturates.
#
# The variance split says WHERE what little variation there is comes from:
#   within-push   variation across the 60 steps of one push
#   between-push  variation of the per-push means
# A channel that is flat within a push AND flat between pushes is a constant.
# =============================================================================
INFORMATIVENESS_OFFSETS = (0.05, 0.10, 0.28)     # relative offsets to price in z


def report_channel_informativeness(df, min_rel_std=0.02):
    print("\n" + "=" * 78)
    print(" CHANNEL INFORMATIVENESS AND SENSITIVITY")
    print("=" * 78)
    print("  Values are in MODEL SPACE: log(x) for the inertial channels "
          "(arm_meff_w, arm_lam_w),\n  raw for the manipulability channels.")

    prefixes = sorted({p for p in ALL_VARIANTS.values() if p is not None})
    rows = []
    for pfx in prefixes:
        W, valid, cols = variant_window(df, prefix=pfx,
                                        log_transform=pfx in LOG_PREFIXES)
        if not cols:
            print(f"\n  [{pfx}*] no columns in this CSV; skipped.")
            continue
        W = np.asarray(W, dtype=np.float64)[valid]
        if len(W) == 0:
            print(f"\n  [{pfx}*] no valid rows; skipped.")
            continue
        is_log = pfx in LOG_PREFIXES

        mean = float(W.mean())
        std = float(W.std())                  # what save_arm_stats stores
        # Relative variation, comparable across channels: for a log channel the
        # std IS the relative spread (exp(std) - 1); for a raw one it is std/|mean|.
        # std/|mean| in log space would depend on the arbitrary log offset.
        rel = (np.exp(std) - 1.0) if is_log else std / max(abs(mean), 1e-12)

        # Variance split. Within = spread across the 60 steps of one push;
        # between = spread of the per-push means.
        row_mean = W.mean(axis=1)
        within = float(np.mean(W.std(axis=1)))
        between = float(row_mean.std())

        print(f"\n  [{pfx}*]   {'log space' if is_log else 'raw'}   "
              f"N={len(W)} pushes x {W.shape[1]} steps")
        print(f"    standardization  mean {mean:+.6f}   std {std:.6f}   "
              f"relative variation {100 * rel:.2f}%")
        if is_log:
            # exp(mean) is the geometric mean in kg; exp(std)-1 the typical
            # relative variation the channel actually shows.
            print(f"    in kg            geometric mean {np.exp(mean):.4f} kg,  "
                  f"typical variation +/-{100 * (np.exp(std) - 1):.2f}%")
        print(f"    variance split   within-push {within:.6f}   "
              f"between-push {between:.6f}   "
              f"(between/within {between / max(within, 1e-12):.2f})")

        # How far a sim-to-real offset lands, in training z units.
        amp = 1.0 / max(std, 1e-12)
        if is_log:
            shifts = ", ".join(f"+{100 * o:.0f}% -> z={np.log1p(o) * amp:+.1f}"
                               for o in INFORMATIVENESS_OFFSETS)
        else:
            shifts = ", ".join(f"+{100 * o:.0f}% -> z={o * abs(mean) * amp:+.1f}"
                               for o in INFORMATIVENESS_OFFSETS)
        print(f"    offset -> z      {shifts}")
        unit = "1.0 in log space (a factor of e)" if is_log else "1.0 raw"
        print(f"    amplification    {unit} = {amp:.1f} z")

        # Does the channel say anything about the object? It should not: the arm
        # follows the same trajectory whatever is on the table. A near-zero
        # correlation with gt_mass is expected and is the point -- the channel is
        # supposed to describe the ARM, not the object.
        for tgt in ("gt_mass", "gt_mu", "obj_yaw_base"):
            if tgt not in df.columns:
                continue
            t = df[tgt].values.astype(np.float64)[valid]
            if t.std() > 0 and row_mean.std() > 0:
                c = float(np.corrcoef(row_mean, t)[0, 1])
                print(f"    corr with {tgt:<13} {c:+.3f}")

        flat = rel < min_rel_std
        if flat:
            print(f"    [WARNING] nearly CONSTANT in training "
                  f"(varies {100 * rel:.2f}% < {100 * min_rel_std:.0f}%).")
            print(f"              The model sees the same value for every push, so this")
            print(f"              channel adds no information, and standardizing divides")
            print(f"              by {std:.5f}: a {100 * INFORMATIVENESS_OFFSETS[-1]:.0f}% "
                  f"sim-to-real offset arrives at "
                  f"z={np.log1p(INFORMATIVENESS_OFFSETS[-1]) * amp if is_log else INFORMATIVENESS_OFFSETS[-1] * abs(mean) * amp:+.0f},")
            print(f"              which saturates the model and collapses its output.")

        # Where the variation lives across the sweep. The pushes ARE multi-angle,
        # so if a channel is meant to describe the arm's configuration it should
        # vary between collect_idx / yaw groups. A channel that is flat BETWEEN
        # angles as well as within a push is constant for the whole dataset.
        for key in ("collect_idx", "push_face_index"):
            if key not in df.columns:
                continue
            grp = pd.Series(row_mean).groupby(df[key].values[valid])
            gm = grp.mean()
            print(f"    across {key:<16} spread {gm.std():.6f} "
                  f"(range {gm.max() - gm.min():.6f}, "
                  f"{100 * (np.exp(gm.max() - gm.min()) - 1) if is_log else 100 * (gm.max() - gm.min()) / max(abs(mean), 1e-12):.2f}% "
                  f"between the extreme groups)")
            print(f"      per-group means: "
                  + ", ".join(f"{k}:{v:.4f}" for k, v in gm.items()))
        if 'obj_yaw_base' in df.columns:
            yaw = np.degrees(df['obj_yaw_base'].values[valid])
            if row_mean.std() > 0:
                print(f"    corr with yaw_deg      "
                      f"{float(np.corrcoef(row_mean, yaw)[0, 1]):+.3f}")

        rows.append((pfx, mean, std, rel, within, between, amp, flat))



    if len(rows) > 1:
        print("\n  " + "-" * 74)
        print(f"  {'channel':<18}{'std':>10}{'variation':>12}{'within':>10}"
              f"{'between':>10}{'x amplif':>10}")
        for pfx, mean, std, rel, within, between, amp, flat in rows:
            print(f"  {pfx:<18}{std:>10.5f}{100 * rel:>11.2f}%{within:>10.5f}"
                  f"{between:>10.5f}{amp:>10.0f}"
                  + ("   <-- flat" if flat else ""))
        least = min(rows, key=lambda r: r[3])
        most = max(rows, key=lambda r: r[3])
        print(f"\n  Most variation: {most[0]} ({100 * most[3]:.2f}%).  "
              f"Least: {least[0]} ({100 * least[3]:.2f}%), "
              f"{most[3] / max(least[3], 1e-12):.0f}x flatter.")
        print("  A flatter channel is both less informative and more fragile: the same")
        print("  real-world offset arrives that many times further out in z.")
    print("=" * 78 + "\n")


# =============================================================================
# PER-PLOT NUMERIC ANALYSIS
# =============================================================================

TARGET_PUSH_SPEED = 0.08     

def _verdict(ok, bad="CHECK"):
    return "OK" if ok else f"*** {bad} ***"

def analyze_velocity(vel, sample_id):
    T = len(vel)
    tail = vel[T // 2:]                       
    plateau, plateau_std = float(tail.mean()), float(tail.std())
    err_pct = 100 * abs(plateau - TARGET_PUSH_SPEED) / TARGET_PUSH_SPEED
    flat_pct = 100 * plateau_std / max(abs(plateau), 1e-9)

    above = np.where(vel >= 0.9 * plateau)[0]
    rise_idx = int(above[0]) if above.size else -1

    print(f"\n  [PLOT 1] Velocity  (sample {sample_id})")
    print(f"    range              : {vel.min():+.4f} .. {vel.max():+.4f} m/s")
    print(f"    plateau (2nd half) : {plateau:.4f} +/- {plateau_std:.4f} m/s")
    print(f"    vs target {TARGET_PUSH_SPEED:.3f}    : {err_pct:5.1f}% off       {_verdict(err_pct < 15)}")
    print(f"    plateau flatness   : {flat_pct:5.1f}% rel std   {_verdict(flat_pct < 15)}")
    print(f"    reaches 90% at t   : {rise_idx} / {T}")
    if vel.min() < 0:
        print(f"    [NOTE] velocity goes negative (min {vel.min():.4f}) -- retraction "
              f"or a frame-sign issue has leaked into the window")

def analyze_acceleration(acc, sample_id):
    T = len(acc)
    peak_idx = int(np.argmin(acc))
    peak_val = float(acc[peak_idx])
    tail_mean = float(np.abs(acc[T // 2:]).mean())

    mask = np.ones(T, dtype=bool)
    mask[max(0, peak_idx - 5):min(T, peak_idx + 6)] = False
    runner_up = float(acc[mask].min()) if mask.any() else float("nan")
    margin = 100 * abs(peak_val - runner_up) / max(abs(peak_val), 1e-9)

    print(f"\n  [PLOT 2] Acceleration  (sample {sample_id})")
    print(f"    range              : {acc.min():+.4f} .. {acc.max():+.4f} m/s^2")
    print(f"    peak (argmin) at t : {peak_idx} / {T}, value {peak_val:+.4f}")
    print(f"    2nd deepest dip    : {runner_up:+.4f} -> margin {margin:5.1f}%   {_verdict(margin > 20)}")
    print(f"    tail mean |acc|    : {tail_mean:.4f} m/s^2    "
          f"{_verdict(tail_mean < 0.5 * abs(peak_val))}")
    if margin <= 20:
        print("    [NOTE] the impact peak is not a clear outlier -- the window anchor "
              "could shift between otherwise identical pushes")

def analyze_force_decomposition(net_sim, f_robot, f_fric_vec, f_calc, sample_id):
    rmse = float(np.sqrt(((f_calc - net_sim) ** 2).mean()))
    scale = float(np.abs(net_sim).mean()) + 1e-9
    corr = float(np.corrcoef(f_calc, net_sim)[0, 1]) if np.std(f_calc) > 1e-12 else float("nan")

    print(f"\n  [PLOT 3] Force decomposition  (sample {sample_id})")
    print(f"    |F_net| (sim) mean : {scale:.4f} N")
    print(f"    F_robot mean       : {f_robot.mean():+.4f} N")
    print(f"    F_fric  mean       : {f_fric_vec.mean():+.4f} N")
    print(f"    RMSE(calc, sim)    : {rmse:.4f} N = {100 * rmse / scale:5.1f}% of |F_net|  "
          f"{_verdict(rmse / scale < 0.25)}")
    print(f"    correlation        : {corr:+.4f}    {_verdict(corr > 0.8)}")

def analyze_newton(net_sim, m_a, acc_sim, gt_mass, sample_id):
    rmse = float(np.sqrt(((net_sim - m_a) ** 2).mean()))
    scale = float(np.abs(net_sim).mean()) + 1e-9

    implied_mass = float(acc_sim.dot(net_sim) / (acc_sim.dot(acc_sim) + 1e-12))
    mass_err_pct = 100 * abs(implied_mass - gt_mass) / max(gt_mass, 1e-9)

    print(f"\n  [PLOT 4] Newton's 2nd law  (sample {sample_id})")
    print(f"    gt_mass            : {gt_mass:.4f} kg")
    print(f"    implied mass (LSQ) : {implied_mass:.4f} kg -> {mass_err_pct:5.1f}% off  "
          f"{_verdict(mass_err_pct < 20)}")
    print(f"    RMSE(F_net, m*a)   : {rmse:.4f} N = {100 * rmse / scale:5.1f}% of |F_net|  "
          f"{_verdict(rmse / scale < 0.25)}")
    if mass_err_pct >= 20:
        print("    [NOTE] the recorded physics does not close here -- this bounds the "
              "achievable estimation accuracy regardless of the model")

def analyze_friction(fric_calc, fric_sim, normal_sim, gt_mu, sample_id):
    rmse = float(np.sqrt(((fric_calc - fric_sim) ** 2).mean()))
    scale = float(np.abs(fric_sim).mean()) + 1e-9
    ratio = float(fric_calc.mean() / (fric_sim.mean() + 1e-9))

    valid = np.abs(normal_sim) > 1e-6
    if valid.any():
        implied_mu = float((fric_sim[valid] / np.abs(normal_sim[valid])).mean())
        mu_err_pct = 100 * abs(implied_mu - gt_mu) / max(gt_mu, 1e-9)
    else:
        implied_mu, mu_err_pct = float("nan"), float("nan")

    print(f"\n  [PLOT 5] Coulomb friction  (sample {sample_id})")
    print(f"    gt_mu              : {gt_mu:.4f}")
    print(f"    implied mu (sensor): {implied_mu:.4f} -> {mu_err_pct:5.1f}% off  "
          f"{_verdict(np.isfinite(mu_err_pct) and mu_err_pct < 20)}")
    print(f"    N_sim mean         : {np.abs(normal_sim).mean():.4f} N")
    print(f"    F_fric sim / calc  : {fric_sim.mean():.4f} / {fric_calc.mean():.4f} N "
          f"(ratio {ratio:.3f})   {_verdict(0.75 < ratio < 1.33)}")
    print(f"    RMSE(calc, sim)    : {rmse:.4f} N = {100 * rmse / scale:5.1f}% of |F_fric|  "
          f"{_verdict(rmse / scale < 0.25)}")


def inspect_samples(df, num_samples=3):
    """
    Extracts elements directly from the DataFrame instead of the PyTorch DataLoader 
    so we can access raw object yaws and unstandardized arm states safely.
    Samples are evenly spaced to ensure variety in object yaw.
    """
    acc_cols = sorted([c for c in df.columns if "input_acc_" in c], key=lambda x: int(x.split('_')[-1]))
    vel_cols = sorted([c for c in df.columns if "input_vel_" in c], key=lambda x: int(x.split('_')[-1]))
    seq_len = len(acc_cols)
    time_steps = np.arange(seq_len)
    
    colors = {
        'vel': '#1f77b4',       
        'acc': '#d62728',       
        'robot': '#2ca02c',     
        'friction': '#ff7f0e',  
        'net_sim': '#7f7f7f',   
        'net_calc': '#9467bd',  
        'theory': '#17becf'     
    }
    
    sns.set_theme(style="whitegrid")
    
    # Grab evenly spaced indices to avoid looking only at the very first yaw
    sample_indices = np.linspace(0, len(df) - 1, min(num_samples, len(df)), dtype=int)
    
    for i in sample_indices:
        row = df.iloc[i]
        
        gt_mass = row['gt_mass']
        gt_mu = row['gt_mu']
        
        # Raw kinematics mapping
        yaw_base = row['obj_yaw_base'] if 'obj_yaw_base' in row else float('nan')

        vel_x = row[vel_cols].values.astype(float)
        acc_x = row[acc_cols].values.astype(float)
        
        st = int(row['start_t'])
        window_range = range(st, st + seq_len)

        # Force mapping dependent on framework
        if FRAME_MODE == "world":
            fz_robot_sim = np.array([row[f"pinn_robot_wrench_t{t}_ax5"] for t in window_range])
            acc_x_sim = np.array([row[f"pinn_RHS_acc_t{t}_ax3"] for t in window_range])
            net_fx_sim = np.array([row[f"pinn_LHS_wrench_t{t}_ax3"] for t in window_range])
            fz_normal_sim = np.array([row[f"pinn_table_wrench_t{t}_ax5"] for t in window_range])
            fx_robot_sim = np.array([row[f"pinn_robot_wrench_t{t}_ax3"] for t in window_range])
        elif FRAME_MODE == "local":
            fz_robot_sim = np.array([row[f"pinn_robot_wrench_t{t}_ax3"] for t in window_range])
            acc_x_sim = np.array([row[f"pinn_RHS_acc_t{t}_ax5"] for t in window_range])
            net_fx_sim = np.array([row[f"pinn_LHS_wrench_t{t}_ax5"] for t in window_range])
            fz_normal_sim = np.array([row[f"pinn_table_wrench_t{t}_ax3"] for t in window_range])
            fx_robot_sim = np.array([row[f"pinn_robot_wrench_t{t}_ax5"] for t in window_range])

        yaw_title_str = f"| Yaw: {np.degrees(yaw_base):.1f}°" if not np.isnan(yaw_base) else ""
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=False)
        
        # -----------------------------------------------------------
        # PLOT 1: Velocity
        # -----------------------------------------------------------
        axes[0].plot(time_steps, vel_x, color=colors['vel'], marker='o', markersize=4, linewidth=2, label='Extracted EE Velocity')
        axes[0].set_title(f"1. Model Input: Kinematics (Velocity) {yaw_title_str}")
        axes[0].set_ylabel("Velocity [m/s]")
        axes[0].legend(loc='upper left')

        print("\n" + "=" * 70)
        face = int(row['push_face_index']) if 'push_face_index' in row else -1
        cidx = int(row['collect_idx']) if 'collect_idx' in row else -1
        push_str = ""
        if 'push_dir_b_x' in row and 'push_dir_b_y' in row:
            push_deg = np.degrees(np.arctan2(row['push_dir_b_y'], row['push_dir_b_x']))
            rel = (push_deg - np.degrees(yaw_base) + 180.0) % 360.0 - 180.0
            push_str = f"   push_dir = {push_deg:+.2f} deg (push - yaw = {rel:+.2f})"
        print(f"SAMPLE {i}  |  gt_mass = {gt_mass:.4f} kg   gt_mu = {gt_mu:.4f}")
        print(f"          yaw_base = {yaw_base:+.4f} rad = {np.degrees(yaw_base):+.2f} deg"
              f"   push_face = {face}   collect_idx = {cidx}{push_str}")
        print("=" * 70)
        analyze_velocity(vel_x, i)
        
        # -----------------------------------------------------------
        # PLOT 2: Acceleration
        # -----------------------------------------------------------
        axes[1].plot(time_steps, acc_x, color=colors['acc'], marker='o', markersize=4, linewidth=2, label='Extracted EE Acceleration')
        axes[1].set_title(f"2. Model Input: Kinematics (Acceleration) {yaw_title_str}")
        axes[1].set_ylabel("Acceleration [m/s\u00b2]")
        axes[1].legend(loc='upper left')

        analyze_acceleration(acc_x, i)

        # -----------------------------------------------------------
        # PHYSICS CALCULATIONS
        # -----------------------------------------------------------
        if FRAME_MODE == "world":
            normal_force_calc = np.clip((gt_mass * G) - fz_robot_sim, 0.0, None)
        elif FRAME_MODE == "local":
            normal_force_calc = np.clip((gt_mass * G) + fz_robot_sim, 0.0, None)
        
        fric_magnitude_calc = gt_mu * normal_force_calc 
        fric_magnitude_sim = gt_mu * np.abs(fz_normal_sim)
        
        fx_friction_vector = -fric_magnitude_sim
        calc_net_force_x = fx_robot_sim + fx_friction_vector
        mass_x_accel = gt_mass * acc_x_sim
        
        # -----------------------------------------------------------
        # PLOT 3: Force Decomposition
        # -----------------------------------------------------------
        axes[2].plot(time_steps, net_fx_sim, label=r'Simulator Net Force ($F_{net}$)', color=colors['net_sim'], linewidth=4, alpha=0.4)
        axes[2].plot(time_steps, fx_robot_sim, label=r'Robot Applied Force ($F_{robot}$)', color=colors['robot'], linewidth=2)
        axes[2].plot(time_steps, fx_friction_vector, label=r'Table Friction ($-F_{fric}$)', color=colors['friction'], linewidth=2)
        axes[2].plot(time_steps, calc_net_force_x, label=r'Calculated Net Force ($F_{robot} - F_{fric}$)', color=colors['net_calc'], linestyle='--', linewidth=2)
        
        axes[2].set_title(f"3. Force Components Decomposition (X-Axis) {yaw_title_str}")
        axes[2].set_ylabel("Force [N]")
        axes[2].legend(loc='upper left')

        analyze_force_decomposition(net_fx_sim, fx_robot_sim, fx_friction_vector, calc_net_force_x, i)
        
        # -----------------------------------------------------------
        # PLOT 4: Newton's Second Law Check
        # -----------------------------------------------------------
        axes[3].plot(time_steps, net_fx_sim, label=r'Simulator Net Force ($F_{net}$)', color=colors['net_sim'], linewidth=4, alpha=0.4)
        axes[3].plot(time_steps, calc_net_force_x, label=r'Force Sum ($F_{robot} - F_{fric}$)', color=colors['net_calc'], linewidth=2)
        axes[3].plot(time_steps, mass_x_accel, label=r"Newton's 2nd Law ($m \cdot a_x$)", color=colors['theory'], linestyle='--', linewidth=2.5)
        
        axes[3].set_title(f"4. Physics Check: Newton's 2nd Law Alignment {yaw_title_str}")
        axes[3].set_ylabel("Force [N]")
        add_min_max_text(axes[3], net_fx_sim, "N")
        axes[3].legend(loc='upper left')

        analyze_newton(net_fx_sim, mass_x_accel, acc_x_sim, gt_mass, i)
    
        # -----------------------------------------------------------
        # PLOT 5: Friction Model Check
        # -----------------------------------------------------------
        axes[4].plot(time_steps, fric_magnitude_calc, label=r'Theoretical Friction ($\mu \cdot N_{calc}$)', color=colors['theory'], linewidth=2)
        axes[4].plot(time_steps, fric_magnitude_sim, label=r'Simulator Friction ($\mu \cdot N_{sim}$)', color=colors['friction'], linestyle='--', linewidth=2.5)
        
        axes[4].set_title(f"5. Physics Check: Coulomb Friction Model (Magnitudes) {yaw_title_str}")
        axes[4].set_ylabel("Force Magnitude [N]")
        axes[4].set_xlabel("Time Step")
        add_min_max_text(axes[4], fric_magnitude_calc, "N")
        axes[4].legend(loc='upper left')

        analyze_friction(fric_magnitude_calc, fric_magnitude_sim, fz_normal_sim, gt_mu, i)
        print()
        
        for ax in axes:
            ax.grid(True, linestyle=':', alpha=0.6)
            
        fig.tight_layout()
        plt.show()


def inspect_velocity_vs_yaw(df):
    """
    Finds a specific set of physical properties (same ground truth mass and mu)
    that has multiple object yaw variations and both push faces. 
    Plots their EE velocity sequences separated by push_face_index to visualize 
    the variance caused strictly by the robot's kinematic configuration.
    """
    if 'obj_yaw_base' not in df.columns or 'push_face_index' not in df.columns:
        print("[WARNING] Missing obj_yaw_base or push_face_index column. Cannot plot velocity vs yaw.")
        return

    # Group by physical properties to find a set that contains a full multi-angle sweep
    grouped = df.groupby(['gt_mass', 'gt_mu'])
    
    target_group = None
    for name, group in grouped:
        # Look for a physical property pair that has a diverse spread of orientations
        # AND contains pushes from both faces
        if group['obj_yaw_base'].nunique() > 5 and group['push_face_index'].nunique() > 1:
            target_group = group
            break
            
    if target_group is None:
        print("[WARNING] Could not find a (mass, mu) pair with multiple yaw angles and both push faces.")
        return

    gt_mass = target_group.iloc[0]['gt_mass']
    gt_mu = target_group.iloc[0]['gt_mu']
    
    vel_cols = sorted([c for c in df.columns if "input_vel_" in c], key=lambda x: int(x.split('_')[-1]))
    seq_len = len(vel_cols)
    time_steps = np.arange(seq_len)
    
    # Sort to ensure colormap maps cleanly
    target_group = target_group.sort_values('obj_yaw_base')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    norm = plt.Normalize(target_group['obj_yaw_base'].min(), target_group['obj_yaw_base'].max())
    sm = plt.cm.ScalarMappable(cmap='twilight', norm=norm)
    
    for face_idx in [0, 1]:
        ax = axes[face_idx]
        sub_group = target_group[target_group['push_face_index'] == face_idx]
        
        for _, row in sub_group.iterrows():
            vel_x = row[vel_cols].values.astype(float)
            yaw = row['obj_yaw_base']
            ax.plot(time_steps, vel_x, color=sm.to_rgba(yaw), linewidth=2, alpha=0.8)
            
        ax.set_title(f"Push Face Index: {face_idx}")
        ax.set_xlabel("Time Step")
        if face_idx == 0:
            ax.set_ylabel("Extracted EE Velocity [m/s]")
        ax.grid(True, linestyle=':', alpha=0.6)
        
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Object Yaw Base (rad)')
    
    fig.suptitle(f"EE Velocity Sequence vs Object Yaw & Push Face\nFixed Properties: Mass = {gt_mass:.4f} kg, Mu = {gt_mu:.4f}")
    
    plt.show()


# =============================================================================
# OBJECT YAW HELPERS
# =============================================================================
def _yaw_deg(yaw_rad):
    """Radians -> degrees, rounded so float noise cannot split one yaw into several."""
    return np.round(np.degrees(np.asarray(yaw_rad, dtype=float)), 2)


def _circ_mean_deg(a):
    """Circular mean in degrees: +179 and -179 average to 180, not 0."""
    r = np.radians(np.asarray(a, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def _circ_spread_deg(a, center):
    """Largest angular distance from `center`, in degrees."""
    d = (np.asarray(a, dtype=float) - center + 180.0) % 360.0 - 180.0
    return float(np.abs(d).max())


def report_object_yaw(df):
    """Object yaw in degrees: the set of yaws, the push direction per
    (yaw x push side), and the collect_idx -> yaw mapping."""
    if 'obj_yaw_base' not in df.columns:
        print("\n  [WARNING] no 'obj_yaw_base' column; cannot report object yaw.")
        return
    d = df.assign(yaw_deg=_yaw_deg(df['obj_yaw_base']))
    uniq = np.sort(d['yaw_deg'].unique())
    print("\n  OBJECT YAW (degrees)")
    print(f"    distinct yaws : {uniq.size}")
    print(f"    range         : [{uniq.min():+.2f}, {uniq.max():+.2f}]")
    if uniq.size > 1:
        gaps = np.diff(uniq)
        print(f"    spacing       : min {gaps.min():.2f}, max {gaps.max():.2f}")
    print(f"    values        : {', '.join(f'{v:+.2f}' for v in uniq)}")

    if {'push_dir_b_x', 'push_dir_b_y'}.issubset(d.columns):
        d['push_deg'] = np.degrees(np.arctan2(d['push_dir_b_y'], d['push_dir_b_x']))
        d['push_rel_deg'] = (d['push_deg'] - d['yaw_deg'] + 180.0) % 360.0 - 180.0
    keys = ['yaw_deg'] + (['push_face_index'] if 'push_face_index' in d.columns else [])
    g = d.groupby(keys)
    table = pd.DataFrame({'rows': g.size()})
    if 'push_deg' in d.columns:
        table['push_dir_deg'] = g['push_deg'].apply(_circ_mean_deg)
        table['push_minus_yaw'] = g['push_rel_deg'].apply(_circ_mean_deg)
        table['max_dev_deg'] = [
            _circ_spread_deg(grp['push_rel_deg'], c)
            for (_, grp), c in zip(g, table['push_minus_yaw'])]
    print("\n  per (object yaw x push side):")
    print(table.round(2).to_string())

    if 'collect_idx' in d.columns:
        g = d.groupby('collect_idx')
        idx = pd.DataFrame({'yaw_deg': g['yaw_deg'].first(),
                            'n_yaws': g['yaw_deg'].nunique(),
                            'rows': g.size()})
        if 'push_face_index' in d.columns:
            idx['push_face'] = g['push_face_index'].first()
        print("\n  collect_idx -> object yaw:")
        print(idx.to_string())
        mixed = idx.index[idx['n_yaws'] > 1].tolist()
        if mixed:
            print(f"  [WARNING] collect_idx {mixed} hold more than one yaw. Each sweep "
                  f"cell should fix one orientation; check how run_collect_sweep "
                  f"maps COLLECT_IDX to yaw.")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return

    print(f"Loading data in chunks to prevent memory crash...")
    
    chunk_list = []
    
    try:
        # Read in chunks of 15,000 rows
        for chunk in pd.read_csv(CSV_PATH, chunksize=15000):
            
            # If the dataset is Multi-Angle, keep only the first 50 environments (out of 512)
            # This perfectly preserves the 20-angle sweeps while dropping ~90% of the massive file
            if MULTI_ANGLE and 'env_id' in chunk.columns:
                chunk = chunk[chunk['env_id'] < 50]
            else:
                # Fallback for Single-Angle dataset: just grab a random 10% slice
                chunk = chunk.sample(frac=0.1, random_state=42)
            
            # Downcast to 32-bit floats to halve the remaining memory footprint
            float_cols = chunk.select_dtypes(include=['float64']).columns
            chunk[float_cols] = chunk[float_cols].astype(np.float32)
            
            if 'gt_fric_force' in chunk.columns:
                chunk['gt_fric_force'] = chunk['gt_fric_force'].apply(clean_force_col)
                
            chunk_list.append(chunk)
            
        df = pd.concat(chunk_list, ignore_index=True)
        print(f"Successfully loaded a representative subset! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
    except MemoryError:
        print("\n[FATAL] Still ran out of memory! Your system RAM is too small even for chunking.")
        return

    # =======================================================
    # DATASET MACRO SUMMARY & DISTRIBUTIONS
    # =======================================================
    print("\n" + "="*50)
    print(f"DATASET MACRO SUMMARY ({FRAME_MODE.upper()} FRAME, "
          f"MULTI_ANGLE={MULTI_ANGLE}, USE_ARM_STATE={USE_ARM_STATE})")
    print("="*50)
    print(f"Representative Subset (Rows): {len(df)}")
    
    if 'gt_mass' in df.columns and 'gt_mu' in df.columns:
        print("\n[Ground Truth Statistics]")
        print(df[['gt_mass', 'gt_mu']].describe().round(4))
        print("="*50 + "\n")
        
        # Plot Global Distributions
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(data=df, x='gt_mass', kde=True, color='#9467bd', ax=axes[0])
        axes[0].set_title(f"Ground Truth Mass Distribution (Subset N={len(df)})")
        axes[0].set_xlabel("Mass [kg]")
        axes[0].set_ylabel("Count")
        
        sns.histplot(data=df, x='gt_mu', kde=True, color='#ff7f0e', ax=axes[1])
        axes[1].set_title(f"Ground Truth Friction Distribution (Subset N={len(df)})")
        axes[1].set_xlabel("Friction Coefficient (\u03bc)")
        axes[1].set_ylabel("Count")
        
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Warning] 'gt_mass' or 'gt_mu' columns not found for distribution plotting.")
    # =======================================================

    if MULTI_ANGLE:
        report_multi_angle_coverage(df)
        report_arm_columns(df)
        plot_kinematic_distributions(df)
        visualize_all_variants(df, standardize=False)
        report_variant_window(df)
        compare_variant_channels(df)
        report_inertial_window(df, log_scale=False)
        report_channel_informativeness(df)
    
    # We only need the DataFrame filtered by domain for this script
    _, _, seq_len, df_filtered, _ = create_dataloaders(
        df, batch_size=64, m_seen_min=M_SEEN_MIN, m_seen_max=M_SEEN_MAX, mu_seen_min=MU_SEEN_MIN, mu_seen_max=MU_SEEN_MAX
    )
    
    print(f"Inspecting training dataloader batches natively from DataFrame...")
    inspect_samples(df_filtered, num_samples=3)
    
    if MULTI_ANGLE:
        print("\nPlotting EE Velocity Variance vs Object Yaw...")
        inspect_velocity_vs_yaw(df_filtered)


if __name__ == "__main__":
    main()