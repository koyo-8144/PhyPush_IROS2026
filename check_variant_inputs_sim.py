"""
What every variant's input sequence looks like, for one (mass, mu, push angle).

WHY
    Each variant feeds the estimator a different second channel, and the channels
    live on wildly different scales -- manipulability around 0.08, Lambda around
    6 kg, m_eff around 3 kg. Comparing a real capture with a simulated one, or one
    push angle with another, means looking at the same five sequences side by side
    rather than at summary statistics.

    This is the SIMULATION side: it reads configs.CSV_PATH, the sweep the estimators
    were trained on. check_variant_inputs_real.py does the same for the real robot,
    with the same panels and the same axis conventions, so the two figures can be
    put next to each other.

WHAT IT SHOWS
    Five panels over the 60-step window:
        v_local_z        end-effector velocity along the push axis
        manipulability   w = sqrt(det(J J^T))
        dir manip        w_dir(u) along the push direction
        osim_lam         mean translational diagonal of Lambda
        eff_mass         effective mass along the push direction
    One line per matching row, with the mean drawn over them.

USAGE
    python check_variant_inputs_sim.py
    python check_variant_inputs_sim.py --mass 0.6 --mu 0.3 --angles -90 0 90
    python check_variant_inputs_sim.py --max-rows 20 --out sim_inputs.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# =============================================================================
# WHAT TO LOOK AT  -- edit here, or pass the same values on the command line
# =============================================================================
TARGET_MASS = 0.297        # kg
TARGET_MU = 0.1799
TARGET_ANGLES = [90.0]      # degrees; [] means every trial of the pair


MASS_TOL = 0.05            # kg, matching tolerance
MU_TOL = 0.02
# ANGLE_TOL = 5.0            # degrees
ANGLE_TOL = 0.0            # degrees

MAX_ROWS_PER_ANGLE = 10    # lines drawn per angle, before the mean
OUT_PNG = "variant_inputs_sim.png"

# Channel prefix per panel, and how it is labelled. The window columns are
# <prefix>0 .. <prefix>59; velocity uses its own prefix.
CHANNELS = [
    ("input_vel_",      "EE velocity", "m/s",   False),
    ("arm_manip_w",     "Manipulability $w$", "", False),
    ("arm_dir_manip_w", "Directional $w_{dir}$", "", False),
    ("arm_lam_w",       r"OSIM $\Lambda$ (mean transl.)", "kg", True),
    ("arm_meff_w",      "Effective mass $m_{eff}$", "kg", True),
]


def window_cols(header, prefix):
    """Columns prefix+digits, ordered by that numeric suffix."""
    cols = [c for c in header if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


# =============================================================================
# THE PUSH ANGLE IN SIMULATION
#
# The sweep CSV does not record the push angle; it records obj_yaw_base and
# push_face_index, and the angle is DERIVED from them. Under
# OFFLINE_DATA_COLLECTION_TRANSLATION with a world-frame action, the task does:
#
#     half       = pi/2
#     yaw_folded = obj_yaw mod half                   # [0, 90)
#     lower      = yaw_folded <= pi/4
#     face 0 ->  target = yaw_folded + half  if lower else yaw_folded
#     face 1 ->  target = yaw_folded - half  if lower else yaw_folded - pi
#     actions[:, 0] = target / pi      ->   push_angle = a0 * pi = target
#
# and the pushset sits at obj + r*[cos(push_angle), sin(push_angle)], exactly as
# the real collector does with its own a0.
#
# WHAT THIS MEANS FOR COMPARING WITH THE ROBOT
#   * The push angle here is ABSOLUTE in the base frame, as it is on the robot, so
#     matching on it compares the same arm geometry. That is the quantity the arm
#     channels depend on, so this is the right key.
#   * But in simulation the angle TRACKS the object's yaw: the folding puts the
#     approach exactly along a face normal every time (the attack angle is 0 by
#     construction). On the robot a0 is chosen independently of how the cube is
#     lying, so the attack angle is whatever the placement happens to give.
#     Two pushes at the same absolute angle can therefore strike the cube
#     differently -- same arm pose, different contact.
#   * Face 0 covers roughly +45..+135 deg and face 1 -135..-45 deg, which is the
#     same lateral window the real MANUAL_A0 set (+-54..126 deg) spans.
# =============================================================================
def sim_push_angle_deg(obj_yaw_rad, face_index):
    """Push angle in degrees, as the task computes it from the object's yaw."""
    yaw = np.asarray(obj_yaw_rad, dtype=float)
    face = np.asarray(face_index, dtype=float)
    half = np.pi / 2.0
    folded = np.remainder(yaw, half)
    lower = folded <= (np.pi / 4.0)
    face0 = np.where(lower, folded + half, folded)
    face1 = np.where(lower, folded - half, folded - np.pi)
    return np.degrees(np.where(face == 0, face0, face1))


def attack_angle_deg(obj_yaw_rad, push_angle_deg):
    """Angle between the push direction and the nearest face normal, in degrees.

    0 means the push is square onto a face. In simulation this is 0 by
    construction; on the robot it is whatever the cube's placement gives, which is
    the part of the geometry the push angle alone does not capture.
    """
    rel = np.radians(push_angle_deg) - np.asarray(obj_yaw_rad, dtype=float)
    return np.degrees(np.remainder(rel + np.pi / 4.0, np.pi / 2.0) - np.pi / 4.0)


def angle_column(header):
    """How to get a push angle in degrees out of this CSV.

    Returns (kind, columns). 'derived' means it is computed from obj_yaw_base and
    push_face_index -- see sim_push_angle_deg -- which is how the multi-angle sweep
    records it.
    """
    if "push_angle_deg" in header:
        return "direct", ["push_angle_deg"]
    if "obj_yaw_base" in header and "push_face_index" in header:
        return "derived", ["obj_yaw_base", "push_face_index"]
    if "obj_yaw_base" in header:
        print("  [WARNING] obj_yaw_base is present but push_face_index is not. The "
              "push angle cannot be derived without the face, so the OBJECT's yaw "
              "is used instead -- it is NOT the push angle, and will not match the "
              "robot's PUSH_ANGLE_DEG.")
        return "objyaw", ["obj_yaw_base"]
    if "collect_idx" in header:
        print("  [note] no angle column; falling back to collect_idx, which indexes "
              "the angle rather than measuring it. --angles then selects indices.")
        return "index", ["collect_idx"]
    return None, []


def chunk_angle_deg(chunk, kind, cols):
    """Push angle in degrees for a chunk, or None when the CSV cannot give one."""
    if kind == "direct":
        return chunk[cols[0]].values.astype(float)
    if kind == "derived":
        return sim_push_angle_deg(chunk["obj_yaw_base"].values,
                                  chunk["push_face_index"].values)
    if kind == "objyaw":
        return np.degrees(chunk[cols[0]].values.astype(float))
    if kind == "index":
        return chunk[cols[0]].values.astype(float)
    return None


def load_rows(csv_path, args):
    """Rows matching (mass, mu) and, where an angle column exists, the angles.

    Read in chunks with only the columns needed: the sweep CSV is tens of
    gigabytes, and the five window families are ~300 of its ~5000 columns.
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    ang_kind, ang_cols = angle_column(header)
    if ang_kind == "derived":
        print("  angle  : derived from obj_yaw_base and push_face_index "
              "(see sim_push_angle_deg)")

    cols = {}
    for prefix, label, _unit, _log in CHANNELS:
        wc = window_cols(header, prefix)
        if not wc:
            print(f"  [note] no '{prefix}*' columns; the {label} panel will be empty.")
        cols[prefix] = wc

    need = ["gt_mass", "gt_mu"] + [c for wc in cols.values() for c in wc]
    for extra in list(ang_cols) + ["seed", "env_id", "start_t"]:
        if extra and extra in header and extra not in need:
            need.append(extra)

    kept = []
    n_seen = 0
    for chunk in pd.read_csv(csv_path, usecols=need, chunksize=200_000):
        n_seen += len(chunk)
        m = (chunk["gt_mass"] - args.mass).abs() <= args.mass_tol
        m &= (chunk["gt_mu"] - args.mu).abs() <= args.mu_tol
        deg = chunk_angle_deg(chunk, ang_kind, ang_cols)
        if deg is not None and args.angles:
            keep = np.zeros(len(chunk), dtype=bool)
            for a in args.angles:
                # Angles wrap: -180 and +180 are the same approach.
                d = np.abs((deg - a + 180.0) % 360.0 - 180.0)
                keep |= d <= args.angle_tol
            m &= keep
        if m.any():
            sub = chunk[m].copy()
            sub["_angle_deg"] = (np.nan if deg is None else deg[m.values])
            sub["_attack_deg"] = (
                attack_angle_deg(sub["obj_yaw_base"].values, sub["_angle_deg"].values)
                if "obj_yaw_base" in sub.columns and ang_kind in ("direct", "derived")
                else np.nan)
            kept.append(sub)

    if not kept:
        sys.exit(f"\n  no rows within mass {args.mass}+-{args.mass_tol} kg, "
                 f"mu {args.mu}+-{args.mu_tol}"
                 + (f", angle {args.angles} +-{args.angle_tol} deg" if args.angles
                    else "")
                 + f" (searched {n_seen} rows). Widen the tolerances, or check the "
                 f"sweep actually contains that pair.")
    df = pd.concat(kept, ignore_index=True)
    print(f"  matched {len(df)} row(s) of {n_seen}")
    if "_attack_deg" in df.columns and df["_attack_deg"].notna().any():
        atk = df["_attack_deg"].abs()
        print(f"  attack angle (push vs the nearest face normal): "
              f"mean {atk.mean():.2f} deg, max {atk.max():.2f} deg")
        if atk.max() < 1.0:
            print("    ~0 by construction here: the sweep folds the angle onto a "
                  "face normal. The robot's angle is set independently of the "
                  "cube's yaw, so its attack angle is whatever the placement gives "
                  "-- same arm pose, different contact.")
    return df, cols, ang_kind


def plot(df, cols, out_path, title, max_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = sorted(df["_angle_deg"].dropna().unique().tolist())
    groups = [(a, df[np.abs(df["_angle_deg"] - a) < 1e-6]) for a in angles] or \
             [(np.nan, df)]

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 2.6 * len(CHANNELS)),
                             sharex=True)
    cmap = plt.get_cmap("tab10")

    for ax, (prefix, label, unit, log) in zip(axes, CHANNELS):
        wc = cols.get(prefix, [])
        if not wc:
            ax.text(0.5, 0.5, f"no '{prefix}*' columns", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_ylabel(label)
            continue
        for gi, (a, g) in enumerate(groups):
            W = g[wc].values.astype(float)
            # A few individual pushes, faint, so the spread is visible; the mean on
            # top, because that is what the standardization statistics describe.
            for row in W[:max_rows]:
                ax.plot(row, color=cmap(gi % 10), alpha=0.25, linewidth=0.8)
            ax.plot(W.mean(axis=0), color=cmap(gi % 10), linewidth=2.2,
                    label=(f"{a:+.0f}$\\degree$ (n={len(g)})" if np.isfinite(a)
                           else f"all (n={len(g)})"))
        ax.set_ylabel(f"{label}" + (f"\n[{unit}]" if unit else ""))
        if log:
            # Lambda and m_eff are log-transformed before standardization, so a log
            # axis is the scale the estimator actually sees.
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=min(4, len(groups)))
    axes[-1].set_xlabel("Step in the 60-step inference window")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  figure -> {out_path}")


def report(df, cols):
    """Per channel and per angle: the numbers behind the panels."""
    print(f"\n  {'channel':<18}{'angle':>8}{'n':>6}{'mean':>12}{'std':>12}"
          f"{'min':>12}{'max':>12}")
    for prefix, label, _unit, log in CHANNELS:
        wc = cols.get(prefix, [])
        if not wc:
            continue
        for a in sorted(df["_angle_deg"].dropna().unique().tolist()) or [np.nan]:
            g = df if not np.isfinite(a) else df[np.abs(df["_angle_deg"] - a) < 1e-6]
            W = g[wc].values.astype(float)
            print(f"  {prefix:<18}{a:>8.0f}{len(g):>6}{W.mean():>12.5f}"
                  f"{W.std():>12.5f}{W.min():>12.5f}{W.max():>12.5f}")
        if log:
            W = df[wc].values.astype(float)
            W = W[np.isfinite(W) & (W > 0)]
            if W.size:
                lv = np.log(W)
                print(f"  {'  (log space)':<18}{'':>8}{'':>6}{lv.mean():>12.5f}"
                      f"{lv.std():>12.5f}"
                      f"   relative variation {100 * (np.exp(lv.std()) - 1):.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="defaults to configs.CSV_PATH")
    ap.add_argument("--mass", type=float, default=TARGET_MASS)
    ap.add_argument("--mu", type=float, default=TARGET_MU)
    ap.add_argument("--angles", type=float, nargs="*", default=TARGET_ANGLES)
    ap.add_argument("--mass-tol", type=float, default=MASS_TOL)
    ap.add_argument("--mu-tol", type=float, default=MU_TOL)
    ap.add_argument("--angle-tol", type=float, default=ANGLE_TOL)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS_PER_ANGLE)
    ap.add_argument("--out", default=OUT_PNG)
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        try:
            from configs import CSV_PATH
            csv_path = CSV_PATH
        except Exception as exc:
            sys.exit(f"could not import CSV_PATH from configs ({exc}); pass --csv")
    if not os.path.exists(csv_path):
        sys.exit(f"not found: {csv_path}")

    print("=" * 74)
    print(" VARIANT INPUT SEQUENCES -- SIMULATION")
    print("=" * 74)
    print(f"  csv    : {csv_path}")
    print(f"  target : mass {args.mass} +-{args.mass_tol} kg, "
          f"mu {args.mu} +-{args.mu_tol}")
    print(f"  angles : {args.angles or 'all'}"
          + (f" +-{args.angle_tol} deg" if args.angles else ""))

    df, cols, ang_kind = load_rows(csv_path, args)
    if ang_kind is None and args.angles:
        print("  [WARNING] the CSV has no angle column, so --angles was ignored and "
              "every matching row is drawn together.")
    report(df, cols)
    plot(df, cols, args.out,
         f"Simulation: mass {args.mass} kg, $\\mu$ {args.mu}", args.max_rows)
    print("=" * 74)


if __name__ == "__main__":
    main()