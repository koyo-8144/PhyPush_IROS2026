"""
output_sim_csv_inputs.py  --  computer 1 (~/PhyPush), SIMULATION side.

Writes ONE wide CSV in which each row is one simulated push:

    ground truth (m, mu)  +  60 timesteps x 5 channels  +  push direction [deg]

The companion script output_real_csv_inputs.py, on computer 2, writes the SAME
schema from the real-robot captures. Copy either file across (scp / git / USB)
and the two can be concatenated and compared directly -- the column names, the
column order and the physical units agree by construction.

WHAT IS AND IS NOT DONE TO THE NUMBERS
    Channels are written in RAW PHYSICAL UNITS: no log transform, no
    standardization, no reduction to a scalar. That is deliberate. The log and
    the (x - mean) / std belong to the comparison step, which needs the
    training sidecar's mean/std to apply them, and keeping the raw values means
    one export serves both the physical comparison and the z-space one.

    Velocity is the same quantity the model is fed (configs.VEL_PREFIX,
    unstandardized), so sim velocity and real v_local_z are directly
    comparable.

WHAT IS FILTERED
    --domains      default 'seen' = domain == 'm_seen_mu_seen', which is the
                   only domain training ever saw. Use 'all' to export the OOD
                   domains too.
    --window-filter
                   default on: drops rows with start_t + 60 > 100, exactly as
                   dataset.create_dataloaders does, so the export covers the
                   same rows the model was trained on.
    Rows are NOT dropped for non-positive arm_lam_w / arm_meff_w. Training
    drops those before the log; here they are kept and counted, because
    whether they exist at all is part of what you are comparing. Use
    --drop-nonpositive to match training exactly.

MEMORY
    The sweep CSV is tens of GB with thousands of columns. This script reads
    the header first, keeps only the ~305 columns it needs, and streams the
    file in chunks, so it never materializes the whole table.

USAGE
    python output_sim_csv_inputs.py
    python output_sim_csv_inputs.py --out sim_channels.csv.gz
    python output_sim_csv_inputs.py --domains all --max-rows 20000
    python output_sim_csv_inputs.py --csv /path/to/other_collection.csv
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

# =============================================================================
# CANONICAL SCHEMA  --  KEEP THIS BLOCK BYTE-IDENTICAL IN BOTH SCRIPTS
#
# output_sim_csv_inputs.py   (computer 1, ~/PhyPush)
# output_real_csv_inputs.py  (computer 2, ~/panda-py/panda_phypush2)
#
# The two CSVs are meant to be concatenated and compared, so the column names,
# the column ORDER and the units must agree exactly. If you change anything
# here, change it in BOTH files and bump SCHEMA_VERSION -- the version is
# written into every row so a mismatched pair is caught at comparison time
# instead of producing a quietly wrong plot.
# =============================================================================
SCHEMA_VERSION = "phypush-channels-1"

SEQ_LEN = 60

# Channel order. This is the order the columns appear in, and it is the order
# configs.VARIANT_TABLE lists them in, so a downstream script can slice any
# variant out of either file by name.
CHANNELS = (
    "input_vel",        # end-effector velocity along the push direction [m/s]
    "arm_manip_w",      # manipulability w                               [-]
    "arm_dir_manip_w",  # directional manipulability w_dir               [-]
    "arm_lam_w",        # mean translational diag of Lambda              [kg]
    "arm_meff_w",       # effective mass along the push direction        [kg]
)

# Units, carried into the header comment of the output so the file is readable
# a year from now without this script.
CHANNEL_UNIT = {
    "input_vel":       "m/s",
    "arm_manip_w":     "-",
    "arm_dir_manip_w": "-",
    "arm_lam_w":       "kg",
    "arm_meff_w":      "kg",
}


# Per-step column names: "<channel>_<t>", t = 0..SEQ_LEN-1.
#
# NOTE this is NOT the sim CSV's own naming. The collector writes input_vel_0
# but arm_manip_w0 (no separator), and the real captures use long-format rows
# with completely different names. Both are mapped onto this one scheme, so the
# two outputs line up column for column.
def step_cols(channel, seq_len=SEQ_LEN):
    return [f"{channel}_{t}" for t in range(seq_len)]


# Identity and label columns, in output order, before the channel blocks.
ID_COLS = (
    "schema_version",   # SCHEMA_VERSION, so a mismatched pair is detectable
    "side",             # "sim" or "real"
    "domain",           # sim: the domain label; real: the condition folder
    "sample_id",        # unique within one side
    "m_gt",             # ground-truth mass [kg]
    "mu_gt",            # ground-truth friction coefficient [-]
    "push_angle_deg",   # push direction in the robot base frame [deg]
    "seq_len",
    "smoothing_window",
    "source",           # which file / run this row came from
)


def output_columns(seq_len=SEQ_LEN):
    cols = list(ID_COLS)
    for ch in CHANNELS:
        cols += step_cols(ch, seq_len)
    return cols
# =============================== end schema ==================================


# =============================================================================
# SIM-SPECIFIC COLUMN NAMING
#
# The collector's own names, which differ per channel:
#   velocity        input_vel_0  .. input_vel_59    (underscore before the index)
#   arm channels    arm_manip_w0 .. arm_manip_w59   (no separator)
# dataset.window_cols matches "prefix + pure digits", which is why velocity has
# to be handled with the prefix "input_vel_" and the others with "arm_*_w".
# =============================================================================
SIM_PREFIX = {
    "input_vel":       "input_vel_",
    "arm_manip_w":     "arm_manip_w",
    "arm_dir_manip_w": "arm_dir_manip_w",
    "arm_lam_w":       "arm_lam_w",
    "arm_meff_w":      "arm_meff_w",
}

# Ground truth and provenance columns read from the sweep CSV.
GT_COLS = ("gt_mass", "gt_mu")
# Push direction comes from the base-frame unit vector the collector recorded.
ANGLE_COLS = ("push_dir_b_x", "push_dir_b_y")
# Extra provenance: kept as output columns so a sim row can be traced back and
# so the angle convention can be re-derived without re-reading the sweep CSV.
EXTRA_COLS = ("obj_yaw_base", "push_face_index", "collect_idx", "seed", "env_id",
              "start_t", "domain")

DEFAULT_OUT = "sim_channels.csv"


def sim_step_cols(header, channel):
    """The collector's columns for one channel, ordered by numeric suffix.

    Strict digit match, same rule as dataset.window_cols: a future column that
    merely shares the prefix cannot silently join the sequence and shift every
    timestep by one.
    """
    pfx = SIM_PREFIX[channel]
    cols = [c for c in header if c.startswith(pfx) and c[len(pfx):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(pfx):]))


def resolve_csv(path_arg):
    """The sweep CSV to read: the argument, else configs' own default."""
    if path_arg:
        return path_arg
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import configs as C
        for name in ("MULTI_ANGLE_CSV_PATH", "CSV_PATH", "DATA_CSV_PATH"):
            p = getattr(C, name, None)
            if p:
                print(f"  [note] CSV from configs.{name}")
                return p
    except Exception as e:                                  # noqa: BLE001
        print(f"  [note] could not read the default from configs.py ({e}); "
              f"pass --csv")
    return None


def push_angle_deg(df, mode="push_dir"):
    """Push direction in degrees, in the robot base frame.

    mode="push_dir"  atan2(push_dir_b_y, push_dir_b_x), the direction the
                     end-effector actually travels. This is the quantity that
                     corresponds to the real side's PUSH_ANGLE_DEG.
    mode="obj_yaw"   degrees(obj_yaw_base). The OBJECT's yaw, not the push
                     direction -- useful only if that is how the real
                     metadata's angle was defined.

    Returns NaN where the source columns are absent, rather than inventing a
    value: a fabricated angle would silently break the angle-resolved
    comparison, which is the main thing this export is for.
    """
    if mode == "obj_yaw":
        if "obj_yaw_base" not in df.columns:
            return np.full(len(df), np.nan)
        return np.degrees(df["obj_yaw_base"].values.astype(np.float64))

    if not all(c in df.columns for c in ANGLE_COLS):
        return np.full(len(df), np.nan)
    x = df["push_dir_b_x"].values.astype(np.float64)
    y = df["push_dir_b_y"].values.astype(np.float64)
    ang = np.degrees(np.arctan2(y, x))
    # A zero-length direction vector has no angle; atan2(0,0) returns 0.0,
    # which would read as a perfectly axial push.
    ang[np.hypot(x, y) < 1e-9] = np.nan
    return ang


def build_usecols(header, args):
    """The columns to load, and the per-channel mapping onto the canonical names."""
    mapping, missing = {}, []
    for ch in CHANNELS:
        cols = sim_step_cols(header, ch)
        if len(cols) == 0:
            missing.append(ch)
            mapping[ch] = None
            continue
        if len(cols) < SEQ_LEN:
            raise SystemExit(
                f"channel '{ch}': the CSV has only {len(cols)} step columns "
                f"({cols[0]}..{cols[-1]}), fewer than the {SEQ_LEN} this schema "
                f"expects. Either the collection used a different window length "
                f"-- in which case change SEQ_LEN in BOTH scripts -- or the "
                f"columns are named differently than SIM_PREFIX assumes.")
        if len(cols) > SEQ_LEN:
            print(f"  [note] channel '{ch}': {len(cols)} step columns present, "
                  f"taking the first {SEQ_LEN} ({cols[0]}..{cols[SEQ_LEN - 1]})")
            cols = cols[:SEQ_LEN]
        mapping[ch] = cols

    if missing:
        print(f"  [WARNING] no columns for {missing} in this CSV; those blocks "
              f"will be written as NaN. An older collection predates them.")

    need = set()
    for cols in mapping.values():
        if cols:
            need.update(cols)
    for c in GT_COLS + ANGLE_COLS + EXTRA_COLS:
        if c in header:
            need.add(c)
    absent_gt = [c for c in GT_COLS if c not in header]
    if absent_gt:
        raise SystemExit(f"the CSV has no {absent_gt}; ground truth is required.")
    return [c for c in header if c in need], mapping


def rows_from_chunk(chunk, mapping, args, offset):
    """One output frame from one input chunk, already filtered.

    Assembled with a single pd.concat rather than ~310 individual column
    assignments: inserting that many columns one at a time into a 20k-row frame
    re-copies the block manager on every insert, which pandas warns about and
    which dominates the runtime on the real sweep CSV.
    """
    n = len(chunk)
    idx = np.arange(n)
    ids = pd.DataFrame({
        "schema_version": np.full(n, SCHEMA_VERSION, dtype=object),
        "side": np.full(n, "sim", dtype=object),
        "domain": (chunk["domain"].values if "domain" in chunk.columns
                   else np.full(n, "unknown", dtype=object)),
        "sample_id": [f"sim_{offset + i}" for i in range(n)],
        "m_gt": chunk["gt_mass"].values.astype(np.float64),
        "mu_gt": chunk["gt_mu"].values.astype(np.float64),
        "push_angle_deg": push_angle_deg(chunk, args.angle_from),
        "seq_len": np.full(n, SEQ_LEN),
        # Sim velocity is recorded as-is; the real side smooths v_local_z with a
        # rolling window, so the number is carried in both files to make an
        # accidental mismatch visible.
        "smoothing_window": np.full(n, 1),
        "source": np.full(n, os.path.basename(args.csv), dtype=object),
    }, index=idx)

    parts = [ids]
    for ch in CHANNELS:
        names = step_cols(ch)
        if mapping[ch] is None:
            block = np.full((n, SEQ_LEN), np.nan)
        else:
            block = chunk[mapping[ch]].values.astype(np.float64)
        parts.append(pd.DataFrame(block, columns=names, index=idx))

    # Provenance, after the channel blocks so the canonical columns keep their
    # fixed positions. The real side writes the same trailing-extras idea.
    prov = {f"sim_{c}": chunk[c].values
            for c in ("obj_yaw_base", "push_face_index", "collect_idx", "seed",
                      "env_id", "start_t")
            if c in chunk.columns}
    if prov:
        parts.append(pd.DataFrame(prov, index=idx))
    return pd.concat(parts, axis=1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="the collected sweep CSV (default: from configs.py)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output CSV; a .gz suffix is compressed")
    ap.add_argument("--domains", default="seen", choices=("seen", "all"),
                    help="'seen' keeps domain == m_seen_mu_seen, what training saw")
    ap.add_argument("--window-filter", dest="window_filter",
                    action="store_true", default=True,
                    help="drop rows with start_t + 60 > 100 (default: on)")
    ap.add_argument("--no-window-filter", dest="window_filter",
                    action="store_false")
    ap.add_argument("--drop-nonpositive", action="store_true",
                    help="also drop rows whose arm_lam_w / arm_meff_w window has "
                         "a non-positive or non-finite step, matching what "
                         "training drops before the log")
    ap.add_argument("--angle-from", default="push_dir",
                    choices=("push_dir", "obj_yaw"),
                    help="push_dir = atan2(push_dir_b_y, push_dir_b_x) [default]")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap the output, sampled uniformly at random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=20000)
    args = ap.parse_args()

    args.csv = resolve_csv(args.csv)
    if not args.csv:
        raise SystemExit("no CSV: pass --csv")
    if not os.path.exists(args.csv):
        raise SystemExit(f"not found: {args.csv}")

    print("=" * 74)
    print(" SIM CHANNEL EXPORT")
    print("=" * 74)
    print(f"  schema : {SCHEMA_VERSION}   seq_len {SEQ_LEN}   "
          f"{len(CHANNELS)} channels")
    print(f"  input  : {args.csv}")
    print(f"  size   : {os.path.getsize(args.csv) / 1e9:.2f} GB")

    header = pd.read_csv(args.csv, nrows=0).columns.tolist()
    usecols, mapping = build_usecols(header, args)
    print(f"  columns: loading {len(usecols)} of {len(header)}")
    for ch in CHANNELS:
        cols = mapping[ch]
        print(f"     {ch:<16} -> " + (f"{cols[0]}..{cols[-1]}" if cols else "MISSING"))

    kept, n_read, n_dom, n_win, n_nonpos = [], 0, 0, 0, 0
    reader = pd.read_csv(args.csv, usecols=usecols, chunksize=args.chunksize,
                         low_memory=False)
    for chunk in reader:
        n_read += len(chunk)

        if args.domains == "seen" and "domain" in chunk.columns:
            before = len(chunk)
            chunk = chunk[chunk["domain"] == "m_seen_mu_seen"]
            n_dom += before - len(chunk)

        if args.window_filter and "start_t" in chunk.columns:
            before = len(chunk)
            chunk = chunk[(chunk["start_t"] + SEQ_LEN) <= 100]
            n_win += before - len(chunk)

        if args.drop_nonpositive:
            before = len(chunk)
            ok = np.ones(len(chunk), dtype=bool)
            for ch in ("arm_lam_w", "arm_meff_w"):
                if mapping[ch]:
                    W = chunk[mapping[ch]].values.astype(np.float64)
                    ok &= np.isfinite(W).all(axis=1) & (W > 0.0).all(axis=1)
            chunk = chunk[ok]
            n_nonpos += before - len(chunk)

        if len(chunk) == 0:
            continue
        kept.append(rows_from_chunk(chunk.reset_index(drop=True), mapping, args,
                                    sum(len(k) for k in kept)))

    print(f"\n  rows read   : {n_read}")
    if n_dom:
        print(f"  dropped     : {n_dom} outside domain m_seen_mu_seen")
    if n_win:
        print(f"  dropped     : {n_win} with start_t + {SEQ_LEN} > 100")
    if n_nonpos:
        print(f"  dropped     : {n_nonpos} with a non-positive inertial step")
    if not kept:
        raise SystemExit("nothing left to write after filtering")

    out = pd.concat(kept, ignore_index=True)
    # sample_id was numbered per chunk; renumber so it is unique over the file.
    out["sample_id"] = [f"sim_{i}" for i in range(len(out))]

    if args.max_rows and len(out) > args.max_rows:
        # Random, not head: the sweep CSV is written in collection order, so the
        # first N rows are a few (mass, mu) pairs rather than a sample of them.
        out = out.sample(n=args.max_rows, random_state=args.seed)
        out = out.sort_index().reset_index(drop=True)
        out["sample_id"] = [f"sim_{i}" for i in range(len(out))]
        print(f"  subsampled  : {args.max_rows} rows (seed {args.seed})")

    cols = output_columns()
    extras = [c for c in out.columns if c not in cols]
    out = out[cols + extras]

    report(out)

    out.to_csv(args.out, index=False)
    print(f"\n  written: {args.out}  "
          f"({len(out)} rows x {len(out.columns)} cols, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB)")
    print("=" * 74)


def report(out):
    """Per-channel summary, and the angle coverage the comparison depends on."""
    print(f"\n  samples: {len(out)}")
    if "domain" in out.columns:
        for d, n in out["domain"].value_counts().items():
            print(f"     {d:<24} {n}")
    print(f"  (mass, mu) pairs: "
          f"{out.groupby(['m_gt', 'mu_gt']).ngroups}")

    ang = out["push_angle_deg"].dropna()
    print(f"\n  push_angle_deg: {len(ang)}/{len(out)} rows have one")
    if len(ang):
        print(f"     range [{ang.min():+.1f}, {ang.max():+.1f}], "
              f"{(ang < 0).sum()} negative / {(ang >= 0).sum()} positive, "
              f"{ang.nunique()} distinct")
    else:
        print("     [WARNING] none. The angle-resolved comparison cannot be done "
              "from this export; check push_dir_b_x / push_dir_b_y.")

    print(f"\n  {'channel':<18}{'unit':>6}{'finite':>10}{'mean':>12}{'std':>12}"
          f"{'min':>12}{'max':>12}")
    for ch in CHANNELS:
        v = out[step_cols(ch)].values.astype(np.float64).ravel()
        f = np.isfinite(v)
        if f.sum() == 0:
            print(f"  {ch:<18}{CHANNEL_UNIT[ch]:>6}{'0':>10}"
                  f"{'--':>12}{'--':>12}{'--':>12}{'--':>12}")
            continue
        print(f"  {ch:<18}{CHANNEL_UNIT[ch]:>6}{f.sum():>10}"
              f"{v[f].mean():>12.5g}{v[f].std():>12.5g}"
              f"{v[f].min():>12.5g}{v[f].max():>12.5g}")
        if ch in ("arm_lam_w", "arm_meff_w"):
            n_bad = int((v[f] <= 0).sum())
            if n_bad:
                print(f"     [note] {n_bad} non-positive sample(s); these cannot "
                      f"be log-transformed. The real side reports the same count, "
                      f"and a large difference is itself a finding.")

    print("\n  CONVENTION CHECK -- do this before plotting anything:")
    print("    The real side's push_angle_deg comes from the collector's")
    print("    PUSH_ANGLE_DEG metadata; this side computes it from the base-frame")
    print("    push direction. Compare the two ranges printed by the two scripts.")
    print("    If the spans differ by a constant, a sign, or 90/180 deg, the zero")
    print("    references differ -- fix it in ONE place before comparing, and do")
    print("    not reconcile it by eye on the plots.")


if __name__ == "__main__":
    main()