import re
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from utils import clean_force_col
from configs import (M_UNSEEN_MAX, MU_UNSEEN_MAX, FRAME_MODE, MULTI_ANGLE,
                     M_SEEN_MIN, M_SEEN_MAX, MU_SEEN_MIN, MU_SEEN_MAX,
                     USE_ARM_STATE, CSV_PATH,
                     INPUT_VARIANT, COND_DIM, INPUT_DIM, VARIANT_WINDOW_PREFIX,
                     VARIANT_LOG_TRANSFORM,
                     MAX_PUSH_LATERAL_OFFSET, MAX_PUSH_COM_OFFSET)

# =============================================================================
# COLUMN GROUPS WRITTEN BY THE MULTI-ANGLE SWEEP
# (_save_offline_data_csv in off_policy_algorithm.py)
# =============================================================================

# Provenance: which sweep cell and which physical sample this row came from.
MULTI_ANGLE_COLS = ['obj_yaw_base', 'push_face_index', 'collect_idx', 'seed', 'env_id']

# Static per-push arm summary, from _arm_static_summary.
# NOTE the last two are NEW relative to the v3 dataset.
ARM_STATIC_COLS = [
    'worst_manipulability',        # min sqrt(det(J J^T)) over the WHOLE push
    'ee_position_error',
    'push_start_reached_flag',
    'push_dir_b_x',
    'push_dir_b_y',
    'worst_dir_manipulability',    # min directional manipulability, whole push
    'push_com_offset',             # POST-push sideways deviation of the object
]

# Arm configuration sampled at the impact index, from _arm_state_a_his.
# 'arm_dir_manip_at_impact' is NEW relative to the v3 dataset.
ARM_STATE_COLS = (
    [f'arm_q{j}' for j in range(7)]                   # joint positions
    + [f'arm_qd{j}' for j in range(7)]                # joint velocities
    + [f'arm_ee_pos_{ax}' for ax in ('x', 'y', 'z')]  # EE position, base frame
    + [f'arm_ee_rot6d_{j}' for j in range(6)]         # EE rotation, 6D
    + ['arm_manip_at_impact']                         # manipulability at impact
    + ['arm_dir_manip_at_impact']                     # directional, at impact
)

ALL_ARM_COLS = ARM_STATIC_COLS + ARM_STATE_COLS

# Columns usable as descriptive features, for inspection/reporting only.
#
# push_start_reached_flag is EXCLUDED: _save_offline_data_csv skips any row where
# push_start_reached or push_end_reached is 0, so every surviving row carries
# 1.0. It is a constant -- no signal, and a divide-by-~0 under standardization.
#
# NOTE: these are NOT what the model consumes. Model inputs are selected by
# INPUT_VARIANT and drawn from the per-step window columns below. This list
# exists so inspect_dataset.py can report on what the CSV contains.
ARM_FEATURE_COLS = [c for c in ARM_STATIC_COLS
                    if c != 'push_start_reached_flag'] + ARM_STATE_COLS

# Per-step window channels, aligned one-to-one with input_vel_0..59.
# These are the columns the input variants actually draw from.
#   arm_manip_w      manipulability w                       (vel_manip_*)
#   arm_dir_manip_w  directional manipulability w_dir       (vel_dirmanip_*)
#   arm_lam_w        mean translational diag of Lambda [kg] (vel_osim_seq)
#   arm_meff_w       effective mass along push dir [kg]     (vel_eff_seq)
WINDOW_PREFIXES = ('arm_manip_w', 'arm_dir_manip_w', 'arm_lam_w', 'arm_meff_w')


def window_cols(df_or_header, prefix):
    """Columns `prefix` + pure digits, ordered by that numeric suffix.

    Strict digit match so a future column sharing the prefix cannot silently
    join the sequence and shift every timestep.
    """
    header = (df_or_header.columns.tolist()
              if hasattr(df_or_header, "columns") else list(df_or_header))
    cols = [c for c in header
            if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


def variant_window(df, prefix=None, log_transform=None):
    """The per-step window channel for a variant, AFTER the variant transform.

    Shared by create_dataloaders and evaluate.py so both apply the identical
    transform before standardization.

    Returns (W, valid, cols):
      W      (N, T) float32. log(x) where log_transform, raw otherwise.
             Invalid rows are filled with 0.0 so W stays finite; drop them
             with `valid`.
      valid  (N,) bool. For log-transformed channels, rows whose every step is
             finite and > 0. The collector writes a failed m_eff as 0.0, which
             is not a real mass. Always all-True for untransformed channels.
      cols   the ordered column names.
    """
    prefix = VARIANT_WINDOW_PREFIX if prefix is None else prefix
    log_transform = VARIANT_LOG_TRANSFORM if log_transform is None else log_transform
    cols = window_cols(df, prefix)
    if not cols:
        return None, None, cols
    W = df[cols].values.astype(np.float64)
    if log_transform:
        valid = np.isfinite(W).all(axis=1) & (W > 0.0).all(axis=1)
        W = np.where(valid[:, None], np.log(np.clip(W, 1e-12, None)), 0.0)
    else:
        valid = np.ones(len(W), dtype=bool)
    return W.astype(np.float32), valid, cols


def _report_column_availability(df):
    """Warn early about missing sweep columns rather than failing deep in a loop."""
    missing_meta = [c for c in MULTI_ANGLE_COLS if c not in df.columns]
    missing_static = [c for c in ARM_STATIC_COLS if c not in df.columns]
    missing_state = [c for c in ARM_STATE_COLS if c not in df.columns]

    if missing_meta:
        print(f"[MULTI_ANGLE][WARNING] Missing provenance columns {missing_meta}.")
        print( "                       Group split will fall back to (gt_mass, gt_mu).")
    if missing_static:
        print(f"[MULTI_ANGLE][WARNING] Missing arm static columns {missing_static}.")
    if missing_state:
        print(f"[MULTI_ANGLE][WARNING] Missing {len(missing_state)} arm state "
              f"columns, e.g. {missing_state[:5]}.")

    for pfx in WINDOW_PREFIXES:
        cols = window_cols(df, pfx)
        if cols:
            print(f"[MULTI_ANGLE] window channel '{pfx}*': {len(cols)} steps "
                  f"({cols[0]} .. {cols[-1]})")
        else:
            print(f"[MULTI_ANGLE][WARNING] No '{pfx}*' columns in the CSV.")

    return not missing_static


def _build_property_groups(df_filtered):
    """Group key identifying one physical (mass, mu) pair.

    In the multi-angle dataset the same pair is observed ~40 times (10 object
    yaws x 2 push sides x seeds). Properties are determined by (seed, env_id),
    so that pair is the natural group.
    """
    if 'seed' in df_filtered.columns and 'env_id' in df_filtered.columns:
        return (df_filtered['seed'].astype(int) * 100000
                + df_filtered['env_id'].astype(int)).values, "seed+env_id"

    return (df_filtered['gt_mass'].round(6).astype(str) + "_"
            + df_filtered['gt_mu'].round(6).astype(str)).values, "gt_mass+gt_mu"


# =============================================================================
# MEMORY-SAFE LOADING
#
# The sweep CSV is ~15 GB with thousands of columns, most of them per-step
# pinn_<family>_t<t>_ax<a> arrays over all 6 axes. Training and evaluation read
# only 4 families on 2 axes (ax3 / ax5, depending on FRAME_MODE). Loading every
# column as float32 needs more RAM than the machine has; the concat then doubles
# the peak and the OS swaps until it freezes.
#
# load_dataset_csv therefore
#   1. reads only the columns something downstream uses (usecols),
#   2. optionally keeps only the seen-domain rows (training uses nothing else),
#   3. estimates the final size from the first chunk and stops BEFORE the
#      machine runs out of memory.
# =============================================================================
PINN_FAMILIES_USED = ('robot_wrench', 'RHS_acc', 'LHS_wrench', 'table_wrench')
PINN_AXES_USED = (3, 5)
_PER_STEP_AXIS_RE = re.compile(r'^(.*)_t(\d+)_ax(\d+)$')


def select_columns(header):
    """Columns to load. Drops every per-step *_t<t>_ax<a> column except the pinn
    families and axes the physics losses read; keeps everything else."""
    keep, dropped = [], {}
    used = {f"pinn_{f}" for f in PINN_FAMILIES_USED}
    for c in header:
        m = _PER_STEP_AXIS_RE.match(c)
        if m is None or (m.group(1) in used and int(m.group(3)) in PINN_AXES_USED):
            keep.append(c)
        else:
            dropped[m.group(1)] = dropped.get(m.group(1), 0) + 1
    return keep, dropped


def _mem_available_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def load_dataset_csv(csv_path=None, chunksize=20000, verbose=True,
                     seen_only=False, max_mem_fraction=0.6):
    """Memory-safe CSV load.

    seen_only=True keeps only rows inside the seen (mass, mu) box, which is all
    create_dataloaders trains on. Leave it False for evaluate.py, which needs the
    OOD domains too.

    Raises MemoryError, before the machine starts swapping, if the projected
    size exceeds max_mem_fraction of the currently available RAM.
    """
    path = csv_path if csv_path is not None else CSV_PATH
    file_bytes = os.path.getsize(path)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols, dropped = select_columns(header)
    if verbose:
        print(f"[load_dataset_csv] {path}  ({file_bytes / 1e9:.1f} GB), "
              f"chunks of {chunksize}, float32")
        print(f"  columns: loading {len(usecols)} of {len(header)}")
        for fam, n in sorted(dropped.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    skipped {fam}_t*_ax*  ({n} cols)")
        if seen_only:
            print(f"  rows: seen domain only  (m in [{M_SEEN_MIN}, {M_SEEN_MAX}], "
                  f"mu in [{MU_SEEN_MIN}, {MU_SEEN_MAX}])")

    avail = _mem_available_gb()
    chunks, total_in, total_kept = [], 0, 0
    kept_bytes, checked = 0, False
    reader = pd.read_csv(path, chunksize=chunksize, usecols=usecols,
                         dtype={c: np.float32 for c in usecols
                                if _PER_STEP_AXIS_RE.match(c)
                                or c.startswith(('input_', 'arm_'))},
                         low_memory=True)
    with reader:
        for chunk in reader:
            total_in += len(chunk)
            if seen_only:
                m, mu = chunk['gt_mass'], chunk['gt_mu']
                chunk = chunk[(m >= M_SEEN_MIN) & (m <= M_SEEN_MAX)
                              & (mu >= MU_SEEN_MIN) & (mu <= MU_SEEN_MAX)]
            float_cols = chunk.select_dtypes(include=['float64']).columns
            if len(float_cols):
                chunk = chunk.astype({c: np.float32 for c in float_cols})
            chunks.append(chunk)
            total_kept += len(chunk)
            kept_bytes += int(chunk.memory_usage(deep=True).sum())

            if not checked and total_in > 0:
                # Project the final size from the first chunk: bytes kept per
                # input row x estimated input rows (file bytes / bytes per row).
                checked = True
                est_rows = file_bytes / max(file_bytes_per_row(path, chunksize), 1)
                projected_gb = kept_bytes / total_in * est_rows / 1e9
                if verbose:
                    avail_str = f"{avail:.1f} GB" if avail is not None else "unknown"
                    print(f"  projected: ~{est_rows:,.0f} rows in file, "
                          f"~{projected_gb:.2f} GB in memory (x2 peak during "
                          f"concat), available RAM {avail_str}")
                if avail is not None and 2 * projected_gb > max_mem_fraction * avail:
                    raise MemoryError(
                        f"Loading would need ~{2 * projected_gb:.1f} GB at peak but only "
                        f"{avail:.1f} GB is available. Use seen_only=True, a smaller "
                        f"CSV, or drop more columns in select_columns().")
            if verbose:
                print(f"  ... {total_in} rows read, {total_kept} kept", end='\r')

    df = pd.concat(chunks, ignore_index=True, copy=False)
    del chunks

    if 'gt_fric_force' in df.columns:
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col).astype(np.float32)

    if verbose:
        mem_gb = df.memory_usage(deep=True).sum() / 1e9
        print(f"\n[load_dataset_csv] loaded {len(df)} rows x {df.shape[1]} cols, "
              f"~{mem_gb:.2f} GB resident")
    return df


def file_bytes_per_row(path, n=200):
    """Average bytes per data line, from the first n lines after the header."""
    with open(path, 'rb') as f:
        f.readline()
        sizes = [len(f.readline()) for _ in range(n)]
    sizes = [x for x in sizes if x > 0]
    return sum(sizes) / len(sizes) if sizes else 1


def gather_pinn_windows(df, seq_len):
    """Vectorized replacement for the per-row pinn_* loop.

    Returns dict of (N, seq_len) float32 arrays: robot_fz, rhs_acc, lhs_net_f,
    table_fz, robot_fx, each read from columns t = start_t .. start_t+seq_len-1,
    with the same FRAME_MODE axis mapping as before.
    """
    if FRAME_MODE == "world":
        spec = {'robot_fz': ('robot_wrench', 5), 'rhs_acc': ('RHS_acc', 3),
                'lhs_net_f': ('LHS_wrench', 3), 'table_fz': ('table_wrench', 5),
                'robot_fx': ('robot_wrench', 3)}
    else:
        spec = {'robot_fz': ('robot_wrench', 3), 'rhs_acc': ('RHS_acc', 5),
                'lhs_net_f': ('LHS_wrench', 5), 'table_fz': ('table_wrench', 3),
                'robot_fx': ('robot_wrench', 5)}

    st = df['start_t'].values.astype(np.int64)
    idx = st[:, None] + np.arange(seq_len)[None, :]                 # (N, T)
    out = {}
    for key, (fam, ax) in spec.items():
        cols = [f"pinn_{fam}_t{t}_ax{ax}" for t in range(100)]
        full = df[cols].values.astype(np.float32, copy=False)       # (N, 100)
        out[key] = np.take_along_axis(full, idx, axis=1)
    return out


# =============================================================================
# VARIANT STATS PERSISTENCE
#
# The conditioning token and the extra input channel are only correct at eval
# time if standardized with the EXACT mean/std used in training. Recomputing at
# eval risks a mismatch (different filtering, split, or float path), which
# silently mis-scales the input. Training writes the stats to a sidecar keyed to
# (CSV, variant) and eval loads them verbatim.
# =============================================================================
def arm_stats_path(csv_path=None):
    base = csv_path if csv_path is not None else CSV_PATH
    return f"{base}.arm_stats.{INPUT_VARIANT}.npz"


def save_arm_stats(feature_cols, mean, std, csv_path=None, seq_len=None):
    path = arm_stats_path(csv_path)
    np.savez(path,
             feature_cols=np.array(feature_cols, dtype=object),
             mean=np.asarray(mean, dtype=np.float32),
             std=np.asarray(std, dtype=np.float32),
             mode=str(INPUT_VARIANT),
             variant=str(INPUT_VARIANT),
             window_prefix=str(VARIANT_WINDOW_PREFIX),
             log_transform=bool(VARIANT_LOG_TRANSFORM),
             seq_len=int(seq_len) if seq_len is not None else -1)
    print(f"[VARIANT] saved standardization stats -> {path}")


def load_arm_stats(csv_path=None):
    """Return (feature_cols, mean, std) written during training."""
    path = arm_stats_path(csv_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Variant stats not found at {path}. Train once with "
            f"INPUT_VARIANT='{INPUT_VARIANT}' so create_dataloaders writes them."
        )
    d = np.load(path, allow_pickle=True)
    cols = list(d['feature_cols'])
    saved = str(d['variant']) if 'variant' in d else (
        str(d['mode']) if 'mode' in d else None)
    if saved is not None and saved != INPUT_VARIANT:
        raise ValueError(
            f"Stats at {path} were saved for variant '{saved}', but "
            f"configs.INPUT_VARIANT is now '{INPUT_VARIANT}'. Retrain or switch back."
        )
    # Sidecars written before the log transform existed carry no flag; they were
    # all linear.
    saved_log = bool(d['log_transform']) if 'log_transform' in d else False
    if saved_log != bool(VARIANT_LOG_TRANSFORM):
        raise ValueError(
            f"Stats at {path} were computed with log_transform={saved_log}, but "
            f"configs says {bool(VARIANT_LOG_TRANSFORM)} for '{INPUT_VARIANT}'. "
            f"The mean/std would be applied on the wrong scale.")
    return cols, d['mean'], d['std']


# Conditioning width, for `cond_dim` when constructing the model.
ARM_DIM = COND_DIM


def _apply_push_quality_filter(df):
    """Drop pushes that were not clean straight-line contacts.

    Lateral offset torques the object; a rotating push violates the
    pure-translation assumption the PINN residuals rest on. push_com_offset is
    filtered only loosely, to catch physics blowups -- it is a post-push outcome,
    not a setup quality, and a large value can simply mean a heavy object slid.
    """
    n0 = len(df)
    notes = []

    if MAX_PUSH_LATERAL_OFFSET is not None and 'push_start_lateral_offset' in df.columns:
        df = df[df['push_start_lateral_offset'] < MAX_PUSH_LATERAL_OFFSET].copy()
        notes.append(f"lateral<{MAX_PUSH_LATERAL_OFFSET}")
    elif MAX_PUSH_LATERAL_OFFSET is not None:
        print("[FILTER][WARNING] MAX_PUSH_LATERAL_OFFSET set but column "
              "'push_start_lateral_offset' is absent; skipping that filter.")

    if MAX_PUSH_COM_OFFSET is not None and 'push_com_offset' in df.columns:
        df = df[df['push_com_offset'] < MAX_PUSH_COM_OFFSET].copy()
        notes.append(f"com<{MAX_PUSH_COM_OFFSET}")

    if notes:
        print(f"[FILTER] push quality ({', '.join(notes)}): "
              f"{len(df)}/{n0} rows kept ({100.0 * len(df) / max(n0, 1):.2f}%)")
    return df


def create_dataloaders(df, batch_size=64, m_seen_min=0.2, m_seen_max=2.0,
                       mu_seen_min=0.15, mu_seen_max=0.5):
    m_unseen_max, m_unseen_min = M_UNSEEN_MAX, m_seen_max
    mu_unseen_max, mu_unseen_min = MU_UNSEEN_MAX, mu_seen_max

    conditions = [
        (df['gt_mass'] >= m_seen_min)   & (df['gt_mass'] <= m_seen_max)   & (df['gt_mu'] >= mu_seen_min)   & (df['gt_mu'] <= mu_seen_max),
        (df['gt_mass'] > m_seen_max)    & (df['gt_mass'] <= m_unseen_max) & (df['gt_mu'] >= mu_seen_min)   & (df['gt_mu'] <= mu_seen_max),
        (df['gt_mass'] >= m_unseen_min) & (df['gt_mass'] < m_seen_min)    & (df['gt_mu'] >= mu_seen_min)   & (df['gt_mu'] <= mu_seen_max),
        (df['gt_mass'] >= m_seen_min)   & (df['gt_mass'] <= m_seen_max)   & (df['gt_mu'] > mu_seen_max)    & (df['gt_mu'] <= mu_unseen_max),
        (df['gt_mass'] >= m_seen_min)   & (df['gt_mass'] <= m_seen_max)   & (df['gt_mu'] >= mu_unseen_min) & (df['gt_mu'] < mu_seen_min),
        (df['gt_mass'] > m_seen_max)    & (df['gt_mass'] <= m_unseen_max) & (df['gt_mu'] > mu_seen_max)    & (df['gt_mu'] <= mu_unseen_max),
        (df['gt_mass'] >= m_unseen_min) & (df['gt_mass'] < m_seen_min)    & (df['gt_mu'] >= mu_unseen_min) & (df['gt_mu'] < mu_seen_min),
        (df['gt_mass'] > m_seen_max)    & (df['gt_mass'] <= m_unseen_max) & (df['gt_mu'] >= mu_unseen_min) & (df['gt_mu'] < mu_seen_min),
        (df['gt_mass'] >= m_unseen_min) & (df['gt_mass'] < m_seen_min)    & (df['gt_mu'] > mu_seen_max)    & (df['gt_mu'] <= mu_unseen_max)
    ]
    choices = [
        'm_seen_mu_seen', 'm_over', 'm_under', 'mu_over', 'mu_under',
        'm_over_mu_over', 'm_under_mu_under', 'm_over_mu_under', 'm_under_mu_over'
    ]

    df['domain'] = np.select(conditions, choices, default='other')

    # Push-quality filter on the FULL frame, before anything splits off it, so the
    # OOD domains evaluate.py selects are filtered the same way the seen domain is.
    have_arm = False
    if MULTI_ANGLE:
        have_arm = _report_column_availability(df)
        df = _apply_push_quality_filter(df)

    df_filtered = df[df['domain'] == 'm_seen_mu_seen'].copy()

    acc_cols = sorted([c for c in df_filtered.columns if "input_acc_" in c], key=lambda x: int(x.split('_')[-1]))
    vel_cols = sorted([c for c in df_filtered.columns if "input_vel_" in c], key=lambda x: int(x.split('_')[-1]))

    num_axes = 1
    seq_len = len(acc_cols) // num_axes

    # The window must fit inside the 100-step recording. This applies ONLY to
    # df_filtered, which the physics loop below iterates -- the returned `df` keeps
    # every row because evaluate.py re-applies this filter itself when selecting
    # each domain.
    valid_mask = (df_filtered['start_t'] + seq_len) <= 100
    df_filtered = df_filtered[valid_mask].copy()

    # Guard: the physics loop indexes pinn_*_t{start_t + seq_len - 1}_*, so a row
    # that survives with start_t + seq_len > 100 raises a KeyError deep in the loop
    # rather than here.
    _max_t = int((df_filtered['start_t'] + seq_len).max()) if len(df_filtered) else 0
    assert _max_t <= 100, (
        f"{(df_filtered['start_t'] + seq_len > 100).sum()} rows have "
        f"start_t + seq_len = {_max_t} > 100. The window filter was not applied to "
        f"the frame the physics loop iterates.")
    print(f"[FILTER] window (start_t + {seq_len} <= 100): {len(df_filtered)} rows kept")

    # Log-transformed inertial channels cannot take a zero / non-finite step.
    # Drop those rows here, BEFORE any tensor is built, so every array and the
    # group split see the same rows.
    if MULTI_ANGLE and USE_ARM_STATE and VARIANT_LOG_TRANSFORM:
        _, _valid, _cols = variant_window(df_filtered)
        if _cols:
            n_bad = int((~_valid).sum())
            if n_bad:
                df_filtered = df_filtered[_valid].copy()
            print(f"[FILTER] '{VARIANT_WINDOW_PREFIX}*' positive & finite: "
                  f"{len(df_filtered)} rows kept ({n_bad} dropped)")


    X_acc_flat = df_filtered[acc_cols].values.astype(np.float32)
    X_vel_flat = df_filtered[vel_cols].values.astype(np.float32)

    X_acc = X_acc_flat.reshape(-1, seq_len, num_axes)
    X_vel = X_vel_flat.reshape(-1, seq_len, num_axes)
    y = df_filtered[['gt_mass', 'gt_mu']].values.astype(np.float32)

    X_acc_tensor = torch.tensor(X_acc)
    X_vel_tensor = torch.tensor(X_vel)
    y_tensor = torch.tensor(y)

    # Vectorized gather of the physics windows (the per-row iterrows loop built
    # five Python lists of ~N x 60 floats each, slow and memory-hungry).
    pw = gather_pinn_windows(df_filtered, seq_len)
    fz_robot_tensor = torch.tensor(pw['robot_fz'], dtype=torch.float32)
    rhs_acc_tensor = torch.tensor(pw['rhs_acc'], dtype=torch.float32)
    lhs_net_f_tensor = torch.tensor(pw['lhs_net_f'], dtype=torch.float32)
    fz_normal_tensor = torch.tensor(pw['table_fz'], dtype=torch.float32)
    fx_robot_tensor = torch.tensor(pw['robot_fx'], dtype=torch.float32)
    del pw
    start_t_tensor = torch.tensor(df_filtered['start_t'].values.astype(np.int64))

    # =================================================================
    # TRAIN / VAL SPLIT (group split under MULTI_ANGLE)
    # =================================================================
    indices = np.arange(len(df_filtered))

    if MULTI_ANGLE:
        groups, group_src = _build_property_groups(df_filtered)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(indices, groups=groups))

        n_groups = len(np.unique(groups))
        print(f"[MULTI_ANGLE] group split on '{group_src}'")
        print(f"[MULTI_ANGLE]   {len(df_filtered)} rows / {n_groups} property groups "
              f"= {len(df_filtered) / max(n_groups, 1):.1f} trajectories per group")
        print(f"[MULTI_ANGLE]   train {len(train_idx)} rows / "
              f"{len(np.unique(groups[train_idx]))} groups")
        print(f"[MULTI_ANGLE]   val   {len(val_idx)} rows / "
              f"{len(np.unique(groups[val_idx]))} groups")

        overlap = np.intersect1d(groups[train_idx], groups[val_idx])
        assert len(overlap) == 0, f"Group leakage: {len(overlap)} groups in both splits"
    else:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        print(f"[SINGLE_ANGLE] random split: train {len(train_idx)} / val {len(val_idx)}")

    # =================================================================
    # VARIANT CHANNELS
    #
    # Every non-baseline variant reads a 60-step window family, so the
    # scalar and the sequence forms describe the same slice of time as the
    # velocity input. The scalar is the MINIMUM over that window, recomputed
    # here -- deliberately not the `worst_*` column, which is the minimum over
    # the whole ~300-step push and therefore a different quantity.
    #
    # Standardization uses TRAIN-split statistics only.
    #   scalar   -> per-feature mean/std, shape (1, 1)
    #   sequence -> ONE scalar mean/std across all (row, timestep) pairs, so the
    #               temporal shape the model reads is preserved rather than
    #               flattened by per-timestep normalization.
    # =================================================================
    arm_tensor = None       # (N, cond_dim) static conditioning token
    seq_tensor = None       # (N, T, 1) extra input channel

    if MULTI_ANGLE and USE_ARM_STATE:
        if not have_arm:
            raise ValueError(
                "INPUT_VARIANT requires the arm columns but they are missing from "
                "the CSV. Re-collect with the arm_state observation group enabled, "
                "or set INPUT_VARIANT='vel_only'."
            )

        W, _, wcols = variant_window(df_filtered)             # (N, T), transformed
        if not wcols:
            raise ValueError(
                f"INPUT_VARIANT='{INPUT_VARIANT}' needs "
                f"'{VARIANT_WINDOW_PREFIX}*' columns, none found in the CSV."
            )
        if len(wcols) != seq_len:
            raise ValueError(
                f"Window channel '{VARIANT_WINDOW_PREFIX}*' has {len(wcols)} steps "
                f"but the velocity input has {seq_len}. They must be the same "
                f"window or the two channels are not time-aligned."
            )

        if COND_DIM > 0:
            # Minimum over the 60-step window -- see the note above.
            raw = W.min(axis=1, keepdims=True)                    # (N, 1)
            mean = raw[train_idx].mean(axis=0, keepdims=True)
            std = raw[train_idx].std(axis=0, keepdims=True) + 1e-8
            arm_tensor = torch.tensor((raw - mean) / std)

            rel = float(std[0, 0] / max(abs(float(mean[0, 0])), 1e-12))
            if rel < 1e-3:
                print(f"[VARIANT][WARNING] conditioning scalar is near-constant "
                      f"(std/|mean| = {rel:.2e}); standardizing will amplify noise "
                      f"to unit scale.")

            save_arm_stats([f"min({VARIANT_WINDOW_PREFIX}*)"], mean, std, seq_len=seq_len)
            print(f"[VARIANT] '{INPUT_VARIANT}': cond token {tuple(arm_tensor.shape)} "
                  f"from min over {len(wcols)} window steps")
            print(f"[VARIANT]   train mean={float(mean[0,0]):.6f} "
                  f"std={float(std[0,0]):.6f} "
                  f"range=[{raw.min():.6f}, {raw.max():.6f}]")

        else:
            # Sequence channel: one scalar mean/std over all (row, timestep).
            mean = np.array([[W[train_idx].mean()]], dtype=np.float32)
            std = np.array([[W[train_idx].std()]], dtype=np.float32) + 1e-8
            seq_tensor = torch.tensor(((W - mean) / std)[:, :, None])   # (N, T, 1)

            feat_name = (f"log({VARIANT_WINDOW_PREFIX}*)" if VARIANT_LOG_TRANSFORM
                         else f"{VARIANT_WINDOW_PREFIX}*")
            save_arm_stats([feat_name], mean, std, seq_len=seq_len)
            print(f"[VARIANT] '{INPUT_VARIANT}': input channel {feat_name} "
                  f"{tuple(seq_tensor.shape)} concatenated to velocity "
                  f"-> input_dim={INPUT_DIM}")
            print(f"[VARIANT]   train mean={float(mean[0,0]):.6f} "
                  f"std={float(std[0,0]):.6f} "
                  f"per-step range=[{W.min():.6f}, {W.max():.6f}]"
                  + (" (log space)" if VARIANT_LOG_TRANSFORM else ""))

    # =================================================================
    # BATCH LAYOUT
    #   0 acc   1 vel   2 y   3 robot_fz   4 rhs_acc   5 lhs_net_f
    #   6 table_fz   7 start_t   8 robot_fx
    #   9  cond   (present only when COND_DIM > 0)
    #   9  seq    (present only when INPUT_DIM > 1)
    # The two are mutually exclusive by construction: a variant is either a
    # scalar condition or a sequence channel, never both. So index 9 is
    # unambiguous given the variant.
    # =================================================================
    train_tensors = [X_acc_tensor[train_idx], X_vel_tensor[train_idx], y_tensor[train_idx],
                     fz_robot_tensor[train_idx], rhs_acc_tensor[train_idx], lhs_net_f_tensor[train_idx],
                     fz_normal_tensor[train_idx], start_t_tensor[train_idx], fx_robot_tensor[train_idx]]
    val_tensors = [X_acc_tensor[val_idx], X_vel_tensor[val_idx], y_tensor[val_idx],
                   fz_robot_tensor[val_idx], rhs_acc_tensor[val_idx], lhs_net_f_tensor[val_idx],
                   fz_normal_tensor[val_idx], start_t_tensor[val_idx], fx_robot_tensor[val_idx]]

    if arm_tensor is not None:
        train_tensors.append(arm_tensor[train_idx])
        val_tensors.append(arm_tensor[val_idx])
    elif seq_tensor is not None:
        train_tensors.append(seq_tensor[train_idx])
        val_tensors.append(seq_tensor[val_idx])

    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Return the FULL frame, not df_filtered. evaluate.py selects OOD domains
    # (m_over, mu_under, ...) off the 'domain' column, and those rows exist only
    # here -- df_filtered is the m_seen_mu_seen cell alone. Returning the filtered
    # frame silently drops every unseen domain from the evaluation summary.
    return train_loader, val_loader, seq_len, df, choices