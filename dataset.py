import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from utils import clean_force_col
from configs import (M_UNSEEN_MAX, MU_UNSEEN_MAX, FRAME_MODE, MULTI_ANGLE,
                     USE_ARM_STATE, CSV_PATH,
                     INPUT_VARIANT, COND_DIM, INPUT_DIM, VARIANT_WINDOW_PREFIX,
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
# These are the columns the four input variants actually draw from.
WINDOW_PREFIXES = ('arm_manip_w', 'arm_dir_manip_w')


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


def load_dataset_csv(csv_path=None, chunksize=50000, verbose=True):
    """Memory-safe CSV load for TRAINING.

    Keeps EVERY row and EVERY column: create_dataloaders reads the pinn_* physics
    arrays and needs the full population. The savings come from chunked reading
    and a float64 -> float32 downcast per chunk.
    """
    path = csv_path if csv_path is not None else CSV_PATH

    if verbose:
        size_gb = os.path.getsize(path) / 1e9 if os.path.exists(path) else float('nan')
        print(f"[load_dataset_csv] {path}  ({size_gb:.1f} GB) "
              f"in chunks of {chunksize}, float32 downcast")

    chunks = []
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        float_cols = chunk.select_dtypes(include=['float64']).columns
        chunk[float_cols] = chunk[float_cols].astype(np.float32)
        chunks.append(chunk)
        total += len(chunk)
        if verbose:
            print(f"  ... {total} rows", end='\r')

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    if 'gt_fric_force' in df.columns:
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col).astype(np.float32)

    if verbose:
        mem_gb = df.memory_usage(deep=True).sum() / 1e9
        print(f"\n[load_dataset_csv] loaded {len(df)} rows x {df.shape[1]} cols, "
              f"~{mem_gb:.1f} GB resident")

    return df


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


    X_acc_flat = df_filtered[acc_cols].values.astype(np.float32)
    X_vel_flat = df_filtered[vel_cols].values.astype(np.float32)

    X_acc = X_acc_flat.reshape(-1, seq_len, num_axes)
    X_vel = X_vel_flat.reshape(-1, seq_len, num_axes)
    y = df_filtered[['gt_mass', 'gt_mu']].values.astype(np.float32)

    X_acc_tensor = torch.tensor(X_acc)
    X_vel_tensor = torch.tensor(X_vel)
    y_tensor = torch.tensor(y)

    robot_fz_list, rhs_acc_list, lhs_net_f_list, table_fz_list, robot_fx_list = [], [], [], [], []
    for idx, row in df_filtered.iterrows():
        st = int(row['start_t'])
        window_range = range(st, st + seq_len)
        if FRAME_MODE == "world":
            robot_fz_list.append([row[f"pinn_robot_wrench_t{t}_ax5"] for t in window_range])
            rhs_acc_list.append([row[f"pinn_RHS_acc_t{t}_ax3"] for t in window_range])
            lhs_net_f_list.append([row[f"pinn_LHS_wrench_t{t}_ax3"] for t in window_range])
            table_fz_list.append([row[f"pinn_table_wrench_t{t}_ax5"] for t in window_range])
            robot_fx_list.append([row[f"pinn_robot_wrench_t{t}_ax3"] for t in window_range])
        elif FRAME_MODE == "local":
            robot_fz_list.append([row[f"pinn_robot_wrench_t{t}_ax3"] for t in window_range])
            rhs_acc_list.append([row[f"pinn_RHS_acc_t{t}_ax5"] for t in window_range])
            lhs_net_f_list.append([row[f"pinn_LHS_wrench_t{t}_ax5"] for t in window_range])
            table_fz_list.append([row[f"pinn_table_wrench_t{t}_ax3"] for t in window_range])
            robot_fx_list.append([row[f"pinn_robot_wrench_t{t}_ax5"] for t in window_range])

    fz_robot_tensor = torch.tensor(np.array(robot_fz_list), dtype=torch.float32)
    rhs_acc_tensor = torch.tensor(np.array(rhs_acc_list), dtype=torch.float32)
    lhs_net_f_tensor = torch.tensor(np.array(lhs_net_f_list), dtype=torch.float32)
    fz_normal_tensor = torch.tensor(np.array(table_fz_list), dtype=torch.float32)
    fx_robot_tensor = torch.tensor(np.array(robot_fx_list), dtype=torch.float32)
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
    # All four non-baseline variants read the SAME 60-step window family, so the
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

        wcols = window_cols(df_filtered, VARIANT_WINDOW_PREFIX)
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

        W = df_filtered[wcols].values.astype(np.float32)          # (N, T)

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

            save_arm_stats([f"{VARIANT_WINDOW_PREFIX}*"], mean, std, seq_len=seq_len)
            print(f"[VARIANT] '{INPUT_VARIANT}': input channel "
                  f"{tuple(seq_tensor.shape)} concatenated to velocity "
                  f"-> input_dim={INPUT_DIM}")
            print(f"[VARIANT]   train mean={float(mean[0,0]):.6f} "
                  f"std={float(std[0,0]):.6f} "
                  f"per-step range=[{W.min():.6f}, {W.max():.6f}]")

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