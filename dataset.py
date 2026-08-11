import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from utils import clean_force_col
from configs import (M_UNSEEN_MAX, MU_UNSEEN_MAX, FRAME_MODE, MULTI_ANGLE,
                     USE_ARM_STATE, ARM_FEATURE_MODE, CSV_PATH)

# =============================================================================
# COLUMN GROUPS WRITTEN BY THE MULTI-ANGLE SWEEP
# (_save_offline_data_csv in phypush_distillation.py)
# =============================================================================

# Provenance: which sweep cell and which physical sample this row came from.
MULTI_ANGLE_COLS = ['obj_yaw_base', 'push_face_index', 'collect_idx', 'seed', 'env_id']

# Static per-push arm summary, from _arm_static_summary (5 values).
ARM_STATIC_COLS = [
    'worst_manipulability',    # min sqrt(det(J J^T)) over the push
    'ee_position_error',       # |push_end_goal - actual_ee_end|
    'push_start_reached_flag', # 1.0 for every saved row (see note below)
    'push_dir_b_x',            # push heading in the robot base frame
    'push_dir_b_y',
]

# Arm configuration sampled at the impact index, from _arm_state_a_his (24 values).
ARM_STATE_COLS = (
    [f'arm_q{j}' for j in range(7)]                   # joint positions
    + [f'arm_qd{j}' for j in range(7)]                # joint velocities
    + [f'arm_ee_pos_{ax}' for ax in ('x', 'y', 'z')]  # EE position, base frame
    + [f'arm_ee_rot6d_{j}' for j in range(6)]         # EE rotation, 6D
    + ['arm_manip_at_impact']                         # manipulability at impact
)

ALL_ARM_COLS = ARM_STATIC_COLS + ARM_STATE_COLS

# Columns usable as model conditioning features.
#
# push_start_reached_flag is EXCLUDED: _save_offline_data_csv skips any row where
# push_start_reached or push_end_reached is 0, so every surviving row carries
# 1.0. It is a constant -- no signal, and a divide-by-~0 during standardization.
ARM_FEATURE_COLS = [c for c in ARM_STATIC_COLS if c != 'push_start_reached_flag'] + ARM_STATE_COLS

# Minimal subset worth trying before the full 28. Two numbers describe the arm's
# reach direction; one describes how degraded the Jacobian got during the push.
ARM_FEATURE_COLS_MINIMAL = ['worst_manipulability', 'push_dir_b_x', 'push_dir_b_y']


# Conditioning presets selected by ARM_FEATURE_MODE in configs.py.
ARM_FEATURE_PRESETS = {
    # Push heading as a unit vector in the robot base frame. One concept,
    # two components. The most complete single descriptor of arm reach.
    "push_dir":       ['push_dir_b_x', 'push_dir_b_y'],
    # Strictly one scalar: how degraded the Jacobian got during the push.
    "manipulability": ['worst_manipulability'],
    # Both of the above.
    "minimal":        ARM_FEATURE_COLS_MINIMAL,
    # Everything: joint positions, joint velocities, EE pose, manipulability.
    "full":           ARM_FEATURE_COLS,
}


def get_arm_feature_cols():
    """Feature list selected by ARM_FEATURE_MODE in configs.py."""
    if ARM_FEATURE_MODE not in ARM_FEATURE_PRESETS:
        raise ValueError(
            f"Unknown ARM_FEATURE_MODE: {ARM_FEATURE_MODE!r}. "
            f"Expected one of {sorted(ARM_FEATURE_PRESETS)}."
        )
    return list(ARM_FEATURE_PRESETS[ARM_FEATURE_MODE])


# Conditioning width, for `cond_dim` when constructing the model.
ARM_DIM = len(ARM_FEATURE_PRESETS.get(ARM_FEATURE_MODE, []))


def _build_property_groups(df_filtered):
    """Group key identifying one physical (mass, mu) pair.

    In the multi-angle dataset the same pair is observed ~20 times (10 object
    yaws x 2 push sides). Properties are determined by (seed, env_id), so that
    pair is the natural group. Falls back to the rounded property values if the
    provenance columns are absent.
    """
    if 'seed' in df_filtered.columns and 'env_id' in df_filtered.columns:
        return (df_filtered['seed'].astype(int) * 100000
                + df_filtered['env_id'].astype(int)).values, "seed+env_id"

    return (df_filtered['gt_mass'].round(6).astype(str) + "_"
            + df_filtered['gt_mu'].round(6).astype(str)).values, "gt_mass+gt_mu"


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
        print(f"[MULTI_ANGLE][WARNING] Missing {len(missing_state)} arm state columns, "
              f"e.g. {missing_state[:5]}.")

    have_arm = not (missing_static or missing_state)
    if have_arm:
        print(f"[MULTI_ANGLE] Arm columns present "
              f"({len(ARM_STATIC_COLS)} static + {len(ARM_STATE_COLS)} state = "
              f"{len(ALL_ARM_COLS)} total, {len(ARM_FEATURE_COLS)} usable as features).")
    return have_arm


def load_dataset_csv(csv_path=None, chunksize=50000, verbose=True):
    """Memory-safe CSV load for TRAINING.

    Unlike the loaders in inspect_dataset.py / compare_datasets.py, this one
    keeps EVERY row and EVERY column: create_dataloaders reads the pinn_* physics
    arrays and needs the full population, so nothing can be dropped or subsampled.
    The savings come only from:

      1. Reading in chunks so the raw file is never fully materialised as float64.
      2. Downcasting float64 -> float32 per chunk, which halves resident memory
         and matches the precision the tensors use downstream anyway.

    A 12 GB CSV loads at roughly half the peak RAM of a plain pd.read_csv, with
    identical contents. Works for any dataset, multi-angle or not.

    Args:
        csv_path: path to load; defaults to configs.CSV_PATH.
        chunksize: rows per chunk. Larger = fewer concat passes but higher peak.
        verbose: print progress.
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
    del chunks  # free the per-chunk copies before create_dataloaders runs

    if 'gt_fric_force' in df.columns:
        # This column is stored as bracketed strings, so the per-chunk float
        # downcast skipped it. Clean, then downcast the resulting float64.
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col).astype(np.float32)

    if verbose:
        mem_gb = df.memory_usage(deep=True).sum() / 1e9
        print(f"\n[load_dataset_csv] loaded {len(df)} rows x {df.shape[1]} cols, "
              f"~{mem_gb:.1f} GB resident")

    return df


# Sidecar file holding the arm-feature standardization stats, so evaluation can
# apply the SAME mean/std the model was trained on. Without this, evaluate.py
# would feed raw push_dir values into a model trained on standardized ones.
ARM_STATS_SUFFIX = ".arm_stats.npz"


def arm_stats_path(csv_path=None):
    from configs import CSV_PATH as _CSV
    base = csv_path if csv_path is not None else _CSV
    return base + ARM_STATS_SUFFIX


def save_arm_stats(feature_cols, mean, std, csv_path=None):
    """Persist arm-feature standardization stats next to the dataset CSV."""
    path = arm_stats_path(csv_path)
    np.savez(path,
             feature_cols=np.array(feature_cols, dtype=object),
             mean=mean.astype(np.float32),
             std=std.astype(np.float32))
    print(f"[ARM_STATE] saved standardization stats -> {path}")


def load_arm_stats(csv_path=None):
    """Load arm-feature stats saved during training. Returns (cols, mean, std).

    Raises FileNotFoundError if training never wrote them -- evaluation must not
    silently fall back to raw or re-derived stats, which would misscale the
    conditioning input.
    """
    path = arm_stats_path(csv_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arm stats not found at {path}. Run train.py once with "
            f"USE_ARM_STATE=True to generate them before evaluating a "
            f"conditioned model."
        )
    d = np.load(path, allow_pickle=True)
    return list(d['feature_cols']), d['mean'], d['std']


def create_dataloaders(df, batch_size=64, m_seen_min=0.2, m_seen_max=2.0, mu_seen_min=0.15, mu_seen_max=0.5):
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
    df_filtered = df[df['domain'] == 'm_seen_mu_seen'].copy()

    have_arm = False
    if MULTI_ANGLE:
        have_arm = _report_column_availability(df)
    
    acc_cols = sorted([c for c in df_filtered.columns if "input_acc_" in c], key=lambda x: int(x.split('_')[-1]))
    vel_cols = sorted([c for c in df_filtered.columns if "input_vel_" in c], key=lambda x: int(x.split('_')[-1]))

    num_axes = 1  
    seq_len = len(acc_cols) // num_axes

    valid_mask = (df_filtered['start_t'] + seq_len) <= 100
    df_filtered = df_filtered[valid_mask].copy()

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
    # TRAIN / VAL SPLIT
    #
    # MULTI_ANGLE: the same (mass, mu) pair appears once per orientation /
    # push-side cell (~20 rows). A random split would leak almost every
    # validation pair into training, so validation error would measure
    # memorization rather than generalization. Split by property group.
    # =================================================================
    indices = np.arange(len(df_filtered))

    if MULTI_ANGLE:
        groups, group_src = _build_property_groups(df_filtered)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(indices, groups=groups))

        n_groups = len(np.unique(groups))
        n_train_groups = len(np.unique(groups[train_idx]))
        n_val_groups = len(np.unique(groups[val_idx]))

        print(f"[MULTI_ANGLE] group split on '{group_src}'")
        print(f"[MULTI_ANGLE]   {len(df_filtered)} rows / {n_groups} property groups "
              f"= {len(df_filtered) / max(n_groups, 1):.1f} trajectories per group")
        print(f"[MULTI_ANGLE]   train {len(train_idx)} rows / {n_train_groups} groups")
        print(f"[MULTI_ANGLE]   val   {len(val_idx)} rows / {n_val_groups} groups")

        overlap = np.intersect1d(groups[train_idx], groups[val_idx])
        assert len(overlap) == 0, f"Group leakage: {len(overlap)} groups in both splits"
    else:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        print(f"[SINGLE_ANGLE] random split: train {len(train_idx)} / val {len(val_idx)}")

    # =================================================================
    # OPTIONAL ARM CONDITIONING FEATURES
    #
    # Standardization statistics come from the TRAIN split only, so no
    # validation information leaks into the input scaling.
    # =================================================================
    arm_tensor = None
    if MULTI_ANGLE and USE_ARM_STATE:
        # Gate on the columns the SELECTED mode actually needs, not on all 29
        # arm columns. ARM_FEATURE_MODE='push_dir' needs only push_dir_b_{x,y};
        # requiring the 24 joint columns too would reject a valid dataset.
        feature_cols = get_arm_feature_cols()
        missing_feats = [c for c in feature_cols if c not in df_filtered.columns]
        if missing_feats:
            raise ValueError(
                f"USE_ARM_STATE=True with ARM_FEATURE_MODE='{ARM_FEATURE_MODE}' "
                f"needs columns {feature_cols}, but the CSV is missing "
                f"{missing_feats}. Re-collect with the arm_state observation "
                f"group enabled, choose a mode whose columns exist, or set "
                f"USE_ARM_STATE=False in configs.py."
            )

        arm_raw = df_filtered[feature_cols].values.astype(np.float32)

        arm_mean = arm_raw[train_idx].mean(axis=0, keepdims=True)
        arm_std = arm_raw[train_idx].std(axis=0, keepdims=True) + 1e-8
        arm_norm = (arm_raw - arm_mean) / arm_std
        arm_tensor = torch.tensor(arm_norm)

        # Persist so evaluate.py can reproduce this exact scaling.
        save_arm_stats(feature_cols, arm_mean, arm_std)

        # Flag near-constant features using a RELATIVE threshold. An absolute
        # cutoff misses columns like arm_ee_rot6d_2 (std 3e-5 about a mean of
        # -1.0, i.e. the gripper always points down) or arm_ee_pos_z (std 4e-4,
        # EE height fixed during the push) -- these are constants in disguise
        # and contribute only noise once standardized.
        scale = np.maximum(np.abs(arm_raw[train_idx]).mean(axis=0), 1e-12)
        rel_var = arm_std.flatten() / scale
        near_const = [c for c, r in zip(feature_cols, rel_var) if r < 1e-3]
        if near_const:
            print(f"[ARM_STATE][WARNING] Near-constant features (relative std < 1e-3, "
                  f"no usable signal): {near_const}")

        print(f"[ARM_STATE] mode='{ARM_FEATURE_MODE}', feature tensor "
              f"{tuple(arm_tensor.shape)} ({len(feature_cols)} features, "
              f"standardized on train split)")
        print(f"[ARM_STATE] features: {feature_cols}")
    
    train_tensors = [X_acc_tensor[train_idx], X_vel_tensor[train_idx], y_tensor[train_idx],
                     fz_robot_tensor[train_idx], rhs_acc_tensor[train_idx], lhs_net_f_tensor[train_idx],
                     fz_normal_tensor[train_idx], start_t_tensor[train_idx], fx_robot_tensor[train_idx]]
    val_tensors = [X_acc_tensor[val_idx], X_vel_tensor[val_idx], y_tensor[val_idx],
                   fz_robot_tensor[val_idx], rhs_acc_tensor[val_idx], lhs_net_f_tensor[val_idx],
                   fz_normal_tensor[val_idx], start_t_tensor[val_idx], fx_robot_tensor[val_idx]]

    if arm_tensor is not None:
        # NOTE: this makes each batch a 10-tuple. Every unpack site must be
        # updated -- see PATCH_evaluate_and_train.md for the five locations.
        train_tensors.append(arm_tensor[train_idx])
        val_tensors.append(arm_tensor[val_idx])

    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)

    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, seq_len, df, choices