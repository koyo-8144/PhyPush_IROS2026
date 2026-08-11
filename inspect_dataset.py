import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configs import (G, M_SEEN_MAX, M_SEEN_MIN, MU_SEEN_MAX, MU_SEEN_MIN,
                     CSV_PATH, FRAME_MODE, MULTI_ANGLE, USE_ARM_STATE)

from dataset import (create_dataloaders, MULTI_ANGLE_COLS, ARM_STATIC_COLS,
                     ARM_STATE_COLS, ARM_FEATURE_COLS)
from utils import clean_force_col, add_min_max_text


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

    if {'obj_yaw_base', 'push_face_index'}.issubset(df.columns):
        pivot = df.pivot_table(index='obj_yaw_base', columns='push_face_index',
                               values='gt_mass', aggfunc='count')
        print("\n  rows per (object yaw x push side):")
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
        print(f"SAMPLE {i}  |  gt_mass = {gt_mass:.4f} kg   gt_mu = {gt_mu:.4f}   yaw_base = {yaw_base:.4f} rad")
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