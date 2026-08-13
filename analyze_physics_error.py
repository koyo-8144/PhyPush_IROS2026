import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configs import CSV_PATH, FRAME_MODE, G

def load_data_subset(csv_path, max_envs=50):
    print(f"Loading data from {csv_path}...")
    chunk_list = []
    for chunk in pd.read_csv(csv_path, chunksize=15000):
        if 'env_id' in chunk.columns:
            chunk = chunk[chunk['env_id'] < max_envs]
        float_cols = chunk.select_dtypes(include=['float64']).columns
        chunk[float_cols] = chunk[float_cols].astype(np.float32)
        chunk_list.append(chunk)
    df = pd.concat(chunk_list, ignore_index=True).copy()
    print(f"Loaded {len(df)} rows.")
    return df

def extract_physics_series(df):
    acc_cols = [c for c in df.columns if "pinn_RHS_acc_t" in c and "_ax" in c]
    seq_len = max([int(c.split('_t')[1].split('_')[0]) for c in acc_cols]) + 1
    
    N = len(df)
    net_fx = np.zeros((N, seq_len), dtype=np.float32)
    acc_x = np.zeros((N, seq_len), dtype=np.float32)
    robot_fx = np.zeros((N, seq_len), dtype=np.float32)
    robot_fz = np.zeros((N, seq_len), dtype=np.float32)
    
    print(f"Extracting {seq_len}-step physics sequences (Frame Mode: {FRAME_MODE.upper()})...")
    for t in range(seq_len):
        if FRAME_MODE == "world":
            net_fx[:, t] = df[f"pinn_LHS_wrench_t{t}_ax3"].values
            acc_x[:, t] = df[f"pinn_RHS_acc_t{t}_ax3"].values
            robot_fx[:, t] = df[f"pinn_robot_wrench_t{t}_ax3"].values
            robot_fz[:, t] = df[f"pinn_robot_wrench_t{t}_ax5"].values
        elif FRAME_MODE == "local":
            net_fx[:, t] = df[f"pinn_LHS_wrench_t{t}_ax5"].values
            acc_x[:, t] = df[f"pinn_RHS_acc_t{t}_ax5"].values
            robot_fx[:, t] = df[f"pinn_robot_wrench_t{t}_ax5"].values
            robot_fz[:, t] = -df[f"pinn_robot_wrench_t{t}_ax3"].values
            
    return net_fx, acc_x, robot_fx, robot_fz

def calculate_physics_residuals(df, net_fx, acc_x, robot_fx, robot_fz):
    print("Calculating Newton's 2nd Law and Kinematic residuals...")
    m_gt = df['gt_mass'].values[:, np.newaxis]
    mu_gt = df['gt_mu'].values[:, np.newaxis]
    
    newton_error_seq = np.abs(net_fx - (m_gt * acc_x))
    normal_force = np.clip((m_gt * G) - robot_fz, 0.0, None)
    fric_force = mu_gt * normal_force
    acc_theory = (robot_fx - fric_force) / (m_gt + 1e-6)
    kinematic_error_seq = np.abs(acc_x - acc_theory)
    
    df['newton_error_mean'] = newton_error_seq[:, :50].mean(axis=1)
    df['kinematic_error_mean'] = kinematic_error_seq[:, :50].mean(axis=1)
    
    # Store full sequences for plotting the worst case later
    df['newton_error_max'] = newton_error_seq[:, :50].max(axis=1)
    
    if 'obj_yaw_base' in df.columns:
        # Bin to nearest 5 degrees to clean up the terminal matrix
        yaw_deg = np.degrees(df['obj_yaw_base'])
        df['yaw_binned'] = (5 * np.round(yaw_deg / 5)).astype(int)
        
    return df

def analyze_and_plot(df, net_fx, acc_x):
    # --- 1. Terminal Matrix Output ---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    
    print("\n" + "="*60)
    print("NEWTON ERROR BREAKDOWN MATRIX (Binned by 5 degrees)")
    print("="*60)
    if 'yaw_binned' in df.columns and 'push_face_index' in df.columns:
        pivot_newton = df.pivot_table(index='push_face_index', columns='yaw_binned', values='newton_error_mean', aggfunc='mean')
        print(pivot_newton.round(3).fillna('-').to_string())
    else:
        print("Missing orientation columns.")

    # --- 2. Find the Worst Trajectory for Time-Domain Mismatch Proof ---
    worst_idx = df['newton_error_max'].idxmax()
    worst_mass = df.iloc[worst_idx]['gt_mass']
    worst_f_net = net_fx[worst_idx, :50]
    worst_m_a = worst_mass * acc_x[worst_idx, :50]
    
    # --- 3. Save Hardcoded Plots to Disk ---
    print("\nGenerating and saving plots to 'physics_diagnostic_plots.png'...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Boxplot of Binned Error
    if 'yaw_binned' in df.columns:
        sns.boxplot(data=df, x='yaw_binned', y='newton_error_mean', hue='push_face_index', ax=axes[0], palette="Set2")
        axes[0].set_title("Newton's 2nd Law Residuals by Binned Orientation")
        axes[0].set_ylabel("Mean Absolute Error [N]")
        axes[0].set_xlabel("Object Yaw [deg]")
    
    # Time-Domain Mismatch Overlay
    time_steps = np.arange(50)
    axes[1].plot(time_steps, worst_f_net, label="F_net (Instantaneous Sensor)", color="purple", linewidth=2.5)
    axes[1].plot(time_steps, worst_m_a, label="m * a (Average Kinematics)", color="orange", linestyle="--", linewidth=2.5)
    axes[1].set_title(f"Time-Domain Mismatch Proof (Worst Trajectory ID: {worst_idx})")
    axes[1].set_ylabel("Force [N]")
    axes[1].set_xlabel("Time Step (First 50 steps)")
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("physics_diagnostic_plots.png", dpi=150)
    print("Saved successfully. Please open 'physics_diagnostic_plots.png' to view the graphs.")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
        
    df = load_data_subset(CSV_PATH)
    net_fx, acc_x, robot_fx, robot_fz = extract_physics_series(df)
    df = calculate_physics_residuals(df, net_fx, acc_x, robot_fx, robot_fz)
    
    analyze_and_plot(df, net_fx, acc_x)

if __name__ == "__main__":
    main()