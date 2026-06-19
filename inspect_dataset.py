import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configs import G, M_SEEN_MAX, M_SEEN_MIN, MU_SEEN_MAX, MU_SEEN_MIN

from dataset import create_dataloaders
from utils import clean_force_col, add_min_max_text

def inspect_dataloader(loader, num_samples=3):
    batch = next(iter(loader))
    X_acc, X_vel, y, fz_robot, rhs_acc, lhs_net_f, fz_normal, start_t, fx_robot = batch
    
    X_acc = X_acc.numpy()
    X_vel = X_vel.numpy()
    y = y.numpy()
    fz_robot = fz_robot.numpy()
    rhs_acc = rhs_acc.numpy()
    lhs_net_f = lhs_net_f.numpy()
    fz_normal = fz_normal.numpy()
    fx_robot = fx_robot.numpy()
    
    seq_len = X_vel.shape[1]
    time_steps = np.arange(seq_len)
    
    sns.set_theme(style="whitegrid")
    
    for i in range(min(num_samples, X_vel.shape[0])):
        fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=False)
        
        gt_mass = y[i, 0]
        gt_mu = y[i, 1]
        
        vel_win = X_vel[i, :, 0]
        axes[0].plot(time_steps, vel_win, 'b-o', linewidth=2, label='Extracted Velocity')
        axes[0].set_title("1. Model Input: Velocity")
        axes[0].set_ylabel("Velocity [m/s]")
        add_min_max_text(axes[0], vel_win, "m/s")
        axes[0].legend(loc='upper left')
        
        info_text = f"M: {gt_mass:.3f}\nMu: {gt_mu:.3f}"
        axes[0].text(0.05, 0.95, info_text, transform=axes[0].transAxes, 
                     verticalalignment='top', fontsize=14,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        acc_win = X_acc[i, :, 0]
        axes[1].plot(time_steps, acc_win, 'r-o', linewidth=2, label='Extracted Acceleration')
        axes[1].set_title("2. Model Input: Acceleration")
        axes[1].set_ylabel("Acceleration [m/s^2]")
        add_min_max_text(axes[1], acc_win, "m/s^2")
        axes[1].legend(loc='upper left')
        
        axes[2].plot(time_steps, lhs_net_f[i], label='LHS (Net Force)', color='purple', linestyle='--', linewidth=2)
        axes[2].plot(time_steps, fx_robot[i], label='Robot Force', color='green', alpha=0.8, linewidth=2)
        axes[2].plot(time_steps, fz_normal[i], label='Table Force (Normal)', color='orange', alpha=0.8, linewidth=2)
        axes[2].set_title("3. Force Components Decomposition")
        axes[2].set_ylabel("Force [N]")
        axes[2].legend(loc='upper left')
        
        sim_win = lhs_net_f[i]
        net_force_calc = gt_mass * rhs_acc[i]
        axes[3].plot(time_steps, sim_win, label='Simulated Net Force', color='purple', linewidth=3, alpha=0.5)
        axes[3].plot(time_steps, net_force_calc, 'k--', label='Calculated (GT Mass * Acc)', linewidth=1.5)
        axes[3].set_title("4. Physics Check: Newton's 2nd Law")
        axes[3].set_ylabel("Force [N]")
        add_min_max_text(axes[3], sim_win, "N")
        axes[3].legend(loc='upper left')
        
        normal_force_calc = np.clip((gt_mass * G) - fz_robot[i], 0.0, None)
        fric_force_calc_profile = gt_mu * normal_force_calc
        fric_force_direct_profile = gt_mu * np.clip(fz_normal[i], 0.0, None)
        
        axes[4].plot(time_steps, fric_force_calc_profile, label='Calculated Friction', color='orange', linewidth=2)
        axes[4].plot(time_steps, fric_force_direct_profile, label='Direct Friction', color='green', linestyle=':', linewidth=2)
        
        axes[4].set_title("5. Physics Check: Friction Model")
        axes[4].set_ylabel("Force [N]")
        axes[4].set_xlabel("Time Step")
        add_min_max_text(axes[4], fric_force_calc_profile, "N")
        axes[4].legend(loc='upper left')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()

def main():
    CSV_PATH = "/home/psxkf4/IsaacLab/source/collected_data/data_tb-3_ta57_emavel1.0_velstd0.0_broad.csv"
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    if 'gt_fric_force' in df.columns:
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col)
        
    print(f"Successfully loaded data! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    train_loader, val_loader, seq_len, df_filtered, choices = create_dataloaders(
        df, batch_size=64, m_seen_min=M_SEEN_MIN, m_seen_max=M_SEEN_MAX, mu_seen_min=MU_SEEN_MIN, mu_seen_max=MU_SEEN_MAX
    )
    
    print(f"Inspecting training dataloader batches...")
    inspect_dataloader(train_loader, num_samples=3)

if __name__ == "__main__":
    main()