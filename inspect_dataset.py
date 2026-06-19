import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configs import G, M_SEEN_MAX, M_SEEN_MIN, MU_SEEN_MAX, MU_SEEN_MIN, CSV_PATH, FRAME_MODE

from dataset import create_dataloaders
from utils import clean_force_col, add_min_max_text

def inspect_dataloader(loader, num_samples=3):
    batch = next(iter(loader))
    
    # Explicitly named variables indicating axis and source
    X_acc, X_vel, y, fz_robot_sim, acc_x_sim, net_fx_sim, fz_normal_sim, start_t, fx_robot_sim = batch
    
    X_acc = X_acc.numpy()
    X_vel = X_vel.numpy()
    y = y.numpy()
    fz_robot_sim = fz_robot_sim.numpy()
    acc_x_sim = acc_x_sim.numpy()
    net_fx_sim = net_fx_sim.numpy()
    fz_normal_sim = fz_normal_sim.numpy()
    fx_robot_sim = fx_robot_sim.numpy()
    
    seq_len = X_vel.shape[1]
    time_steps = np.arange(seq_len)
    
    # Tableau 10 color palette for high contrast and academic readability
    colors = {
        'vel': '#1f77b4',       # Muted Blue
        'acc': '#d62728',       # Brick Red
        'robot': '#2ca02c',     # Forest Green
        'friction': '#ff7f0e',  # Safety Orange
        'net_sim': '#7f7f7f',   # Neutral Grey
        'net_calc': '#9467bd',  # Muted Purple
        'theory': '#17becf'     # Cyan
    }
    
    sns.set_theme(style="whitegrid")
    
    for i in range(min(num_samples, X_vel.shape[0])):
        fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=False)
        
        gt_mass = y[i, 0]
        gt_mu = y[i, 1]
        
        # -----------------------------------------------------------
        # PLOT 1: Velocity
        # -----------------------------------------------------------
        vel_x = X_vel[i, :, 0]
        axes[0].plot(time_steps, vel_x, color=colors['vel'], marker='o', markersize=4, linewidth=2, label='Extracted EE Velocity')
        axes[0].set_title("1. Model Input: Kinematics (Velocity)")
        axes[0].set_ylabel("Velocity [m/s]")
        # add_min_max_text(axes[0], vel_x, "m/s")
        axes[0].legend(loc='upper left')
        
        # info_text = f"Ground Truth | Mass: {gt_mass:.3f} kg | Friction (\u03bc): {gt_mu:.3f}"
        # axes[0].text(0.02, 0.95, info_text, transform=axes[0].transAxes, 
        #              verticalalignment='top', fontsize=12, fontweight='bold',
        #              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=colors['net_sim'], alpha=0.9))
        
        # -----------------------------------------------------------
        # PLOT 2: Acceleration
        # -----------------------------------------------------------
        acc_x = X_acc[i, :, 0]
        axes[1].plot(time_steps, acc_x, color=colors['acc'], marker='o', markersize=4, linewidth=2, label='Extracted EE Acceleration')
        axes[1].set_title("2. Model Input: Kinematics (Acceleration)")
        axes[1].set_ylabel("Acceleration [m/s\u00b2]")
        # add_min_max_text(axes[1], acc_x, "m/s\u00b2")
        axes[1].legend(loc='upper left')

        # -----------------------------------------------------------
        # PHYSICS CALCULATIONS
        # -----------------------------------------------------------
        if FRAME_MODE == "world":
            normal_force_calc = np.clip((gt_mass * G) - fz_robot_sim[i], 0.0, None)
        elif FRAME_MODE == "local":
            normal_force_calc = np.clip((gt_mass * G) + fz_robot_sim[i], 0.0, None)
        fric_magnitude_calc = gt_mu * normal_force_calc 
        fric_magnitude_sim = gt_mu * np.abs(fz_normal_sim[i])
        
        # Friction vector opposes the direction of motion (push is +X, friction is -X)
        fx_friction_vector = -fric_magnitude_sim
        calc_net_force_x = fx_robot_sim[i] + fx_friction_vector
        mass_x_accel = gt_mass * acc_x_sim[i]
        
        # -----------------------------------------------------------
        # PLOT 3: Force Decomposition
        # -----------------------------------------------------------
        axes[2].plot(time_steps, net_fx_sim[i], label=r'Simulator Net Force ($F_{net}$)', color=colors['net_sim'], linewidth=4, alpha=0.4)
        axes[2].plot(time_steps, fx_robot_sim[i], label=r'Robot Applied Force ($F_{robot}$)', color=colors['robot'], linewidth=2)
        axes[2].plot(time_steps, fx_friction_vector, label=r'Table Friction ($-F_{fric}$)', color=colors['friction'], linewidth=2)
        axes[2].plot(time_steps, calc_net_force_x, label=r'Calculated Net Force ($F_{robot} - F_{fric}$)', color=colors['net_calc'], linestyle='--', linewidth=2)
        
        axes[2].set_title("3. Force Components Decomposition (X-Axis)")
        axes[2].set_ylabel("Force [N]")
        axes[2].legend(loc='upper left')
        
        # -----------------------------------------------------------
        # PLOT 4: Newton's Second Law Check
        # -----------------------------------------------------------
        axes[3].plot(time_steps, net_fx_sim[i], label=r'Simulator Net Force ($F_{net}$)', color=colors['net_sim'], linewidth=4, alpha=0.4)
        axes[3].plot(time_steps, calc_net_force_x, label=r'Force Sum ($F_{robot} - F_{fric}$)', color=colors['net_calc'], linewidth=2)
        axes[3].plot(time_steps, mass_x_accel, label=r"Newton's 2nd Law ($m \cdot a_x$)", color=colors['theory'], linestyle='--', linewidth=2.5)
        
        axes[3].set_title("4. Physics Check: Newton's 2nd Law Alignment")
        axes[3].set_ylabel("Force [N]")
        add_min_max_text(axes[3], net_fx_sim[i], "N")
        axes[3].legend(loc='upper left')
    
        # -----------------------------------------------------------
        # PLOT 5: Friction Model Check
        # -----------------------------------------------------------
        axes[4].plot(time_steps, fric_magnitude_calc, label=r'Theoretical Friction ($\mu \cdot N_{calc}$)', color=colors['theory'], linewidth=2)
        axes[4].plot(time_steps, fric_magnitude_sim, label=r'Simulator Friction ($\mu \cdot N_{sim}$)', color=colors['friction'], linestyle='--', linewidth=2.5)
        
        axes[4].set_title("5. Physics Check: Coulomb Friction Model (Magnitudes)")
        axes[4].set_ylabel("Force Magnitude [N]")
        axes[4].set_xlabel("Time Step")
        add_min_max_text(axes[4], fric_magnitude_calc, "N")
        axes[4].legend(loc='upper left')
        
        for ax in axes:
            ax.grid(True, linestyle=':', alpha=0.6)
            
        plt.tight_layout()
        plt.show()

def main():
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