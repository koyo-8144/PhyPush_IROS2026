import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models import PhysicsTransformerEstimator
from dataset import create_dataloaders
from utils import set_seed, clean_force_col
from configs import CSV_PATH

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
OUTPUT_DIR = "paper_results/sim/attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_SHOW = False 
BATCH_SIZE = 64 


# --- UPDATE THIS PATH TO MATCH YOUR TIMESTAMP RUN ---
BASE_RUN_DIR = "/home/psxkf4/phypush_training/results/checkpoints/from_20260316_YOUR_TIMESTAMP_HERE"

# --- MANUAL MODEL SELECTION ---
# We map to the domain_evaluation_summary.csv just as a reference point to find the checkpoint directory
MODELS_TO_COMPARE = {
    "PhyPush (Data loss)": os.path.join(BASE_RUN_DIR, "data_tcri-log1p_mse_task10.0/domain_evaluation_summary.csv"),
    "PhyPush ($m$ best)": os.path.join(BASE_RUN_DIR, "pinn_pcri-L1_p10c10.0/domain_evaluation_summary.csv"),
    "PhyPush ($\\mu$ best)": os.path.join(BASE_RUN_DIR, "pinn_annstartepo300_rampd600_pcri-L1_p5c10.0_p2-2c5.0/domain_evaluation_summary.csv"),
    r"$\mathcal{L}_{\text{acc}}$": os.path.join(BASE_RUN_DIR, "pinn_pcri-L1_p10c10.0/domain_evaluation_summary.csv"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. PLOTTING HELPER
# ==========================================
def plot_emphasized_kinematics(ax, data_matrix, title, weights, color, ylabel):
    """Plots individual kinematics and emphasizes indices with max average attention."""
    t_steps = np.arange(data_matrix.shape[1])
    
    # Plot individual trajectories with low alpha to show the distribution
    max_plots = min(data_matrix.shape[0], 300) # Limit to 300 to keep rendering fast
    for i in range(max_plots):
        ax.plot(t_steps, data_matrix[i], color=color, alpha=0.03)
        
    # Plot the mean trajectory darker for reference
    ax.plot(t_steps, np.mean(data_matrix, axis=0), color='black', alpha=0.8, linewidth=1.5, label='Mean Velocity')
    
    # Find all indices with the maximum attention weight
    max_val = np.max(weights)
    max_indices = np.where(weights == max_val)[0]
    
    for i, idx in enumerate(max_indices):
        ax.axvspan(idx - 0.4, idx + 0.4, color='red', alpha=0.3, label='Max Avg Attention' if i == 0 else "")
    
    ax.set_title(title)
    ax.set_xticks(t_steps[::5]) # Only label every 5th tick to prevent crowding
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

# ==========================================
# 3. MAIN ANALYSIS FUNCTION (Averaged)
# ==========================================
def generate_average_attention_plot(model, model_name, val_loader, save_dir):
    model.eval()
    
    # Storage lists for aggregating the whole dataset
    storage = {'dec_enc': [], 'mass_net': [], 'fric_fric': []}
    all_vel = []

    def get_attn_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # Append batch of attention maps to the list
                storage[name].append(output[1].detach().cpu().numpy())
        return hook

    # Safely check if cross_attn exists
    has_cross_attn = hasattr(model, 'cross_attn')

    # Register Hooks
    hooks = []
    if has_cross_attn:
        hooks.append(model.cross_attn.register_forward_hook(get_attn_hook('dec_enc')))
        
    if hasattr(model, 'mass_attn'):
        hooks.append(model.mass_attn.register_forward_hook(get_attn_hook('mass_net')))
    if hasattr(model, 'fric_attn'):
        hooks.append(model.fric_attn.register_forward_hook(get_attn_hook('fric_fric')))

    print("   -> Running dataset through model individually in batches...")
    with torch.no_grad():
        for batch in val_loader:
            batch_acc, batch_vel, *_ = batch
            
            x_vel = batch_vel.clone().to(device)
            # Ensure proper dimensions
            if x_vel.dim() == 2:
                x_vel = x_vel.unsqueeze(-1)
            
            # Model Pass (Evaluates each individual sample and saves its unique attention!)
            _ = model(x_vel.float()) 
            
            all_vel.append(x_vel.detach().cpu().numpy())

    # Remove hooks
    for h in hooks: 
        h.remove()

    # --- AGGREGATE AND AVERAGE OVER DATASET ---
    if len(all_vel) == 0:
        print(f"Warning: No valid data processed for {model_name}.")
        return

    vel_concat = np.concatenate(all_vel, axis=0) # Shape: (N_samples, Seq_len, 1)
    vel_matrix = vel_concat.squeeze(-1)          # Shape: (N_samples, Seq_len)
    num_samples = vel_matrix.shape[0]
    
    # Average the attention maps across all samples
    avg_attn_maps = {}
    if has_cross_attn and len(storage['dec_enc']) > 0:
        avg_attn_maps['dec_enc'] = np.concatenate(storage['dec_enc'], axis=0).mean(axis=0)
        
    if len(storage['mass_net']) > 0:
        avg_attn_maps['mass_net'] = np.concatenate(storage['mass_net'], axis=0).mean(axis=0)
        
    if len(storage['fric_fric']) > 0:
        avg_attn_maps['fric_fric'] = np.concatenate(storage['fric_fric'], axis=0).mean(axis=0)

    safe_title = str(model_name).replace('$', '').replace('\\', '').replace('{', '').replace('}', '').replace('_', ' ')

    # ------------------------------------------------------------
    # PLOTTING GRID (2 Columns: Velocity Dist & Avg Heatmap)
    # ------------------------------------------------------------
    if has_cross_attn and 'dec_enc' in avg_attn_maps:
        fig = plt.figure(figsize=(18, 35))
        gs = fig.add_gridspec(6, 2, hspace=0.5, wspace=0.3)
        row_offset = 4  
        
        # --- Rows 0 to 3: Cross-Attention Heads 1-4 ---
        cross_weights = avg_attn_maps['dec_enc'] 
        for h_idx in range(min(4, cross_weights.shape[0])):
            h_weights = cross_weights[h_idx]
            head_importance = np.mean(h_weights, axis=0) 
            
            # Left: Velocity Distribution
            ax_v = fig.add_subplot(gs[h_idx, 0])
            plot_emphasized_kinematics(ax_v, vel_matrix, f"Head {h_idx+1}: Velocity Distribution", head_importance, 'green', "m/s")
            
            # Right: Average Heatmap
            ax_m = fig.add_subplot(gs[h_idx, 1])
            sns.heatmap(h_weights.T, ax=ax_m, cmap="magma", cbar=True)
            ax_m.set_title(f"Head {h_idx+1} Avg Alignment\nQuery: q_dec | Key: h_enc")
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)
        row_offset = 0

    # --- Row for Mass Readout ---
    if 'mass_net' in avg_attn_maps:
        mass_weights = avg_attn_maps['mass_net'].flatten()
        ax_mv = fig.add_subplot(gs[row_offset, 0])
        plot_emphasized_kinematics(ax_mv, vel_matrix, "Mass: Velocity Focus", mass_weights, 'green', "m/s")
        
        ax_mm = fig.add_subplot(gs[row_offset, 1])
        sns.heatmap(avg_attn_maps['mass_net'].T, ax=ax_mm, cmap="viridis", cbar=True)
        ax_mm.set_title("Average Mass Spotlight\nQuery: q_m_batch | Key: net_f_est")

    # --- Row for Friction Readout ---
    if 'fric_fric' in avg_attn_maps:
        fric_weights = avg_attn_maps['fric_fric'].flatten()
        ax_fv = fig.add_subplot(gs[row_offset + 1, 0])
        plot_emphasized_kinematics(ax_fv, vel_matrix, "Friction: Velocity Focus", fric_weights, 'green', "m/s")
        
        ax_ff = fig.add_subplot(gs[row_offset + 1, 1])
        sns.heatmap(avg_attn_maps['fric_fric'].T, ax=ax_ff, cmap="viridis", cbar=True)
        ax_ff.set_title("Average Friction Spotlight\nQuery: q_f_batch | Key: fric_f_est")

    plt.suptitle(f"Global Average Attention Analysis: {model_name}\n(Averaged over {num_samples} validation samples)", fontsize=22, y=0.94)
    
    # Save
    save_name = f"attn_global_avg_{safe_title.replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Average Map: {save_path}")
    
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
def main():
    set_seed(42)
    print(f"Using Device: {device}")
    print(f"Starting Global Average Attention Analysis...")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}")
        return
        
    # Load raw dataset once
    df_raw = pd.read_csv(CSV_PATH)
    if 'gt_fric_force' in df_raw.columns:
        df_raw['gt_fric_force'] = df_raw['gt_fric_force'].apply(clean_force_col)
    
    for model_name, csv_path in MODELS_TO_COMPARE.items():
        print(f"\nProcessing Model: {model_name}")
        
        # 1. Deduce checkpoint paths
        ckpt_dir = os.path.dirname(csv_path)
        model_weights_path = os.path.join(ckpt_dir, "transformer_epoch1000.pth") 
        config_path = os.path.join(ckpt_dir, "config.json")
        
        if not os.path.exists(config_path):
            print(f"Config not found at {config_path}. Skipping.")
            continue
            
        if not os.path.exists(model_weights_path):
            print(f"Weights not found at {model_weights_path}. Skipping.")
            continue

        # 2. Load model-specific config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 3. Create Dataloader tailored to this config's bounds
        _, val_loader, seq_len, _, _ = create_dataloaders(
            df_raw, 
            batch_size=BATCH_SIZE, 
            m_seen_min=config.get('m_seen_min', 0.2), 
            m_seen_max=config.get('m_seen_max', 2.0)
        )
            
        # 4. Initialize Model Architecture dynamically
        model = PhysicsTransformerEstimator(
            input_dim=1, 
            d_model=config.get('d_model', 64), 
            nhead=4, 
            num_encoder_layers=config.get('num_enc', 4), 
            seq_len=config.get('seq_len', seq_len), 
            dropout=config.get('dropout', 0.1), 
            sharpness=config.get('sharpness', 1.0), 
            cross_sharpness=config.get('cross_sharpness', 5.0), 
            m_sharpness=config.get('m_sharpness', 5.0), 
            mu_sharpness=config.get('mu_sharpness', 5.0), 
            version=config.get('transformer_ver', 5),
            max_mass_scale=config.get('last_layer_ms', 2.0), 
            max_mu_scale=config.get('last_layer_mus', 1.0)
        ).to(device)

        # 5. Load the weights into the initialized model
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        
        # 6. Generate the global average plot
        generate_average_attention_plot(model, model_name, val_loader, OUTPUT_DIR)

if __name__ == "__main__":
    main()