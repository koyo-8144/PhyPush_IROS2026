import os
import glob
import json
import csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from models import PhysicsTransformerEstimator
from dataset import create_dataloaders
from utils import set_seed, clean_force_col
from configs import M_SEEN_MAX, M_SEEN_MIN, MU_SEEN_MAX, MU_SEEN_MIN, M_UNSEEN_MAX, MU_UNSEEN_MAX, GLOBAL_M_RANGE, GLOBAL_MU_RANGE, GLOBAL_FRIC_RANGE, REAL_M_RANGE, REAL_MU_RANGE, REAL_FRIC_RANGE, INCLUDE_UNSEEN, CSV_PATH

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
PLOT_SHOW = False
SMOOTHING_WINDOW_SIZE = 3
TOP_NUM = 5

CHECKPOINT_DIR = "./results/checkpoints/from_20260618/20260618_191529/pinn_pcri-L1_p5c10.0" 

WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "transformer_epoch1000.pth")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.json")
EVAL_CHECKPOINT_DIR = CHECKPOINT_DIR
OFFLINE_DATA_DIR = "/home/psxkf4/panda_phypush/csv_data/offline_collection"

# --- Load Config to set dynamic ranges ---
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
else:
    print(f"Error: config.json not found at {CONFIG_PATH}")
    config = {}



# ==========================================
# EXPERIMENT FUNCTIONS
# ==========================================
def calculate_metrics(gt, est, range_val):
    gt = np.array(gt).flatten()
    est = np.array(est).flatten()
    
    raw_errors = est - gt
    mean_err = np.mean(raw_errors)
    mean_err_pct = (mean_err / range_val) * 100 if range_val > 0 else 0
    
    abs_errors = np.abs(raw_errors)
    
    mae = np.mean(abs_errors)
    nmae_pct = (mae / range_val) * 100 if range_val > 0 else 0
    rmse = np.sqrt(np.mean(abs_errors**2))
    nrmse_pct = (rmse / range_val) * 100 if range_val > 0 else 0
    
    err_std_dev = np.std(abs_errors)
    err_std_pct = (err_std_dev / range_val) * 100 if range_val > 0 else 0
    
    est_std_dev = np.std(est)
    
    denominator = (np.abs(gt) + np.abs(est)) / 2
    smape_pct = np.mean(abs_errors / np.maximum(denominator, 1e-8)) * 100
    
    return {
        "mean_err": mean_err,           
        "mean_err_pct": mean_err_pct,   
        "mae": mae, 
        "nmae_pct": nmae_pct, 
        "nrmse_pct": nrmse_pct, 
        "smape_pct": smape_pct,
        "std_dev": err_std_dev,
        "std_pct": err_std_pct,
        "est_std_dev": est_std_dev
    }

def _plot_experiment_1_manifold(mass_gt, mu_gt, mass_est, mu_est, domain_label, m_range, mu_range):
    m_gt_np, mu_gt_np = mass_gt.cpu().numpy(), mu_gt.cpu().numpy()
    m_est_np, mu_est_np = mass_est.cpu().numpy(), mu_est.cpu().numpy()
    
    m_stats = calculate_metrics(m_gt_np, m_est_np, m_range)
    mu_stats = calculate_metrics(mu_gt_np, mu_est_np, mu_range)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sc1 = axes[0].scatter(m_gt_np, mu_gt_np, c=np.abs(m_est_np - m_gt_np), cmap='viridis', vmin=0, vmax=0.5)
    plt.colorbar(sc1, ax=axes[0]).set_label('Abs Mass Error [kg]')
    axes[0].set_title(f'Mass Estimation Error\nMAE: {m_stats["mae"]:.4f} kg | STD: ±{m_stats["std_pct"]:.2f}%')
    
    sc2 = axes[1].scatter(m_gt_np, mu_gt_np, c=np.abs(mu_est_np - mu_gt_np), cmap='plasma', vmin=0, vmax=0.1)
    plt.colorbar(sc2, ax=axes[1]).set_label('Abs Friction Coeff Error')
    axes[1].set_title(f'Friction Coeff Error\nMAE: {mu_stats["mae"]:.4f} | STD: ±{mu_stats["std_pct"]:.2f}%')
    
    plt.suptitle(f"Exp 1: Manifold Interpolation - {domain_label}")
    plt.savefig(os.path.join(EVAL_CHECKPOINT_DIR, f"exp1_manifold_{domain_label}.png"), dpi=100)
    
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)
    return {"mass": m_stats, "mu": mu_stats}

def _plot_experiment_2_consistency(mass_est, mu_est, fric_f_gt, fz_robot_tensor, domain_label, seq_len):
    m_est_val = mass_est.cpu().numpy()
    mu_est_val = mu_est.cpu().numpy()
    
    mean_robot_f_z = fz_robot_tensor[:, seq_len//2:].mean(dim=1).cpu().numpy()
    mean_fric_gt = fric_f_gt[:, seq_len//2:].mean(dim=1).cpu().numpy()

    calc_normal_force = np.maximum((m_est_val * 9.81) - mean_robot_f_z, 0.0)
    f_pred = mu_est_val * calc_normal_force
    
    r2 = r2_score(mean_fric_gt, f_pred) if len(mean_fric_gt) > 1 else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].scatter(mean_fric_gt, f_pred, c=mu_est_val, cmap='plasma', alpha=0.6)
    axes[1].scatter(mean_fric_gt, f_pred, c=m_est_val, cmap='viridis', alpha=0.6)
    
    for ax in axes:
        max_val = max(mean_fric_gt.max(), f_pred.max(), 1)
        ax.plot([0, max_val], [0, max_val], 'r--')
        ax.set_xlabel("GT Friction Force (N)")
        ax.set_ylabel("Predicted Friction Force (N)")
    
    plt.suptitle(f"Exp 2: Friction Consistency - {domain_label}\nR²: {r2:.4f}")
    plt.savefig(os.path.join(EVAL_CHECKPOINT_DIR, f"exp2_consistency_{domain_label}.png"), dpi=100)
    
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)
    return {"fric_r2": r2}

def _plot_experiment_3_consistency(mass_est, rhs_acc_tensor, net_f_gt, domain_label, seq_len):
    m_est_val = mass_est.cpu().numpy()
    
    mean_acc_x = rhs_acc_tensor[:, :seq_len//2].mean(dim=1).cpu().numpy()
    mean_net_f_gt = net_f_gt[:, :seq_len//2].mean(dim=1).cpu().numpy()

    f_pred = m_est_val * mean_acc_x
    valid = np.abs(mean_acc_x) > 0.01
    r2 = r2_score(mean_net_f_gt[valid], f_pred[valid]) if np.sum(valid) > 5 else 0.0
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(mean_net_f_gt, f_pred, c=m_est_val, cmap='viridis', alpha=0.6)
    
    min_val = min(mean_net_f_gt.min(), f_pred.min())
    max_val = max(mean_net_f_gt.max(), f_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("GT Net Force (N)")
    plt.ylabel("Predicted Net Force (N)")
    plt.title(f"Exp 3: Net Force Consistency - {domain_label}\nR²: {r2:.4f}")
    plt.savefig(os.path.join(EVAL_CHECKPOINT_DIR, f"exp3_dynamics_{domain_label}.png"), dpi=100)
    
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)
    return {"net_r2": r2}

def log_results_to_csv(m1, m2, m3, domain_label, is_first=False):
    file_path = os.path.join(EVAL_CHECKPOINT_DIR, "domain_evaluation_summary.csv")
    
    entry = {
        "domain": domain_label,
        "mass_mean_err": round(m1["mass"]["mean_err"], 4),
        "mass_mean_err_pct": round(m1["mass"]["mean_err_pct"], 2),
        "mass_nmae_pct": round(m1["mass"]["nmae_pct"], 2),
        "mass_nrmse_pct": round(m1["mass"]["nrmse_pct"], 2),
        "mass_smape_pct": round(m1["mass"]["smape_pct"], 2),
        "mass_std_dev": round(m1["mass"]["std_dev"], 4),
        "mass_est_std_dev": round(m1["mass"]["est_std_dev"], 4),
        "mass_std_pct": round(m1["mass"]["std_pct"], 2),
        "mu_mean_err": round(m1["mu"]["mean_err"], 4),
        "mu_mean_err_pct": round(m1["mu"]["mean_err_pct"], 2),
        "mu_nmae_pct": round(m1["mu"]["nmae_pct"], 2),
        "mu_nrmse_pct": round(m1["mu"]["nrmse_pct"], 2),
        "mu_smape_pct": round(m1["mu"]["smape_pct"], 2),
        "mu_std_dev": round(m1["mu"]["std_dev"], 4),
        "mu_est_std_dev": round(m1["mu"]["est_std_dev"], 4),
        "mu_std_pct": round(m1["mu"]["std_pct"], 2),
        "fric_r2": round(m2["fric_r2"], 4),
        "net_f_r2": round(m3["net_r2"], 4)
    }
    
    mode = 'w' if is_first else 'a'
    with open(file_path, mode=mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if mode == 'w': writer.writeheader()
        writer.writerow(entry)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    if not config:
        print("Stopping execution due to missing configuration.")
        return

    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    if 'gt_fric_force' in df.columns:
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col)

    # Dataloader using the constants directly
    _, _, seq_len, df, _ = create_dataloaders(
        df, config['batch_size'], M_SEEN_MIN, M_SEEN_MAX
    )
    
    acc_cols = sorted([c for c in df.columns if "input_acc_" in c], key=lambda x: int(x.split('_')[-1]))
    vel_cols = sorted([c for c in df.columns if "input_vel_" in c], key=lambda x: int(x.split('_')[-1]))

    model = PhysicsTransformerEstimator(
        input_dim=1, 
        d_model=config['d_model'], 
        nhead=4, 
        num_encoder_layers=config['num_enc'], 
        seq_len=config.get('seq_len', seq_len), 
        dropout=config['dropout'], 
        sharpness=config['sharpness'], 
        cross_sharpness=config['cross_sharpness'], 
        m_sharpness=config['m_sharpness'], 
        mu_sharpness=config['mu_sharpness'], 
        version=config['transformer_ver'],
        max_mass_scale=config['last_layer_ms'], 
        max_mu_scale=config['last_layer_mus']
    ).to(device)

    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        return

    if not INCLUDE_UNSEEN:
        domain_choices = ['m_seen_light', 'm_seen_middle', 'm_seen_heavy']
    else:
        domain_choices = [
            'm_seen_mu_seen', 'm_seen_light', 'm_seen_middle', 'm_seen_heavy',
            'm_over', 'm_under', 'mu_over', 'mu_under',
            'm_over_mu_over', 'm_under_mu_under', 'm_over_mu_under', 'm_under_mu_over'
        ]

    # ==========================================
    # EVALUATION LOOP: SIMULATION DATA
    # ==========================================
    print("\n--- Starting Simulation Evaluation ---")
    for i, target_domain in enumerate(domain_choices):
        base_domain = 'm_seen_mu_seen' if 'm_seen' in target_domain else target_domain
        
        df_domain = df[(df['domain'] == base_domain) & 
                      (df['start_t'] + seq_len <= 100)].copy()
        
        if target_domain == 'm_seen_light':
            df_domain = df_domain[df_domain['gt_mass'] < 0.8]
        elif target_domain == 'm_seen_middle':
            df_domain = df_domain[(df_domain['gt_mass'] >= 0.8) & (df_domain['gt_mass'] < 1.4)]
        elif target_domain == 'm_seen_heavy':
            df_domain = df_domain[df_domain['gt_mass'] >= 1.4]
        
        if len(df_domain) == 0: 
            print(f">>> Skipping Domain: {target_domain} (No samples fit criteria)")
            continue
        
        print(f"\n>>> Processing Domain: {target_domain} ({len(df_domain)} samples)")

        X_acc = torch.tensor(df_domain[acc_cols].values.reshape(-1, seq_len, 1)).float().to(device)
        X_vel = torch.tensor(df_domain[vel_cols].values.reshape(-1, seq_len, 1)).float().to(device)
        y_gt = torch.tensor(df_domain[['gt_mass', 'gt_mu']].values).float().to(device)

        robot_fz_list, rhs_acc_list = [], []
        lhs_net_f_list, table_fz_list = [], []
        
        for _, row in df_domain.iterrows():
            st = int(row['start_t'])
            window = range(st, st + seq_len) 
            robot_fz_list.append([row[f"pinn_robot_wrench_t{t}_ax5"] for t in window])
            rhs_acc_list.append([row[f"pinn_RHS_acc_t{t}_ax3"] for t in window])
            lhs_net_f_list.append([row[f"pinn_LHS_wrench_t{t}_ax3"] for t in window])
            table_fz_list.append([row[f"pinn_table_wrench_t{t}_ax5"] for t in window])
        
        fz_robot_tensor = torch.tensor(np.array(robot_fz_list)).float().to(device)
        rhs_acc_tensor = torch.tensor(np.array(rhs_acc_list)).float().to(device)
        net_f_gt_tensor = torch.tensor(np.array(lhs_net_f_list)).float().to(device)
        
        normal_f_gt = torch.abs(torch.tensor(np.array(table_fz_list)).float().to(device))
        fric_f_gt_tensor = y_gt[:, 1].unsqueeze(1) * normal_f_gt

        model.eval()
        with torch.no_grad():
            preds, _, _ = model(X_vel)
            mass_est, mu_est = preds[:, 0], preds[:, 1]
            mass_gt, mu_gt = y_gt[:, 0], y_gt[:, 1]

        m1 = _plot_experiment_1_manifold(mass_gt, mu_gt, mass_est, mu_est, target_domain, GLOBAL_M_RANGE, GLOBAL_MU_RANGE)
        m2 = _plot_experiment_2_consistency(mass_est, mu_est, fric_f_gt_tensor, fz_robot_tensor, target_domain, seq_len)
        m3 = _plot_experiment_3_consistency(mass_est, rhs_acc_tensor, net_f_gt_tensor, target_domain, seq_len)
        
        log_results_to_csv(m1, m2, m3, target_domain, is_first=(i == 0))

    print(f"\nDone. Summary saved in {EVAL_CHECKPOINT_DIR}")

    # ==========================================
    # EVALUATION LOOP: REAL DATA
    # ==========================================
    real_eval_csv_path = os.path.join(EVAL_CHECKPOINT_DIR, "real_evaluation_summary.csv")
    real_detailed_csv_path = os.path.join(EVAL_CHECKPOINT_DIR, "real_detailed_inference.csv")

    all_runs = []
    if os.path.exists(OFFLINE_DATA_DIR):
        condition_folders = [f.path for f in os.scandir(OFFLINE_DATA_DIR) if f.is_dir()]
        
        model.eval()
        with torch.no_grad():
            for folder in condition_folders:
                condition_name = os.path.basename(folder)
                gt_path = os.path.join(folder, "ground_truth.csv")
                
                if not os.path.exists(gt_path):
                    continue
                    
                gt_df = pd.read_csv(gt_path)
                gt_dict = dict(zip(gt_df['Parameter'], gt_df['Value']))
                
                m_gt = float(gt_dict.get('M_GT', -1))
                raw_mu = gt_dict.get('MU_GT', 'None')
                mu_gt = float(raw_mu) if raw_mu not in ['None', 'N/A', 'NaN'] else np.nan

                csv_pattern = os.path.join(folder, "*_100steps.csv")
                for file_path in glob.glob(csv_pattern):
                    run_name = os.path.basename(file_path).split('_100steps')[0]
                    df_real = pd.read_csv(file_path)

                    df_real['v_y_smoothed'] = df_real['v_y'].rolling(window=SMOOTHING_WINDOW_SIZE, min_periods=1).mean()
                    df_inf = df_real[df_real['is_inference_region'] == 1].copy()
                    
                    if len(df_inf) != seq_len: 
                        continue
                        
                    vel_data = df_inf['v_y_smoothed'].values
                    X_vel_real = torch.tensor(vel_data).unsqueeze(0).unsqueeze(-1).float().to(device)
                    
                    preds, _, _ = model(X_vel_real)
                    m_est = preds[0, 0].item()
                    mu_est = preds[0, 1].item()
                    
                    fric_f_gt = m_gt * mu_gt * G if not np.isnan(mu_gt) else np.nan
                    fric_f_est = m_est * mu_est * G

                    all_runs.append({
                        "domain": condition_name,
                        "run": run_name,
                        "m_gt": m_gt,
                        "m_est": m_est,
                        "mu_gt": mu_gt,
                        "mu_est": mu_est,
                        "fric_f_gt": fric_f_gt,      
                        "fric_f_est": fric_f_est,    
                        "abs_mass_err": abs(m_est - m_gt),
                        "abs_mu_err": abs(mu_est - mu_gt) if not np.isnan(mu_gt) else np.nan,
                        "abs_fric_f_err": abs(fric_f_est - fric_f_gt) if not np.isnan(fric_f_gt) else np.nan
                    })
    else:
        print(f"Warning: Offline data directory not found at {OFFLINE_DATA_DIR}. Skipping Real World Inference.")

    df_runs = pd.DataFrame(all_runs)
    if not df_runs.empty:
        df_runs.to_csv(real_detailed_csv_path, index=False)
        
        df_sorted_mass = df_runs.sort_values(by=['domain', 'abs_mass_err'], ascending=True)
        df_best_mass_runs = df_sorted_mass.groupby('domain').head(TOP_NUM).copy()

        df_sorted_mu = df_runs.sort_values(by=['domain', 'abs_mu_err'], ascending=True)
        df_best_mu_runs = df_sorted_mu.groupby('domain').head(TOP_NUM).copy()
        
        df_sorted_fric = df_runs.sort_values(by=['domain', 'abs_fric_f_err'], ascending=True)
        df_best_fric_runs = df_sorted_fric.groupby('domain').head(TOP_NUM).copy()

        with open(real_eval_csv_path, mode='w', newline='') as f:
            fieldnames = [
                "domain", 
                "mass_mean_err", "mass_mean_err_pct", "mass_nmae_pct", "mass_nrmse_pct", "mass_smape_pct", "mass_std_dev", "mass_est_std_dev", "mass_std_pct",
                "mu_mean_err", "mu_mean_err_pct", "mu_nmae_pct", "mu_nrmse_pct", "mu_smape_pct", "mu_std_dev", "mu_est_std_dev", "mu_std_pct",
                "fric_f_mean_err", "fric_f_mean_err_pct", "fric_f_nmae_pct", "fric_f_nrmse_pct", "fric_f_smape_pct", "fric_f_std_dev", "fric_f_est_std_dev", "fric_f_std_pct",
                
                "mass_mean_err_pct_real_range", "mass_nmae_pct_real_range", "mass_nrmse_pct_real_range", "mass_std_pct_real_range",
                "mu_mean_err_pct_real_range", "mu_nmae_pct_real_range", "mu_nrmse_pct_real_range", "mu_std_pct_real_range",
                "fric_f_mean_err_pct_real_range", "fric_f_nmae_pct_real_range", "fric_f_nrmse_pct_real_range", "fric_f_std_pct_real_range",
                
                "best_mass_mean_err", "best_mass_mean_err_pct", "best_mass_nmae_pct", "best_mass_nrmse_pct", "best_mass_smape_pct", "best_mass_std_dev", "best_mass_est_std_dev", "best_mass_est_raw",  
                "best_mu_mean_err", "best_mu_mean_err_pct", "best_mu_nmae_pct", "best_mu_nrmse_pct", "best_mu_smape_pct", "best_mu_std_dev", "best_mu_est_std_dev", "best_mu_est_raw",    
                "best_fric_f_mean_err", "best_fric_f_mean_err_pct", "best_fric_f_nmae_pct", "best_fric_f_nrmse_pct", "best_fric_f_smape_pct", "best_fric_f_std_dev", "best_fric_f_est_std_dev", "best_fric_f_est_raw", 
                
                "best_mass_nmae_pct_real_range", "best_mass_nrmse_pct_real_range",
                "best_mu_nmae_pct_real_range", "best_mu_nrmse_pct_real_range",
                "best_fric_f_nmae_pct_real_range", "best_fric_f_nrmse_pct_real_range"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            domains = df_runs['domain'].unique()
            
            for dom in domains:
                dom_data_all = df_runs[df_runs['domain'] == dom]
                m_gt_all, m_est_all = dom_data_all['m_gt'].values, dom_data_all['m_est'].values
                mu_gt_all, mu_est_all = dom_data_all['mu_gt'].values, dom_data_all['mu_est'].values
                f_gt_all, f_est_all = dom_data_all['fric_f_gt'].values, dom_data_all['fric_f_est'].values
                
                dom_data_best_mass = df_best_mass_runs[df_best_mass_runs['domain'] == dom]
                m_gt_best, m_est_best = dom_data_best_mass['m_gt'].values, dom_data_best_mass['m_est'].values
                
                dom_data_best_mu = df_best_mu_runs[df_best_mu_runs['domain'] == dom]
                mu_gt_best, mu_est_best = dom_data_best_mu['mu_gt'].values, dom_data_best_mu['mu_est'].values
                
                dom_data_best_fric = df_best_fric_runs[df_best_fric_runs['domain'] == dom]
                f_gt_best, f_est_best = dom_data_best_fric['fric_f_gt'].values, dom_data_best_fric['fric_f_est'].values
                
                m_stats_avg_global = calculate_metrics(m_gt_all, m_est_all, GLOBAL_M_RANGE)
                m_stats_avg_real = calculate_metrics(m_gt_all, m_est_all, REAL_M_RANGE)
                m_stats_best_global = calculate_metrics(m_gt_best, m_est_best, GLOBAL_M_RANGE)
                m_stats_best_real = calculate_metrics(m_gt_best, m_est_best, REAL_M_RANGE)
                
                empty_stats = {"mean_err": np.nan, "mean_err_pct": np.nan, "nmae_pct": np.nan, "nrmse_pct": np.nan, "smape_pct": np.nan, "std_dev": np.nan, "std_pct": np.nan, "est_std_dev": np.nan}
                
                if not np.isnan(mu_gt_all).all():
                    mu_stats_avg_global = calculate_metrics(mu_gt_all, mu_est_all, GLOBAL_MU_RANGE)
                    mu_stats_avg_real = calculate_metrics(mu_gt_all, mu_est_all, REAL_MU_RANGE)
                    mu_stats_best_global = calculate_metrics(mu_gt_best, mu_est_best, GLOBAL_MU_RANGE)
                    mu_stats_best_real = calculate_metrics(mu_gt_best, mu_est_best, REAL_MU_RANGE)
                    
                    f_stats_avg_global = calculate_metrics(f_gt_all, f_est_all, GLOBAL_FRIC_RANGE)
                    f_stats_avg_real = calculate_metrics(f_gt_all, f_est_all, REAL_FRIC_RANGE)
                    f_stats_best_global = calculate_metrics(f_gt_best, f_est_best, GLOBAL_FRIC_RANGE)
                    f_stats_best_real = calculate_metrics(f_gt_best, f_est_best, REAL_FRIC_RANGE)
                else:
                    mu_stats_avg_global = mu_stats_avg_real = mu_stats_best_global = mu_stats_best_real = empty_stats
                    f_stats_avg_global = f_stats_avg_real = f_stats_best_global = f_stats_best_real = empty_stats
                
                writer.writerow({
                    "domain": dom,
                    "mass_mean_err": round(m_stats_avg_global["mean_err"], 4),
                    "mass_mean_err_pct": round(m_stats_avg_global["mean_err_pct"], 2),
                    "mass_nmae_pct": round(m_stats_avg_global["nmae_pct"], 2),
                    "mass_nrmse_pct": round(m_stats_avg_global["nrmse_pct"], 2),
                    "mass_smape_pct": round(m_stats_avg_global["smape_pct"], 2),
                    "mass_std_dev": round(m_stats_avg_global["std_dev"], 4), 
                    "mass_est_std_dev": round(m_stats_avg_global["est_std_dev"], 4), 
                    "mass_std_pct": round(m_stats_avg_global["std_pct"], 2),
                    
                    "mu_mean_err": round(mu_stats_avg_global["mean_err"], 4) if not np.isnan(mu_stats_avg_global["mean_err"]) else "NaN",
                    "mu_mean_err_pct": round(mu_stats_avg_global["mean_err_pct"], 2) if not np.isnan(mu_stats_avg_global["mean_err_pct"]) else "NaN",
                    "mu_nmae_pct": round(mu_stats_avg_global["nmae_pct"], 2) if not np.isnan(mu_stats_avg_global["nmae_pct"]) else "NaN",
                    "mu_nrmse_pct": round(mu_stats_avg_global["nrmse_pct"], 2) if not np.isnan(mu_stats_avg_global["nrmse_pct"]) else "NaN",
                    "mu_smape_pct": round(mu_stats_avg_global["smape_pct"], 2) if not np.isnan(mu_stats_avg_global["smape_pct"]) else "NaN",
                    "mu_std_dev": round(mu_stats_avg_global["std_dev"], 4) if not np.isnan(mu_stats_avg_global["std_dev"]) else "NaN", 
                    "mu_est_std_dev": round(mu_stats_avg_global["est_std_dev"], 4) if not np.isnan(mu_stats_avg_global["est_std_dev"]) else "NaN", 
                    "mu_std_pct": round(mu_stats_avg_global["std_pct"], 2) if not np.isnan(mu_stats_avg_global["std_pct"]) else "NaN",

                    "fric_f_mean_err": round(f_stats_avg_global["mean_err"], 4) if not np.isnan(f_stats_avg_global["mean_err"]) else "NaN",
                    "fric_f_mean_err_pct": round(f_stats_avg_global["mean_err_pct"], 2) if not np.isnan(f_stats_avg_global["mean_err_pct"]) else "NaN",
                    "fric_f_nmae_pct": round(f_stats_avg_global["nmae_pct"], 2) if not np.isnan(f_stats_avg_global["nmae_pct"]) else "NaN",
                    "fric_f_nrmse_pct": round(f_stats_avg_global["nrmse_pct"], 2) if not np.isnan(f_stats_avg_global["nrmse_pct"]) else "NaN",
                    "fric_f_smape_pct": round(f_stats_avg_global["smape_pct"], 2) if not np.isnan(f_stats_avg_global["smape_pct"]) else "NaN",
                    "fric_f_std_dev": round(f_stats_avg_global["std_dev"], 4) if not np.isnan(f_stats_avg_global["std_dev"]) else "NaN", 
                    "fric_f_est_std_dev": round(f_stats_avg_global["est_std_dev"], 4) if not np.isnan(f_stats_avg_global["est_std_dev"]) else "NaN", 
                    "fric_f_std_pct": round(f_stats_avg_global["std_pct"], 2) if not np.isnan(f_stats_avg_global["std_pct"]) else "NaN",
                    
                    "mass_mean_err_pct_real_range": round(m_stats_avg_real["mean_err_pct"], 2),
                    "mass_nmae_pct_real_range": round(m_stats_avg_real["nmae_pct"], 2),
                    "mass_nrmse_pct_real_range": round(m_stats_avg_real["nrmse_pct"], 2),
                    "mass_std_pct_real_range": round(m_stats_avg_real["std_pct"], 2),
                    
                    "mu_mean_err_pct_real_range": round(mu_stats_avg_real["mean_err_pct"], 2) if not np.isnan(mu_stats_avg_real["mean_err_pct"]) else "NaN",
                    "mu_nmae_pct_real_range": round(mu_stats_avg_real["nmae_pct"], 2) if not np.isnan(mu_stats_avg_real["nmae_pct"]) else "NaN",
                    "mu_nrmse_pct_real_range": round(mu_stats_avg_real["nrmse_pct"], 2) if not np.isnan(mu_stats_avg_real["nrmse_pct"]) else "NaN",
                    "mu_std_pct_real_range": round(mu_stats_avg_real["std_pct"], 2) if not np.isnan(mu_stats_avg_real["std_pct"]) else "NaN",

                    "fric_f_mean_err_pct_real_range": round(f_stats_avg_real["mean_err_pct"], 2) if not np.isnan(f_stats_avg_real["mean_err_pct"]) else "NaN",
                    "fric_f_nmae_pct_real_range": round(f_stats_avg_real["nmae_pct"], 2) if not np.isnan(f_stats_avg_real["nmae_pct"]) else "NaN",
                    "fric_f_nrmse_pct_real_range": round(f_stats_avg_real["nrmse_pct"], 2) if not np.isnan(f_stats_avg_real["nrmse_pct"]) else "NaN",
                    "fric_f_std_pct_real_range": round(f_stats_avg_real["std_pct"], 2) if not np.isnan(f_stats_avg_real["std_pct"]) else "NaN",
                    
                    "best_mass_mean_err": round(m_stats_best_global["mean_err"], 4),
                    "best_mass_mean_err_pct": round(m_stats_best_global["mean_err_pct"], 2),
                    "best_mass_nmae_pct": round(m_stats_best_global["nmae_pct"], 2),
                    "best_mass_nrmse_pct": round(m_stats_best_global["nrmse_pct"], 2),
                    "best_mass_smape_pct": round(m_stats_best_global["smape_pct"], 2),
                    "best_mass_std_dev": round(m_stats_best_global["std_dev"], 4), 
                    "best_mass_est_std_dev": round(m_stats_best_global["est_std_dev"], 4), 
                    "best_mass_est_raw": round(np.mean(m_est_best), 4), 
                    
                    "best_mu_mean_err": round(mu_stats_best_global["mean_err"], 4) if not np.isnan(mu_stats_best_global["mean_err"]) else "NaN",
                    "best_mu_mean_err_pct": round(mu_stats_best_global["mean_err_pct"], 2) if not np.isnan(mu_stats_best_global["mean_err_pct"]) else "NaN",
                    "best_mu_nmae_pct": round(mu_stats_best_global["nmae_pct"], 2) if not np.isnan(mu_stats_best_global["nmae_pct"]) else "NaN",
                    "best_mu_nrmse_pct": round(mu_stats_best_global["nrmse_pct"], 2) if not np.isnan(mu_stats_best_global["nrmse_pct"]) else "NaN",
                    "best_mu_smape_pct": round(mu_stats_best_global["smape_pct"], 2) if not np.isnan(mu_stats_best_global["smape_pct"]) else "NaN",
                    "best_mu_std_dev": round(mu_stats_best_global["std_dev"], 4) if not np.isnan(mu_stats_best_global["std_dev"]) else "NaN", 
                    "best_mu_est_std_dev": round(mu_stats_best_global["est_std_dev"], 4) if not np.isnan(mu_stats_best_global["est_std_dev"]) else "NaN", 
                    "best_mu_est_raw": round(np.mean(mu_est_best), 4) if len(mu_est_best) > 0 and not np.isnan(mu_est_best).all() else "NaN", 
                    
                    "best_fric_f_mean_err": round(f_stats_best_global["mean_err"], 4) if not np.isnan(f_stats_best_global["mean_err"]) else "NaN",
                    "best_fric_f_mean_err_pct": round(f_stats_best_global["mean_err_pct"], 2) if not np.isnan(f_stats_best_global["mean_err_pct"]) else "NaN",
                    "best_fric_f_nmae_pct": round(f_stats_best_global["nmae_pct"], 2) if not np.isnan(f_stats_best_global["nmae_pct"]) else "NaN",
                    "best_fric_f_nrmse_pct": round(f_stats_best_global["nrmse_pct"], 2) if not np.isnan(f_stats_best_global["nrmse_pct"]) else "NaN",
                    "best_fric_f_smape_pct": round(f_stats_best_global["smape_pct"], 2) if not np.isnan(f_stats_best_global["smape_pct"]) else "NaN",
                    "best_fric_f_std_dev": round(f_stats_best_global["std_dev"], 4) if not np.isnan(f_stats_best_global["std_dev"]) else "NaN",
                    "best_fric_f_est_std_dev": round(f_stats_best_global["est_std_dev"], 4) if not np.isnan(f_stats_best_global["est_std_dev"]) else "NaN",
                    "best_fric_f_est_raw": round(np.mean(f_est_best), 4) if len(f_est_best) > 0 and not np.isnan(f_est_best).all() else "NaN", 

                    "best_mass_nmae_pct_real_range": round(m_stats_best_real["nmae_pct"], 2),
                    "best_mass_nrmse_pct_real_range": round(m_stats_best_real["nrmse_pct"], 2),
                    "best_mu_nmae_pct_real_range": round(mu_stats_best_real["nmae_pct"], 2) if not np.isnan(mu_stats_best_real["nmae_pct"]) else "NaN",
                    "best_mu_nrmse_pct_real_range": round(mu_stats_best_real["nrmse_pct"], 2) if not np.isnan(mu_stats_best_real["nrmse_pct"]) else "NaN",
                    "best_fric_f_nmae_pct_real_range": round(f_stats_best_real["nmae_pct"], 2) if not np.isnan(f_stats_best_real["nmae_pct"]) else "NaN",
                    "best_fric_f_nrmse_pct_real_range": round(f_stats_best_real["nrmse_pct"], 2) if not np.isnan(f_stats_best_real["nrmse_pct"]) else "NaN"
                })
                
        print(f"Real-world aggregated summary (with Friction Force & Raw Estimates) saved to: {real_eval_csv_path}")

        fig, ax = plt.subplots(figsize=(9, 8))
        
        sc = ax.scatter(df_runs['m_gt'], df_runs['m_est'], 
                        c=np.abs(df_runs['m_est'] - df_runs['m_gt']), 
                        cmap='viridis', vmin=0, vmax=0.5, s=40, alpha=0.5, label='All Runs')
                        
        ax.scatter(df_best_mass_runs['m_gt'], df_best_mass_runs['m_est'], 
                   edgecolors='red', facecolors='none', linewidths=1.5, s=90, label=f'Top {TOP_NUM} Runs (Mass)')
                   
        plt.colorbar(sc, ax=ax).set_label('Absolute Error [kg]')

        min_val = min(df_runs['m_gt'].min(), df_runs['m_est'].min())
        max_val = max(df_runs['m_gt'].max(), df_runs['m_est'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Estimation')

        ax.set_xlabel('Ground Truth Mass (kg)')
        ax.set_ylabel('Estimated Mass (kg)')
        
        global_avg_m_stats = calculate_metrics(df_runs['m_gt'].values, df_runs['m_est'].values, GLOBAL_M_RANGE)
        global_best_m_stats = calculate_metrics(df_best_mass_runs['m_gt'].values, df_best_mass_runs['m_est'].values, GLOBAL_M_RANGE)
        
        ax.set_title(f"Real-World Mass Estimation\n"
                     f"AVERAGE (All pushes) - Mean Err: {global_avg_m_stats['mean_err']:+.4f} kg | nMAE: {global_avg_m_stats['nmae_pct']:.2f}% | STD: ±{global_avg_m_stats['std_pct']:.2f}%\n"
                     f"BEST {TOP_NUM} PUSHES - Mean Err: {global_best_m_stats['mean_err']:+.4f} kg | nMAE: {global_best_m_stats['nmae_pct']:.2f}%")
        
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(EVAL_CHECKPOINT_DIR, "exp1_real_world_mass_accuracy.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Real-world plot saved to: {plot_path}")
        
        if PLOT_SHOW:
            plt.show()
        plt.close(fig)
    else:
        print("No valid 60-step real-world data found to process.")

if __name__ == "__main__":
    main()