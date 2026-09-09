import os
import random
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Core framework imports
from dataset import create_dataloaders, load_dataset_csv
from utils import clean_force_col
from configs import (M_SEEN_MIN, M_SEEN_MAX, MU_SEEN_MIN, MU_SEEN_MAX, 
                     M_UNSEEN_MAX, MU_UNSEEN_MAX, INCLUDE_UNSEEN, FRAME_MODE)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

PLOT_SHOW = False
checkpoint_dir = "./results/checkpoints"
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ==========================================
# EXPERIMENT METRICS HELPER FUNCTIONS
# ==========================================
def calculate_metrics(gt, est, range_val):
    gt = gt.flatten()
    est = est.flatten()
    raw_errors = est - gt
    mean_err = np.mean(raw_errors)
    mean_err_pct = (mean_err / range_val) * 100 if range_val > 0 else 0
    abs_errors = np.abs(raw_errors)
    mae = np.mean(abs_errors)
    nmae_pct = (mae / range_val) * 100 if range_val > 0 else 0
    rmse = np.sqrt(np.mean(abs_errors**2))
    nrmse_pct = (rmse / range_val) * 100 if range_val > 0 else 0
    std_dev = np.std(abs_errors)
    std_pct = (std_dev / range_val) * 100 if range_val > 0 else 0
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
        "std_dev": std_dev,             
        "std_pct": std_pct,
        "est_std_dev": est_std_dev      
    }

def _run_exp_1(mass_gt, mu_gt, mass_est, mu_est, m_range, mu_range):
    m_stats = calculate_metrics(mass_gt.cpu().numpy(), mass_est.cpu().numpy(), m_range)
    mu_stats = calculate_metrics(mu_gt.cpu().numpy(), mu_est.cpu().numpy(), mu_range)
    return {"mass": m_stats, "mu": mu_stats}

def _run_exp_2(mass_est, mu_est, fric_f_est, fz_robot_tensor_dom, seq_len):
    m_est_val = mass_est.cpu().numpy()
    mu_est_val = mu_est.cpu().numpy()
    mean_robot_f_z = fz_robot_tensor_dom[:, :seq_len//2].mean(dim=1).cpu().numpy()
    
    if FRAME_MODE == "local":
        mean_robot_f_z = -mean_robot_f_z
        
    mean_fric_pred = fric_f_est[:, :seq_len//2].mean(dim=1).squeeze(-1).cpu().numpy()
    calc_normal_force = np.maximum((m_est_val * 9.81) - mean_robot_f_z, 0.0)
    f_theoretical = mu_est_val * calc_normal_force
    r2 = r2_score(f_theoretical, mean_fric_pred) if len(f_theoretical) > 1 else 0.0
    return {"fric_r2": r2}

def _run_exp_3(mass_est, rhs_acc_tensor_dom, phys_net_est, seq_len):
    m_est_val = mass_est.cpu().numpy()
    mean_acc_x = rhs_acc_tensor_dom[:, :seq_len//2].mean(dim=1).cpu().numpy()
    mean_net_f_pred = phys_net_est[:, :seq_len//2].mean(dim=1).squeeze(-1).cpu().numpy()
    f_theoretical = m_est_val * mean_acc_x
    valid = np.abs(mean_acc_x) > 0.01
    r2 = r2_score(mean_net_f_pred[valid], f_theoretical[valid]) if np.sum(valid) > 5 else 0.0
    return {"net_r2": r2}

def _run_exp_4(net_f_gt, fric_f_gt, net_f_est, fric_f_est, net_range, fric_range):
    net_stats = calculate_metrics(net_f_gt.cpu().numpy(), net_f_est.cpu().numpy(), net_range)
    fric_stats = calculate_metrics(fric_f_gt.cpu().numpy(), fric_f_est.cpu().numpy(), fric_range)
    return {"net_f": net_stats, "fric_f": fric_stats}

def log_results_to_csv(m1, m2, m3, m4, domain_label, eval_checkpoint_dir, is_first=False):
    file_path = os.path.join(eval_checkpoint_dir, "domain_evaluation_summary.csv")
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
        "net_f_r2": round(m3["net_r2"], 4),
        "net_f_mean_err_pct": round(m4["net_f"]["mean_err_pct"], 2),
        "net_f_nmae_pct": round(m4["net_f"]["nmae_pct"], 2),
        "net_f_std_dev": round(m4["net_f"]["std_dev"], 4),
        "net_f_est_std_dev": round(m4["net_f"]["est_std_dev"], 4),
        "net_f_std_pct": round(m4["net_f"]["std_pct"], 2),
        "fric_f_mean_err_pct": round(m4["fric_f"]["mean_err_pct"], 2),
        "fric_f_nmae_pct": round(m4["fric_f"]["nmae_pct"], 2),
        "fric_f_std_dev": round(m4["fric_f"]["std_dev"], 4),
        "fric_f_est_std_dev": round(m4["fric_f"]["est_std_dev"], 4),
        "fric_f_std_pct": round(m4["fric_f"]["std_pct"], 2)
    }
    mode = 'w' if is_first else 'a'
    with open(file_path, mode=mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if mode == 'w': writer.writeheader()
        writer.writerow(entry)

# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    set_seed(42)
    
    # 1. FORCE CPU TO AVOID DEADLOCKS
    # PyTorch CUDA + sklearn joblib n_jobs=-1 causes fork deadlocks.
    device = torch.device("cpu")
    print(f"Forcing RF evaluation to: {device} to prevent multiprocessing hang.")

    # 2. LOAD DATA & INITIALIZE DATALOADERS
    df = load_dataset_csv()
    if 'gt_fric_force' in df.columns:
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col)

    batch_size = 64
    train_loader, val_loader, seq_len, df, choices = create_dataloaders(
        df, batch_size, M_SEEN_MIN, M_SEEN_MAX, MU_SEEN_MIN, MU_SEEN_MAX
    )

    # 3. EXTRACT RF TRAINING DATA
    X_vel_train = train_loader.dataset.tensors[1]
    y_train = train_loader.dataset.tensors[2]
    rhs_acc_train = train_loader.dataset.tensors[4]
    fx_robot_train = train_loader.dataset.tensors[8]

    eval_checkpoint_dir = os.path.join(checkpoint_dir, "baseline_random_forest")
    os.makedirs(eval_checkpoint_dir, exist_ok=True)

    print("\n" + "="*50)
    print("TRAINING PAPER BASELINE: MULTI-OUTPUT RANDOM FOREST")
    print("="*50)

    def extract_windowed_stats(signal_tensor):
        _, sequence_length = signal_tensor.shape
        w_len = sequence_length // 3
        w1 = signal_tensor[:, :w_len]
        w2 = signal_tensor[:, w_len:2*w_len]
        w3 = signal_tensor[:, 2*w_len:]
        
        features = []
        for w in [w1, w2, w3]:
            features.extend([
                torch.mean(w, dim=1, keepdim=True),
                torch.std(w, dim=1, keepdim=True, unbiased=False),
                torch.sqrt(torch.mean(w**2, dim=1, keepdim=True))
            ])
        return torch.cat(features, dim=1)

    delta_t = 0.01
    u_x_train = torch.cumsum(rhs_acc_train, dim=1) * delta_t 

    X_train_rf = torch.cat([
        extract_windowed_stats(fx_robot_train),
        extract_windowed_stats(u_x_train),
        extract_windowed_stats(X_vel_train.squeeze(-1))
    ], dim=1).numpy()

    y_train_rf = y_train.numpy()

    regr_multirf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
    )
    print(f"Training on {len(X_train_rf)} samples...")
    regr_multirf.fit(X_train_rf, y_train_rf)
    print("Baseline Training Complete!")

    # 4. EVALUATION DOMAIN SCORING
    if INCLUDE_UNSEEN:
        domain_choices = [
            'm_seen_mu_seen', 'm_seen_light', 'm_seen_middle', 'm_seen_heavy',
            'm_over', 'm_under', 'mu_over', 'mu_under',
            'm_over_mu_over', 'm_under_mu_under', 'm_over_mu_under', 'm_under_mu_over'
        ]
        global_m_range = M_UNSEEN_MAX - M_SEEN_MIN  
        global_mu_range = MU_UNSEEN_MAX - MU_SEEN_MIN 
    else:
        domain_choices = ['m_seen_light', 'm_seen_middle', 'm_seen_heavy']
        global_m_range = M_SEEN_MAX - M_SEEN_MIN  
        global_mu_range = MU_SEEN_MAX - MU_SEEN_MIN 

    acc_cols = sorted([c for c in df.columns if "input_acc_" in c], key=lambda x: int(x.split('_')[-1]))
    vel_cols = sorted([c for c in df.columns if "input_vel_" in c], key=lambda x: int(x.split('_')[-1]))

    print("Calculating global force bounds (memory safe)...")
    valid_mask = (df['start_t'] + seq_len <= 100).values
    valid_indices = np.where(valid_mask)[0]
    
    # Avoid allocating a massive copy of the 12GB dataset; extract directly using numpy arrays
    sample_indices = np.random.choice(valid_indices, size=min(2000, len(valid_indices)), replace=False)
    
    all_net_f_vals, all_table_fz_vals = [], []
    for idx in sample_indices:
        row = df.iloc[idx].to_dict()
        st = int(row['start_t'])
        window = range(st, st + seq_len)
        if FRAME_MODE == "world":
            all_net_f_vals.extend([row[f"pinn_LHS_wrench_t{t}_ax3"] for t in window])
            all_table_fz_vals.extend([row[f"pinn_table_wrench_t{t}_ax5"] for t in window])
        else:
            all_net_f_vals.extend([row[f"pinn_LHS_wrench_t{t}_ax5"] for t in window])
            all_table_fz_vals.extend([row[f"pinn_table_wrench_t{t}_ax3"] for t in window])

    global_net_f_range = np.max(all_net_f_vals) - np.min(all_net_f_vals)
    global_fric_f_range = df['gt_mu'].max() * np.max(np.abs(all_table_fz_vals))

    for i, target_domain in enumerate(domain_choices):
        base_domain = 'm_seen_mu_seen' if 'm_seen' in target_domain else target_domain
        df_domain = df[(df['domain'] == base_domain) & (df['start_t'] + seq_len <= 100)]
        
        if target_domain == 'm_seen_light':
            df_domain = df_domain[df_domain['gt_mass'] < 0.8]
        elif target_domain == 'm_seen_middle':
            df_domain = df_domain[(df_domain['gt_mass'] >= 0.8) & (df_domain['gt_mass'] < 1.4)]
        elif target_domain == 'm_seen_heavy':
            df_domain = df_domain[df_domain['gt_mass'] >= 1.4]
        
        if len(df_domain) == 0: 
            continue
        
        print(f"   Evaluating Domain: {target_domain} ({len(df_domain)} samples)")

        X_acc = torch.tensor(df_domain[acc_cols].values.reshape(-1, seq_len, 1)).float().to(device)
        X_vel = torch.tensor(df_domain[vel_cols].values.reshape(-1, seq_len, 1)).float().to(device)
        y_gt = torch.tensor(df_domain[['gt_mass', 'gt_mu']].values).float().to(device)

        robot_fz_list, rhs_acc_list = [], []
        lhs_net_f_list, table_fz_list, robot_fx_list = [], [], []
        
        # O(1) dictionary parsing prevents Pandas Series allocation blocks
        df_records = df_domain.to_dict('records')
        for row in df_records:
            st = int(row['start_t'])
            window = range(st, st + seq_len) 
            
            if FRAME_MODE == "world":
                robot_fz_list.append([row[f"pinn_robot_wrench_t{t}_ax5"] for t in window])
                rhs_acc_list.append([row[f"pinn_RHS_acc_t{t}_ax3"] for t in window])
                lhs_net_f_list.append([row[f"pinn_LHS_wrench_t{t}_ax3"] for t in window])
                table_fz_list.append([row[f"pinn_table_wrench_t{t}_ax5"] for t in window])
                robot_fx_list.append([row[f"pinn_robot_wrench_t{t}_ax3"] for t in window])
            else:
                robot_fz_list.append([row[f"pinn_robot_wrench_t{t}_ax3"] for t in window])
                rhs_acc_list.append([row[f"pinn_RHS_acc_t{t}_ax5"] for t in window])
                lhs_net_f_list.append([row[f"pinn_LHS_wrench_t{t}_ax5"] for t in window])
                table_fz_list.append([row[f"pinn_table_wrench_t{t}_ax3"] for t in window])
                robot_fx_list.append([row[f"pinn_robot_wrench_t{t}_ax5"] for t in window])
        
        fz_robot_tensor_dom = torch.tensor(np.array(robot_fz_list)).float().to(device)
        rhs_acc_tensor_dom = torch.tensor(np.array(rhs_acc_list)).float().to(device)
        fx_robot_tensor_dom = torch.tensor(np.array(robot_fx_list)).float().to(device)
        net_f_gt_tensor = torch.tensor(np.array(lhs_net_f_list)).float().to(device)
        normal_f_gt = torch.abs(torch.tensor(np.array(table_fz_list)).float().to(device))
        fric_f_gt_tensor = y_gt[:, 1].unsqueeze(1) * normal_f_gt

        u_x_dom = torch.cumsum(rhs_acc_tensor_dom, dim=1) * delta_t 
        u_px_dom = X_vel.squeeze(-1)
        
        X_rf_domain = torch.cat([
            extract_windowed_stats(fx_robot_tensor_dom),
            extract_windowed_stats(u_x_dom),
            extract_windowed_stats(u_px_dom)
        ], dim=1).cpu().numpy()

        preds_rf = regr_multirf.predict(X_rf_domain)
        mass_est = torch.tensor(preds_rf[:, 0]).float().to(device)
        mu_est = torch.tensor(preds_rf[:, 1]).float().to(device)
        mass_gt, mu_gt = y_gt[:, 0], y_gt[:, 1]

        phys_net = mass_est.unsqueeze(1) * rhs_acc_tensor_dom
        mean_robot_f_z = fz_robot_tensor_dom
        if FRAME_MODE == "local":
            mean_robot_f_z = -mean_robot_f_z
            
        calc_normal = torch.clamp((mass_est.unsqueeze(1) * 9.81) - mean_robot_f_z, min=0.0)
        phys_fric = mu_est.unsqueeze(1) * calc_normal

        m1 = _run_exp_1(mass_gt, mu_gt, mass_est, mu_est, global_m_range, global_mu_range)
        m2 = _run_exp_2(mass_est, mu_est, phys_fric, fz_robot_tensor_dom, seq_len)
        m3 = _run_exp_3(mass_est, rhs_acc_tensor_dom, phys_net, seq_len)
        m4 = _run_exp_4(net_f_gt_tensor, fric_f_gt_tensor, phys_net, phys_fric, global_net_f_range, global_fric_f_range)
        
        log_results_to_csv(m1, m2, m3, m4, target_domain, eval_checkpoint_dir, is_first=(i == 0))

    print(f"\nDone. RF Baseline Summary saved in: {eval_checkpoint_dir}/domain_evaluation_summary.csv")

if __name__ == "__main__":
    main()