import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import clean_force_col
from configs import M_UNSEEN_MAX, MU_UNSEEN_MAX, FRAME_MODE

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

    indices = np.arange(len(df_filtered))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_acc_tensor[train_idx], X_vel_tensor[train_idx], y_tensor[train_idx],
                                  fz_robot_tensor[train_idx], rhs_acc_tensor[train_idx], lhs_net_f_tensor[train_idx],
                                  fz_normal_tensor[train_idx], start_t_tensor[train_idx], fx_robot_tensor[train_idx])
    val_dataset = TensorDataset(X_acc_tensor[val_idx], X_vel_tensor[val_idx], y_tensor[val_idx],
                                fz_robot_tensor[val_idx], rhs_acc_tensor[val_idx], lhs_net_f_tensor[val_idx],
                                fz_normal_tensor[val_idx], start_t_tensor[val_idx], fx_robot_tensor[val_idx])

    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, seq_len, df, choices