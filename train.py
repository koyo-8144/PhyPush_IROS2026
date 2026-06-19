import os
import time
import datetime
import torch
import json
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils import set_seed, build_model_string, clean_force_col
from models import PhysicsTransformerEstimator
from losses import log_mse_loss, PinnLossCalculator
from dataset import create_dataloaders
from configs import used_config, CSV_PATH

def main():
    set_seed(42)
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    if 'gt_fric_force' in df.columns:
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # ==========================================
    # 1. HYPERPARAMETERS (Stored in a Dictionary)
    # ==========================================
    config = used_config

    # Override pinn4 coefficients if diff_coeffs_pinn4 is set to 1
    if config['diff_coeffs_pinn4']:
        config['pinn_coeffs'].update({'p4': 1.0, 'p4_1': 10.0, 'p4_2': 10.0, 'p4_3': 1.0})

    # --- UNPACK CONFIG INTO LOCAL VARIABLES ---
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr_optimizer = config['lr_optimizer']
    lr_scheduler = config['lr_scheduler']
    loss_type = config['loss_type']
    task_coeff = config['task_coeff']
    task_criterion = config['task_criterion']
    c_entropy_coeff = config['c_entropy_coeff']
    m_entropy_coeff = config['m_entropy_coeff']
    f_entropy_coeff = config['f_entropy_coeff']
    force_coeff = config['force_coeff']
    force_criterion = config['force_criterion']
    d_model = config['d_model']
    num_enc = config['num_enc']
    last_layer_ms = config['last_layer_ms']
    last_layer_mus = config['last_layer_mus']
    dropout = config['dropout']
    sharpness = config['sharpness']
    cross_sharpness = config['cross_sharpness']
    m_sharpness = config['m_sharpness']
    mu_sharpness = config['mu_sharpness']
    init_lr = config['init_lr']
    pinn_criterion_str = config['pinn_criterion']
    mass_scale = config['mass_scale']
    fric_scale = config['fric_scale']
    pinn_coeff_annealing = config['pinn_coeff_annealing']
    annealing_start_epoch = config['annealing_start_epoch']
    ramp_duration = config['ramp_duration']
    m_seen_max = config['m_seen_max']
    m_seen_min = config['m_seen_min']
    mu_seen_max = config['mu_seen_max']
    mu_seen_min = config['mu_seen_min']
    acc_filter_threshold = config['acc_filter_threshold']
    vel_filter_threshold = config['vel_filter_threshold']
    transformer_ver = config['transformer_ver']
    diff_coeffs_pinn4 = config['diff_coeffs_pinn4']

    p_c = config['pinn_coeffs']
    pinn_coeff_1 = p_c['p1']
    pinn_coeff_2 = p_c['p2']
    pinn_coeff_2_2 = p_c['p2-2']
    pinn_coeff_3 = p_c['p3']
    pinn_coeff_4 = p_c['p4']
    pinn_coeff_4_1 = p_c['p4_1']
    pinn_coeff_4_2 = p_c['p4_2']
    pinn_coeff_4_3 = p_c['p4_3']
    pinn_coeff_5 = p_c['p5']
    pinn_coeff_6 = p_c['p6']
    pinn_coeff_7 = p_c['p7']
    pinn_coeff_8 = p_c['p8']
    pinn_coeff_9 = p_c['p9']
    pinn_coeff_9_2 = p_c['p9-2']
    pinn_coeff_9_3 = p_c['p9-3']
    pinn_coeff_10 = p_c['p10']
    pinn_coeff_11 = p_c['p11']
    pinn_coeff_11_2 = p_c['p11-2']

    # --- Initialize Utility Objects ---
    plot_inter = 100
    model_save_inter = 200
    printout_inter = 50

    mse_criterion = nn.MSELoss(reduction='none')
    log1p_mse_criterion = log_mse_loss

    c_head_coeffs = torch.tensor([c_entropy_coeff] * 4).to(device)

    # --- DATA LOADERS ---
    train_loader, val_loader, seq_len, df, choices = create_dataloaders(
        df, batch_size, m_seen_min, m_seen_max, mu_seen_min, mu_seen_max
    )
    config['seq_len'] = seq_len


    # ==========================================
    # 2. PATHS & SAVING CONFIG
    # ==========================================
    train_from = "from_20260618"
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    training_model = build_model_string(config) 
    parent_results_dir = "./results"
    
    checkpoint_dir = os.path.join(parent_results_dir, "checkpoints", f"{train_from}", f"{current_time}", training_model)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_save_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_save_path}")

    # ==========================================
    # 3. MODEL SETUP & INIT
    # ==========================================
    model = PhysicsTransformerEstimator(
        input_dim=1,          
        d_model=d_model,           
        nhead=4,              
        num_encoder_layers=num_enc, 
        seq_len=seq_len,
        dropout=dropout,
        sharpness=sharpness,
        cross_sharpness=cross_sharpness,
        m_sharpness=m_sharpness,
        mu_sharpness=mu_sharpness,
        version=transformer_ver,
        max_mass_scale=last_layer_ms,
        max_mu_scale=last_layer_mus,
    ).to(device)

    if lr_optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-3)
    elif lr_optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-3)

    if lr_scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-8
        )
    elif lr_scheduler == "OneCycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=init_lr, steps_per_epoch=len(train_loader), 
            epochs=num_epochs, pct_start=0.1, anneal_strategy='cos'
        )
        
    print(f"Model Initialized with input_dim=1 and seq_len={seq_len}")

    # ==========================================
    # 4. TRAINING LOOP WITH LIVE PLOTTING & SAVING
    # ==========================================
    best_val_loss = float('inf')
    block_start_time = time.time()

    train_losses, val_losses, lrs = [], [], []
    weighted_task_history, weighted_force_history = [], []
    weighted_entropy_history, weighted_pinn_history = [], []

    c_head_histories = [[], [], [], []] 
    m_entropy_history, f_entropy_history = [], []
    m_est_history, mu_est_history = [], []
    m_gt_history, mu_gt_history = [], []

    if task_criterion == "mse":
        task_cri = mse_criterion
    elif task_criterion == "log1p_mse":
        task_cri = log1p_mse_criterion

    if force_criterion == "mse":
        force_cri = mse_criterion
    elif force_criterion == "log1p_mse":
        force_cri = log1p_mse_criterion

    if pinn_criterion_str == "mse":
        pinn_cri = nn.MSELoss(reduction='none')
    elif pinn_criterion_str == "log1p_mse":
        pinn_cri = log1p_mse_criterion
    elif pinn_criterion_str == "L1":
        pinn_cri = nn.L1Loss(reduction='none')
        
    phys = PinnLossCalculator(pinn_cri)

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):

        if pinn_coeff_annealing == 1:
            if epoch < annealing_start_epoch:
                current_pinn_coeff = 0.0
            else:
                ramp = min(1.0, (epoch - annealing_start_epoch) / ramp_duration)
                current_pinn_coeff = ramp
        else:
            current_pinn_coeff = 1.0
        
        # --- TRAIN ---
        model.train()
        running_total_loss = 0.0
        running_weighted_task = 0.0
        running_weighted_force = 0.0
        running_weighted_entropy = 0.0
        running_weighted_pinn = 0.0
        running_c_heads = torch.zeros(4).to(device)
        running_m_ent = 0.0
        running_f_ent = 0.0
        
        for b_acc, b_vel, b_y, b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz, b_start_t, b_robot_fx in train_loader:
            b_acc, b_vel, b_y = b_acc.to(device), b_vel.to(device), b_y.to(device)
            b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz = b_robot_fz.to(device), b_rhs_acc.to(device), b_lhs_net_f.to(device), b_table_fz.to(device)
            b_robot_fx = b_robot_fx.to(device)
            b_start_t = b_start_t.to(device)
            optimizer.zero_grad()
            
            output, (c_weights, m_weights, f_weights), (net_f_est, fric_f_est) = model(b_vel)

            m_gt = b_y[:, 0].unsqueeze(1)
            mu_gt = b_y[:, 1].unsqueeze(1)
            gravity_term = m_gt * 9.81
            
            net_f_gt = b_lhs_net_f
            normal_f_gt = torch.abs(b_table_fz) 
            fric_f_gt = mu_gt * torch.clamp(normal_f_gt, min=0.0)
            force_gt = torch.cat([net_f_gt, fric_f_gt], dim=-1)
            
            m_est = output[:, 0].unsqueeze(1)
            mu_est = output[:, 1].unsqueeze(1)
            net_f_est = net_f_est.squeeze(-1)
            fric_f_est = fric_f_est.squeeze(-1)
            force_est = torch.cat([net_f_est, fric_f_est], dim=-1)
        
            if loss_type == "data":        
                task_loss = task_cri(output, b_y) 
                weighted_task_loss = task_coeff * task_loss
                force_loss = force_cri(force_gt, force_est)
                weighted_force_loss = force_coeff * force_loss

                total_loss = weighted_task_loss + weighted_force_loss
                weighted_pinn_loss = torch.tensor(0.0).to(device)
            elif loss_type == "pinn" or loss_type == "hybrid":
                # =======================================================
                # DYNAMIC PHYSICS MASKING
                # =======================================================
                is_accelerating_mask = (torch.abs(b_rhs_acc) > acc_filter_threshold).float()
                mask_net = is_accelerating_mask[:, :seq_len//2]
                
                is_sliding_mask = (torch.abs(b_vel.squeeze(-1)) > vel_filter_threshold).float()
                mask_fric = is_sliding_mask[:, seq_len//2:]
                
                def apply_mask(loss_tensor, mask):
                    active_frames = torch.clamp(mask.sum(), min=1.0)
                    return (loss_tensor * mask).sum() / active_frames

                # --- PART A: NET FORCE ---
                pinn_net_1 = apply_mask(phys.net_force_law(m_est, b_rhs_acc[:, :seq_len//2], net_f_est[:, :seq_len//2]), mask_net)
                pinn_net_2 = apply_mask(phys.net_force_law(m_est, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2]), mask_net)
                pinn_net_2_ann = pinn_net_2 
                pinn_net_3 = apply_mask(phys.net_force_law(m_gt, b_rhs_acc[:, :seq_len//2], net_f_est[:, :seq_len//2]), mask_net)
                pinn_net_4 = apply_mask(phys.net_force_law_smoothed(m_est, b_rhs_acc[:, :seq_len//2], net_f_est[:, :seq_len//2]), mask_net)
                pinn_net_4_ann = apply_mask(phys.net_force_law_smoothed(m_est, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2]), mask_net)
                pinn_acc_inertia = apply_mask(phys.inertia_acceleration_law(m_est, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net)
                pinn_pos_inertia = apply_mask(phys.inertia_position_law(m_est, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net)

                # --- PART B: FRICTION ---
                pinn_fric_1 = apply_mask(phys.friction_law(m_est, mu_est, b_robot_fz[:, seq_len//2:], fric_f_est[:, seq_len//2:]), mask_fric)
                pinn_fric_2 = apply_mask(phys.friction_law(m_est, mu_est, b_robot_fz[:, seq_len//2:], fric_f_gt[:, seq_len//2:]), mask_fric)
                pinn_fric_3 = apply_mask(phys.friction_direct_law(mu_gt, torch.abs(b_table_fz[:, seq_len//2:]), fric_f_est[:, seq_len//2:]), mask_fric)
                pinn_fric_4 = apply_mask(phys.friction_law(m_gt, mu_est, b_robot_fz[:, seq_len//2:], fric_f_gt[:, seq_len//2:]), mask_fric)
                pinn_fric_5 = apply_mask(phys.friction_law_robust(m_est, mu_est, b_robot_fz[:, seq_len//2:], fric_f_est[:, seq_len//2:]), mask_fric)
                pinn_fric_5_ann = apply_mask(phys.friction_law_robust(m_est, mu_est, b_robot_fz[:, seq_len//2:], fric_f_gt[:, seq_len//2:]), mask_fric)
                pinn_acc_consistency = apply_mask(phys.kinematic_consistency(m_gt, mu_est, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric)
                pinn_acc_consistency_2 = apply_mask(phys.kinematic_consistency(m_est, mu_est, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric)
                pinn_pos_kinematic = apply_mask(phys.kinematic_position_consistency(m_gt, mu_est, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric)
                pinn_pos_kinematic_2 = apply_mask(phys.kinematic_position_consistency(m_est, mu_est, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric)
                
                # --- AGGREGATION ---
                pinn_loss_1 = (mass_scale * pinn_net_1) + (fric_scale * pinn_fric_1)
                if pinn_coeff_annealing == 1:
                    pinn_loss_2 = ((mass_scale * pinn_net_2) + (fric_scale * pinn_fric_2)) * current_pinn_coeff
                else:
                    pinn_loss_2 = (mass_scale * pinn_net_2) + (fric_scale * pinn_fric_2)
                
                pinn_loss_3 = ((mass_scale * pinn_net_2) + (fric_scale * pinn_fric_2) 
                                +(mass_scale * pinn_net_3) + (fric_scale * pinn_fric_3))
                
                if diff_coeffs_pinn4:
                    pinn_loss_4 = (pinn_coeff_4_1 * ((mass_scale * pinn_net_2) + (fric_scale * pinn_fric_2))
                                    +pinn_coeff_4_2 * ((mass_scale * pinn_net_3) + (fric_scale * pinn_fric_3))
                                    +pinn_coeff_4_3 * ((mass_scale * pinn_net_1) + (fric_scale * pinn_fric_1)))      
                else:
                    pinn_loss_4 = ((mass_scale * pinn_net_2) + (fric_scale * pinn_fric_2)
                                    +(mass_scale * pinn_net_3) + (fric_scale * pinn_fric_3)
                                    +(mass_scale * pinn_net_1) + (fric_scale * pinn_fric_1))
                
                pinn_loss_5 = (mass_scale * pinn_net_2) + (fric_scale * pinn_fric_4)
                pinn_loss_6 = ((mass_scale * pinn_net_2) + (fric_scale * pinn_fric_4)
                                +(mass_scale * pinn_net_3) + (fric_scale * pinn_fric_3))
                pinn_loss_7 = ((mass_scale * pinn_net_2) + (fric_scale * pinn_fric_4)
                                +(mass_scale * pinn_net_3) + (fric_scale * pinn_fric_3)
                                +(mass_scale * pinn_net_1) + (fric_scale * pinn_fric_1))
                
                if pinn_coeff_annealing == 1:
                    pinn_loss_8 = ((mass_scale * pinn_net_4_ann) + (fric_scale * pinn_fric_5_ann)) * current_pinn_coeff
                else:
                    pinn_loss_8 = (mass_scale * pinn_net_4) + (fric_scale * pinn_fric_5)
    
                pinn_loss_9 = (mass_scale * pinn_net_2) + (fric_scale * pinn_acc_consistency) 
                pinn_loss_10 = (mass_scale * pinn_acc_inertia) + (fric_scale * pinn_acc_consistency) 
                pinn_loss_11 = (mass_scale * pinn_pos_inertia) + (fric_scale * pinn_pos_kinematic)  
                
                if pinn_coeff_annealing == 1:
                    pinn_loss_2_2 = (fric_scale * pinn_fric_2) * current_pinn_coeff
                    pinn_loss_9_2 = (fric_scale * pinn_acc_consistency_2) * current_pinn_coeff
                    pinn_loss_11_2 = (fric_scale * pinn_pos_kinematic_2) * current_pinn_coeff
                else:
                    pinn_loss_2_2 = fric_scale * pinn_fric_2 
                    pinn_loss_9_2 = fric_scale * pinn_acc_consistency_2 
                    pinn_loss_11_2 = fric_scale * pinn_pos_kinematic_2 

                weighted_pinn_loss = (pinn_coeff_1 * pinn_loss_1 + pinn_coeff_2 * pinn_loss_2 + pinn_coeff_3 * pinn_loss_3 
                                        + pinn_coeff_4 * pinn_loss_4 + pinn_coeff_5 * pinn_loss_5 + pinn_coeff_6 * pinn_loss_6
                                        + pinn_coeff_7 * pinn_loss_7 + pinn_coeff_8 * pinn_loss_8 + pinn_coeff_9 * pinn_loss_9
                                        + pinn_coeff_2_2 * pinn_loss_2_2 + pinn_coeff_9_2 * pinn_loss_9_2
                                        + pinn_coeff_10 * pinn_loss_10 + pinn_coeff_11 * pinn_loss_11 + pinn_coeff_11_2 * pinn_loss_11_2)

                if loss_type == "pinn":
                    total_loss = weighted_pinn_loss
                    weighted_task_loss, weighted_force_loss = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                elif loss_type == "hybrid":
                    task_loss = task_cri(output, b_y) 
                    weighted_task_loss = task_coeff * task_loss
                    force_loss = force_cri(force_gt, force_est)
                    weighted_force_loss = force_coeff * force_loss
                    total_loss = weighted_task_loss + weighted_force_loss + weighted_pinn_loss
                
            # Attention Entropy calculation
            if c_weights is not None:
                c_ent_all = -torch.sum(c_weights * torch.log(c_weights + 1e-10), dim=-1) 
                c_ent = c_ent_all.mean() 
                batch_head_entropies = c_ent_all.mean(dim=(0, 2))
                weighted_c_entropy = torch.sum(batch_head_entropies * c_head_coeffs)
            else:
                batch_head_entropies = torch.zeros_like(c_head_coeffs)
                weighted_c_entropy = torch.tensor(0.0, device=c_head_coeffs.device)
            
            m_ent = -torch.sum(m_weights * torch.log(m_weights + 1e-10), dim=-1).mean()
            f_ent = -torch.sum(f_weights * torch.log(f_weights + 1e-10), dim=-1).mean()
            weighted_m_entropy = m_entropy_coeff * m_ent
            weighted_f_entropy = f_entropy_coeff * f_ent
            
            total_weighted_entropy = weighted_c_entropy + weighted_m_entropy + weighted_f_entropy
            
            total_loss += total_weighted_entropy
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if lr_scheduler == "OneCycle":
                scheduler.step()
            
            batch_size_curr = b_acc.size(0)
            running_total_loss += total_loss.item() * batch_size_curr
            running_weighted_task += weighted_task_loss.item() * batch_size_curr
            running_weighted_force += weighted_force_loss.item() * batch_size_curr
            running_weighted_entropy += total_weighted_entropy.item() * batch_size_curr
            running_weighted_pinn += weighted_pinn_loss.item() * batch_size_curr
            
            running_c_heads += batch_head_entropies * batch_size_curr
            running_m_ent += m_ent.item() * batch_size_curr
            running_f_ent += f_ent.item() * batch_size_curr
        
        # Epoch Averages
        train_samples = len(train_loader.dataset)
        train_losses.append(running_total_loss / train_samples)
        weighted_task_history.append(running_weighted_task / train_samples)
        weighted_force_history.append(running_weighted_force / train_samples)
        weighted_entropy_history.append(running_weighted_entropy / train_samples)
        weighted_pinn_history.append(running_weighted_pinn / train_samples)
        
        avg_heads = running_c_heads / train_samples
        for i in range(4):
            c_head_histories[i].append(avg_heads[i].item())
        m_entropy_history.append(running_m_ent / train_samples)
        f_entropy_history.append(running_f_ent / train_samples)

        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        
        epoch_m_est_sum = 0.0
        epoch_mu_est_sum = 0.0
        epoch_m_gt_sum = 0.0
        epoch_mu_gt_sum = 0.0
        val_samples_count = 0
        
        with torch.no_grad():
            for b_acc, b_vel, b_y, b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz, b_start_t, b_robot_fx in val_loader:
                b_acc, b_vel, b_y = b_acc.to(device), b_vel.to(device), b_y.to(device)
                b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz = b_robot_fz.to(device), b_rhs_acc.to(device), b_lhs_net_f.to(device), b_table_fz.to(device)
                b_robot_fx = b_robot_fx.to(device)
                output, _, (net_f_est_val, fric_f_est_val) = model(b_vel) 
                
                m_gt_val = b_y[:, 0].unsqueeze(1)
                mu_gt_val = b_y[:, 1].unsqueeze(1)
                m_est_val = output[:, 0].unsqueeze(1)
                mu_est_val = output[:, 1].unsqueeze(1)
                
                batch_sz = b_acc.size(0)
                epoch_m_est_sum += m_est_val.sum().item()
                epoch_mu_est_sum += mu_est_val.sum().item()
                epoch_m_gt_sum += m_gt_val.sum().item()
                epoch_mu_gt_sum += mu_gt_val.sum().item()
                val_samples_count += batch_sz

                net_f_gt_val = b_lhs_net_f 
                normal_f_gt_val = torch.abs(b_table_fz)
                fric_f_gt_val = mu_gt_val * torch.clamp(normal_f_gt_val, min=0.0)
                force_gt_val = torch.cat([net_f_gt_val, fric_f_gt_val], dim=-1)
                force_est_val = torch.cat([net_f_est_val.squeeze(-1), fric_f_est_val.squeeze(-1)], dim=-1)

                if loss_type == "data":
                    task_v = task_cri(output, b_y)
                    force_v = force_cri(force_gt_val, force_est_val)
                    batch_loss = (task_coeff * task_v) + (force_coeff * force_v)

                elif loss_type == "pinn" or loss_type == "hybrid":
                    is_accelerating_mask_v = (torch.abs(b_rhs_acc) > acc_filter_threshold).float()
                    mask_net_v = is_accelerating_mask_v[:, :seq_len//2]
                    
                    is_sliding_mask_v = (torch.abs(b_vel.squeeze(-1)) > vel_filter_threshold).float()
                    mask_fric_v = is_sliding_mask_v[:, seq_len//2:]
                    
                    def apply_mask_v(loss_tensor, mask):
                        active_frames = torch.clamp(mask.sum(), min=1.0)
                        return (loss_tensor * mask).sum() / active_frames

                    # --- PART A ---
                    pinn_net_v1 = apply_mask_v(phys.net_force_law(m_est_val, b_rhs_acc[:, :seq_len//2], net_f_est_val[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v2 = apply_mask_v(phys.net_force_law(m_est_val, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v2_ann = pinn_net_v2
                    pinn_net_v3 = apply_mask_v(phys.net_force_law(m_gt_val, b_rhs_acc[:, :seq_len//2], net_f_est_val[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v4 = apply_mask_v(phys.net_force_law_smoothed(m_est_val, b_rhs_acc[:, :seq_len//2], net_f_est_val[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v4_ann = apply_mask_v(phys.net_force_law_smoothed(m_est_val, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_acc_inertia_v = apply_mask_v(phys.inertia_acceleration_law(m_est_val, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net_v)
                    pinn_pos_inertia_v = apply_mask_v(phys.inertia_position_law(m_est_val, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net_v)
                    
                    # --- PART B ---
                    pinn_fric_v1 = apply_mask_v(phys.friction_law(m_est_val, mu_est_val, b_robot_fz[:, seq_len//2:], fric_f_est_val[:, seq_len//2:].squeeze(-1)), mask_fric_v)
                    pinn_fric_v2 = apply_mask_v(phys.friction_law(m_est_val, mu_est_val, b_robot_fz[:, seq_len//2:], fric_f_gt_val[:, seq_len//2:].squeeze(-1)), mask_fric_v)
                    pinn_fric_v3 = apply_mask_v(phys.friction_direct_law(mu_gt_val, torch.abs(b_table_fz[:, seq_len//2:]), fric_f_est_val[:, seq_len//2:].squeeze(-1)), mask_fric_v)
                    pinn_fric_v4 = apply_mask_v(phys.friction_law(m_gt_val, mu_est_val, b_robot_fz[:, seq_len//2:], fric_f_gt_val[:, seq_len//2:].squeeze(-1)), mask_fric_v)
                    pinn_fric_v5 = apply_mask_v(phys.friction_law_robust(m_est_val, mu_est_val, b_robot_fz[:, seq_len//2:], fric_f_est_val[:, seq_len//2:].squeeze(-1)), mask_fric_v)
                    pinn_fric_v5_ann = apply_mask_v(phys.friction_law_robust(m_est_val, mu_est_val, b_robot_fz[:, seq_len//2:], fric_f_gt_val[:, seq_len//2:].squeeze(-1)), mask_fric_v)
                    pinn_acc_consistency_v = apply_mask_v(phys.kinematic_consistency(m_gt_val, mu_est_val, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric_v)
                    pinn_acc_consistency_v_2 = apply_mask_v(phys.kinematic_consistency(m_est_val, mu_est_val, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric_v)
                    pinn_pos_kinematic_v = apply_mask_v(phys.kinematic_position_consistency(m_gt_val, mu_est_val, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric_v)
                    pinn_pos_kinematic_v_2 = apply_mask_v(phys.kinematic_position_consistency(m_est_val, mu_est_val, b_robot_fx[:, seq_len//2:], b_robot_fz[:, seq_len//2:], b_rhs_acc[:, seq_len//2:]), mask_fric_v)
                    
                    # --- AGGREGATION ---
                    pinn_loss_v1 = (mass_scale * pinn_net_v1) + (fric_scale * pinn_fric_v1)
                    if pinn_coeff_annealing == 1:
                        pinn_loss_v2 = ((mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v2)) * current_pinn_coeff
                    else:
                        pinn_loss_v2 = (mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v2)
                    pinn_loss_v3 = ((mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v2) 
                                    +(mass_scale * pinn_net_v3) + (fric_scale * pinn_fric_v3))
                    if diff_coeffs_pinn4:
                        pinn_loss_v4 = (pinn_coeff_4_1 * ((mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v2))
                                        +pinn_coeff_4_2 * ((mass_scale * pinn_net_v3) + (fric_scale * pinn_fric_v3))
                                        +pinn_coeff_4_3 * ((mass_scale * pinn_net_v1) + (fric_scale * pinn_fric_v1)))      
                    else:
                        pinn_loss_v4 = ((mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v2)
                                        +(mass_scale * pinn_net_v3) + (fric_scale * pinn_fric_v3)
                                        +(mass_scale * pinn_net_v1) + (fric_scale * pinn_fric_v1))
                    pinn_loss_v5 = (mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v4)
                    pinn_loss_v6 = ((mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v4)
                                    +(mass_scale * pinn_net_v3) + (fric_scale * pinn_fric_v3))
                    pinn_loss_v7 = ((mass_scale * pinn_net_v2) + (fric_scale * pinn_fric_v4)
                                    +(mass_scale * pinn_net_v3) + (fric_scale * pinn_fric_v3)
                                    +(mass_scale * pinn_net_v1) + (fric_scale * pinn_fric_v1))
                    if pinn_coeff_annealing == 1:
                        pinn_loss_v8 = ((mass_scale * pinn_net_v4_ann) + (fric_scale * pinn_fric_v5_ann)) * current_pinn_coeff
                    else:
                        pinn_loss_v8 = (mass_scale * pinn_net_v4) + (fric_scale * pinn_fric_v5)
                    
                    pinn_loss_v9 = (mass_scale * pinn_net_v2) + (fric_scale * pinn_acc_consistency_v)
                    pinn_loss_v10 = (mass_scale * pinn_acc_inertia_v) + (fric_scale * pinn_acc_consistency_v)
                    pinn_loss_v11 = (mass_scale * pinn_pos_inertia_v) + (fric_scale * pinn_pos_kinematic_v) 
            
                    if pinn_coeff_annealing == 1:
                        pinn_loss_v2_2 = (fric_scale * pinn_fric_v2) * current_pinn_coeff
                        pinn_loss_v9_2 = (fric_scale * pinn_acc_consistency_v_2) * current_pinn_coeff
                        pinn_loss_v11_2 = (fric_scale * pinn_pos_kinematic_v_2) * current_pinn_coeff
                    else:
                        pinn_loss_v2_2 = fric_scale * pinn_fric_v2
                        pinn_loss_v9_2 = fric_scale * pinn_acc_consistency_v_2
                        pinn_loss_v11_2 = fric_scale * pinn_pos_kinematic_v_2
                    
                    weighted_pinn_loss_val = ((pinn_coeff_1 * pinn_loss_v1) + (pinn_coeff_2 * pinn_loss_v2) 
                                    + (pinn_coeff_3 * pinn_loss_v3) + (pinn_coeff_4 * pinn_loss_v4) 
                                    + (pinn_coeff_5 * pinn_loss_v5) + (pinn_coeff_6 * pinn_loss_v6)
                                    + (pinn_coeff_7 * pinn_loss_v7) + (pinn_coeff_8 * pinn_loss_v8)
                                    + (pinn_coeff_9 * pinn_loss_v9) + (pinn_coeff_2_2 * pinn_loss_v2_2)
                                    + (pinn_coeff_9_2 * pinn_loss_v9_2) + (pinn_coeff_10 * pinn_loss_v10)
                                    + (pinn_coeff_11 * pinn_loss_v11) + (pinn_coeff_11_2 * pinn_loss_v11_2))

                    if loss_type == "pinn":
                        batch_loss = weighted_pinn_loss_val
                    elif loss_type == "hybrid":
                        task_v = task_cri(output, b_y)
                        force_v = force_cri(force_gt_val, force_est_val)
                        batch_loss = (task_coeff * task_v) + (force_coeff * force_v) + weighted_pinn_loss_val
                    
                val_running_loss += batch_loss.item() * b_acc.size(0)
        
        m_est_history.append(epoch_m_est_sum / val_samples_count)
        mu_est_history.append(epoch_mu_est_sum / val_samples_count)
        m_gt_history.append(epoch_m_gt_sum / val_samples_count)
        mu_gt_history.append(epoch_mu_gt_sum / val_samples_count)
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        if lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % model_save_inter == 0:
            save_path = os.path.join(checkpoint_dir, f"transformer_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    Regular Checkpoint saved: {save_path}")
        
        # ==========================================
        # PASTE THE PRINTOUT BLOCK HERE
        # ==========================================
        if (epoch + 1) % printout_inter == 0:
            print(f"Epoch [{epoch + 1:04d}/{num_epochs}] "
                    f"| Train Total: {train_losses[-1]:.4e} "
                    f"| Val Total: {val_losses[-1]:.4e} "
                    f"| LR: {current_lr:.2e}")
            print(f"    -> Est Mass: {m_est_history[-1]:.3f} kg (GT: {m_gt_history[-1]:.3f} kg)")
            print(f"    -> Est Fric: {mu_est_history[-1]:.3f}    (GT: {mu_gt_history[-1]:.3f})")

        # --- PLOT SAVING BLOCK ---
        if (epoch + 1) % plot_inter == 0:
            fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(12, 21)) 
            
            ax1.plot(train_losses, label='Total Train Loss', color='tab:blue', alpha=0.4)
            ax1.plot(val_losses, label='Val Task Loss', color='tab:orange', linewidth=2)
            ax1.axhline(y=best_val_loss, color='green', linestyle=':', alpha=0.6)
            ax1.set_yscale('log')
            ax1.set_ylabel('Loss (Log Scale)', color='tab:blue')
            ax1.set_title(f"Global Training Performance (Epoch {epoch+1})")
            ax1.legend(loc='upper left')
            ax1.grid(True, which="both", ls="-", alpha=0.1)

            ax2 = ax1.twinx() 
            ax2.plot(lrs, label='LR', color='tab:red', linestyle='--')
            ax2.set_yscale('log')
            ax2.set_ylabel('Learning Rate', color='tab:red')
            ax2.legend(loc='upper right')

            ax3.plot(weighted_task_history, label='Weighted Task Loss (Physics)', color='tab:purple', lw=2)
            ax3.plot(weighted_force_history, label='Weighted Force Loss (Physics)', color='tab:blue', lw=2)
            ax3.plot(weighted_entropy_history, label='Weighted Entropy (Attention)', color='tab:green', lw=2)
            ax3.plot(weighted_pinn_history, label='Weighted PINN Loss (Physics)', color='tab:orange', lw=2)
            ax3.set_yscale('log')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Component Loss (Log Scale)')
            ax3.set_title("Newtonian Objective vs. Attention Sharpness")
            ax3.legend(loc='upper right')
            ax3.grid(True, which="both", ls="-", alpha=0.1) 

            colors = ['cyan', 'deepskyblue', 'blue', 'darkblue']
            for i in range(4):
                ax4.plot(c_head_histories[i], label=f'C-Head {i+1}', color=colors[i], alpha=0.7)
            max_entropy = np.log(seq_len)
            ax4.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.3, label=f'Uniform Limit (ln({seq_len}))')
            ax4.plot(m_entropy_history, label='Mass-Attn', color='magenta', linestyle='--', lw=2)
            ax4.plot(f_entropy_history, label='Fric-Attn', color='yellow', linestyle=':', lw=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Raw Entropy')
            ax4.set_title("Divergence of Attention Experts")
            ax4.legend(loc='upper right', fontsize='x-small', ncol=2)
            
            ax5.plot(m_est_history, label='Estimated Mass (m)', color='magenta', linewidth=2)
            ax5.plot(m_gt_history, label='GT Mass', color='magenta', linestyle='--', alpha=0.5)
            ax5.plot(mu_est_history, label='Estimated Fric (μ)', color='orange', linewidth=2)
            ax5.plot(mu_gt_history, label='GT Fric', color='orange', linestyle='--', alpha=0.5)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Parameter Value')
            ax5.set_title("Global Property Estimation Progress (Validation)")
            ax5.legend(loc='upper right', ncol=2)
            ax5.grid(True, alpha=0.2)

            plt.tight_layout()
            
            # Save the plot to the checkpoint directory, continually overwriting the same file
            plot_save_path = os.path.join(checkpoint_dir, "training_progress.png")
            plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
            
            # Free memory by closing the figure explicitly
            plt.close(fig)



    # ==========================================
    # FINAL SAVING (After Loop)
    # ==========================================
    fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(12, 21))
    
    ax1.plot(train_losses, label='Total Train Loss', color='tab:blue', alpha=0.4)
    ax1.plot(val_losses, label='Val Task Loss', color='tab:orange', linewidth=2)
    ax1.axhline(y=best_val_loss, color='green', linestyle=':', alpha=0.6)
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss (Log Scale)', color='tab:blue')
    ax1.set_title(f"Global Training Performance (Epoch {epoch+1})")
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="-", alpha=0.1)
    
    ax2 = ax1.twinx()
    ax2.plot(lrs, label='LR', color='tab:red', linestyle='--')
    ax2.set_yscale('log')
    ax2.set_ylabel('Learning Rate', color='tab:red')
    ax2.legend(loc='upper right')
    
    ax3.plot(weighted_task_history, label='Weighted Task Loss (Physics)', color='tab:purple', lw=2)
    ax3.plot(weighted_force_history, label='Weighted Force Loss (Physics)', color='tab:blue', lw=2)
    ax3.plot(weighted_entropy_history, label='Weighted Entropy (Attention)', color='tab:green', lw=2)
    ax3.plot(weighted_pinn_history, label='Weighted PINN Loss (Physics)', color='tab:orange', lw=2)
    ax3.set_yscale('log') 
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Component Loss (Log Scale)')
    ax3.set_title("Newtonian Objective vs. Attention Sharpness")
    ax3.legend(loc='upper right')
    ax3.grid(True, which="both", ls="-", alpha=0.1) 
    
    colors = ['cyan', 'deepskyblue', 'blue', 'darkblue']
    for i in range(4):
        ax4.plot(c_head_histories[i], label=f'C-Head {i+1}', color=colors[i], alpha=0.7)
    max_entropy = np.log(seq_len)
    ax4.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.3, label=f'Uniform Limit (ln({seq_len}))')
    ax4.plot(m_entropy_history, label='Mass-Attn', color='magenta', linestyle='--', lw=2)
    ax4.plot(f_entropy_history, label='Fric-Attn', color='yellow', linestyle=':', lw=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Raw Entropy')
    ax4.set_title("Divergence of Attention Experts")
    ax4.legend(loc='upper right', fontsize='x-small', ncol=2)

    ax5.plot(m_est_history, label='Estimated Mass (m)', color='magenta', linewidth=2)
    ax5.plot(m_gt_history, label='GT Mass', color='magenta', linestyle='--', alpha=0.5)
    ax5.plot(mu_est_history, label='Estimated Fric (μ)', color='orange', linewidth=2)
    ax5.plot(mu_gt_history, label='GT Fric', color='orange', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Parameter Value')
    ax5.set_title("Global Property Estimation Progress (Validation)")
    ax5.legend(loc='upper right', ncol=2)
    ax5.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    plot_save_path = os.path.join(checkpoint_dir, "training_dynamics_full_report.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Final report saved to: {plot_save_path}")

if __name__ == "__main__":
    main()