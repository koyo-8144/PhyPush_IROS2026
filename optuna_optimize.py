import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner

from utils import set_seed, clean_force_col, build_model_string
from models import PhysicsTransformerEstimator
from losses import log_mse_loss, PinnLossCalculator
from dataset import create_dataloaders
from configs import used_config, CSV_PATH

def objective(trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 1. OPTUNA HYPERPARAMETER SEARCH SPACE
    # ==========================================
    config = copy.deepcopy(used_config)
    
    # Override fixed variables
    config['num_epochs'] = 500
    
    # Suggest variables to optimize
    config['init_lr'] = trial.suggest_float('init_lr', 1e-5, 5e-3, log=True)
    config['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 256])
    config['d_model'] = trial.suggest_categorical('d_model', [64, 128, 256])
    config['num_enc'] = trial.suggest_int('num_enc', 2, 6)
    config['dropout'] = trial.suggest_float('dropout', 0.0, 0.4)
    
    # Newly added scaling and sharpness parameters
    config['mass_scale'] = trial.suggest_float('mass_scale', 1.0, 10.0)
    config['fric_scale'] = trial.suggest_float('fric_scale', 0.1, 5.0)
    config['last_layer_ms'] = trial.suggest_float('last_layer_ms', 1.0, 5.0)
    config['last_layer_mus'] = trial.suggest_float('last_layer_mus', 0.5, 2.0)
    config['cross_sharpness'] = trial.suggest_float('cross_sharpness', 1.0, 10.0)
    config['m_sharpness'] = trial.suggest_float('m_sharpness', 1.0, 10.0)
    config['mu_sharpness'] = trial.suggest_float('mu_sharpness', 1.0, 10.0)
    
    # # Tuning the specific PINN coefficient p5
    # config['pinn_coeffs']['p5'] = trial.suggest_float('pinn_p5', 1.0, 20.0)

    # if config['diff_coeffs_pinn4']:
    #     config['pinn_coeffs'].update({'p4': 1.0, 'p4_1': 10.0, 'p4_2': 10.0, 'p4_3': 1.0})

    # ==========================================
    # 2. DATA PREPARATION
    # ==========================================
    df = pd.read_csv(CSV_PATH)
    if 'gt_fric_force' in df.columns:
        df['gt_fric_force'] = df['gt_fric_force'].apply(clean_force_col)

    train_loader, val_loader, seq_len, _, _ = create_dataloaders(
        df, 
        config['batch_size'], 
        config['m_seen_min'], config['m_seen_max'],
        config['mu_seen_min'], config['mu_seen_max']
    )
    config['seq_len'] = seq_len
    frame_mode = config['frame_mode']

    # ==========================================
    # 3. MODEL, OPTIMIZER, LOSS SETUP
    # ==========================================
    model = PhysicsTransformerEstimator(
        input_dim=1,          
        d_model=config['d_model'],           
        nhead=4,              
        num_encoder_layers=config['num_enc'], 
        seq_len=seq_len,
        dropout=config['dropout'],
        sharpness=config['sharpness'],
        cross_sharpness=config['cross_sharpness'],
        m_sharpness=config['m_sharpness'],
        mu_sharpness=config['mu_sharpness'],
        version=config['transformer_ver'],
        max_mass_scale=config['last_layer_ms'],
        max_mu_scale=config['last_layer_mus'],
    ).to(device)

    if config['lr_optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['init_lr'], weight_decay=1e-3)
    elif config['lr_optimizer'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['init_lr'], steps_per_epoch=len(train_loader), 
        epochs=config['num_epochs'], pct_start=0.1, anneal_strategy='cos'
    )

    if config['task_criterion'] == "mse":
        task_cri = nn.MSELoss(reduction='none')
    else:
        task_cri = log_mse_loss

    if config['force_criterion'] == "mse":
        force_cri = nn.MSELoss(reduction='none')
    else:
        force_cri = log_mse_loss

    if config['pinn_criterion'] == "mse":
        pinn_cri = nn.MSELoss(reduction='none')
    elif config['pinn_criterion'] == "log1p_mse":
        pinn_cri = log_mse_loss
    elif config['pinn_criterion'] == "L1":
        pinn_cri = nn.L1Loss(reduction='none')
        
    phys = PinnLossCalculator(pinn_cri)
    
    c_head_coeffs = torch.tensor([config['c_entropy_coeff']] * 4).to(device)
    best_val_loss = float('inf')

    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    for epoch in range(config['num_epochs']):
        
        current_pinn_coeff = 1.0
        if config['pinn_coeff_annealing'] == 1:
            if epoch < config['annealing_start_epoch']:
                current_pinn_coeff = 0.0
            else:
                current_pinn_coeff = min(1.0, (epoch - config['annealing_start_epoch']) / config['ramp_duration'])

        model.train()
        
        for b_acc, b_vel, b_y, b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz, b_start_t, b_robot_fx in train_loader:
            b_acc, b_vel, b_y = b_acc.to(device), b_vel.to(device), b_y.to(device)
            b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz = b_robot_fz.to(device), b_rhs_acc.to(device), b_lhs_net_f.to(device), b_table_fz.to(device)
            b_robot_fx = b_robot_fx.to(device)

            if frame_mode == "local":
                b_robot_fz = -b_robot_fz

            optimizer.zero_grad()
            
            output, (c_weights, m_weights, f_weights), (net_f_est, fric_f_est) = model(b_vel)

            m_gt = b_y[:, 0].unsqueeze(1)
            mu_gt = b_y[:, 1].unsqueeze(1)
            
            net_f_gt = b_lhs_net_f
            normal_f_gt = torch.abs(b_table_fz) 
            fric_f_gt = mu_gt * torch.clamp(normal_f_gt, min=0.0)
            force_gt = torch.cat([net_f_gt, fric_f_gt], dim=-1)
            
            m_est = output[:, 0].unsqueeze(1)
            mu_est = output[:, 1].unsqueeze(1)
            net_f_est = net_f_est.squeeze(-1)
            fric_f_est = fric_f_est.squeeze(-1)
            force_est = torch.cat([net_f_est, fric_f_est], dim=-1)

            if config['loss_type'] == "data":        
                task_loss = task_cri(output, b_y) 
                weighted_task_loss = config['task_coeff'] * task_loss
                force_loss = force_cri(force_gt, force_est)
                weighted_force_loss = config['force_coeff'] * force_loss
                total_loss = weighted_task_loss + weighted_force_loss
                weighted_pinn_loss = torch.tensor(0.0).to(device)
                
            elif config['loss_type'] in ["pinn", "hybrid"]:
                is_accelerating_mask = (torch.abs(b_rhs_acc) > config['acc_filter_threshold']).float()
                mask_net = is_accelerating_mask[:, :seq_len//2]
                
                is_sliding_mask = (torch.abs(b_vel.squeeze(-1)) > config['vel_filter_threshold']).float()
                mask_fric = is_sliding_mask[:, seq_len//2:]
                
                def apply_mask(loss_tensor, mask):
                    active_frames = torch.clamp(mask.sum(), min=1.0)
                    return (loss_tensor * mask).sum() / active_frames

                pinn_net_1 = apply_mask(phys.net_force_law(m_est, b_rhs_acc[:, :seq_len//2], net_f_est[:, :seq_len//2]), mask_net)
                pinn_net_2 = apply_mask(phys.net_force_law(m_est, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2]), mask_net)
                pinn_net_2_ann = pinn_net_2 
                pinn_net_3 = apply_mask(phys.net_force_law(m_gt, b_rhs_acc[:, :seq_len//2], net_f_est[:, :seq_len//2]), mask_net)
                pinn_net_4 = apply_mask(phys.net_force_law_smoothed(m_est, b_rhs_acc[:, :seq_len//2], net_f_est[:, :seq_len//2]), mask_net)
                pinn_net_4_ann = apply_mask(phys.net_force_law_smoothed(m_est, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2]), mask_net)
                pinn_acc_inertia = apply_mask(phys.inertia_acceleration_law(m_est, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net)
                pinn_pos_inertia = apply_mask(phys.inertia_position_law(m_est, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net)

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

                p_c = config['pinn_coeffs']
                ms = config['mass_scale']
                fs = config['fric_scale']
                
                pinn_loss_1 = (ms * pinn_net_1) + (fs * pinn_fric_1)
                pinn_loss_2 = ((ms * pinn_net_2) + (fs * pinn_fric_2)) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else (ms * pinn_net_2) + (fs * pinn_fric_2)
                pinn_loss_3 = ((ms * pinn_net_2) + (fs * pinn_fric_2) + (ms * pinn_net_3) + (fs * pinn_fric_3))
                
                if config['diff_coeffs_pinn4']:
                    pinn_loss_4 = (p_c['p4_1'] * ((ms * pinn_net_2) + (fs * pinn_fric_2)) + p_c['p4_2'] * ((ms * pinn_net_3) + (fs * pinn_fric_3)) + p_c['p4_3'] * ((ms * pinn_net_1) + (fs * pinn_fric_1)))      
                else:
                    pinn_loss_4 = ((ms * pinn_net_2) + (fs * pinn_fric_2) + (ms * pinn_net_3) + (fs * pinn_fric_3) + (ms * pinn_net_1) + (fs * pinn_fric_1))
                
                pinn_loss_5 = (ms * pinn_net_2) + (fs * pinn_fric_4)
                pinn_loss_6 = ((ms * pinn_net_2) + (fs * pinn_fric_4) + (ms * pinn_net_3) + (fs * pinn_fric_3))
                pinn_loss_7 = ((ms * pinn_net_2) + (fs * pinn_fric_4) + (ms * pinn_net_3) + (fs * pinn_fric_3) + (ms * pinn_net_1) + (fs * pinn_fric_1))
                pinn_loss_8 = ((ms * pinn_net_4_ann) + (fs * pinn_fric_5_ann)) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else (ms * pinn_net_4) + (fs * pinn_fric_5)
    
                pinn_loss_9 = (ms * pinn_net_2) + (fs * pinn_acc_consistency) 
                pinn_loss_10 = (ms * pinn_acc_inertia) + (fs * pinn_acc_consistency) 
                pinn_loss_11 = (ms * pinn_pos_inertia) + (fs * pinn_pos_kinematic)  
                
                pinn_loss_2_2 = (fs * pinn_fric_2) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else fs * pinn_fric_2 
                pinn_loss_9_2 = (fs * pinn_acc_consistency_2) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else fs * pinn_acc_consistency_2 
                pinn_loss_11_2 = (fs * pinn_pos_kinematic_2) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else fs * pinn_pos_kinematic_2 

                weighted_pinn_loss = (p_c['p1'] * pinn_loss_1 + p_c['p2'] * pinn_loss_2 + p_c['p3'] * pinn_loss_3 
                                        + p_c['p4'] * pinn_loss_4 + p_c['p5'] * pinn_loss_5 + p_c['p6'] * pinn_loss_6
                                        + p_c['p7'] * pinn_loss_7 + p_c['p8'] * pinn_loss_8 + p_c['p9'] * pinn_loss_9
                                        + p_c['p2-2'] * pinn_loss_2_2 + p_c['p9-2'] * pinn_loss_9_2
                                        + p_c['p10'] * pinn_loss_10 + p_c['p11'] * pinn_loss_11 + p_c['p11-2'] * pinn_loss_11_2)

                if config['loss_type'] == "pinn":
                    total_loss = weighted_pinn_loss
                elif config['loss_type'] == "hybrid":
                    task_loss = task_cri(output, b_y) 
                    weighted_task_loss = config['task_coeff'] * task_loss
                    force_loss = force_cri(force_gt, force_est)
                    weighted_force_loss = config['force_coeff'] * force_loss
                    total_loss = weighted_task_loss + weighted_force_loss + weighted_pinn_loss
            
            if c_weights is not None:
                c_ent_all = -torch.sum(c_weights * torch.log(c_weights + 1e-10), dim=-1) 
                batch_head_entropies = c_ent_all.mean(dim=(0, 2))
                weighted_c_entropy = torch.sum(batch_head_entropies * c_head_coeffs)
            else:
                weighted_c_entropy = torch.tensor(0.0, device=c_head_coeffs.device)
            
            m_ent = -torch.sum(m_weights * torch.log(m_weights + 1e-10), dim=-1).mean()
            f_ent = -torch.sum(f_weights * torch.log(f_weights + 1e-10), dim=-1).mean()
            total_weighted_entropy = weighted_c_entropy + (config['m_entropy_coeff'] * m_ent) + (config['f_entropy_coeff'] * f_ent)
            
            total_loss += total_weighted_entropy
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # ==========================================
        # 5. VALIDATION LOOP
        # ==========================================
        model.eval()
        val_running_loss = 0.0
        
        with torch.no_grad():
            for b_acc, b_vel, b_y, b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz, b_start_t, b_robot_fx in val_loader:
                b_acc, b_vel, b_y = b_acc.to(device), b_vel.to(device), b_y.to(device)
                b_robot_fz, b_rhs_acc, b_lhs_net_f, b_table_fz = b_robot_fz.to(device), b_rhs_acc.to(device), b_lhs_net_f.to(device), b_table_fz.to(device)
                b_robot_fx = b_robot_fx.to(device)

                if frame_mode == "local":
                    b_robot_fz = -b_robot_fz

                output, _, (net_f_est_val, fric_f_est_val) = model(b_vel) 
                
                m_gt_val = b_y[:, 0].unsqueeze(1)
                mu_gt_val = b_y[:, 1].unsqueeze(1)
                m_est_val = output[:, 0].unsqueeze(1)
                mu_est_val = output[:, 1].unsqueeze(1)
                
                net_f_gt_val = b_lhs_net_f 
                normal_f_gt_val = torch.abs(b_table_fz)
                fric_f_gt_val = mu_gt_val * torch.clamp(normal_f_gt_val, min=0.0)
                force_gt_val = torch.cat([net_f_gt_val, fric_f_gt_val], dim=-1)
                force_est_val = torch.cat([net_f_est_val.squeeze(-1), fric_f_est_val.squeeze(-1)], dim=-1)

                if config['loss_type'] == "data":
                    task_v = task_cri(output, b_y)
                    force_v = force_cri(force_gt_val, force_est_val)
                    batch_loss = (config['task_coeff'] * task_v) + (config['force_coeff'] * force_v)

                elif config['loss_type'] in ["pinn", "hybrid"]:
                    is_accelerating_mask_v = (torch.abs(b_rhs_acc) > config['acc_filter_threshold']).float()
                    mask_net_v = is_accelerating_mask_v[:, :seq_len//2]
                    
                    is_sliding_mask_v = (torch.abs(b_vel.squeeze(-1)) > config['vel_filter_threshold']).float()
                    mask_fric_v = is_sliding_mask_v[:, seq_len//2:]
                    
                    def apply_mask_v(loss_tensor, mask):
                        active_frames = torch.clamp(mask.sum(), min=1.0)
                        return (loss_tensor * mask).sum() / active_frames

                    pinn_net_v1 = apply_mask_v(phys.net_force_law(m_est_val, b_rhs_acc[:, :seq_len//2], net_f_est_val[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v2 = apply_mask_v(phys.net_force_law(m_est_val, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v2_ann = pinn_net_v2
                    pinn_net_v3 = apply_mask_v(phys.net_force_law(m_gt_val, b_rhs_acc[:, :seq_len//2], net_f_est_val[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v4 = apply_mask_v(phys.net_force_law_smoothed(m_est_val, b_rhs_acc[:, :seq_len//2], net_f_est_val[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_net_v4_ann = apply_mask_v(phys.net_force_law_smoothed(m_est_val, b_rhs_acc[:, :seq_len//2], b_lhs_net_f[:, :seq_len//2].squeeze(-1)), mask_net_v)
                    pinn_acc_inertia_v = apply_mask_v(phys.inertia_acceleration_law(m_est_val, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net_v)
                    pinn_pos_inertia_v = apply_mask_v(phys.inertia_position_law(m_est_val, b_lhs_net_f[:, :seq_len//2], b_rhs_acc[:, :seq_len//2]), mask_net_v)
                    
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
                    
                    p_c = config['pinn_coeffs']
                    ms = config['mass_scale']
                    fs = config['fric_scale']

                    pinn_loss_v1 = (ms * pinn_net_v1) + (fs * pinn_fric_v1)
                    pinn_loss_v2 = ((ms * pinn_net_v2) + (fs * pinn_fric_v2)) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else (ms * pinn_net_v2) + (fs * pinn_fric_v2)
                    pinn_loss_v3 = ((ms * pinn_net_v2) + (fs * pinn_fric_v2) + (ms * pinn_net_v3) + (fs * pinn_fric_v3))
                    
                    if config['diff_coeffs_pinn4']:
                        pinn_loss_v4 = (p_c['p4_1'] * ((ms * pinn_net_v2) + (fs * pinn_fric_v2)) + p_c['p4_2'] * ((ms * pinn_net_v3) + (fs * pinn_fric_v3)) + p_c['p4_3'] * ((ms * pinn_net_v1) + (fs * pinn_fric_v1)))      
                    else:
                        pinn_loss_v4 = ((ms * pinn_net_v2) + (fs * pinn_fric_v2) + (ms * pinn_net_v3) + (fs * pinn_fric_v3) + (ms * pinn_net_v1) + (fs * pinn_fric_v1))
                    
                    pinn_loss_v5 = (ms * pinn_net_v2) + (fs * pinn_fric_v4)
                    pinn_loss_v6 = ((ms * pinn_net_v2) + (fs * pinn_fric_v4) + (ms * pinn_net_v3) + (fs * pinn_fric_v3))
                    pinn_loss_v7 = ((ms * pinn_net_v2) + (fs * pinn_fric_v4) + (ms * pinn_net_v3) + (fs * pinn_fric_v3) + (ms * pinn_net_v1) + (fs * pinn_fric_v1))
                    pinn_loss_v8 = ((ms * pinn_net_v4_ann) + (fs * pinn_fric_v5_ann)) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else (ms * pinn_net_v4) + (fs * pinn_fric_v5)
                    
                    pinn_loss_v9 = (ms * pinn_net_v2) + (fs * pinn_acc_consistency_v)
                    pinn_loss_v10 = (ms * pinn_acc_inertia_v) + (fs * pinn_acc_consistency_v)
                    pinn_loss_v11 = (ms * pinn_pos_inertia_v) + (fs * pinn_pos_kinematic_v) 
            
                    pinn_loss_v2_2 = (fs * pinn_fric_v2) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else fs * pinn_fric_v2
                    pinn_loss_v9_2 = (fs * pinn_acc_consistency_v_2) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else fs * pinn_acc_consistency_v_2
                    pinn_loss_v11_2 = (fs * pinn_pos_kinematic_v_2) * current_pinn_coeff if config['pinn_coeff_annealing'] == 1 else fs * pinn_pos_kinematic_v_2
                    
                    weighted_pinn_loss_val = ((p_c['p1'] * pinn_loss_v1) + (p_c['p2'] * pinn_loss_v2) 
                                  + (p_c['p3'] * pinn_loss_v3) + (p_c['p4'] * pinn_loss_v4) 
                                  + (p_c['p5'] * pinn_loss_v5) + (p_c['p6'] * pinn_loss_v6)
                                  + (p_c['p7'] * pinn_loss_v7) + (p_c['p8'] * pinn_loss_v8)
                                  + (p_c['p9'] * pinn_loss_v9) + (p_c['p2-2'] * pinn_loss_v2_2)
                                  + (p_c['p9-2'] * pinn_loss_v9_2) + (p_c['p10'] * pinn_loss_v10)
                                  + (p_c['p11'] * pinn_loss_v11) + (p_c['p11-2'] * pinn_loss_v11_2))

                    if config['loss_type'] == "pinn":
                        batch_loss = weighted_pinn_loss_val
                    elif config['loss_type'] == "hybrid":
                        task_v = task_cri(output, b_y)
                        force_v = force_cri(force_gt_val, force_est_val)
                        batch_loss = (config['task_coeff'] * task_v) + (config['force_coeff'] * force_v) + weighted_pinn_loss_val
                    
                val_running_loss += batch_loss.item() * b_acc.size(0)
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        # Optuna Pruning step
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            
    return best_val_loss

if __name__ == "__main__":
    print("Starting Hyperparameter Optimization with Optuna...")
    
    # 1. Ensure the output directory exists
    os.makedirs("./results/optuna", exist_ok=True)
    
    # 2. Define a persistent SQLite database path and a name for this study
    storage_name = "sqlite:///results/optuna/phypush_tuning.db"
    study_name = "phypush_hyperparam_search"
    
    # 3. Create or load the study using the storage
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # This allows you to stop the script and resume later!
        direction="minimize", 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    )
    
    # Run 1000 trials
    study.optimize(objective, n_trials=1000)

    print("\nOptimization Finished.")
    print("Best Trial:")
    best_trial = study.best_trial

    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters to JSON
    with open("./results/optuna/best_params.json", "w") as f:
        json.dump(best_trial.params, f, indent=4)