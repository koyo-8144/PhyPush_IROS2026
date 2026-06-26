import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def clean_force_col(x):
    if isinstance(x, str):
        return float(x.strip('[]'))
    return x

def get_sequence_data(row, prefix, seq_len=100, num_axes=6):
    """Reconstructs array from flattened columns like 'prefix_t0_ax0'"""
    data = np.zeros((seq_len, num_axes))
    for t in range(seq_len):
        for ax in range(num_axes):
            col = f"{prefix}_t{t}_ax{ax}"
            if col in row:
                data[t, ax] = row[col]
    return data

def get_flat_window(row, prefix, window_size=10):
    """Reconstructs 1D array from columns like 'prefix_0', 'prefix_1'"""
    data = np.zeros(window_size)
    for i in range(window_size):
        col = f"{prefix}_{i}"
        if col in row:
            data[i] = row[col]
    return data

def add_min_max_text(ax, data, unit=""):
    dmin, dmax = np.min(data), np.max(data)
    stats_text = f"Min: {dmin:.3f} {unit}\nMax: {dmax:.3f} {unit}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            horizontalalignment='right', verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

def build_model_string(config):
    parts = [config.get('loss_type', 'pinn')]
    
    if config.get('pinn_coeff_annealing', 0) != 0:
        parts.append(f"annstartepo{config.get('annealing_start_epoch')}")
        parts.append(f"rampd{config.get('ramp_duration')}")
    
    if config.get('task_coeff', 0.0) != 0.0:
        parts.append(f"tcri-{config.get('task_criterion')}") 
        parts.append(f"task{config.get('task_coeff')}")
        
    if config.get('force_coeff', 0.0) != 0.0:
        parts.append(f"fcri-{config.get('force_criterion')}")
        parts.append(f"force{config.get('force_coeff')}")

    pinn_coeffs = config.get('pinn_coeffs', {})
    if any(pinn_coeffs.values()):
        parts.append(f"pcri-{config.get('pinn_criterion')}")
        
        for i in range(1, 12):
            val = pinn_coeffs.get(f"p{i}", 0.0)
            if val != 0.0:
                parts.append(f"p{i}c{val}")
        
        for key in ["p2-2", "p9-2", "p9-3", "p11-2"]:
            val = pinn_coeffs.get(key, 0.0)
            if val != 0.0:
                parts.append(f"{key}c{val}") 

    return "_".join(parts)