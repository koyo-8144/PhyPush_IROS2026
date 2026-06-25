M_SEEN_MAX = 2.0
M_SEEN_MIN = 0.2
MU_SEEN_MAX = 0.5
MU_SEEN_MIN = 0.15

# Unseen (OOD) boundaries
M_UNSEEN_MAX = 3.0
MU_UNSEEN_MAX = 0.7

INCLUDE_UNSEEN = True
# Calculated Global Ranges for Simulation Evaluation
if INCLUDE_UNSEEN:
    GLOBAL_M_RANGE = M_UNSEEN_MAX - M_SEEN_MIN  
    GLOBAL_MU_RANGE = MU_UNSEEN_MAX - MU_SEEN_MIN 
else:
    GLOBAL_M_RANGE = M_SEEN_MAX - M_SEEN_MIN  
    GLOBAL_MU_RANGE = MU_SEEN_MAX - MU_SEEN_MIN 

G = 9.81
GLOBAL_FRIC_RANGE = (M_SEEN_MAX * MU_SEEN_MAX * G) - (M_SEEN_MIN * MU_SEEN_MIN * G)

# Real-World Ranges
REAL_M_RANGE = 1.856 - 0.175
REAL_MU_RANGE = 0.4626 - 0.1580
REAL_FRIC_RANGE = (1.856 * 0.4626 * G) - (0.175 * 0.1580 * G)

# FRAME_MODE = "world"
FRAME_MODE = "local"

if FRAME_MODE == "world":
    CSV_PATH = "/home/psxkf4/IsaacLab/source/collected_data/data_tb-3_ta57_emavel1.0_velstd0.0_broad.csv" # force
elif FRAME_MODE == "local":
    # CSV_PATH = "/home/psxkf4/IsaacLab/source/collected_data/data_trans_cube.csv" # force_v2, force_v3
    CSV_PATH = "/home/psxkf4/IsaacLab/source/collected_data/data_cube_closed_gripper.csv" # force_v4


config_data = {
    'batch_size': 64,
    'num_epochs': 1000,
    'lr_optimizer': "AdamW",
    'lr_scheduler': "OneCycle",
    'loss_type': "pinn",
    'task_coeff': 10.0,
    'task_criterion': "log1p_mse",
    'c_entropy_coeff': 0.0,
    'm_entropy_coeff': 0.0,
    'f_entropy_coeff': 0.0,
    'force_coeff': 0.0,
    'force_criterion': "log1p_mse",
    'd_model': 64,
    'num_enc': 4,
    'last_layer_ms': 2.0,
    'last_layer_mus': 1.0,
    'dropout': 0.1,
    'sharpness': 1.0,
    'cross_sharpness': 5.0,
    'm_sharpness': 5.0,
    'mu_sharpness': 5.0,
    'init_lr': 3e-4,
    'pinn_criterion': "L1",
    'diff_coeffs_pinn4': 0,
    'pinn_coeffs': {
        'p1': 0.0, 'p2': 0.0, 'p2-2': 0.0, 'p3': 0.0,
        'p4': 0.0, 'p4_1': 0.0, 'p4_2': 0.0, 'p4_3': 0.0,
        'p5': 0.0, 'p6': 0.0, 'p7': 0.0, 'p8': 0.0,
        'p9': 0.0, 'p9-2': 0.0, 'p9-3': 0.0,
        'p10': 0.0, 'p11': 0.0, 'p11-2': 0.0
    },
    'mass_scale': 7.5,
    'fric_scale': 1.0,
    'pinn_coeff_annealing': 0,
    'annealing_start_epoch': 300,
    'ramp_duration': 600,
    'm_seen_max': M_SEEN_MAX,
    'm_seen_min': M_SEEN_MIN,
    'mu_seen_max': MU_SEEN_MAX,
    'mu_seen_min': MU_SEEN_MIN,
    'acc_filter_threshold': 0.3,
    'vel_filter_threshold': 0.01,
    'transformer_ver': 5,
    'frame_mode': FRAME_MODE,
}


config_force = {
    'batch_size': 64,
    'num_epochs': 1000,
    'lr_optimizer': "AdamW",
    'lr_scheduler': "OneCycle",
    'loss_type': "pinn",
    'task_coeff': 0.0,
    'task_criterion': "log1p_mse",
    'c_entropy_coeff': 0.0,
    'm_entropy_coeff': 0.0,
    'f_entropy_coeff': 0.0,
    'force_coeff': 0.0,
    'force_criterion': "log1p_mse",
    'd_model': 64,
    'num_enc': 4,
    'last_layer_ms': 2.0,
    'last_layer_mus': 1.0,
    'dropout': 0.1,
    'sharpness': 1.0,
    'cross_sharpness': 5.0,
    'm_sharpness': 5.0,
    'mu_sharpness': 5.0,
    'init_lr': 3e-4,
    'pinn_criterion': "L1",
    'diff_coeffs_pinn4': 0,
    'pinn_coeffs': {
        'p1': 0.0, 'p2': 0.0, 'p2-2': 0.0, 'p3': 0.0,
        'p4': 0.0, 'p4_1': 0.0, 'p4_2': 0.0, 'p4_3': 0.0,
        'p5': 10.0, 'p6': 0.0, 'p7': 0.0, 'p8': 0.0,
        'p9': 0.0, 'p9-2': 0.0, 'p9-3': 0.0,
        'p10': 0.0, 'p11': 0.0, 'p11-2': 0.0
    },
    'mass_scale': 7.5,
    'fric_scale': 1.0,
    'pinn_coeff_annealing': 0,
    'annealing_start_epoch': 300,
    'ramp_duration': 600,
    'm_seen_max': M_SEEN_MAX,
    'm_seen_min': M_SEEN_MIN,
    'mu_seen_max': MU_SEEN_MAX,
    'mu_seen_min': MU_SEEN_MIN,
    'acc_filter_threshold': 0.3,
    'vel_filter_threshold': 0.01,
    'transformer_ver': 5,
    'frame_mode': FRAME_MODE,
}


config_force_v2 = {
    'batch_size': 64,
    'num_epochs': 1000,
    'lr_optimizer': "AdamW",
    'lr_scheduler': "OneCycle",
    'loss_type': "pinn",
    'task_coeff': 0.0,
    'task_criterion': "log1p_mse",
    'c_entropy_coeff': 0.0,
    'm_entropy_coeff': 0.0,
    'f_entropy_coeff': 0.0,
    'force_coeff': 0.0,
    'force_criterion': "log1p_mse",
    'd_model': 128,
    'num_enc': 4,
    'last_layer_ms': 3.24267634686027,
    'last_layer_mus': 0.9333627947697261,
    'dropout': 0.020576753511250212,
    'sharpness': 1.0,
    'cross_sharpness': 7.43379868747758,
    'm_sharpness': 9.20706918441079,
    'mu_sharpness': 1.0318866842435197,
    'init_lr': 1.4143827128448119e-05,
    'pinn_criterion': "L1",
    'diff_coeffs_pinn4': 0,
    'pinn_coeffs': {
        'p1': 0.0, 'p2': 0.0, 'p2-2': 0.0, 'p3': 0.0,
        'p4': 0.0, 'p4_1': 0.0, 'p4_2': 0.0, 'p4_3': 0.0,
        'p5': 10.0, 'p6': 0.0, 'p7': 0.0, 'p8': 0.0,
        'p9': 0.0, 'p9-2': 0.0, 'p9-3': 0.0,
        'p10': 0.0, 'p11': 0.0, 'p11-2': 0.0
    },
    'mass_scale': 1.7914125228896531,
    'fric_scale': 4.01667630154419,
    'pinn_coeff_annealing': 0,
    'annealing_start_epoch': 300,
    'ramp_duration': 600,
    'm_seen_max': M_SEEN_MAX,
    'm_seen_min': M_SEEN_MIN,
    'mu_seen_max': MU_SEEN_MAX,
    'mu_seen_min': MU_SEEN_MIN,
    'acc_filter_threshold': 0.3,
    'vel_filter_threshold': 0.01,
    'transformer_ver': 5,
    'frame_mode': FRAME_MODE,
}


config_force_v3 = {
    'batch_size': 64,
    'num_epochs': 1000,
    'lr_optimizer': "AdamW",
    'lr_scheduler': "OneCycle",
    'loss_type': "pinn",
    'task_coeff': 0.0,
    'task_criterion': "log1p_mse",
    'c_entropy_coeff': 0.0,
    'm_entropy_coeff': 0.0,
    'f_entropy_coeff': 0.0,
    'force_coeff': 0.0,
    'force_criterion': "log1p_mse",
    'd_model': 64,
    'num_enc': 6,
    'last_layer_ms': 3.1557037339649225,
    'last_layer_mus': 1.3244787508344682,
    'dropout': 0.03707939352522528,
    'sharpness': 1.0,
    'cross_sharpness': 7.43379868747758,
    'm_sharpness': 9.20706918441079,
    'mu_sharpness': 1.0318866842435197,
    'cross_sharpness': 9.628721199478191,
    'm_sharpness': 9.994619264743257,
    'mu_sharpness': 7.8646360451996005,
    'init_lr': 0.0010060140244026493,
    'pinn_criterion': "L1",
    'diff_coeffs_pinn4': 0,
    'pinn_coeffs': {
        'p1': 0.0, 'p2': 0.0, 'p2-2': 0.0, 'p3': 0.0,
        'p4': 0.0, 'p4_1': 0.0, 'p4_2': 0.0, 'p4_3': 0.0,
        'p5': 10.0, 'p6': 0.0, 'p7': 0.0, 'p8': 0.0,
        'p9': 0.0, 'p9-2': 0.0, 'p9-3': 0.0,
        'p10': 0.0, 'p11': 0.0, 'p11-2': 0.0
    },
    'mass_scale': 1.2296503610329115,
    'fric_scale': 0.2826182834972651,
    'pinn_coeff_annealing': 0,
    'annealing_start_epoch': 300,
    'ramp_duration': 600,
    'm_seen_max': M_SEEN_MAX,
    'm_seen_min': M_SEEN_MIN,
    'mu_seen_max': MU_SEEN_MAX,
    'mu_seen_min': MU_SEEN_MIN,
    'acc_filter_threshold': 0.3,
    'vel_filter_threshold': 0.01,
    'transformer_ver': 5,
    'frame_mode': FRAME_MODE,
}


config_force_v4 = {
    'batch_size': 256,
    'num_epochs': 1000,
    'lr_optimizer': "AdamW",
    'lr_scheduler': "OneCycle",
    'loss_type': "pinn",
    'task_coeff': 0.0,
    'task_criterion': "log1p_mse",
    'c_entropy_coeff': 0.0,
    'm_entropy_coeff': 0.0,
    'f_entropy_coeff': 0.0,
    'force_coeff': 0.0,
    'force_criterion': "log1p_mse",
    'd_model': 64,
    'num_enc': 4,
    'last_layer_ms': 1.192172043937462,
    'last_layer_mus': 1.7910809746812413,
    'dropout': 0.0004615346900806658,
    'sharpness': 1.0,
    'cross_sharpness': 1.2965844927099692,
    'm_sharpness': 3.9822290389819024,
    'mu_sharpness': 9.814362222938573,
    'init_lr': 9.223299520640666e-05,
    'pinn_criterion': "L1",
    'diff_coeffs_pinn4': 0,
    'pinn_coeffs': {
        'p1': 0.0, 'p2': 0.0, 'p2-2': 0.0, 'p3': 0.0,
        'p4': 0.0, 'p4_1': 0.0, 'p4_2': 0.0, 'p4_3': 0.0,
        'p5': 10.0, 'p6': 0.0, 'p7': 0.0, 'p8': 0.0,
        'p9': 0.0, 'p9-2': 0.0, 'p9-3': 0.0,
        'p10': 0.0, 'p11': 0.0, 'p11-2': 0.0
    },
    'mass_scale': 1.0036750460870603,
    'fric_scale': 0.49229772763197466,
    'pinn_coeff_annealing': 0,
    'annealing_start_epoch': 300,
    'ramp_duration': 600,
    'm_seen_max': M_SEEN_MAX,
    'm_seen_min': M_SEEN_MIN,
    'mu_seen_max': MU_SEEN_MAX,
    'mu_seen_min': MU_SEEN_MIN,
    'acc_filter_threshold': 0.3,
    'vel_filter_threshold': 0.01,
    'transformer_ver': 5,
    'frame_mode': FRAME_MODE,
}



used_config = config_force_v4