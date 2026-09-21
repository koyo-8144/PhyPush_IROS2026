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
REAL_M_MAX = 0.702
REAL_M_MIN = 0.340
REAL_MU_MAX = 0.4334
REAL_MU_MIN = 0.255
REAL_M_RANGE = REAL_M_MAX - REAL_M_MIN
REAL_MU_RANGE = REAL_MU_MAX - REAL_MU_MIN
REAL_FRIC_RANGE = (REAL_M_MAX * REAL_MU_MAX * G) - (REAL_M_MIN * REAL_MU_MIN * G)

# FRAME_MODE = "world"
FRAME_MODE = "local"

# =============================================================================
# MULTI_ANGLE
#
# True  -> dataset from run_collect_sweep_sb.sh: each (mass, mu) pair observed
#          from 10 object yaws x 2 push sides (COLLECT_IDX 0..19) across several
#          RANDOM_SEED values.
# False -> original single-orientation dataset.
#
# When True, dataset.py switches to a GROUP split keyed on the property pair.
# A plain random split would place the same (mass, mu) pair in both train and
# val (it appears ~20 times), making validation error meaningless.
# =============================================================================
MULTI_ANGLE = True

# =============================================================================
# INPUT_VARIANT
#
# The 60-step end-effector VELOCITY sequence is ALWAYS the input. This selects
# what, if anything, is supplied alongside it.
#
#   "vel_only"           velocity sequence only.                    baseline
#                        input_dim=1, cond_dim=0
#
#   "vel_manip_cond"     + worst (minimum) manipulability over the
#                        SAME 60 steps, as a static conditioning
#                        token prepended to the encoder sequence.
#                        input_dim=1, cond_dim=1
#
#   "vel_dirmanip_cond"  + worst (minimum) DIRECTIONAL manipulability
#                        over the same 60 steps, as a static token.
#                        input_dim=1, cond_dim=1
#
#   "vel_manip_seq"      + the full 60-step manipulability sequence,
#                        as a second input CHANNEL alongside velocity.
#                        input_dim=2, cond_dim=0
#
#   "vel_dirmanip_seq"   + the full 60-step directional manipulability
#                        sequence, as a second input channel.
#                        input_dim=2, cond_dim=0
#
#   "vel_osim_seq"       + the 60-step OPERATIONAL-SPACE INERTIA sequence,
#                        as a second input channel. The CSV stores one
#                        scalar per step (arm_lam_w*): the mean of the
#                        translational diagonal of
#                        Lambda = (J M^-1 J^T)^-1, not the full 6x6 matrix.
#                        input_dim=2, cond_dim=0
#
#   "vel_eff_seq"        + the 60-step EFFECTIVE MASS sequence along the
#                        push direction u, m_eff = 1 / (u^T Lambda^-1 u)
#                        (arm_meff_w*), as a second input channel.
#                        input_dim=2, cond_dim=0
#
# The two inertial channels are LOG-transformed before standardization
# (VARIANT_LOG_TRANSFORM). Lambda is a matrix inverse and becomes heavy-tailed
# near kinematic singularities; a single linear mean/std would let a few such
# rows set the scale for everything else. Rows with a non-positive or
# non-finite value in the channel (a failed solve, written as 0.0 by the
# collector) are dropped for these variants.
#
# Note on the two scalar variants: the minimum is taken over the 60-step
# INFERENCE WINDOW, recomputed here from arm_manip_w* / arm_dir_manip_w*. It is
# NOT the `worst_manipulability` / `worst_dir_manipulability` column, which is
# the minimum over the WHOLE push (~300 steps, including approach and
# post-impact). Taking it over the window keeps all four variants describing the
# same slice of time as the velocity input, so they are comparable.
# =============================================================================
# INPUT_VARIANT = "vel_only"
# INPUT_VARIANT = "vel_manip_cond"
# INPUT_VARIANT = "vel_dirmanip_cond"
# INPUT_VARIANT = "vel_manip_seq"
INPUT_VARIANT = "vel_dirmanip_seq"
# INPUT_VARIANT = "vel_osim_seq"
# INPUT_VARIANT = "vel_eff_seq"

GRIPPER_CLOSED = True

# Single source of truth for every variant.
#   name -> (window column prefix, cond_dim, input_dim, log-transform channel)
VARIANT_TABLE = {
    "vel_only":          (None,              0, 1, False),
    "vel_manip_cond":    ("arm_manip_w",     1, 1, False),
    "vel_dirmanip_cond": ("arm_dir_manip_w", 1, 1, False),
    "vel_manip_seq":     ("arm_manip_w",     0, 2, False),
    "vel_dirmanip_seq":  ("arm_dir_manip_w", 0, 2, False),
    "vel_osim_seq":      ("arm_lam_w",       0, 2, True),
    "vel_eff_seq":       ("arm_meff_w",      0, 2, True),
}
_VARIANTS = tuple(VARIANT_TABLE)
if INPUT_VARIANT not in VARIANT_TABLE:
    raise ValueError(f"Unknown INPUT_VARIANT {INPUT_VARIANT!r}. Expected one of {_VARIANTS}.")

# Derived, so train.py / evaluate.py never hardcode these.
(VARIANT_WINDOW_PREFIX, COND_DIM, INPUT_DIM,
 VARIANT_LOG_TRANSFORM) = VARIANT_TABLE[INPUT_VARIANT]
USE_ARM_STATE = INPUT_VARIANT != "vel_only"

# Kept for backwards compatibility with older sidecars / scripts that still read
# ARM_FEATURE_MODE. It now just mirrors the variant.
ARM_FEATURE_MODE = INPUT_VARIANT


if FRAME_MODE == "world":
    CSV_PATH = "/home/psxkf4/IsaacLab/source/collected_data/data_tb-3_ta57_emavel1.0_velstd0.0_broad.csv"
elif FRAME_MODE == "local":
    if MULTI_ANGLE:
        if GRIPPER_CLOSED:
            CSV_PATH = ("/home/psxkf4/IsaacLab/source/collected_data/"
                        "data_cube_closed_gripper_multi_angle_sb3/"
                        "data_cube_closed_gripper_multi_angle_sb3.csv")
        else:
            CSV_PATH = ("/home/psxkf4/IsaacLab/source/collected_data/"
                                "data_cube_opened_gripper_multi_angle_sb3/"
                                "data_cube_opened_gripper_multi_angle_sb3.csv")
    else:
        CSV_PATH = "/home/psxkf4/IsaacLab/source/collected_data/data_cube_closed_gripper.csv"


# =============================================================================
# PUSH QUALITY FILTER
#
# push_start_lateral_offset is the lateral distance from the object centre to
# the ACHIEVED push line, measured BEFORE the push. Only lateral error torques
# the object, and a rotating push violates the pure-translation assumption the
# PINN residuals rest on.
#
# Set to None to disable. 0.005 (5 mm) matches the threshold the collection
# diagnostic reports against.
#
# NOT the same as push_com_offset, which is the object's POST-push sideways
# deviation -- an outcome, not a setup quality, and occasionally metres wide
# when a physics blowup throws the object off the table.
# =============================================================================
MAX_PUSH_LATERAL_OFFSET = 0.005
MAX_PUSH_COM_OFFSET = 0.05          # loose: drops physics blowups only


config_multi_angle = {
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
    'multi_angle': MULTI_ANGLE,
    # --- variant plumbing ---
    'input_variant': INPUT_VARIANT,
    'use_arm_state': USE_ARM_STATE,
    'arm_feature_mode': ARM_FEATURE_MODE,
    'cond_dim': COND_DIM,
    'input_dim': INPUT_DIM,
    'variant_window_prefix': VARIANT_WINDOW_PREFIX,
    'variant_log_transform': VARIANT_LOG_TRANSFORM,
    'max_push_lateral_offset': MAX_PUSH_LATERAL_OFFSET,
    'max_push_com_offset': MAX_PUSH_COM_OFFSET,
}

used_config = config_multi_angle