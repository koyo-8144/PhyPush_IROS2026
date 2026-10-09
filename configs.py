import os

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
# A variant is now just an ORDERED LIST OF PER-STEP CHANNELS fed to the encoder,
# plus an optional static conditioning scalar. Velocity is no longer assumed:
# it is channel 'input_vel', and a variant that omits it is a genuine
# velocity-free ablation.
#
#   --- velocity-based (unchanged behaviour) ---
#   "vel_only"           velocity only.                               baseline
#   "vel_manip_cond"     velocity + worst (minimum) manipulability over the
#                        SAME 60 steps, as a static conditioning token
#                        prepended to the encoder sequence.
#   "vel_dirmanip_cond"  velocity + worst (minimum) DIRECTIONAL manipulability
#                        over the same 60 steps, as a static token.
#   "vel_manip_seq"      velocity + full 60-step manipulability, 2nd channel.
#   "vel_dirmanip_seq"   velocity + full 60-step directional manipulability.
#   "vel_osim_seq"       velocity + 60-step OPERATIONAL-SPACE INERTIA
#                        (arm_lam_w*: the mean of the translational diagonal of
#                        Lambda = (J M^-1 J^T)^-1, one scalar per step).
#   "vel_eff_seq"        velocity + 60-step EFFECTIVE MASS along the push
#                        direction u, m_eff = 1 / (u^T Lambda^-1 u)
#                        (arm_meff_w*).
#
#   --- NEW: velocity REPLACED by an inertial channel ---
#   "osim_only"          arm_lam_w  only.                   input_dim=1
#   "osim_manip"         arm_lam_w  + manipulability.       input_dim=2
#   "osim_dirmanip"      arm_lam_w  + directional manip.    input_dim=2
#   "eff_only"           arm_meff_w only.                   input_dim=1
#   "eff_manip"          arm_meff_w + manipulability.       input_dim=2
#   "eff_dirmanip"       arm_meff_w + directional manip.    input_dim=2
#
#   These six carry NO velocity channel, by design: they answer whether the
#   arm's own inertial signature alone identifies the object, independently of
#   the measured object motion. The velocity sequence is still loaded and still
#   drives the PINN sliding mask (vel_filter_threshold) -- it is withheld from
#   the MODEL INPUT only. If you meant "velocity PLUS these two channels",
#   add the entry with VEL_PREFIX first; nothing else needs to change.
#
# The two inertial channels are LOG-transformed before standardization
# (see CHANNEL_LOG). Lambda is a matrix inverse and becomes heavy-tailed near
# kinematic singularities; a single linear mean/std would let a few such rows
# set the scale for everything else. Rows with a non-positive or non-finite
# value in ANY log channel the variant uses (a failed solve, written as 0.0 by
# the collector) are dropped.
#
# Note on the scalar (cond) variants: the minimum is taken over the 60-step
# INFERENCE WINDOW, recomputed from arm_manip_w* / arm_dir_manip_w*. It is NOT
# the `worst_manipulability` / `worst_dir_manipulability` column, which is the
# minimum over the WHOLE push (~300 steps, including approach and post-impact).
# =============================================================================
# INPUT_VARIANT = "vel_only"
# INPUT_VARIANT = "vel_manip_cond"
# INPUT_VARIANT = "vel_dirmanip_cond"
# INPUT_VARIANT = "vel_manip_seq"
# INPUT_VARIANT = "vel_dirmanip_seq"
# INPUT_VARIANT = "vel_osim_seq"
# INPUT_VARIANT = "vel_eff_seq"
# INPUT_VARIANT = "osim_only"
# INPUT_VARIANT = "osim_manip"
# INPUT_VARIANT = "osim_dirmanip"
INPUT_VARIANT = "eff_only"
# INPUT_VARIANT = "eff_manip"
# INPUT_VARIANT = "eff_dirmanip"

# --- Environment override -----------------------------------------------------
# run_variants.sh sweeps variants by setting these, so a sweep never has to
# rewrite this file (which would race if two runs overlap). Unset -> whatever is
# selected above. A bad value is caught by the VARIANT_TABLE check below, which
# prints the full list of valid names.
INPUT_VARIANT = os.environ.get("PHYPUSH_INPUT_VARIANT", INPUT_VARIANT)

GRIPPER_CLOSED = True
_gc = os.environ.get("PHYPUSH_GRIPPER_CLOSED")
if _gc is not None:
    if _gc.strip().lower() not in ("0", "1", "true", "false"):
        raise ValueError(
            f"PHYPUSH_GRIPPER_CLOSED must be 0/1/true/false, got {_gc!r}.")
    GRIPPER_CLOSED = _gc.strip().lower() in ("1", "true")

# =============================================================================
# CHANNEL REGISTRY
#
# prefix -> is it log-transformed before standardization?
#
# 'input_vel' is a MARKER, not a window prefix: velocity comes from the
# input_vel_0..59 columns via the existing vel_cols path and is fed RAW
# (unstandardized), exactly as before. It is never passed to window_cols().
# =============================================================================
VEL_PREFIX = "input_vel"

CHANNEL_LOG = {
    VEL_PREFIX:        False,   # raw, never standardized
    "arm_manip_w":     False,   # manipulability w
    "arm_dir_manip_w": False,   # directional manipulability w_dir
    # "arm_lam_w":       True,    # mean translational diag of Lambda [kg]
    # "arm_meff_w":      True,    # effective mass along push dir [kg]
    "arm_lam_w":       False,    # mean translational diag of Lambda [kg]
    "arm_meff_w":      False,    # effective mass along push dir [kg]
}

# Single source of truth for every variant.
#   name -> (ordered tuple of sequence channels, cond channel or None)
VARIANT_TABLE = {
    "vel_only":          ((VEL_PREFIX,),                       None),
    "vel_manip_cond":    ((VEL_PREFIX,),                       "arm_manip_w"),
    "vel_dirmanip_cond": ((VEL_PREFIX,),                       "arm_dir_manip_w"),
    "vel_manip_seq":     ((VEL_PREFIX, "arm_manip_w"),         None),
    "vel_dirmanip_seq":  ((VEL_PREFIX, "arm_dir_manip_w"),     None),
    "vel_osim_seq":      ((VEL_PREFIX, "arm_lam_w"),           None),
    "vel_eff_seq":       ((VEL_PREFIX, "arm_meff_w"),          None),
    # --- velocity-free ---
    "osim_only":         (("arm_lam_w",),                      None),
    "osim_manip":        (("arm_lam_w", "arm_manip_w"),        None),
    "osim_dirmanip":     (("arm_lam_w", "arm_dir_manip_w"),    None),
    "eff_only":          (("arm_meff_w",),                     None),
    "eff_manip":         (("arm_meff_w", "arm_manip_w"),       None),
    "eff_dirmanip":      (("arm_meff_w", "arm_dir_manip_w"),   None),
}
_VARIANTS = tuple(VARIANT_TABLE)
if INPUT_VARIANT not in VARIANT_TABLE:
    raise ValueError(f"Unknown INPUT_VARIANT {INPUT_VARIANT!r}. Expected one of {_VARIANTS}.")

# =============================================================================
# DERIVED -- train.py / evaluate.py / dataset.py never hardcode any of this.
# =============================================================================
VARIANT_CHANNELS, VARIANT_COND_PREFIX = VARIANT_TABLE[INPUT_VARIANT]

_unknown = [c for c in VARIANT_CHANNELS if c not in CHANNEL_LOG]
if _unknown:
    raise ValueError(f"Variant {INPUT_VARIANT!r} references unregistered channels {_unknown}.")
if VARIANT_COND_PREFIX is not None and VARIANT_COND_PREFIX not in CHANNEL_LOG:
    raise ValueError(f"Variant {INPUT_VARIANT!r} cond channel {VARIANT_COND_PREFIX!r} is unregistered.")
if VARIANT_COND_PREFIX == VEL_PREFIX:
    raise ValueError("Velocity cannot be a static conditioning scalar.")
if len(set(VARIANT_CHANNELS)) != len(VARIANT_CHANNELS):
    raise ValueError(f"Variant {INPUT_VARIANT!r} repeats a channel.")
if not VARIANT_CHANNELS:
    raise ValueError(f"Variant {INPUT_VARIANT!r} has no input channels.")

USE_VEL = VEL_PREFIX in VARIANT_CHANNELS

# The per-step WINDOW channels (everything except velocity), in model order.
VARIANT_SEQ_PREFIXES = tuple(c for c in VARIANT_CHANNELS if c != VEL_PREFIX)
VARIANT_SEQ_LOG = tuple(CHANNEL_LOG[c] for c in VARIANT_SEQ_PREFIXES)

COND_DIM = 1 if VARIANT_COND_PREFIX is not None else 0
VARIANT_COND_LOG = CHANNEL_LOG[VARIANT_COND_PREFIX] if COND_DIM else False

INPUT_DIM = len(VARIANT_CHANNELS)
N_SEQ_EXTRA = len(VARIANT_SEQ_PREFIXES)

# Batch slot 9 holds EITHER the cond token OR the extra sequence channels, and
# dataset.py / train.py / evaluate.py rely on that being unambiguous.
if COND_DIM > 0 and N_SEQ_EXTRA > 0:
    raise ValueError(
        f"Variant {INPUT_VARIANT!r} asks for both a conditioning scalar and extra "
        f"sequence channels. The batch layout reserves one slot for the two. Give "
        f"the dataset a second slot before adding such a variant.")

USE_ARM_STATE = (N_SEQ_EXTRA > 0) or (COND_DIM > 0)

# The channels whose standardization stats must be persisted, in stats order.
VARIANT_STATS_CHANNELS = ((VARIANT_COND_PREFIX,) if COND_DIM
                          else VARIANT_SEQ_PREFIXES)
VARIANT_STATS_LOG = ((VARIANT_COND_LOG,) if COND_DIM else VARIANT_SEQ_LOG)
VARIANT_STATS_KIND = "cond" if COND_DIM else "seq"

# --- Legacy aliases. First stats channel only; kept so older sidecars and
#     scripts that read these names still import. New code should use
#     VARIANT_STATS_CHANNELS / VARIANT_STATS_LOG, which describe every channel.
VARIANT_WINDOW_PREFIX = VARIANT_STATS_CHANNELS[0] if VARIANT_STATS_CHANNELS else None
VARIANT_LOG_TRANSFORM = VARIANT_STATS_LOG[0] if VARIANT_STATS_LOG else False
ARM_FEATURE_MODE = INPUT_VARIANT


if FRAME_MODE == "world":
    CSV_PATH = "/home/psxkf4/IsaacLab/source/collected_data/data_tb-3_ta57_emavel1.0_velstd0.0_broad.csv"
elif FRAME_MODE == "local":
    if MULTI_ANGLE:
        if GRIPPER_CLOSED:
            CSV_PATH = ("/home/psxkf4/IsaacLab/source/collected_data/"
                        "data_cube_closed_gripper_multi_angle_sb3_v2/"
                        "data_cube_closed_gripper_multi_angle_sb3_v2.csv")
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
    'use_vel': USE_VEL,
    'variant_channels': list(VARIANT_CHANNELS),
    'variant_cond_prefix': VARIANT_COND_PREFIX,
    'variant_seq_prefixes': list(VARIANT_SEQ_PREFIXES),
    'variant_seq_log': list(VARIANT_SEQ_LOG),
    'variant_cond_log': VARIANT_COND_LOG,
    # legacy keys, first stats channel only -- kept so old checkpoints and
    # sidecar readers still load.
    'variant_window_prefix': VARIANT_WINDOW_PREFIX,
    'variant_log_transform': VARIANT_LOG_TRANSFORM,
    'max_push_lateral_offset': MAX_PUSH_LATERAL_OFFSET,
    'max_push_com_offset': MAX_PUSH_COM_OFFSET,
}

used_config = config_multi_angle