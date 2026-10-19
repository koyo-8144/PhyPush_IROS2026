"""
plot_estimator_polar_sim.py  --  the SIMULATION twin of plot_estimator_polar.py.

One image per estimator, one polar chart per object-surface configuration, drawn
with the same layout, the same angular convention and the same radial scale as
the real-robot figures, so a sim panel and a real panel can be put side by side
and read as one comparison.

WHAT IS THE SAME AS THE REAL SCRIPT
    The 27 configurations, by (mass, mu).  The push-angle grid (+/-54 .. 126 deg
    in 18 deg steps).  The radius (NRMSE as a percentage of the real-world
    range).  The broken ring, the side tables, the shared radial limit, the
    per-configuration singles under a folder named after the estimator.

WHAT CANNOT BE THE SAME, AND MATTERS
    1. THE OBJECT IS ALWAYS THE CUBE.  The sweep is data_cube_*; there is no tin
       can and no plastic container in it.  A panel labelled "Tin can 0.211 kg"
       is a 0.211 kg CUBE with mu = 0.3654.  This script reproduces the real
       object's mass and friction, not its shape, size or inertia tensor.  Any
       sim-to-real gap that comes from geometry is invisible here by
       construction.

    2. THE PAIR IS MATCHED, NOT SAMPLED.  The sweep draws (mass, mu) at random,
       so it holds no row at exactly 0.388 kg / 0.4334.  Rows within
       MASS_TOL / MU_TOL are used instead, and the realised mean is reported per
       configuration so the substitution is visible.

    3. ONE ANGLE HOLDS MANY PUSHES.  The real robot pushes once per angle, so its
       per-cell "NRMSE" is that push's own absolute error.  Here a cell can hold
       dozens of rows from different object yaws, and the NRMSE is a genuine
       root-mean-square over them.  The n column in the side table says how many.

    4. THE SIM ANGLE TRACKS THE OBJECT'S YAW.  The task folds the approach onto a
       face normal, so the attack angle is ~0 by construction; on the robot a0 is
       set independently of how the object is lying.  Two pushes at the same
       absolute base-frame angle therefore strike the object differently in the
       two settings.  The absolute angle is still the right key -- it is what the
       arm's geometry depends on -- but it is not the whole contact.

HELD-OUT ONLY, BY DEFAULT
    Each (mass, mu) pair appears ~40 times in the sweep, so scoring every row
    would mostly measure memorization.  HELDOUT=on replays training's own
    GroupShuffleSplit (seed+env_id, test_size=0.2, random_state=42) and keeps the
    validation groups.  The real objects were never trained on, so the held-out
    split is the honest comparison; HELDOUT=off is a diagnostic, and the figures
    say which was used.

USAGE
    python plot_estimator_polar_sim.py
    HELDOUT=off python plot_estimator_polar_sim.py
    MASS_TOL=0.08 MU_TOL=0.03 python plot_estimator_polar_sim.py
    DOMAINS="cube0.922-rubber_0.4334 tincan0.618-woodsmooth_0.2724" \
        LABELS=on python plot_estimator_polar_sim.py
    SINGLES=only python plot_estimator_polar_sim.py
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# THE ESTIMATORS
#
# variant -> (checkpoint run folder, model string, checkpoint file)
# Under CKPT_ROOT. config.json beside the weights supplies d_model, seq_len, the
# channel list and the variant name, which is checked against the key here -- a
# VARIANT_SPEC row pointing at the wrong folder is the easiest mistake to make
# and the hardest to see in a finished figure.
# =============================================================================
# CKPT_ROOT = os.environ.get(
#     "CKPT_ROOT",
#     os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                  "results", "checkpoints", "from_20260916", "gripper_closed"))
CKPT_ROOT = os.environ.get(
    "CKPT_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "checkpoints", "from_20261016", "gripper_closed"))

CKPT_FILE = os.environ.get("CKPT_FILE", "transformer_epoch1000.pth")

# Hand-written rows. Normally EMPTY: the table below is discovered from disk, by
# reading each run's own config.json. A hardcoded timestamp goes stale the moment
# a variant is retrained, and the stale row points at a directory that either does
# not exist under the current CKPT_ROOT or -- worse -- holds a different variant.
#
# Fill a row in only to break a genuine ambiguity: two runs of one variant under
# one root, where the directory names cannot say which is meant. An entry here
# WINS over discovery.
VARIANT_SPEC_OVERRIDE = {
    # "vel_manip_seq": ("20261016_163158",
    #                   "pinn_pcri-L1_p5c10.0_multiangle_vel_manip_seq_vn1434u"),
}


def discover_variant_spec(root):
    """{variant: (run, model_string)} read from the checkpoints themselves.

    Each run folder holds exactly one model-string subfolder, and its config.json
    records the input_variant it was trained for. That is the only place the
    pairing is actually recorded -- the model string usually ends in the variant
    name, but not for vel_only (whose string has no variant suffix) and not when
    a tag like _vn1434u is appended, so parsing the folder name is guesswork and
    reading config.json is not.

    Returns ({}, [note]) rather than raising, so a missing root still reaches the
    normal error path with its own message.
    """
    found, notes, dupes = {}, [], {}
    for cfg_path in sorted(glob.glob(os.path.join(root, "*", "*", "config.json"))):
        model_dir = os.path.dirname(cfg_path)
        run = os.path.basename(os.path.dirname(model_dir))
        model_string = os.path.basename(model_dir)
        try:
            v = json.load(open(cfg_path)).get("input_variant")
        except Exception as exc:                              # noqa: BLE001
            notes.append(f"  [note] unreadable {cfg_path}: {type(exc).__name__}")
            continue
        if not v:
            continue
        if not os.path.exists(os.path.join(model_dir, CKPT_FILE)):
            notes.append(f"  [note] {run}/{model_string}: no {CKPT_FILE}, skipped")
            continue
        if v in found:
            dupes.setdefault(v, [found[v]]).append((run, model_string))
            continue
        found[v] = (run, model_string)
    for v, rows in dupes.items():
        notes.append(
            f"  [WARNING] {len(rows)} runs under this root say input_variant="
            f"'{v}': {[r for r, _m in rows]}. Keeping {rows[0][0]}. Put the one "
            f"you mean in VARIANT_SPEC_OVERRIDE.")
    return found, notes


_DISCOVERED, _DISCOVERY_NOTES = discover_variant_spec(CKPT_ROOT)
VARIANT_SPEC = dict(_DISCOVERED)
VARIANT_SPEC.update(VARIANT_SPEC_OVERRIDE)

VARIANTS = os.environ.get("VARIANTS", " ".join(VARIANT_SPEC)).split()

# =============================================================================
# THE 27 REAL CONFIGURATIONS
#
# Copied verbatim from the real evaluation's detail files, domain string
# included, so this script's per-angle CSV can be joined to
# estimator_polar_per_angle.csv on (domain, push_angle_deg) with no renaming.
#
# The object name is kept for the labels and for the seen/unseen split, but in
# simulation every one of these is a cube -- see note 1 in the module docstring.
# =============================================================================
REAL_CONFIGS = (
    ("cube0.388-rubber_0.4334",        0.388, 0.4334),
    ("cube0.388-woodrough_0.255",      0.388, 0.2550),
    ("cube0.388-woodsmooth_0.3421",    0.388, 0.3421),
    ("cube0.599-rubber_0.4334",        0.599, 0.4334),
    ("cube0.599-woodrough_0.255",      0.599, 0.2550),
    ("cube0.599-woodsmooth_0.3421",    0.599, 0.3421),
    ("cube0.922-rubber_0.4334",        0.922, 0.4334),
    ("cube0.922-woodrough_0.255",      0.922, 0.2550),
    ("cube0.922-woodsmooth_0.3421",    0.922, 0.3421),
    ("cube1.384-rubber_0.4334",        1.384, 0.4334),
    ("cube1.384-woodrough_0.255",      1.384, 0.2550),
    ("cube1.384-woodsmooth_0.3421",    1.384, 0.3421),
    ("cube1.98-rubber_0.4334",         1.980, 0.4334),
    ("cube1.98-woodrough_0.255",       1.980, 0.2550),
    ("cube1.98-woodsmooth_0.3421",     1.980, 0.3421),
    ("placon0.297-rubber_0.2987",      0.297, 0.2987),
    ("placon0.297-woodrough_0.1799",   0.297, 0.1799),
    ("placon0.297-woodsmooth_0.158",   0.297, 0.1580),
    ("placon0.874-rubber_0.2987",      0.874, 0.2987),
    ("placon0.874-woodrough_0.1799",   0.874, 0.1799),
    ("placon0.874-woodsmooth_0.158",   0.874, 0.1580),
    ("tincan0.211-rubber_0.3654",      0.211, 0.3654),
    ("tincan0.211-woodrough_0.3747",   0.211, 0.3747),
    ("tincan0.211-woodsmooth_0.2724",  0.211, 0.2724),
    ("tincan0.618-rubber_0.3654",      0.618, 0.3654),
    ("tincan0.618-woodrough_0.3747",   0.618, 0.3747),
    ("tincan0.618-woodsmooth_0.2724",  0.618, 0.2724),
)

# How close a sweep row has to be to count as that configuration. The sweep
# samples (mass, mu) continuously, so an exact match does not exist; these are
# the same defaults check_variant_inputs_sim.py uses.
#
# Wider tolerance means more rows per cell and a less specific object. The
# printed table gives the realised mean and spread per configuration, which is
# how to tell whether a tolerance was too generous: a configuration whose
# matched mass spans 0.05 kg is not one object any more.
MASS_TOL = float(os.environ.get("MASS_TOL", 0.05))
MU_TOL = float(os.environ.get("MU_TOL", 0.02))

# =============================================================================
# NORMALIZATION -- MUST MATCH THE REAL SIDE
#
# The radius is NRMSE as a percentage of a range. For a sim panel to be readable
# beside a real one, the DENOMINATOR has to be identical, and it is not derivable
# from anything on this machine: the real script takes it from
# utils.move_to_xyz.REAL_M_RANGE / REAL_MU_RANGE on the robot computer.
#
# Default: the span of the 27 configurations themselves, which is what the real
# objects actually cover -- 1.769 kg and 0.2754.
#
# VERIFIED, not assumed: dividing mass_err by mass_nrmse_pct across every row of
# the robot side's estimator_polar_top3.csv recovers its denominator, and it is
# 1.768999 for mass and 0.275400 for mu. So these defaults already match the real
# figures and need no override.
#
# Re-derive it the same way if the real objects ever change:
#     100 * abs(mass_err) / mass_nrmse_pct   ->  REAL_M_RANGE
# and set M_RANGE / MU_RANGE here. Two figures drawn on different denominators
# look comparable and are not.
#
# NOTE this is NOT configs.REAL_M_RANGE on this machine, which is 0.702 - 0.340 =
# 0.362 and describes some earlier object set; using it would inflate every
# radius here by ~4.9x.
# =============================================================================
_M_SPAN = max(c[1] for c in REAL_CONFIGS) - min(c[1] for c in REAL_CONFIGS)
_MU_SPAN = max(c[2] for c in REAL_CONFIGS) - min(c[2] for c in REAL_CONFIGS)
M_RANGE = float(os.environ.get("M_RANGE", _M_SPAN))
MU_RANGE = float(os.environ.get("MU_RANGE", _MU_SPAN))

# =============================================================================
# DATA
# =============================================================================
# The sweep the from_20260916 velocity estimators were trained on. This must be
# the CSV whose sidecar holds their standardization statistics -- resolve_data()
# below checks that the sidecar exists beside it and says what it found.
SIM_CSV = os.environ.get("SIM_CSV", "")
SIDECAR_BASE = os.environ.get("SIDECAR_BASE", "")
COLLECTED_ROOT = os.environ.get(
    "COLLECTED_ROOT", "/home/psxkf4/IsaacLab/source/collected_data")

# =============================================================================
# EE VELOCITY OBSERVATION NOISE -- the same injection train.py performs
#
# DEFAULT OFF, and that default is the right one for the headline figure:
# evaluation measures what a model does with the data it will actually meet, and
# the simulated channel is the data. Adding noise here does not make the sim
# result "more correct"; it answers a different question.
#
# The question it DOES answer is worth having. The real captures carry the
# hardware noise floor; the sweep does not. So a sim angle-profile is measured on
# a cleaner input than the real one, and part of any sim-to-real difference in
# that profile is just the missing noise. Turning this on with the FITTED sigma
# puts the two inputs on the same footing, which is the comparison
# angle_profile_agreement.py is making.
#
#   VEL_NOISE = "off"      no injection                            (default)
#               "on"       sigma_inject from VEL_NOISE_JSON
#               "0.0015"   that sigma, in m/s, by hand
#
# Same three forms, same file and same vel_noise.VelNoise class train.py uses,
# so a figure drawn with "on" carries the sigma the models were trained with and
# not a second, independently chosen number.
#
# Velocity is fed RAW and unstandardized in this pipeline, so sigma goes in
# directly in m/s with no z-space conversion -- hence apply(..., std=None),
# exactly as in train.py. A pipeline that standardized velocity would have to
# pass the channel's training std instead.
#
# Output goes to a SEPARATE folder when this is on (see OUT_DIR below), because
# a noised run and a clean run of the same checkpoint produce different numbers
# under identical file names, and the run tag has to stay parseable by
# split_tag() in the real-side scripts.
# =============================================================================
VEL_NOISE = os.environ.get("VEL_NOISE", "off")
VEL_NOISE_JSON = os.environ.get("VEL_NOISE_JSON", "vel_noise_model.json")
# Fixed, so the figure is reproducible. Re-running must not move a panel.
VEL_NOISE_SEED = int(os.environ.get("VEL_NOISE_SEED", 42))


def make_vel_noise():
    """The VelNoise train.py would have built from the same switch."""
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from vel_noise import VelNoise
    v = VEL_NOISE.strip().lower()
    if v in ("", "off", "0", "false", "none"):
        return VelNoise.disabled()
    if v in ("on", "1", "true"):
        return VelNoise.from_json(VEL_NOISE_JSON)
    try:
        return VelNoise(float(VEL_NOISE))
    except ValueError:
        raise SystemExit(
            f"VEL_NOISE must be off | on | a sigma in m/s, got {VEL_NOISE!r}")


VEL_NOISE_OBJ = make_vel_noise()

SEQ_LEN = int(os.environ.get("SEQ_LEN", 60))
CHUNKSIZE = int(os.environ.get("CHUNKSIZE", 200_000))

HELDOUT = os.environ.get("HELDOUT", "on")
GROUP_SPLIT_SEED = int(os.environ.get("GROUP_SPLIT_SEED", 42))

# The seen box and the push-quality thresholds training used. Replayed here in
# the same combination, because the group split has to see the same row set or it
# is a different split and "held out" means nothing.
M_SEEN_MIN, M_SEEN_MAX = 0.2, 2.0
MU_SEEN_MIN, MU_SEEN_MAX = 0.15, 0.5
MAX_PUSH_LATERAL_OFFSET = 0.005
MAX_PUSH_COM_OFFSET = 0.05

BATCH = int(os.environ.get("BATCH", 512))

# =============================================================================
# OUTPUT -- mirrors the real script's folder shape
# =============================================================================
# The folder carries the noise tag, so a noised run cannot overwrite a clean one.
# The TAG, not the run tag: tag_for() has to keep the "<variant>_<root>_
# transformer_sim" shape that split_tag() in the real-side scripts parses, and
# appending to it would break that parse.
_OUT_SUFFIX = "" if not VEL_NOISE_OBJ.enabled else f"_{VEL_NOISE_OBJ.tag()}"
OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", f"estimator_polar_sim{_OUT_SUFFIX}"))

SINGLES = os.environ.get("SINGLES", "on")
LABELS = os.environ.get("LABELS", "auto")
LABEL_PANEL_MAX = int(os.environ.get("LABEL_PANEL_MAX", 6))
RSCALE = os.environ.get("RSCALE", "shared")
RMAX_CLIP = float(os.environ.get("RMAX_CLIP", 60.0))
NCOLS = int(os.environ.get("NCOLS", 0))
TOP_N = int(os.environ.get("TOP_N", 3))
STYLE = os.environ.get("STYLE", "light")
DPI = int(os.environ.get("DPI", 150))
REPORT = os.environ.get("REPORT", "on")
DOMAINS = os.environ.get("DOMAINS", "")

A0_MAGNITUDES = (0.30, 0.40, 0.50, 0.60, 0.70)
PUSH_ANGLES_DEG = tuple(sorted(round(s * m * 180.0, 6)
                               for m in A0_MAGNITUDES for s in (-1, 1)))
# The real script uses 8.0, because its pushes land ON the grid and a loose
# tolerance could only mis-assign. Here the angle is continuous, so the default
# tiles the 18 deg spacing without overlap: every push joins its nearest target,
# none is double-counted, and widening further would merge two targets.
ANGLE_TOL_DEG = float(os.environ.get("ANGLE_TOL_DEG", 9.0))

# =============================================================================
# PALETTE / LABELS -- identical to the real script
# =============================================================================
THEME = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e",
                  muted="#8a8984", grid="#d9d8d4",
                  mass="#2a78d6", fric="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7",
                 muted="#8a8984", grid="#3a3a38",
                 mass="#3987e5", fric="#d95926"),
}
SERIES = {"mass": dict(label=r"mass $m$", ls="-", marker="o"),
          "fric": dict(label=r"friction $\mu$", ls="--", marker="s")}
METRICS = (("mass", "m_est", "m_gt", M_RANGE, "m", 3),
           ("fric", "mu_est", "mu_gt", MU_RANGE, r"$\mu$", 4))

OBJECT_DISPLAY = {"cube": "Plastic cube", "tincan": "Tin can",
                  "placon": "Plastic container", "container": "Plastic container",
                  "woodenbox": "Wooden box"}
SURFACE_DISPLAY = {"rubber": "rubber", "woodrough": "rough wood",
                   "woodsmooth": "smooth wood"}
DOMAIN_RE = re.compile(r"^(?P<obj>[A-Za-z]+)(?P<m>[\d.]*)[-_](?P<surf>[A-Za-z]+)"
                       r"(?:_(?P<mu>[\d.]+))?$")

METHOD_LABEL = {
    "vel_only": "PhyPush (vel only)", "vel_manip_cond": r"PhyPush ($w$ cond)",
    "vel_dirmanip_cond": r"PhyPush ($w_{dir}$ cond)",
    "vel_manip_seq": r"PhyPush ($w$ seq)",
    "vel_dirmanip_seq": r"PhyPush ($w_{dir}$ seq)",
    "vel_osim_seq": r"PhyPush ($\Lambda$ seq)",
    "vel_eff_seq": r"PhyPush ($m_{eff}$ seq)",
}

SEEN_OBJECTS = {"Plastic cube"}


class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, t):
        self._stream.write(t); self._fh.write(t); self._fh.flush(); return len(t)

    def flush(self):
        self._stream.flush(); self._fh.flush()

    def __getattr__(self, n):
        return getattr(self._stream, n)


# =============================================================================
# NAMING
# =============================================================================
def safe_name(s):
    """A configuration name as a file name; the real script's rule, unchanged."""
    return re.sub(r"[^A-Za-z0-9._+-]", "_", str(s))


def tag_for(variant):
    """The run tag that names this estimator's files.

    Shaped like the real side's -- <variant>_<root>_transformer -- so the sim and
    real images for one estimator sort next to each other and split_tag() in the
    real scripts parses it unchanged. "_sim" marks which machine produced it.
    """
    root = os.path.basename(os.path.dirname(CKPT_ROOT.rstrip(os.sep)))
    return f"{variant}_{root}_transformer_sim"


def title_for(variant):
    label = METHOD_LABEL.get(variant, variant.replace("_", " "))
    return f"{label}  --  simulation"


def display_domain(d):
    m = DOMAIN_RE.match(str(d))
    if not m:
        return str(d), ""
    return (OBJECT_DISPLAY.get(m.group("obj").lower(), m.group("obj").title()),
            SURFACE_DISPLAY.get(m.group("surf").lower(), m.group("surf").title()))


# =============================================================================
# RESOLVING THE SWEEP AND ITS SIDECAR
# =============================================================================
def sidecar_for(base, variant):
    return f"{base}.arm_stats.{variant}.npz"


def variant_needs_sidecar(variant):
    """Does this variant standardize any channel?

    vel_only does not: velocity is fed RAW and unstandardized, so training never
    wrote a sidecar for it -- which is why six sidecars serve seven variants.
    Asking for one is not a missing file, it is a category error, and both the
    resolve_data() check and predict() have to know the difference.
    """
    try:
        _w, cpath = checkpoint_paths(variant)
        seqs, cond = variant_channels(json.load(open(cpath)), variant)
    except Exception:                                         # noqa: BLE001
        return True          # unreadable: assume it needs one and let it fail loudly
    return bool(cond) or bool(seqs)


def root_word():
    """The collection root CKPT_ROOT sits under, e.g. 'from_20261016'."""
    return os.path.basename(os.path.dirname(CKPT_ROOT.rstrip(os.sep)))


def root_spec_entry():
    """configs.ROOT_SPEC[<this root>], or None.

    configs.py is this project's single source of truth for which sweep a
    collection root READS and which stem its sidecars are WRITTEN under -- the
    two come apart for several roots, from_20261016 among them (reads ..._sb3,
    writes ..._sb3_n1). Guessing that pairing from the files on disk is what
    made resolve_data() ambiguous in the first place, so ask the table.

    Imported defensively: configs.py prints, reads TRAIN_FROM from the
    environment, and raises when TRAIN_FROM and GRIPPER_CLOSED disagree. None of
    that should take this figure down, so a failure just falls back to the glob.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import configs
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [note] configs.py not importable ({type(exc).__name__}); "
              f"falling back to searching for the sidecars on disk")
        return None
    spec = getattr(configs, "ROOT_SPEC", {}).get(root_word())
    if spec is None:
        return None
    collected = getattr(configs, "COLLECTED_DATA_ROOT", COLLECTED_ROOT)
    csv_stem, stats_stem, logs = spec
    return (os.path.join(collected, csv_stem, f"{csv_stem}.csv"),
            os.path.join(collected, stats_stem, f"{stats_stem}.csv"),
            dict(logs))


def resolve_data():
    """(csv_path, sidecar_base) for the sweep these checkpoints were trained on.

    The sidecar is not optional and not interchangeable: it holds the mean/std the
    frozen model was trained to expect, and standardizing with another sweep's
    statistics silently mis-scales every input. So the CSV is accepted only when
    every requested variant's sidecar sits beside it.

    configs.CSV_PATH is not trusted here. It tracks whatever TRAIN_FROM is set to
    right now (the velocity-free v2/v4 sweeps, by default), which is not the sweep
    the from_20260916 velocity estimators were trained on.
    """
    if SIM_CSV and SIDECAR_BASE:
        return SIM_CSV, SIDECAR_BASE
    if SIM_CSV:
        return SIM_CSV, SIM_CSV

    # configs.ROOT_SPEC first: it STATES the pairing instead of inferring it,
    # which is the only way to be right when the CSV stem and the sidecar stem
    # differ. The glob below cannot distinguish ..._sb3 from ..._sb3_n1 -- both
    # are closed-gripper sweeps carrying sidecars for every variant -- and that
    # is exactly the case this root is.
    entry = root_spec_entry()
    if entry is not None:
        csv_path, base, logs = entry
        print(f"  root spec  : configs.ROOT_SPEC[{root_word()!r}]")
        print(f"     reads   {csv_path}")
        print(f"     sidecar {base}"
              + ("   (redirected)" if base != csv_path else ""))
        print(f"     log     {logs}")
        missing = [v for v in VARIANTS
                   if variant_needs_sidecar(v)
                   and not os.path.exists(sidecar_for(base, v))]
        if missing:
            raise SystemExit(
                f"configs.ROOT_SPEC says this root's sidecars live under\n"
                f"    {base}\n"
                f"  but there is none for {missing}.\n"
                f"  Either those variants were never trained under "
                f"{root_word()}, or the stats were written elsewhere -- set "
                f"SIDECAR_BASE to override.")
        return csv_path, base

    # Every sidecar on disk for the variants asked for, grouped by its base.
    bases = {}
    for v in VARIANTS:
        for p in glob.glob(os.path.join(COLLECTED_ROOT, "*",
                                        f"*.arm_stats.{v}.npz")):
            base = p[: -len(f".arm_stats.{v}.npz")]
            bases.setdefault(base, set()).add(v)

    complete = sorted(b for b, vs in bases.items() if set(VARIANTS) <= vs)
    print(f"  sidecar search: {COLLECTED_ROOT}/*/*.arm_stats.<variant>.npz")
    for b, vs in sorted(bases.items()):
        mark = "  <- has every requested variant" if b in complete else ""
        print(f"     {os.path.basename(b)}   {sorted(vs)}{mark}")

    # The gripper state is not a preference, it is part of the training identity:
    # a checkpoint under .../gripper_closed/ was trained on a closed-gripper
    # sweep, and an opened-gripper sweep's statistics describe a different
    # end-effector. CKPT_ROOT carries that word, so a closed/opened pair of
    # candidates is decided here rather than handed back as a question.
    if len(complete) > 1:
        want = ("closed" if "gripper_closed" in CKPT_ROOT
                else "opened" if "gripper_opened" in CKPT_ROOT else "")
        if want:
            narrowed = [b for b in complete
                        if f"{want}_gripper" in os.path.basename(b)]
            if len(narrowed) == 1:
                dropped = [os.path.basename(b) for b in complete
                           if b not in narrowed]
                print(f"  the checkpoint path says gripper_{want}, so only a "
                      f"'{want}_gripper' sweep can be its training data; "
                      f"dropping {dropped}")
                complete = narrowed

    if len(complete) == 1 and os.path.exists(complete[0]):
        print(f"  using {complete[0]}")
        return complete[0], complete[0]
    if not complete:
        raise SystemExit(
            f"No sweep under {COLLECTED_ROOT} has a sidecar for every variant in "
            f"{VARIANTS}.\n"
            f"  Set SIM_CSV (and SIDECAR_BASE if the stats live beside a "
            f"different file) to the sweep these checkpoints were trained on.\n"
            f"  The sidecar is named <base>.arm_stats.<variant>.npz and is "
            f"written by dataset.create_dataloaders during training.")
    raise SystemExit(
        f"Several sweeps carry sidecars for {VARIANTS}:\n    "
        + "\n    ".join(complete)
        + "\n  Pass SIM_CSV=<the one these checkpoints were trained on>. Picking "
          "the wrong one standardizes the input with another sweep's statistics, "
          "which changes every number without raising anything.")


def load_sidecar(base, variant):
    """(mean, std, logs, channels) as written during training."""
    path = sidecar_for(base, variant)
    if not os.path.exists(path):
        raise SystemExit(f"sidecar not found: {path}")
    d = np.load(path, allow_pickle=True)
    saved = str(d["variant"]) if "variant" in d else (
        str(d["mode"]) if "mode" in d else None)
    if saved is not None and saved != variant:
        raise SystemExit(f"{path} was saved for variant '{saved}', not '{variant}'")
    mean = np.asarray(d["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(d["std"], dtype=np.float32).reshape(1, -1)
    logs = (np.atleast_1d(d["log_transform"]).astype(bool).tolist()
            if "log_transform" in d else [False])
    chans = ([str(c) for c in np.atleast_1d(d["channels"]).tolist()]
             if "channels" in d else [])

    # Cross-check the sidecar's own log flags against configs.ROOT_SPEC for this
    # collection root. This is what catches the wrong STEM: ..._sb3 and
    # ..._sb3_n1 both carry a sidecar for every variant, and nothing else in the
    # pipeline would notice that the stats came from the other one. A sweep
    # standardized in log space and one standardized linearly differ by orders of
    # magnitude, so a mismatch here is never cosmetic.
    if ROOT_LOGS_RESOLVED and chans:
        for c, lg in zip(chans, logs):
            want = ROOT_LOGS_RESOLVED.get(c)
            if want is not None and bool(want) != bool(lg):
                raise SystemExit(
                    f"{path} has log_transform={bool(lg)} for '{c}', but "
                    f"configs.ROOT_SPEC[{root_word()!r}] says {bool(want)}.\n"
                    f"  The sidecar and the collection root disagree, which "
                    f"usually means this is the other sweep's stem. Check "
                    f"ROOT_SPEC, or set SIDECAR_BASE explicitly.")
    return mean, std, logs, chans


def checkpoint_paths(variant):
    run, model_string = VARIANT_SPEC[variant]
    folder = os.path.join(CKPT_ROOT, run, model_string)
    return (os.path.join(folder, CKPT_FILE),
            os.path.join(folder, "config.json"))


def window_cols(header, prefix):
    """prefix + pure digits, ordered numerically."""
    header = list(header.columns) if hasattr(header, "columns") else list(header)
    cols = [c for c in header
            if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


# =============================================================================
# LOADING
# =============================================================================
def variant_channels(cfg, variant):
    """(seq prefixes, cond prefix) for a checkpoint, from its own config.json.

    Read from the checkpoint rather than from configs.py: configs.py describes
    whatever INPUT_VARIANT is selected now, while config.json describes the model
    that is about to be loaded.
    """
    seqs = [c for c in (cfg.get("variant_seq_prefixes") or [])]
    cond = cfg.get("variant_cond_prefix")
    if not seqs and cond is None:
        # Pre-table checkpoint: fall back to the legacy single-prefix keys.
        pfx = cfg.get("variant_window_prefix")
        if cfg.get("cond_dim", 0):
            cond = pfx
        elif pfx:
            seqs = [pfx]
    return seqs, cond


def needed_prefixes():
    """Every window channel any requested variant reads."""
    out = []
    for v in VARIANTS:
        _, cpath = checkpoint_paths(v)
        if not os.path.exists(cpath):
            continue
        seqs, cond = variant_channels(json.load(open(cpath)), v)
        for p in list(seqs) + ([cond] if cond else []):
            if p and p not in out:
                out.append(p)
    return out


def load_sweep(csv_path, prefixes):
    """Sweep rows inside the seen box, filtered as training filtered them.

    Read in chunks with only the columns needed: the sweep is tens of GB and
    these are ~200 of its ~5000 columns.
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()

    vel_cols = window_cols(header, "input_vel_")
    if len(vel_cols) != SEQ_LEN:
        print(f"  [WARNING] {len(vel_cols)} input_vel_* columns, expected {SEQ_LEN}")

    missing_dir = [c for c in ("push_dir_b_x", "push_dir_b_y") if c not in header]
    if missing_dir:
        raise SystemExit(
            f"{csv_path} has no {missing_dir}. The push angle in the base frame "
            f"cannot be derived without the push direction, and matching the real "
            f"angles is the whole point of this figure.")

    need = list(vel_cols) + ["gt_mass", "gt_mu", "start_t",
                             "push_dir_b_x", "push_dir_b_y"]
    for p in prefixes:
        wc = window_cols(header, p)
        if not wc:
            raise SystemExit(f"{csv_path} has no '{p}*' columns, which a "
                             f"requested variant needs.")
        need += wc
    for c in ("seed", "env_id", "push_start_lateral_offset", "push_com_offset",
              "obj_yaw_base", "push_face_index", "collect_idx"):
        if c in header:
            need.append(c)
    need = list(dict.fromkeys(c for c in need if c in header))

    print(f"  reading {len(need)} of {len(header)} columns, chunks of {CHUNKSIZE}")
    parts, n_in = [], 0
    for chunk in pd.read_csv(csv_path, usecols=need, chunksize=CHUNKSIZE):
        n_in += len(chunk)
        m, mu = chunk["gt_mass"], chunk["gt_mu"]
        keep = ((m >= M_SEEN_MIN) & (m <= M_SEEN_MAX)
                & (mu >= MU_SEEN_MIN) & (mu <= MU_SEEN_MAX)
                & ((chunk["start_t"] + SEQ_LEN) <= 100))
        if MAX_PUSH_LATERAL_OFFSET is not None and "push_start_lateral_offset" in chunk:
            keep &= chunk["push_start_lateral_offset"] < MAX_PUSH_LATERAL_OFFSET
        if MAX_PUSH_COM_OFFSET is not None and "push_com_offset" in chunk:
            keep &= chunk["push_com_offset"] < MAX_PUSH_COM_OFFSET
        sub = chunk[keep]
        if len(sub):
            parts.append(sub.astype({c: np.float32 for c in sub.columns
                                     if sub[c].dtype == np.float64}))
        print(f"  ... {n_in} rows read, {sum(len(p) for p in parts)} kept",
              end="\r")
    if not parts:
        raise SystemExit("\nno row survived the seen-box and push-quality filters")
    df = pd.concat(parts, ignore_index=True)
    print(f"\n  {n_in} rows -> {len(df)} after the training filters")
    return df, vel_cols


def hold_out(df):
    """Training's own group split, replayed; returns the validation rows.

    GroupShuffleSplit partitions the UNIQUE group labels, so the result depends
    on the set of groups and the seed, not on row order -- which is why applying
    training's filters in a different order still reproduces its split, as long
    as the same rows survive.
    """
    if HELDOUT != "on":
        print(f"  [note] HELDOUT={HELDOUT}: scoring every row, including the "
              f"property groups these checkpoints trained on. Diagnostic only.")
        return df, None
    if not {"seed", "env_id"} <= set(df.columns):
        print("  [WARNING] no seed/env_id columns; grouping on (gt_mass, gt_mu) "
              "instead, which is what dataset.py falls back to.")
        groups = (df["gt_mass"].round(6).astype(str) + "_"
                  + df["gt_mu"].round(6).astype(str)).values
    else:
        groups = (df["seed"].astype(int) * 100000
                  + df["env_id"].astype(int)).values
    idx = np.arange(len(df))
    _, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2,
        random_state=GROUP_SPLIT_SEED).split(idx, groups=groups))
    out = df.iloc[test_idx].reset_index(drop=True)
    print(f"  held-out: {len(out)} rows / "
          f"{len(np.unique(groups[test_idx]))} property groups "
          f"(of {len(np.unique(groups))})")
    return out, groups[test_idx]


def wrap180(a):
    a = np.asarray(a, float)
    return -((-a + 180.0) % 360.0 - 180.0) + 0.0


def bin_angles(df):
    """Attach the nearest real push angle, and drop what lands outside the grid.

    The angle comes from push_dir_b_x / push_dir_b_y: the realised push direction
    in the base frame, which is the same quantity the robot's PUSH_ANGLE_DEG
    measures. Deriving it from obj_yaw_base and push_face_index instead would
    reproduce the task's INTENDED angle, not the one the arm actually took.
    """
    ang = wrap180(np.degrees(np.arctan2(df["push_dir_b_y"].values,
                                        df["push_dir_b_x"].values)))
    tgt = np.array(PUSH_ANGLES_DEG, float)
    d = np.abs(wrap180(ang[:, None] - tgt[None, :]))
    k = np.argmin(d, axis=1)
    ok = d[np.arange(len(ang)), k] <= ANGLE_TOL_DEG
    out = df[ok].copy()
    out["push_angle_deg"] = ang[ok]
    out["_bin"] = tgt[k][ok]
    return out, int((~ok).sum()), ang


def match_configs(df):
    """Attach a 'domain' column by matching (mass, mu) to the real configurations.

    A row can satisfy two configurations when the tolerances overlap (0.297 and
    0.211 are 0.086 apart, so MASS_TOL above 0.043 makes them compete). The
    NEAREST configuration wins, measured in units of the tolerances so mass and
    mu weigh equally, and the count of contested rows is reported -- a large one
    means the tolerance is wider than the configurations are separated.
    """
    m = df["gt_mass"].values.astype(float)
    mu = df["gt_mu"].values.astype(float)
    doms = [c[0] for c in REAL_CONFIGS]
    tm = np.array([c[1] for c in REAL_CONFIGS])[None, :]
    tu = np.array([c[2] for c in REAL_CONFIGS])[None, :]

    dm = np.abs(m[:, None] - tm) / MASS_TOL
    du = np.abs(mu[:, None] - tu) / MU_TOL
    within = (dm <= 1.0) & (du <= 1.0)
    n_hits = within.sum(axis=1)
    score = np.where(within, np.hypot(dm, du), np.inf)
    best = np.argmin(score, axis=1)

    hit = n_hits > 0
    out = df[hit].copy()
    out["domain"] = np.array(doms)[best[hit]]
    contested = int((n_hits > 1).sum())
    return out, contested, int((~hit).sum())


# =============================================================================
# INFERENCE
# =============================================================================
def build_input(df, vel_cols, variant, cfg, mean, std, logs, device):
    """The encoder input, assembled exactly as dataset.py assembles it.

    Velocity is channel 0 and stays RAW. Window channels follow in the config's
    own order, standardized with the training sidecar. A cond variant takes the
    MINIMUM over the window -- the same reduction dataset.py applies, and not the
    worst_* column, which is the minimum over the whole ~300-step push.
    """
    import torch
    seqs, cond_pfx = variant_channels(cfg, variant)
    X = df[vel_cols].values.astype(np.float32)[:, :, None]

    # Observation noise on the velocity channel, as train.py applies it: white,
    # one independent draw per timestep, in m/s on the RAW channel, BEFORE any
    # downstream use. Channel 0 is velocity here by construction -- it is laid
    # down first, above, and the window channels are concatenated after it.
    #
    # Drawn from a seeded generator so the figure is reproducible: an unseeded
    # draw would move every panel slightly on each run and there would be no way
    # to tell that from a real change.
    if VEL_NOISE_OBJ.enabled:
        rng = np.random.default_rng(VEL_NOISE_SEED)
        X[:, :, 0] = VEL_NOISE_OBJ.apply(X[:, :, 0], generator=rng)

    COND = None

    if cond_pfx:
        wc = window_cols(df, cond_pfx)
        if len(wc) != X.shape[1]:
            raise SystemExit(f"'{cond_pfx}*' has {len(wc)} steps, velocity has "
                             f"{X.shape[1]}; the two are not time-aligned.")
        W = df[wc].values.astype(np.float64)
        if logs and logs[0]:
            W = np.log(np.clip(W, 1e-12, None))
        raw = W.min(axis=1, keepdims=True)
        COND = ((raw - mean) / std).astype(np.float32)
    else:
        for k, p in enumerate(seqs):
            wc = window_cols(df, p)
            if len(wc) != X.shape[1]:
                raise SystemExit(f"'{p}*' has {len(wc)} steps, velocity has "
                                 f"{X.shape[1]}; the two are not time-aligned.")
            W = df[wc].values.astype(np.float64)
            if k < len(logs) and logs[k]:
                W = np.log(np.clip(W, 1e-12, None))
            Z = ((W - mean[0, k]) / std[0, k]).astype(np.float32)
            X = np.concatenate([X, Z[:, :, None]], axis=-1)

    want = int(cfg.get("input_dim", X.shape[-1]))
    if X.shape[-1] != want:
        raise SystemExit(f"built {X.shape[-1]} input channel(s) but config.json "
                         f"says input_dim={want}")
    return (torch.tensor(X).float().to(device),
            torch.tensor(COND).float().to(device) if COND is not None else None)


def predict(df, vel_cols, variant, device):
    """(N, 2) estimates for one variant."""
    import torch
    from models import PhysicsTransformerEstimator

    w_path, c_path = checkpoint_paths(variant)
    for p in (w_path, c_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    cfg = json.load(open(c_path))

    ck_v = cfg.get("input_variant")
    if ck_v is not None and ck_v != variant:
        raise SystemExit(
            f"VARIANT_SPEC['{variant}'] points at {os.path.dirname(w_path)}, "
            f"whose config.json says input_variant='{ck_v}'. Fix the mapping "
            f"rather than the check -- the figure would be mislabelled.")

    seqs, cond_pfx = variant_channels(cfg, variant)
    expect = [cond_pfx] if cond_pfx else list(seqs)
    if expect:
        mean, std, logs, chans = load_sidecar(SIDECAR_BASE_RESOLVED, variant)
    else:
        # Velocity only: nothing is standardized, so there is no sidecar to read
        # and build_input() never touches mean/std.
        mean, std, logs, chans = None, None, [], []
    if chans and chans != expect:
        raise SystemExit(f"sidecar holds stats for {chans} but the checkpoint "
                         f"standardizes {expect}")

    cond_dim = int(cfg.get("cond_dim", 1 if cond_pfx else 0))
    model = PhysicsTransformerEstimator(
        input_dim=int(cfg.get("input_dim", 1)), d_model=cfg["d_model"], nhead=4,
        num_encoder_layers=cfg["num_enc"], seq_len=int(cfg.get("seq_len", SEQ_LEN)),
        dropout=0.0, sharpness=cfg["sharpness"],
        cross_sharpness=cfg["cross_sharpness"], m_sharpness=cfg["m_sharpness"],
        mu_sharpness=cfg["mu_sharpness"], version=cfg["transformer_ver"],
        max_mass_scale=cfg["last_layer_ms"], max_mu_scale=cfg["last_layer_mus"],
        cond_dim=cond_dim).to(device)
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()

    print(f"     {os.path.relpath(w_path, CKPT_ROOT)}")
    # mean/std are None for a variant that standardizes nothing (vel_only), so
    # the sidecar part of this line has to be conditional -- ravel() on None is
    # an AttributeError, and main()'s except would turn it into a skipped
    # estimator with a message that says nothing about a print statement.
    _stats = (f"sidecar mean={mean.ravel()} std={std.ravel()}" if mean is not None
              else "no sidecar (velocity is fed raw, nothing standardized)")
    print(f"     channels {expect or ['input_vel']}   "
          f"input_dim={cfg.get('input_dim')}   cond_dim={cond_dim}   {_stats}")

    X, COND = build_input(df, vel_cols, variant, cfg, mean, std, logs, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            c = COND[i:i + BATCH] if COND is not None else None
            out.append(model(X[i:i + BATCH], cond=c)[0].cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


# =============================================================================
# AGGREGATION -- the same per-cell quantities the real script computes
# =============================================================================
def per_domain(df, domain):
    """{angle: {quantity: stats}} for one configuration.

    Unlike the real script, a cell can hold many pushes, so the NRMSE is a real
    root-mean-square over them rather than one push's absolute error. 'n' carries
    the count into the side table, because a cell of 1 and a cell of 40 are not
    equally trustworthy and nothing else in the figure would say so.
    """
    sub = df[df["domain"].astype(str) == domain]
    if sub.empty:
        return None
    out = {}
    for a, g in sub.groupby("_bin"):
        rec = {}
        for key, est_c, gt_c, rng, _s, _d in METRICS:
            est = g[est_c].values.astype(float)
            gt = g[gt_c].values.astype(float)
            m = np.isfinite(est) & np.isfinite(gt)
            if not m.any():
                rec[key] = None
                continue
            e = est[m] - gt[m]
            rec[key] = dict(n=int(m.sum()), est=float(est[m].mean()),
                            gt=float(gt[m].mean()),
                            nrmse=100.0 * float(np.sqrt(np.mean(e ** 2))) / rng)
        out[float(a)] = rec
    return out


def domains_of(df):
    """The configurations present, in the real script's order: object family,
    then true mass, then mu -- so the panels sit in the same places."""
    fam = re.compile(r"^([A-Za-z]+)")
    order = [c[0] for c in REAL_CONFIGS if c[0] in set(df["domain"].astype(str))]
    gts = {}
    for d, g in df.groupby(df["domain"].astype(str)):
        gts[d] = (float(g["m_gt"].mean()), float(g["mu_gt"].mean()))
    order.sort(key=lambda d: ((fam.match(d).group(1) if fam.match(d) else d),
                              gts[d][0], gts[d][1]))
    return order, gts


def overall(df):
    out = {}
    for key, est_c, gt_c, rng, _s, _d in METRICS:
        est = df[est_c].values.astype(float)
        gt = df[gt_c].values.astype(float)
        m = np.isfinite(est) & np.isfinite(gt)
        if m.sum() < 3:
            continue
        e = est[m] - gt[m]
        sd = float(np.std(gt[m]))
        rmse, bias = float(np.sqrt(np.mean(e ** 2))), float(np.mean(e))
        resid = np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))
        out[key] = dict(nrmse=100 * rmse / rng, bias=bias,
                        gain=float(np.std(est[m])) / sd if sd > 1e-12 else np.nan,
                        skill=(1 - (resid / sd) ** 2) * 100 if sd > 1e-12 else np.nan,
                        base=100 * sd / rng)
    return out


# =============================================================================
# DRAWING -- lifted from plot_estimator_polar.py so the two figures match
# =============================================================================
def nice_ticks(lim, want=3):
    raw = lim / max(want, 1)
    mag = 10.0 ** np.floor(np.log10(max(raw, 1e-9)))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    t = np.arange(step, lim + 1e-9, step)
    return t[t < lim * 0.995]


def arcs(angles):
    """Contiguous runs of the push grid, so the two 108-degree gaps stay open."""
    xs = sorted(angles)
    if len(xs) < 2:
        return [xs]
    step = np.median(np.diff(xs))
    runs, cur = [], [xs[0]]
    for a, b in zip(xs, xs[1:]):
        if (b - a) > 2.0 * step:
            runs.append(cur); cur = [b]
        else:
            cur.append(b)
    runs.append(cur)
    return runs


def side_table(dom, gts, d, n):
    """The block beside one panel. Carries the REALISED mass and mu, not the real
    object's, because the sweep was matched within a tolerance and the difference
    is exactly what a reader needs to judge the substitution."""
    obj, surf = display_domain(dom)
    want = dict((c[0], (c[1], c[2])) for c in REAL_CONFIGS).get(dom)
    gt = gts[dom]
    out = [f"{obj} on {surf}", f"{dom}", ""]
    if want:
        out += [f"real object     m = {want[0]:.3f} kg   mu = {want[1]:.4f}",
                f"sim matched     m = {gt[0]:.3f} kg   mu = {gt[1]:.4f}", ""]
    else:
        out += [f"ground truth   m = {gt[0]:.3f} kg", f"{'':15}mu = {gt[1]:.4f}", ""]
    scored = []
    for a in d:
        vals = [d[a][k]["nrmse"] for k, *_ in METRICS if d[a].get(k)]
        if vals:
            scored.append((float(np.mean(vals)), a))
    scored.sort()
    out.append(f"best {min(n, len(scored))} push direction(s)")
    out.append(f"{'angle':>6}{'m_est':>9}{'NRMSE':>8}{'mu_est':>9}{'NRMSE':>8}{'n':>5}")
    for _s, a in scored[:n]:
        m, u = d[a].get("mass"), d[a].get("fric")
        out.append(f"{a:>+5.0f}°"
                   + (f"{m['est']:>9.3f}{m['nrmse']:>7.1f}%" if m else f"{'--':>9}{'--':>8}")
                   + (f"{u['est']:>9.4f}{u['nrmse']:>7.1f}%" if u else f"{'--':>9}{'--':>8}")
                   + f"{(m['n'] if m else 0):>5}")
    tot = []
    for key, _e, _g, _r, sym, _d in METRICS:
        rs = [d[a][key] for a in d if d[a].get(key)]
        if rs:
            num = sum(r["nrmse"] ** 2 * r["n"] for r in rs)
            den = sum(r["n"] for r in rs)
            tot.append(f"{sym} {np.sqrt(num / den):.1f}%")
    n_push = sum(d[a]["mass"]["n"] for a in d if d[a].get("mass"))
    out += ["", f"all {len(d)} angles   " + "   ".join(tot),
            f"{n_push} push(es) in total"]
    return "\n".join(out)


def label_for(a, rec):
    lines = [f"{a:+.0f}$\\degree$"]
    for key, _e, _g, _r, sym, dec in METRICS:
        r = rec.get(key)
        lines.append(f"{sym} --" if not r else
                     f"{sym} {r['est']:.{dec}f}  {r['est'] - r['gt']:+.{dec}f}"
                     f"  {r['nrmse']:.0f}%")
    return "\n".join(lines)


def draw_panel(ax, d, t, lim, labels):
    """One configuration's polar chart. The grid and the singles both come
    through here, so the two cannot drift into showing the same data
    differently."""
    clipped = 0
    ax.set_facecolor(t["surface"])
    # +90 to the RIGHT and -90 to the LEFT, with the two 108-degree gaps at the
    # bottom and the top -- the real script's convention, so a sim panel and a
    # real panel can be overlaid.
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_thetagrids([a % 360 for a in PUSH_ANGLES_DEG],
                      labels=([""] * 10 if labels else
                              [f"{a:+.0f}" for a in PUSH_ANGLES_DEG]))
    ax.tick_params(axis="x", labelsize=5.5, colors=t["muted"], pad=-2)
    ax.set_rlim(0, lim)
    ax.set_rlabel_position(0.0)
    ax.set_rticks(nice_ticks(lim))
    ax.tick_params(axis="y", labelsize=5.5, colors=t["muted"])
    ax.grid(True, color=t["grid"], lw=0.5, alpha=0.9)
    ax.spines["polar"].set_color(t["grid"])

    for key, *_ in METRICS:
        spec = SERIES[key]
        for run in arcs([a for a in d if d[a].get(key)]):
            ax.plot(np.radians(run),
                    np.clip([d[a][key]["nrmse"] for a in run], 0, lim),
                    color=t[key], ls=spec["ls"], lw=1.6, zorder=3,
                    solid_capstyle="round")
        for a in d:
            r = d[a].get(key)
            if not r:
                continue
            over = r["nrmse"] > lim
            clipped += int(over)
            ax.plot([np.radians(a)], [min(r["nrmse"], lim)],
                    marker=spec["marker"], ms=5 if not labels else 7,
                    mfc=t["surface"] if over else t[key], mec=t[key],
                    mew=1.6 if over else 1.1, zorder=4, linestyle="none")

    if labels:
        for i, a in enumerate(sorted(d)):
            rad = np.radians(a)
            ha = ("left" if np.cos(rad) > 0.15
                  else "right" if np.cos(rad) < -0.15 else "center")
            va = ("bottom" if np.sin(rad) > 0.15
                  else "top" if np.sin(rad) < -0.15 else "center")
            ax.text(rad, lim * (1.06 if i % 2 == 0 else 1.58),
                    label_for(a, d[a]), fontsize=6.2, color=t["text2"],
                    ha=ha, va=va, linespacing=1.4, zorder=5)
    return clipped


def series_legend(fig, nrow, labels):
    handles = [Line2D([], [], color=THEME[STYLE][k], ls=SERIES[k]["ls"], lw=1.8,
                      marker=SERIES[k]["marker"], ms=6, label=SERIES[k]["label"])
               for k, *_ in METRICS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=10,
               bbox_to_anchor=(0.5, (-0.07 if labels else -0.012) / nrow))


def draw(variant, data, gts, order, t, path, lim, labels, rows_df):
    n = len(order)
    ncol = NCOLS or int(np.clip(np.ceil(np.sqrt(n)), 1, 6))
    nrow = int(np.ceil(n / ncol))
    size = 6.4 if labels else 3.6
    side = 0.95 if not labels else 1.35
    fig = plt.figure(figsize=(size * (1 + side) * ncol, size * nrow * 1.08))
    gs = fig.add_gridspec(nrow, 2 * ncol, width_ratios=[1, side] * ncol,
                          wspace=0.05, hspace=0.42 if not labels else 0.9)
    axes, texts = [], []
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol)
        axes.append(fig.add_subplot(gs[r, 2 * c], projection="polar"))
        texts.append(fig.add_subplot(gs[r, 2 * c + 1]))

    clipped = 0
    for ax, dom in zip(axes, order):
        clipped += draw_panel(ax, data[dom], t, lim, labels)
    for tx, dom in zip(texts, order):
        tx.axis("off")
        tx.text(0.0 if not labels else 0.30, 0.5,
                side_table(dom, gts, data[dom], TOP_N),
                transform=tx.transAxes, ha="left", va="center",
                fontsize=7.4 if not labels else 9.0, color=t["text2"],
                family="monospace", linespacing=1.5)
    for ax in axes[n:]:
        ax.set_visible(False)
    for tx in texts[n:]:
        tx.set_visible(False)

    series_legend(fig, nrow, labels)

    o = overall(rows_df)
    head = title_for(variant)
    sub = (f"radius = NRMSE % of the real range   ·   {n} configuration(s)   ·   "
           f"{len(rows_df)} sweep push(es), "
           f"{'held-out groups' if HELDOUT == 'on' else 'ALL groups (diagnostic)'}")
    sub += (f"\nmatched within mass +/-{MASS_TOL:g} kg, mu +/-{MU_TOL:g}   ·   "
            f"the sim object is a CUBE at the real object's mass and mu, not its "
            f"shape")
    if "mass" in o:
        sub += (f"\nwhole run:  mass NRMSE {o['mass']['nrmse']:.1f}%  "
                f"(a constant at the mean scores {o['mass']['base']:.1f}%)   "
                f"bias {o['mass']['bias']:+.3f} kg   gain {o['mass']['gain']:.2f}   "
                f"skill {o['mass']['skill']:.1f}%")
    if clipped:
        sub += f"   ·   {clipped} point(s) beyond {RMAX_CLIP:g}% drawn hollow on the rim"
    fig.suptitle(head + "\n" + sub, fontsize=12, color=t["text"],
                 y=0.995, va="bottom")
    fig.subplots_adjust(top=0.995 - (0.10 / nrow), bottom=0.03 / nrow + 0.01,
                        left=0.015, right=0.99)
    fig.patch.set_facecolor(t["surface"])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    return path


SIDECAR_BASE_RESOLVED = ""
# {channel: log flag} from configs.ROOT_SPEC for this collection root, filled by
# resolve_data(). Empty when configs.py had no entry, in which case load_sidecar
# has nothing to cross-check against and trusts the sidecar.
ROOT_LOGS_RESOLVED = {}


def main():
    global SIDECAR_BASE_RESOLVED, ROOT_LOGS_RESOLVED
    if LABELS not in ("on", "off", "auto"):
        raise SystemExit(f"LABELS must be on|off|auto, got {LABELS!r}")
    if SINGLES not in ("on", "off", "only"):
        raise SystemExit(f"SINGLES must be on|off|only, got {SINGLES!r}")
    os.makedirs(OUT_DIR, exist_ok=True)
    t = THEME[STYLE]

    print("=" * 88)
    print(" PER-ESTIMATOR POLAR CHARTS  --  SIMULATION")
    print("=" * 88)
    print(f"  checkpoints: {CKPT_ROOT}")
    for _n in _DISCOVERY_NOTES:
        print(_n)
    if not VARIANT_SPEC:
        raise SystemExit(
            f"No run under {CKPT_ROOT} has a config.json naming an "
            f"input_variant.\n"
            f"  Expected <CKPT_ROOT>/<run>/<model_string>/config.json beside "
            f"{CKPT_FILE}.\n"
            f"  Set CKPT_ROOT to the collection root you mean, or fill in "
            f"VARIANT_SPEC_OVERRIDE.")
    print(f"  variants   : {VARIANTS}")
    print(f"  discovered : {len(_DISCOVERED)} run(s) from "
          f"{CKPT_ROOT}/*/*/config.json"
          + (f", {len(VARIANT_SPEC_OVERRIDE)} overridden"
             if VARIANT_SPEC_OVERRIDE else ""))
    for _v in VARIANTS:
        if _v in VARIANT_SPEC:
            _r, _m = VARIANT_SPEC[_v]
            _src = "override" if _v in VARIANT_SPEC_OVERRIDE else "config.json"
            print(f"     {_v:<20} {_r}/{_m}   [{_src}]")
        else:
            print(f"     {_v:<20} NOT FOUND under this root")
    print(f"  configs    : {len(REAL_CONFIGS)} real object-surface pairs")
    print(f"  held-out   : {HELDOUT}")
    if VEL_NOISE_OBJ.enabled:
        print(f"  vel noise  : {VEL_NOISE_OBJ.describe()}")
        print(f"               injected on the RAW velocity channel, seed "
              f"{VEL_NOISE_SEED}, the same way train.py does.")
        print(f"               This is a DIAGNOSTIC: it asks how the model "
              f"behaves when the simulated")
        print(f"               input carries the hardware noise floor. The "
              f"headline sim figure is VEL_NOISE=off.")
    else:
        print(f"  vel noise  : off (the simulated channel as collected)")
    print(f"  normalize  : m /{M_RANGE:.4f} kg   mu /{MU_RANGE:.4f}")
    if abs(M_RANGE - _M_SPAN) < 1e-6 and abs(MU_RANGE - _MU_SPAN) < 1e-6:
        print(f"               = the span of the 27 configurations, which is what "
              f"the robot side uses")
        print(f"               (recovered from its own CSV as 1.768999 / "
              f"0.275400). Comparable as-is.")
    else:
        print(f"               OVERRIDDEN from the default {_M_SPAN:.4f} / "
              f"{_MU_SPAN:.4f}. The real figures use the default,")
        print(f"               so these radii are NOT comparable with them unless "
              f"the real side was overridden to match.")

    csv_path, sidecar_base = resolve_data()
    SIDECAR_BASE_RESOLVED = sidecar_base
    _entry = root_spec_entry()
    ROOT_LOGS_RESOLVED = _entry[2] if _entry else {}
    if not os.path.exists(csv_path):
        raise SystemExit(f"sweep not found: {csv_path}")
    print(f"  sweep      : {csv_path}")
    if sidecar_base != csv_path:
        print(f"  sidecars   : {sidecar_base}.arm_stats.<variant>.npz")

    prefixes = needed_prefixes()
    print(f"  channels   : {prefixes or 'velocity only'}")

    df, vel_cols = load_sweep(csv_path, prefixes)
    df, heldout_groups = hold_out(df)

    df, n_off_grid, _ang = bin_angles(df)
    print(f"  angle grid : {len(PUSH_ANGLES_DEG)} targets, +/-{ANGLE_TOL_DEG:g} deg; "
          f"{n_off_grid} push(es) outside every bin, {len(df)} kept")

    df, contested, n_unmatched = match_configs(df)
    print(f"  matched    : {len(df)} push(es) to {df['domain'].nunique()} of "
          f"{len(REAL_CONFIGS)} configuration(s); {n_unmatched} row(s) matched "
          f"none")
    if contested:
        print(f"  [note] {contested} row(s) were within tolerance of more than one "
              f"configuration; each went to the nearest. If that count is large, "
              f"MASS_TOL/MU_TOL exceed the spacing between configurations.")
    if df.empty:
        raise SystemExit("no sweep row matched any configuration; widen MASS_TOL "
                         "/ MU_TOL, or check the sweep covers these properties")

    missing = [c[0] for c in REAL_CONFIGS
               if c[0] not in set(df["domain"].astype(str))]
    if missing:
        print(f"  [WARNING] {len(missing)} configuration(s) have no matching sweep "
              f"row and will be absent from the figures:")
        for d in missing:
            w = dict((c[0], (c[1], c[2])) for c in REAL_CONFIGS)[d]
            print(f"     {d:<32} m={w[0]:.3f} mu={w[1]:.4f}")
        print(f"            The sweep samples (mass, mu) at random, so a corner of "
              f"the box can simply be unpopulated after the held-out split.")

    df["m_gt"] = df["gt_mass"].astype(float)
    df["mu_gt"] = df["gt_mu"].astype(float)

    # ---- per-configuration coverage, before any model runs ------------------
    print(f"\n  coverage per configuration (pushes / angles covered):")
    print(f"  {'configuration':<32}{'pushes':>8}{'angles':>8}"
          f"{'m matched':>12}{'mu matched':>12}")
    for d, g in sorted(df.groupby("domain"),
                       key=lambda kv: [c[0] for c in REAL_CONFIGS].index(kv[0])):
        print(f"  {d:<32}{len(g):>8}{g['_bin'].nunique():>8}"
              f"{g['m_gt'].mean():>9.3f}±{g['m_gt'].std(ddof=0):<.3f}"
              f"{g['mu_gt'].mean():>8.4f}±{g['mu_gt'].std(ddof=0):<.4f}")

    # ---- inference ----------------------------------------------------------
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"\n  device     : {device}")

    loaded = {}
    for v in VARIANTS:
        if v not in VARIANT_SPEC:
            print(f"  [skip] no VARIANT_SPEC entry for {v!r}")
            continue
        print(f"\n--- {v} ---")
        try:
            pr = predict(df, vel_cols, v, device)
        except SystemExit as e:
            print(f"  [ERROR] {e}")
            continue
        except Exception as e:                               # noqa: BLE001
            print(f"  [ERROR] {type(e).__name__}: {e}")
            continue
        d = df.copy()
        d["m_est"] = pr[:, 0]
        d["mu_est"] = pr[:, 1]
        order, gts = domains_of(d)
        if DOMAINS.strip():
            want = DOMAINS.split()
            bad = [x for x in want if x not in order]
            if bad:
                raise SystemExit(f"configuration(s) not matched in the sweep: {bad}")
            order = want
        data = {x: per_domain(d, x) for x in order}
        data = {x: y for x, y in data.items() if y}
        order = [x for x in order if x in data]
        if not order:
            print(f"  [skip] {v}: no usable configuration")
            continue
        loaded[v] = (d, data, gts, order)
        o = overall(d)
        print(f"  mass NRMSE {o['mass']['nrmse']:6.2f}%  "
              f"(constant-at-the-mean {o['mass']['base']:.2f}%)   "
              f"gain {o['mass']['gain']:.2f}   skill {o['mass']['skill']:.1f}%")
        if "fric" in o:
            print(f"  mu   NRMSE {o['fric']['nrmse']:6.2f}%  "
                  f"(constant-at-the-mean {o['fric']['base']:.2f}%)   "
                  f"gain {o['fric']['gain']:.2f}   skill {o['fric']['skill']:.1f}%")

    if not loaded:
        raise SystemExit("no variant produced predictions")

    # ---- one radial scale across everything --------------------------------
    vals = [r[k]["nrmse"] for _d, data, _g, _o in loaded.values()
            for dd in data.values() for r in dd.values()
            for k, *_ in METRICS if r.get(k)]
    shared = min(max(vals + [1.0]) * 1.08, RMAX_CLIP)
    if RSCALE == "shared":
        print(f"\n  radial max : {shared:.0f}%  (shared by every panel and image)")
        print(f"               To overlay these on the real figures, run the real "
              f"script with RMAX_CLIP={RMAX_CLIP:g} and compare the printed "
              f"radial max; the larger of the two should be forced on both.")

    n_panels = max(len(o) for _d, _da, _g, o in loaded.values())
    labels = (LABELS == "on" or (LABELS == "auto" and n_panels <= LABEL_PANEL_MAX))
    print(f"  labels     : {'on' if labels else 'off'}  "
          f"({n_panels} panel(s) per image; the singles inherit this)")

    rows, n_singles = [], 0
    print("\n  written:")
    for v, (d, data, gts, order) in loaded.items():
        lim = shared
        if RSCALE != "shared":
            own = [r[k]["nrmse"] for dd in data.values() for r in dd.values()
                   for k, *_ in METRICS if r.get(k)]
            lim = min(max(own + [1.0]) * 1.08, RMAX_CLIP)

        tag = tag_for(v)
        if SINGLES != "only":
            png = os.path.join(OUT_DIR, f"estimator_polar_{tag}.png")
            print(f"  {draw(v, data, gts, order, t, png, lim, labels, d)}")

        single_dir = os.path.join(OUT_DIR, tag)
        if SINGLES != "off":
            os.makedirs(single_dir, exist_ok=True)

        for dom in order:
            if SINGLES != "off":
                sp = os.path.join(single_dir, f"{safe_name(dom)}.png")
                # The grid function with a one-element order: same panel, same
                # side table, same radial limit, saved on its own.
                draw(v, data, gts, [dom], t, sp, lim, labels, d)
                n_singles += 1
            for a in sorted(data[dom]):
                rec = {"tag": tag, "variant": v, "domain": dom,
                       "push_angle_deg": a}
                for key, _e, _g, _r, _s, _dc in METRICS:
                    st = data[dom][a].get(key)
                    if st:
                        rec[f"{key}_gt"] = round(st["gt"], 6)
                        rec[f"{key}_est"] = round(st["est"], 6)
                        rec[f"{key}_err"] = round(st["est"] - st["gt"], 6)
                        rec[f"{key}_nrmse_pct"] = round(st["nrmse"], 3)
                        rec[f"{key}_n"] = st["n"]
                rows.append(rec)
        if SINGLES != "off":
            print(f"  {single_dir}{os.sep}   {len(order)} configuration(s)")

    out = pd.DataFrame(rows)
    out["heldout"] = HELDOUT
    out["mass_tol"] = MASS_TOL
    out["mu_tol"] = MU_TOL
    out["m_range"] = M_RANGE
    out["mu_range"] = MU_RANGE
    csv_out = os.path.join(OUT_DIR, "sim_estimator_polar_per_angle.csv")
    out.to_csv(csv_out, index=False)
    print(f"  {csv_out}")

    if TOP_N > 0 and len(out):
        sc = out.assign(_score=out[[c for c in ("mass_nrmse_pct", "fric_nrmse_pct")
                                    if c in out.columns]].mean(axis=1))
        best = (sc.sort_values(["tag", "domain", "_score"])
                  .groupby(["tag", "domain"]).head(TOP_N).drop(columns="_score"))
        p = os.path.join(OUT_DIR, f"sim_estimator_polar_top{TOP_N}.csv")
        best.to_csv(p, index=False)
        print(f"  {p}")
        print(f"\n  [note] this top-{TOP_N} ranks by the MEAN of the two NRMSEs, so "
              f"one set of angles serves both quantities. The per-quantity tables "
              f"in make_result_tables.py rank mass and mu independently and will "
              f"not agree with it.")

    print(f"\n  OVERALL, per estimator")
    # gain and skill for FRICTION as well as mass. NRMSE alone cannot tell a
    # model that tracks mu badly from one that ignores mu and emits a constant:
    # both land near the constant-at-the-mean score. gain (the ratio of
    # predicted spread to true spread) goes to ~0 for a constant, and skill goes
    # to ~0 with it, so the two columns separate the cases at a glance.
    print(f"  {'estimator':<40}{'NRMSE_m':>9}{'bias':>8}{'gain':>7}{'skill':>8}"
          f"{'NRMSE_u':>9}{'gain_u':>8}{'skill_u':>9}{'base_u':>8}")
    for v, (d, _da, _g, _o) in loaded.items():
        o = overall(d)
        if "mass" not in o:
            continue
        print(f"  {title_for(v)[:39]:<40}{o['mass']['nrmse']:>8.1f}%"
              f"{o['mass']['bias']:>8.3f}{o['mass']['gain']:>7.2f}"
              f"{o['mass']['skill']:>7.1f}%"
              + (f"{o['fric']['nrmse']:>8.1f}%{o['fric']['gain']:>8.2f}"
                 f"{o['fric']['skill']:>8.1f}%{o['fric']['base']:>7.1f}%"
                 if "fric" in o else f"{'--':>9}{'--':>8}{'--':>9}{'--':>8}"))

    print("\n  gain_u near 0 with skill_u near 0 means mu is not being tracked: "
          "the model emits")
    print("  something close to a constant, and its NRMSE_u is then just base_u, "
          "what a")
    print("  constant at the mean scores. A real mu estimate has gain_u near 1 "
          "and skill_u well")
    print("  above 0.")
    print("\n  READING THESE AGAINST THE REAL FIGURES")
    print("  Same radius, same angles, same configurations -- but the sim object is")
    print("  a cube at the real object's mass and friction, and each cell here")
    print("  averages many pushes where the robot had one. A gap between the two")
    print("  figures therefore mixes the sim-to-real gap with the shape difference")
    print("  and with that averaging; it is a comparison of trends, not of cells.")
    if HELDOUT != "on":
        print("\n  HELDOUT IS OFF: these groups include ones the checkpoints trained")
        print("  on, so the sim numbers are optimistic and not comparable with the")
        print("  real ones at all. Re-run with HELDOUT=on before reporting.")
    print("=" * 88)


if __name__ == "__main__":
    if REPORT == "off":
        main()
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        rp = os.path.join(OUT_DIR, "sim_estimator_polar_report.txt")
        saved = sys.stdout
        with open(rp, "w") as fh:
            sys.stdout = _Tee(saved, fh)
            try:
                main()
            except BaseException:
                import traceback
                traceback.print_exc(file=sys.stdout)
                raise
            finally:
                sys.stdout = saved
        print(f"\n  report: {rp}")