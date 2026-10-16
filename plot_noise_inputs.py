"""
plot_noise_inputs.py  --  what the model actually saw during a noised run.

Rebuilds the encoder input the way train.py assembles it -- raw velocity in
channel 0, the variant's window channels standardized with the sidecar THIS run
wrote -- injects the same observation noise, and draws the result.

    TRAIN_FROM=from_20261016 PHYPUSH_INPUT_VARIANT=vel_manip_seq \
        python plot_noise_inputs.py

The sigma comes from the trained run's own config.json, not from
vel_noise_model.json. They are the same number today; they stop being the same
number the moment the fit is re-run, and this script is supposed to show what a
PARTICULAR run was trained on, not what the current model file says.

WHAT EACH PANEL ANSWERS
  (a)(b)  does the noised velocity look like the real robot's? -- the same
          check compare_noise_injected_channels.py makes on the robot computer,
          here on the actual training tensor.
  (c)     is the injected noise the size it was supposed to be? The difference
          between (b) and (a) IS the noise, so its std is a direct measurement
          rather than a restatement of the config.
  (d)     the assembled input, every channel at its true scale. Velocity is fed
          RAW (m/s) while the window channels are STANDARDIZED, so the channels
          arrive at the projection two orders of magnitude apart. That is a
          property of the pipeline, not of the noise -- but it is the context
          in which the noise has to matter, and it is invisible in (a)-(c).
  (e)     the spectrum, with the band the sigma was fitted in. The noised floor
          should sit at sigma.
  (f)     the PINN sliding mask, which train.py builds from the CLEAN velocity.
          This panel measures what would have changed had the noise reached it.
"""

import os
import re
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import (CSV_PATH, INPUT_VARIANT, TRAIN_FROM, GRIPPER_CLOSED,
                     VEL_PREFIX, USE_VEL, VARIANT_SEQ_PREFIXES, VARIANT_SEQ_LOG,
                     VARIANT_COND_PREFIX, VARIANT_COND_LOG, COND_DIM,
                     M_SEEN_MIN, M_SEEN_MAX, MU_SEEN_MIN, MU_SEEN_MAX,
                     MAX_PUSH_LATERAL_OFFSET, MAX_PUSH_COM_OFFSET,
                     used_config)
from dataset import window_cols, variant_windows, variant_window, load_arm_stats
from vel_noise import VelNoise

# =============================================================================
# CONFIG
# =============================================================================
# Rows to read from the sweep CSV. The file is ~15 GB and this script needs a
# few hundred usable pushes, so it reads a slice rather than the whole thing --
# the standardization stats come from the SIDECAR, not from this slice, so a
# partial read changes nothing about what is drawn.
N_ROWS = int(os.environ.get("N_ROWS", 40000))
SKIP = int(os.environ.get("SKIP", 0))        # start later in the file
N_SHOW = int(os.environ.get("N_SHOW", 6))    # example traces per panel
SEED = int(os.environ.get("SEED", 0))
FS = float(os.environ.get("FS", 100.0))
OUT = os.environ.get("OUT", "")
DPI = int(os.environ.get("DPI", 160))
CHECKPOINTS_ROOT = os.environ.get("CHECKPOINTS_ROOT", "./results/checkpoints")

# Palette and encodings match the rest of the project: slot 1 blue, slot 2
# orange, slot 3 aqua, identity never carried by colour alone.
C_CLEAN, C_NOISE, C_EXTRA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


def theme():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": TEXT,
        "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.edgecolor": GRID, "grid.color": GRID, "axes.titlecolor": TEXT,
        "axes.grid": True, "grid.alpha": 0.5, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
        "legend.frameon": False, "lines.linewidth": 1.4, "figure.dpi": DPI,
    })


# =============================================================================
# THE RUN'S OWN SIGMA
# =============================================================================
def run_config():
    """The config.json of the trained run for this (TRAIN_FROM, variant).

    Found by matching input_variant inside the root's run directories rather
    than by rebuilding the model-string path, for the same reason evaluate.py
    now reads the model directory off disk: the name carries a noise tag this
    script cannot derive.
    """
    gripper = "gripper_closed" if GRIPPER_CLOSED else "gripper_opened"
    pattern = os.path.join(CHECKPOINTS_ROOT, TRAIN_FROM or "*", gripper,
                           "*", "*", "config.json")
    hits = []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as fh:
                cfg = json.load(fh)
        except Exception:                                    # noqa: BLE001
            continue
        if cfg.get("input_variant") == INPUT_VARIANT:
            hits.append((p, cfg))
    if not hits:
        return None, None
    if len(hits) > 1:
        print(f"  [note] {len(hits)} runs of '{INPUT_VARIANT}' under "
              f"'{TRAIN_FROM}'; using the most recent:")
        for p, _ in hits:
            print(f"     {os.path.dirname(p)}")
    return hits[-1]


def resolve_noise():
    """(VelNoise, where it came from). Prefers the run's own recorded sigma."""
    path, cfg = run_config()
    if cfg is not None and float(cfg.get("vel_noise_sigma", 0.0) or 0.0) > 0:
        s = float(cfg["vel_noise_sigma"])
        return VelNoise(s, source=path), f"the run's config.json ({path})"
    if cfg is None:
        # Distinguished from the case below on purpose: "the model file is
        # missing" would send you looking for vel_noise_model.json when what is
        # actually missing is the trained run this script is supposed to depict.
        print(f"  [WARNING] no trained run of '{INPUT_VARIANT}' found under "
              f"'{TRAIN_FROM}' in {CHECKPOINTS_ROOT}.")
        print(f"            Falling back to the current model file, so the "
              f"figure shows what such a run WOULD have been trained on rather "
              f"than what one was.")
    if cfg is not None:
        print(f"  [WARNING] {path} records vel_noise_sigma = "
              f"{cfg.get('vel_noise_sigma')!r}.")
        print(f"            That run was trained WITHOUT injected noise. The "
              f"panels below would show noise it never saw, so the sigma is "
              f"taken from the model file instead and the figure is labelled "
              f"as hypothetical.")
    vn = VelNoise.from_json(os.environ.get("VEL_NOISE_JSON",
                                           "vel_noise_model.json"))
    return vn, "vel_noise_model.json (NOT this run's recorded sigma)"


# =============================================================================
# DATA
# =============================================================================
def needed_columns(header):
    cols = ["gt_mass", "gt_mu", "start_t"]
    cols += [c for c in header if c.startswith("input_vel_")
             and c[len("input_vel_"):].isdigit()]
    cols += [c for c in header if c.startswith("input_acc_")
             and c[len("input_acc_"):].isdigit()]
    for p in list(VARIANT_SEQ_PREFIXES) + ([VARIANT_COND_PREFIX] if COND_DIM else []):
        if p and p != VEL_PREFIX:
            cols += window_cols(header, p)
    for c in ("push_start_lateral_offset", "push_com_offset"):
        if c in header:
            cols.append(c)
    return [c for c in dict.fromkeys(cols) if c in header]


def load_slice():
    """A filtered slice of the sweep, matching create_dataloaders' own filters.

    Every filter here exists in create_dataloaders. Drawing rows it would have
    dropped would show the model an input it was never given.
    """
    header = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    use = needed_columns(header)
    skip = range(1, SKIP + 1) if SKIP > 0 else None
    df = pd.read_csv(CSV_PATH, usecols=use, nrows=N_ROWS, skiprows=skip)
    n0 = len(df)

    df = df[(df["gt_mass"] >= M_SEEN_MIN) & (df["gt_mass"] <= M_SEEN_MAX)
            & (df["gt_mu"] >= MU_SEEN_MIN) & (df["gt_mu"] <= MU_SEEN_MAX)]
    n_seen = len(df)

    if MAX_PUSH_LATERAL_OFFSET is not None and "push_start_lateral_offset" in df:
        df = df[df["push_start_lateral_offset"] < MAX_PUSH_LATERAL_OFFSET]
    if MAX_PUSH_COM_OFFSET is not None and "push_com_offset" in df:
        df = df[df["push_com_offset"] < MAX_PUSH_COM_OFFSET]
    n_quality = len(df)

    vel_cols = sorted([c for c in df.columns if c.startswith("input_vel_")],
                      key=lambda x: int(x.split("_")[-1]))
    seq_len = len(vel_cols)
    df = df[(df["start_t"] + seq_len) <= 100].copy()

    print(f"  rows   : {n0} read -> {n_seen} in the seen box -> "
          f"{n_quality} after push quality -> {len(df)} after the window filter")
    if len(df) == 0:
        raise SystemExit(
            "no usable rows in this slice. Raise N_ROWS, or set SKIP to read "
            "from further into the file.")
    print(f"  objects: {df.groupby(['gt_mass', 'gt_mu']).ngroups} distinct "
          f"(mass, mu) pair(s), mass {df.gt_mass.min():.3f}-{df.gt_mass.max():.3f}")
    return df, vel_cols, seq_len


def build_input_blocks(df, vel_cols, seq_len, noise):
    """(clean, noised, channel_names) -- the encoder input, both ways.

    Mirrors train.build_input exactly: velocity raw in channel 0, the window
    channels standardized with the SIDECAR's mean/std, concatenated in
    VARIANT_CHANNELS order. The noise is added to the velocity channel only,
    and only in the noised copy.
    """
    V = df[vel_cols].values.astype(np.float32)[:, :, None]       # (N, T, 1)
    names = []
    blocks = []
    if USE_VEL:
        blocks.append(V)
        names.append(f"{VEL_PREFIX}  [m/s, raw]")

    if VARIANT_SEQ_PREFIXES:
        A, valid, _ = variant_windows(df, VARIANT_SEQ_PREFIXES,
                                      VARIANT_SEQ_LOG, seq_len=seq_len)
        _, mean, std = load_arm_stats()                          # (1, K) each
        print(f"  stats  : sidecar mean {np.round(mean.reshape(-1), 6)} "
              f"std {np.round(std.reshape(-1), 6)}")
        A = (A - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        if not valid.all():
            keep = valid
            print(f"  [filter] {int((~keep).sum())} row(s) dropped: a log "
                  f"channel was non-positive, as create_dataloaders does")
            A, V = A[keep], V[keep]
            blocks = [V] if USE_VEL else []
            df = df[keep]
        blocks.append(A)
        for p, lg in zip(VARIANT_SEQ_PREFIXES, VARIANT_SEQ_LOG):
            names.append(f"{p}  [{'log, ' if lg else ''}standardized]")

    clean = np.concatenate(blocks, axis=-1) if blocks else V
    noised = clean.copy()
    if USE_VEL and noise.enabled:
        rng = np.random.default_rng(SEED)
        noised[:, :, 0] = clean[:, :, 0] + rng.normal(
            0.0, noise.sigma, size=clean[:, :, 0].shape)
    return clean, noised, names, df


# =============================================================================
# SPECTRUM
# =============================================================================
def psd(W, fs):
    N = W.shape[1]
    tt = np.arange(N)
    A = np.vstack([tt, np.ones(N)]).T
    coef, *_ = np.linalg.lstsq(A, W.T, rcond=None)
    Xd = W - (A @ coef).T
    w = np.hanning(N)
    F = np.fft.rfft(Xd * w, axis=1)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    S = 2.0 * np.abs(F) ** 2 / (fs * np.sum(w ** 2))
    S[:, 0] /= 2.0
    if N % 2 == 0:
        S[:, -1] /= 2.0
    return f, S.mean(axis=0)


# =============================================================================
# FIGURE
# =============================================================================
def draw(clean, noised, names, noise, src, seq_len, n_obj):
    theme()
    tt = np.arange(seq_len)
    vc, vn_ = clean[:, :, 0], noised[:, :, 0]
    show = min(N_SHOW, len(vc))

    fig = plt.figure(figsize=(13.0, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.27)

    # ---- (a)(b) velocity, clean and noised, shared y --------------------
    lo = min(vc[:show].min(), vn_[:show].min())
    hi = max(vc[:show].max(), vn_[:show].max())
    pad = 0.07 * (hi - lo if hi > lo else 1.0)
    for k, (lab, W, col) in enumerate(
            [("(a) velocity as collected", vc, C_CLEAN),
             ("(b) velocity as the model saw it", vn_, C_NOISE)]):
        ax = fig.add_subplot(gs[0, k])
        for row in W[:show]:
            ax.plot(tt, row, color=col, lw=1.0, alpha=0.8)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{lab}   n={show}", loc="left")
        ax.set_xlabel("timestep")
        if k == 0:
            ax.set_ylabel("EE velocity [m/s]")
        else:
            ax.tick_params(labelleft=False)

    # ---- (c) the injected noise itself ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    d = (vn_ - vc).ravel() * 1e3
    if noise.enabled and d.size:
        q = float(np.percentile(np.abs(d), 99.5))
        ax.hist(d, bins=np.linspace(-q, q, 61), density=True,
                histtype="stepfilled", facecolor=C_NOISE, edgecolor=C_NOISE,
                alpha=0.35, linewidth=1.4)
        meas = float(np.std(d)) * 1e-3
        ax.axvline(noise.sigma * 1e3, color=TEXT2, ls=(0, (4, 3)), lw=1.1)
        ax.axvline(-noise.sigma * 1e3, color=TEXT2, ls=(0, (4, 3)), lw=1.1)
        ax.set_title(f"(c) injected noise   measured sd {meas:.6f}\n"
                     f"configured $\\sigma$ {noise.sigma:.6f}  "
                     f"ratio {meas / noise.sigma:.3f}", loc="left")
    else:
        ax.text(0.5, 0.5, "no noise injected", ha="center", va="center",
                transform=ax.transAxes, color=TEXT2)
        ax.set_title("(c) injected noise", loc="left")
    ax.set_xlabel("noised $-$ clean  [mm/s]")
    ax.set_ylabel("density")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    # ---- (d) every channel at its true scale ----------------------------
    ax = fig.add_subplot(gs[1, 0])
    cols = [C_NOISE, C_EXTRA, "#eda100", "#6250d6"]
    for c in range(noised.shape[-1]):
        ax.plot(tt, noised[0, :, c], color=cols[c % len(cols)],
                ls="-" if c == 0 else (0, (4, 2)),
                label=names[c] if c < len(names) else f"channel {c}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("channel value")
    # The velocity trace looks flat here, and that IS the finding -- but a
    # reader should not have to infer the size of the gap from a squashed line,
    # so it is stated.
    sds = [float(noised[:, :, c].std()) for c in range(noised.shape[-1])]
    # On its own line: appended to the title it ran into panel (e)'s.
    gap = (f"\nvelocity sd {sds[0]:.4f} vs "
           f"{'/'.join(f'{s:.2f}' for s in sds[1:])} standardized, "
           f"{max(sds[1:]) / max(sds[0], 1e-12):.0f}x apart"
           if len(sds) > 1 else "")
    ax.set_title(f"(d) the assembled input, one push, true scales{gap}",
                 loc="left", fontsize=9)
    ax.legend(loc="best", fontsize=7.5)

    # ---- (e) spectrum ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for lab, W, col, ls in (("clean", vc, C_CLEAN, "-"),
                            ("+ noise", vn_, C_NOISE, (0, (3, 2)))):
        f, S = psd(W, FS)
        ok = f > 0
        ax.plot(f[ok], S[ok], color=col, ls=ls, label=lab)
    if noise.enabled:
        floor = noise.sigma ** 2 * 2.0 / FS
        ax.axhline(floor, color=TEXT2, ls=(0, (1, 2)), lw=1.1,
                   label=r"$\sigma^2\,2/f_s$")
    ax.set_yscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"PSD [(m/s)$^2$/Hz]")
    ax.set_title("(e) spectrum of the velocity channel", loc="left")
    ax.legend(loc="best", fontsize=7.5)

    # ---- (f) the PINN mask the noise does NOT touch ---------------------
    ax = fig.add_subplot(gs[1, 2])
    thr = float(used_config.get("vel_filter_threshold", 0.01))
    half = seq_len // 2
    m_clean = (np.abs(vc) > thr).astype(float)
    m_noisy = (np.abs(vn_) > thr).astype(float)
    ax.plot(tt, m_clean.mean(axis=0), color=C_CLEAN, lw=1.6,
            label="from clean velocity  (what train.py uses)")
    # Drawn thinner and dashed over the clean curve: when the noise changes
    # nothing the two coincide exactly, and an invisible second series reads as
    # a plotting failure rather than as the result.
    ax.plot(tt, m_noisy.mean(axis=0), color=C_NOISE, ls=(0, (2, 3)), lw=2.4,
            alpha=0.85, label="if the noise had reached it")
    ax.axvspan(half, seq_len - 1, color=TEXT2, alpha=0.08, linewidth=0, zorder=0)
    flips_half = int((m_clean[:, half:] != m_noisy[:, half:]).sum())
    n_half = m_clean[:, half:].size
    flips_all = int((m_clean != m_noisy).sum())
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel(f"timestep   shaded: the friction half the mask is sliced to")
    ax.set_ylabel("fraction of pushes sliding")
    coincide = ("  -- the two curves coincide" if flips_all == 0 else "")
    ax.set_title(f"(f) PINN sliding mask, |v| > {thr:g}{coincide}\n"
                 f"would-be flips: {flips_half}/{n_half} in the friction half, "
                 f"{flips_all} over the window", loc="left", fontsize=9)
    ax.legend(loc="lower right", fontsize=7.5)

    state = ("$\\sigma$ = %.6f m/s" % noise.sigma if noise.enabled
             else "NO NOISE")
    fig.suptitle(
        f"Training input for {INPUT_VARIANT} under {TRAIN_FROM}   --   "
        f"{state}\n"
        f"{len(clean)} push(es) from {n_obj} (mass, $\\mu$) pair(s)   --   "
        f"sigma from {src}",
        fontsize=10.5, color=TEXT, y=1.01)
    return fig


def main():
    print("=" * 84)
    print(" TRAINING INPUT WITH INJECTED NOISE")
    print("=" * 84)
    print(f"  variant: {INPUT_VARIANT}   train_from: {TRAIN_FROM}")
    print(f"  data   : {CSV_PATH}")
    noise, src = resolve_noise()
    print(f"  sigma  : {noise.sigma:.6f} m/s   <- {src}")
    if not USE_VEL:
        raise SystemExit(
            f"variant '{INPUT_VARIANT}' has no velocity channel, so there is no "
            f"noised input to draw. Pick a vel_* variant.")

    df, vel_cols, seq_len = load_slice()
    n_obj = df.groupby(["gt_mass", "gt_mu"]).ngroups
    clean, noised, names, df = build_input_blocks(df, vel_cols, seq_len, noise)
    print(f"  input  : {clean.shape}  channels {names}")

    for c, nm in enumerate(names):
        col = noised[:, :, c]
        print(f"     ch{c} {nm:38s} mean {col.mean():+.5f}  std {col.std():.5f}"
              f"  range [{col.min():+.4f}, {col.max():+.4f}]")

    fig = draw(clean, noised, names, noise, src, seq_len, n_obj)
    out = OUT or f"noise_input_{INPUT_VARIANT}_{TRAIN_FROM or 'none'}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  written: {os.path.abspath(out)}")
    print("=" * 84)


if __name__ == "__main__":
    main()