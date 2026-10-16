"""
check_top3_sim.py  --  the best 3 push directions per object, SIMULATION side.

The twin of check_top3_real.py. Same columns, same ranking rule, same angle
bookkeeping, so the two output files join directly.

PREREQUISITE
    plot_estimator_polar_sim.py must have been run first: this script reads the
    per-angle table it writes,

        results/estimator_polar_sim/sim_estimator_polar_per_angle.csv

    rather than loading the checkpoints again. One inference pass, one set of
    numbers -- a second pass could differ (dropout, device, a changed sidecar)
    and then the top-3 file and the polar figure would disagree about the same
    run with nothing to say which was right.

    So: to check a different estimator here, add it to VARIANT_SPEC in
    plot_estimator_polar_sim.py, re-run that, then run this.

RANKED PER QUANTITY, NOT BY A COMBINED SCORE
    sim_estimator_polar_top3.csv ranks by the MEAN of the two NRMSEs, so one set
    of three angles serves both quantities. This script ranks mass by mass NRMSE
    and mu by mu NRMSE, independently -- "the best 3 angles for mass" meaning
    exactly that. The two files will disagree; this is the one that answers the
    question asked, and the real-side script does the same thing.

THE ANGLE CONVENTION
    The real robot's push angle is the NEGATIVE of the simulation's for the same
    geometry. Every row carries all three forms:
        push_angle_deg   as recorded on this side (the simulation angle)
        angle_sim_deg    the same number, named for joining
        angle_real_deg   its counterpart in the robot's frame
    so a row here and a row from check_top3_real.py describing the same geometry
    share (domain, quantity, angle_real_deg).

OUTPUT
    results/estimator_polar_sim/top3_check/
        top3_sim_<tag>.csv           one file per estimator
        top3_sim_all.csv             the same rows, concatenated
        top3_sim_report.txt          a readable version of both

USAGE
    python check_top3_sim.py
    TOP_N=5 python check_top3_sim.py
    PER_ANGLE_CSV=/path/to/sim_estimator_polar_per_angle.csv python check_top3_sim.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker

# =============================================================================
# WHERE THE DATA IS
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
POLAR_DIR = os.environ.get(
    "POLAR_DIR", os.path.join(base_dir, "results", "estimator_polar_sim"))
PER_ANGLE_CSV = os.environ.get(
    "PER_ANGLE_CSV", os.path.join(POLAR_DIR, "sim_estimator_polar_per_angle.csv"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(POLAR_DIR, "top3_check"))

TOP_N = int(os.environ.get("TOP_N", 3))

# =============================================================================
# RANK_MODE -- how the top N is decided
#
#   "perquantity"  mass ranked by mass NRMSE, mu by mu NRMSE, independently.
#                  The two quantities generally name different angles.
#   "mean"         one ranking per configuration on the mean of the two NRMSEs,
#                  so one set of angles serves both. This is the rule
#                  estimator_polar_top3.csv uses.
#   "both"         (default) every output is produced twice, once per rule, with
#                  _perq / _mean in the file name and a rank_mode column in the
#                  rows, plus a combined *_bothmodes_all.csv.
#
# They answer different questions and will disagree -- on this data the two
# rankings shared 2 of 27 mass cells. Neither is wrong; "mean" asks which push
# is best overall, "perquantity" asks which push is best for the quantity you
# care about.
# =============================================================================
RANK_MODE = os.environ.get("RANK_MODE", "both")
if RANK_MODE not in ("perquantity", "mean", "both"):
    raise SystemExit(f"RANK_MODE must be perquantity|mean|both, got {RANK_MODE!r}")
MODES = (("perquantity", "mean") if RANK_MODE == "both" else (RANK_MODE,))
MODE_SUFFIX = {"perquantity": "perq", "mean": "mean"}
MODE_LABEL = {"perquantity": "ranked per quantity by its own NRMSE",
              "mean": "ranked by the mean of the mass and mu NRMSEs"}

DOMAINS = os.environ.get("DOMAINS", "")
VARIANTS = os.environ.get("VARIANTS", "")
REPORT = os.environ.get("REPORT", "on")

A0_MAGNITUDES = (0.30, 0.40, 0.50, 0.60, 0.70)
PUSH_ANGLES_DEG = tuple(sorted(round(s * m * 180.0, 6)
                               for m in A0_MAGNITUDES for s in (-1, 1)))

# See check_top3_real.py: one constant, because it is a convention rather than a
# measurement, and the report prints the resulting mapping so a wrong setting
# shows up in the output rather than hiding in it.
ANGLE_SIGN_FLIP = bool(int(os.environ.get("ANGLE_SIGN_FLIP", 1)))

# The per-angle CSV names mu's columns "fric_*"; the output uses "mu" throughout
# so both sides' files carry the same vocabulary.
#   output quantity -> (est col, gt col, nrmse col, n col, decimals)
QUANTITIES = (("mass", "mass_est", "mass_gt", "mass_nrmse_pct", "mass_n", 3),
              ("mu", "fric_est", "fric_gt", "fric_nrmse_pct", "fric_n", 4))

# Quantity order for the figures, independent of the column layout above so the
# plotting code is identical in both scripts.
QUANTITY_KEYS = (("mass",), ("mu",))

OBJECT_DISPLAY = {"cube": "Plastic cube", "tincan": "Tin can",
                  "placon": "Plastic container", "container": "Plastic container",
                  "woodenbox": "Wooden box"}
SURFACE_DISPLAY = {"rubber": "rubber", "woodrough": "rough wood",
                   "woodsmooth": "smooth wood"}
DOMAIN_RE = re.compile(r"^(?P<obj>[A-Za-z]+)(?P<m>[\d.]*)[-_](?P<surf>[A-Za-z]+)"
                       r"(?:_(?P<mu>[\d.]+))?$")


# =============================================================================
# FIGURES
#
# The job of these panels is "which push directions made the cut, and how good
# were they" -- identity (which angle) crossed with magnitude (NRMSE) across the
# configurations. That is a rank map: one row per configuration, one column per
# angle, a mark where that angle reached the top N.
#
# The encoding is deliberately split so neither channel has to carry two jobs:
#   position  which angle, which configuration
#   size      the rank (1 is largest) -- the thing being reported
#   colour    the NRMSE, a single hue light-to-dark as the error rises
# A rank-1 mark is therefore big, and a good one is also pale; the eye finds the
# large marks first, which is the ordering the question asks for.
#
# The hue per quantity (blue for mass, orange for mu) is the one the polar
# figures already use, so a quantity looks the same everywhere. The pair was
# checked rather than eyeballed: Delta E 33.6 for normal vision and 24.7 under
# protanopia, both well clear of the floor.
#
# THE X AXIS IS THE REAL ROBOT'S FRAME ON BOTH SIDES, so the simulation figure
# and the real figure can be laid side by side and their columns line up. Each
# tick carries both numbers, real above simulation, and the panel title says
# which frame ordered the axis.
# =============================================================================
FIGS = os.environ.get("FIGS", "on")
# simple   just the angles and their rank -- the usual question
# detailed the rank map with the errors encoded as well
# hist     only the per-estimator angle histograms
# both     (default) one of each, named apart
#
# The histogram is written under every setting except "off", because it answers
# a question the other two do not: which directions this estimator favours
# across the whole object set, rather than per object.
FIG_STYLE = os.environ.get("FIG_STYLE", "both")
DPI = int(os.environ.get("DPI", 150))

FIG = dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e",
           muted="#8a8984", grid="#e2e1dd")
QUANT_HUE = {"mass": "#2a78d6", "mu": "#eb6834"}
QUANT_CMAP = {"mass": "Blues", "mu": "Oranges"}
QUANT_TITLE = {"mass": "Mass $m$", "mu": r"Friction $\mu$"}


def _angle_axis():
    """(real-frame angles ascending, tick labels carrying both frames)."""
    reals = sorted(PUSH_ANGLES_DEG)
    labs = []
    for a in reals:
        other = -a if ANGLE_SIGN_FLIP else a
        labs.append(f"{a:+.0f}\n{other:+.0f}")
    return reals, labs


def _domain_order(top):
    """Configurations ordered as the polar figures order them: object family,
    then true mass, then mu -- so a row here is the same object as the panel
    with that name there."""
    fam = re.compile(r"^([A-Za-z]+)")
    key = {}
    for d, g in top.groupby("domain"):
        m = g[g["quantity"] == "mass"]["gt"]
        u = g[g["quantity"] == "mu"]["gt"]
        key[d] = ((fam.match(d).group(1) if fam.match(d) else d),
                  float(m.iloc[0]) if len(m) else 0.0,
                  float(u.iloc[0]) if len(u) else 0.0)
    return sorted(key, key=lambda d: key[d])


def _rank_size(rank, n):
    """Rank 1 largest. Linear, floored, so rank N is still clearly a mark."""
    if n <= 1:
        return 170.0
    return float(170.0 - 105.0 * (rank - 1) / (n - 1))


def print_mode_tables(combined, simple, side, mode):
    """The readable tables for one ranking rule."""
    print("\n" + "#" * 92)
    print(f" RANKING: {MODE_LABEL[mode].upper()}")
    print("#" * 92)
    for tag in combined["tag"].unique():
        t = combined[combined["tag"] == tag]
        v, root, filt, backend = split_tag(tag)
        print("\n" + "=" * 92)
        print(f" {tag}")
        print(f"   variant {v}   backend {backend}   "
              f"velocity {pretty_filter(filt)}")
        print("=" * 92)
        for q, *rest in QUANTITY_KEYS:
            dec = 3 if q == "mass" else 4
            sub = t[t["quantity"] == q]
            if sub.empty:
                continue
            print(f"\n  {q.upper()}   (best {TOP_N} angles per object, "
                  f"{MODE_LABEL[mode]})")
            print(f"  {'configuration':<32}{'gt':>9}" +
                  "".join(f"{f'#{i} angle':>10}{'(other)':>9}{'est':>9}{'NRMSE':>8}"
                          for i in range(1, TOP_N + 1)))
            for dom in sub["domain"].unique():
                d = sub[sub["domain"] == dom].sort_values("rank")
                line = f"  {dom:<32}{d['gt'].iloc[0]:>9.{dec}f}"
                other = "angle_sim_deg" if side == "real" else "angle_real_deg"
                for _, r in d.iterrows():
                    line += (f"{r['push_angle_deg']:>+9.0f}°"
                             f"{r[other]:>+9.0f}"
                             f"{r['est']:>9.{dec}f}{r['nrmse_pct']:>7.1f}%")
                print(line)
    print_simple_angles(simple, side, mode)


def write_simple_angles(combined, path, side, mode="perquantity"):
    """The angles alone, one row per (estimator, configuration, quantity).

    The per-estimator CSVs carry the estimate, the error, the count and both
    angle frames, which is what you want when checking a number. This is the
    other thing people actually want: just the directions, wide, one line each,
    sortable and diffable.

    angle1..angleN are this side's own angles, numeric, so the file sorts and
    filters. The two string columns are for reading and for pasting into a
    caption -- the same three directions written in each frame.
    """
    rows = []
    for (tag, dom, q), g in combined.groupby(["tag", "domain", "quantity"],
                                             sort=False):
        g = g.sort_values("rank")
        rec = {"side": side, "rank_mode": mode, "tag": tag,
               "variant": g["variant"].iloc[0], "filter": g["filter"].iloc[0],
               "backend": g["backend"].iloc[0], "domain": dom, "quantity": q}
        for i in range(1, TOP_N + 1):
            r = g[g["rank"] == i]
            rec[f"angle{i}"] = (float(r["push_angle_deg"].iloc[0])
                                if len(r) else np.nan)
        rec["angles_real"] = ", ".join(f"{a:+.0f}" for a in g["angle_real_deg"])
        rec["angles_sim"] = ", ".join(f"{a:+.0f}" for a in g["angle_sim_deg"])
        rec["nrmse_best_pct"] = round(float(g["nrmse_pct"].iloc[0]), 3)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def print_simple_angles(simple, side, mode="perquantity"):
    """The same thing on screen, narrow enough to scan."""
    native = "real" if side == "real" else "sim"
    other = "sim" if side == "real" else "real"
    print("\n" + "=" * 92)
    print(f" TOP-{TOP_N} ANGLES ONLY   ({native} frame, with the {other} frame "
          f"alongside)")
    print(f" {MODE_LABEL[mode]}")
    print("=" * 92)
    for tag, g in simple.groupby("tag", sort=False):
        print(f"\n  {tag}")
        print(f"    {'configuration':<32}{'quantity':<10}"
              f"{f'{native} frame':<22}{f'{other} frame':<22}{'best':>7}")
        for _, r in g.iterrows():
            a_native = r["angles_real"] if side == "real" else r["angles_sim"]
            a_other = r["angles_sim"] if side == "real" else r["angles_real"]
            print(f"    {r['domain']:<32}{r['quantity']:<10}"
                  f"{a_native:<22}{a_other:<22}{r['nrmse_best_pct']:>6.1f}%")


def plot_angle_hist(top, tag, path, side, mode="perquantity"):
    """Per estimator: how often each push angle reaches the top N.

    One bar per angle, stacked by rank, so the height says "this direction is
    useful for N configurations" and the segments say how often it was the best
    one rather than merely third.

    THE DASHED LINE IS WHAT MAKES IT READABLE. With C configurations each
    contributing N picks over A angles, pure chance puts C*N/A picks on every
    angle. A bar below that line is a direction the estimator AVOIDS; only bars
    above it are evidence that the direction carries information. Without the
    reference, every histogram of a top-N selection looks like a result, because
    the bars are never zero.
    """
    reals, labs = _angle_axis()
    xpos = {round(a, 6): i for i, a in enumerate(reals)}
    quants = [q for q, *_ in QUANTITY_KEYS if (top["quantity"] == q).any()]
    if not quants:
        return None

    fig, axes = plt.subplots(1, len(quants), figsize=(7.0 * len(quants), 3.9),
                             squeeze=False)
    for c, q in enumerate(quants):
        ax = axes[0][c]
        ax.set_facecolor(FIG["surface"])
        sub = top[top["quantity"] == q]
        n_conf = sub["domain"].nunique()

        cmap = plt.get_cmap(QUANT_CMAP[q])
        counts = np.zeros((TOP_N, len(reals)), dtype=int)
        for _, r in sub.iterrows():
            xi = xpos.get(round(float(r["angle_real_deg"]), 6))
            if xi is not None and 1 <= int(r["rank"]) <= TOP_N:
                counts[int(r["rank"]) - 1, xi] += 1

        bottom = np.zeros(len(reals))
        for k in range(TOP_N):
            shade = 1.0 - 0.55 * k / max(TOP_N - 1, 1)
            ax.bar(range(len(reals)), counts[k], bottom=bottom, width=0.66,
                   color=cmap(0.35 + 0.5 * shade),
                   # A surface-coloured edge is the 2px gap between stacked
                   # segments; without it the ranks merge into one block.
                   edgecolor=FIG["surface"], linewidth=1.4,
                   label=f"rank {k + 1}", zorder=3)
            bottom += counts[k]

        tot = counts.sum(axis=0)
        for i, n in enumerate(tot):
            if n:
                ax.text(i, n + 0.08 * max(tot.max(), 1), str(int(n)),
                        ha="center", va="bottom", fontsize=7,
                        color=FIG["text2"], zorder=4)

        if n_conf and len(reals):
            chance = n_conf * TOP_N / len(reals)
            ax.axhline(chance, color=FIG["muted"], lw=1.1, ls="--", zorder=2)
            ax.text(len(reals) - 0.45, chance, f" chance {chance:.1f}",
                    fontsize=7, color=FIG["muted"], va="center", ha="left")

        ax.set_xlim(-0.7, len(reals) - 0.3)
        ax.set_ylim(0, max(tot.max() * 1.3, 1.6))
        ax.set_xticks(range(len(reals)))
        ax.set_xticklabels(labs, fontsize=7.0, color=FIG["text2"])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        ax.tick_params(axis="both", labelsize=7.5, colors=FIG["muted"], length=0)
        ax.grid(True, axis="y", color=FIG["grid"], lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(FIG["grid"])
        ax.set_ylabel(f"configurations with this angle in the top {TOP_N}",
                      fontsize=8, color=FIG["text2"])
        ax.set_xlabel("push angle [deg]   real frame above, simulation below",
                      fontsize=8, color=FIG["text2"])
        ax.set_title(f"{QUANT_TITLE[q]}   ({n_conf} configuration(s))",
                     fontsize=10.5, color=FIG["text"], pad=6)
        ax.legend(fontsize=7.5, frameon=False, ncol=TOP_N,
                  loc="upper left", bbox_to_anchor=(0.0, 1.0))

    v, root, filt, backend = split_tag(tag)
    fig.suptitle(
        f"how often each push angle is in the best {TOP_N}   --   {side.upper()}"
        f"\n{tag}   ·   stacked by rank   ·   {MODE_LABEL[mode]}"
        f"\na bar below the dashed line is a direction this estimator avoids",
        fontsize=10.5, color=FIG["text"], y=0.995, va="bottom")
    fig.patch.set_facecolor(FIG["surface"])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FIG["surface"])
    plt.close(fig)
    return path


def plot_angles_simple(top, tag, path, side, mode="perquantity"):
    """Just the angles: which three, for every configuration, mass and mu.

    No colourbar, no size scale, no marginal counts -- the only thing encoded
    beyond position is the rank, printed as its own digit inside the mark. The
    rank-map figure beside this one carries the errors; this one answers
    "which directions?" and nothing else, which is usually the question.

    Rank is shaded dark-to-light 1..N as well as printed, so the ordering is
    readable at a glance and still exact when read closely.
    """
    reals, labs = _angle_axis()
    xpos = {round(a, 6): i for i, a in enumerate(reals)}
    doms = _domain_order(top)
    quants = [q for q, *_ in QUANTITY_KEYS if (top["quantity"] == q).any()]
    if not doms or not quants:
        return None

    nrow = len(doms)
    fig, axes = plt.subplots(
        1, len(quants), figsize=(6.6 * len(quants), max(3.0, 0.30 * nrow + 2.2)),
        squeeze=False, sharey=True)

    for c, q in enumerate(quants):
        ax = axes[0][c]
        ax.set_facecolor(FIG["surface"])
        cmap = plt.get_cmap(QUANT_CMAP[q])

        for r in range(nrow):
            ax.axhline(r, color=FIG["grid"], lw=0.6, zorder=0)
        for i in range(len(reals)):
            ax.axvline(i, color=FIG["grid"], lw=0.6, zorder=0)

        for _, row in top[top["quantity"] == q].iterrows():
            if row["domain"] not in doms:
                continue
            xi = xpos.get(round(float(row["angle_real_deg"]), 6))
            if xi is None:
                continue
            yi = doms.index(row["domain"])
            rank = int(row["rank"])
            shade = 1.0 - 0.55 * (rank - 1) / max(TOP_N - 1, 1)
            ax.scatter([xi], [yi], s=165, marker="o",
                       color=cmap(0.35 + 0.5 * shade),
                       edgecolors=QUANT_HUE[q], linewidths=0.8, zorder=3)
            ax.text(xi, yi, str(rank), ha="center", va="center",
                    fontsize=7.2, fontweight="bold", zorder=4,
                    color="#ffffff" if shade > 0.6 else FIG["text"])

        ax.set_xlim(-0.6, len(reals) - 0.4)
        ax.set_ylim(nrow - 0.5, -0.5)
        ax.set_xticks(range(len(reals)))
        ax.set_xticklabels(labs, fontsize=7.0, color=FIG["text2"])
        ax.set_yticks(range(nrow))
        # Set the labels on EVERY panel and hide them with tick_params instead
        # of passing [] to the right-hand one: these axes share a y axis, so
        # clearing its labels anywhere clears them everywhere, including the
        # panel that was meant to keep them.
        ax.set_yticklabels(doms, fontsize=6.8, color=FIG["text2"])
        ax.tick_params(axis="both", length=0, labelleft=(c == 0))
        for s in ax.spines.values():
            s.set_color(FIG["grid"])
        ax.set_title(f"{QUANT_TITLE[q]}", fontsize=11, color=FIG["text"], pad=6)
        ax.set_xlabel("push angle [deg]   real frame above, simulation below",
                      fontsize=8, color=FIG["text2"])

    v, root, filt, backend = split_tag(tag)
    fig.suptitle(
        f"best {TOP_N} push angles per configuration   --   {side.upper()}\n"
        f"{tag}   ·   the digit is the rank, 1 is best   ·   {MODE_LABEL[mode]}",
        fontsize=10.5, color=FIG["text"], y=0.995, va="bottom")
    fig.patch.set_facecolor(FIG["surface"])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FIG["surface"])
    plt.close(fig)
    return path


def plot_rank_map(top, tag, path, side, mode="perquantity"):
    """One figure per estimator: the top-N angles for every configuration."""
    reals, labs = _angle_axis()
    xpos = {round(a, 6): i for i, a in enumerate(reals)}
    doms = _domain_order(top)
    if not doms:
        return None
    quants = [q for q, *_ in QUANTITY_KEYS if (top["quantity"] == q).any()]
    if not quants:
        return None

    nrow, ncol = len(doms), len(quants)
    fig = plt.figure(figsize=(7.6 * ncol, max(4.2, 0.30 * nrow + 3.4)))
    gs = fig.add_gridspec(2, ncol, height_ratios=[max(nrow, 6), 3.0],
                          hspace=0.08, wspace=0.08)

    for c, q in enumerate(quants):
        sub = top[top["quantity"] == q]
        ax = fig.add_subplot(gs[0, c])
        ax.set_facecolor(FIG["surface"])

        # A recessive guide per configuration, so a row with one mark still
        # reads as a row rather than as a floating point.
        for r in range(nrow):
            ax.axhline(r, color=FIG["grid"], lw=0.6, zorder=0)
        for i in range(len(reals)):
            ax.axvline(i, color=FIG["grid"], lw=0.6, zorder=0)

        X, Y, S, C = [], [], [], []
        for _, row in sub.iterrows():
            if row["domain"] not in doms:
                continue
            xi = xpos.get(round(float(row["angle_real_deg"]), 6))
            if xi is None:
                continue
            X.append(xi); Y.append(doms.index(row["domain"]))
            S.append(_rank_size(int(row["rank"]), TOP_N))
            C.append(float(row["nrmse_pct"]))
        if not X:
            ax.axis("off")
            continue

        vmax = float(np.nanpercentile(C, 95)) if len(C) > 3 else float(max(C))
        vmax = max(vmax, 1e-6)
        # The ramp starts at 0.25 rather than 0: a near-white mark on a near-white
        # surface is invisible, and the best cells are exactly the ones worth
        # seeing.
        cmap = plt.get_cmap(QUANT_CMAP[q])
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{q}_sub", cmap(np.linspace(0.25, 0.95, 256)))
        sc = ax.scatter(X, Y, s=S, c=C, cmap=cmap, vmin=0.0, vmax=vmax,
                        edgecolors=QUANT_HUE[q], linewidths=0.8, zorder=3)

        # Only rank 1 is labelled. A number on every mark would be 81 numbers.
        for _, row in sub[sub["rank"] == 1].iterrows():
            xi = xpos.get(round(float(row["angle_real_deg"]), 6))
            if xi is None or row["domain"] not in doms:
                continue
            ax.text(xi + 0.26, doms.index(row["domain"]),
                    f"{row['nrmse_pct']:.0f}%", fontsize=6.0,
                    color=FIG["text2"], va="center", ha="left", zorder=4)

        ax.set_xlim(-0.6, len(reals) - 0.4)
        ax.set_ylim(nrow - 0.5, -0.5)
        ax.set_xticks(range(len(reals)))
        ax.set_xticklabels([])
        ax.set_yticks(range(nrow))
        # Only the leftmost panel carries them: the panels share rows, and a
        # repeat on the right collides with the previous panel's colorbar.
        ax.set_yticklabels(doms if c == 0 else [], fontsize=6.6,
                           color=FIG["text2"])
        ax.tick_params(axis="both", length=0)
        for s in ax.spines.values():
            s.set_color(FIG["grid"])
        ax.set_title(f"{QUANT_TITLE[q]}   --   best {TOP_N} push directions"
                     f"\nmark size = rank (1 is largest), colour = NRMSE,"
                     f" label = rank-1 NRMSE",
                     fontsize=10, color=FIG["text"], pad=8)

        cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.035)
        cb.set_label("NRMSE [%]", fontsize=8, color=FIG["text2"])
        cb.ax.tick_params(labelsize=7, colors=FIG["muted"], length=2)
        cb.outline.set_edgecolor(FIG["grid"])

        # ---- marginal: how often each angle made the cut -------------------
        axb = fig.add_subplot(gs[1, c])
        axb.set_facecolor(FIG["surface"])
        counts = np.zeros(len(reals), dtype=int)
        for x in X:
            counts[x] += 1
        axb.bar(range(len(reals)), counts, color=QUANT_HUE[q], alpha=0.85,
                width=0.62, zorder=3)
        for i, n in enumerate(counts):
            if n:
                axb.text(i, n, str(n), ha="center", va="bottom", fontsize=6.6,
                         color=FIG["text2"])
        axb.set_xlim(-0.6, len(reals) - 0.4)
        axb.set_ylim(0, max(counts.max() * 1.25, 1))
        axb.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
        axb.set_xticks(range(len(reals)))
        axb.set_xticklabels(labs, fontsize=6.8, color=FIG["text2"])
        axb.set_ylabel(f"in top {TOP_N}", fontsize=8, color=FIG["text2"])
        axb.tick_params(axis="both", labelsize=7, colors=FIG["muted"], length=0)
        axb.grid(True, axis="y", color=FIG["grid"], lw=0.6)
        axb.set_axisbelow(True)
        for s in axb.spines.values():
            s.set_color(FIG["grid"])
        axb.set_xlabel("push angle [deg]   real frame above, "
                       "simulation frame below", fontsize=8,
                       color=FIG["text2"])

    v, root, filt, backend = split_tag(tag)
    head = (f"{tag}\n{side.upper()}   ·   variant {v}   ·   backend {backend}"
            f"   ·   velocity {pretty_filter(filt) if side == 'real' else 'as trained'}"
            f"   ·   {nrow} configuration(s)   ·   {MODE_LABEL[mode]}")
    fig.suptitle(head, fontsize=11, color=FIG["text"], y=0.995, va="bottom")
    fig.patch.set_facecolor(FIG["surface"])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FIG["surface"])
    plt.close(fig)
    return path


def plot_summary(combined, path, side, mode="perquantity"):
    """Across estimators: which angles they favour, and how good rank 1 was."""
    reals, labs = _angle_axis()
    xpos = {round(a, 6): i for i, a in enumerate(reals)}
    tags = list(dict.fromkeys(combined["tag"]))
    quants = [q for q, *_ in QUANTITY_KEYS if (combined["quantity"] == q).any()]
    if not tags or not quants:
        return None

    fig = plt.figure(figsize=(7.8 * len(quants), 2.2 + 0.52 * len(tags) * 2))
    gs = fig.add_gridspec(2, len(quants), height_ratios=[1.35, 1.0],
                          hspace=0.55, wspace=0.3)

    for c, q in enumerate(quants):
        sub = combined[combined["quantity"] == q]

        # ---- counts, estimator x angle --------------------------------------
        M = np.zeros((len(tags), len(reals)), dtype=int)
        for _, r in sub.iterrows():
            xi = xpos.get(round(float(r["angle_real_deg"]), 6))
            if xi is not None:
                M[tags.index(r["tag"]), xi] += 1

        ax = fig.add_subplot(gs[0, c])
        cmap = plt.get_cmap(QUANT_CMAP[q])
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{q}_sub", cmap(np.linspace(0.12, 0.92, 256)))
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0)
        for i in range(len(tags)):
            for j in range(len(reals)):
                if M[i, j]:
                    # Ink, not the series colour: the cell behind it carries the
                    # identity, the number only has to be readable on it.
                    shade = M[i, j] / max(M.max(), 1)
                    ax.text(j, i, str(M[i, j]), ha="center", va="center",
                            fontsize=6.8,
                            color="#ffffff" if shade > 0.6 else FIG["text2"])
        ax.set_xticks(range(len(reals)))
        ax.set_xticklabels(labs, fontsize=6.8, color=FIG["text2"])
        ax.set_yticks(range(len(tags)))
        # Leftmost column only: the columns share rows, so a repeat on the right
        # lands on the previous panel's colorbar.
        ax.set_yticklabels([t[:46] for t in tags] if c == 0 else [],
                           fontsize=6.8, color=FIG["text2"])
        ax.tick_params(length=0)
        ax.set_title(f"{QUANT_TITLE[q]}   --   how often each angle is in the "
                     f"top {TOP_N}", fontsize=10, color=FIG["text"], pad=8)
        ax.set_xlabel("push angle [deg]   real frame above, simulation below",
                      fontsize=8, color=FIG["text2"])
        cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
        cb.set_label(f"configurations", fontsize=8, color=FIG["text2"])
        cb.ax.tick_params(labelsize=7, colors=FIG["muted"], length=2)
        cb.outline.set_edgecolor(FIG["grid"])

        # ---- how good was rank 1 --------------------------------------------
        ax2 = fig.add_subplot(gs[1, c])
        ax2.set_facecolor(FIG["surface"])
        r1 = sub[sub["rank"] == 1]
        med = [float(r1[r1["tag"] == t]["nrmse_pct"].median())
               if (r1["tag"] == t).any() else np.nan for t in tags]
        y = np.arange(len(tags))
        ax2.barh(y, med, color=QUANT_HUE[q], alpha=0.85, height=0.6, zorder=3)
        for i, m in enumerate(med):
            if np.isfinite(m):
                ax2.text(m, i, f" {m:.1f}%", va="center", fontsize=7,
                         color=FIG["text2"])
        ax2.set_yticks(y)
        ax2.set_yticklabels([t[:46] for t in tags] if c == 0 else [],
                            fontsize=6.8, color=FIG["text2"])
        ax2.invert_yaxis()
        ax2.set_xlabel(f"median rank-1 NRMSE over the configurations [%]",
                       fontsize=8, color=FIG["text2"])
        ax2.tick_params(axis="both", labelsize=7, colors=FIG["muted"], length=0)
        ax2.grid(True, axis="x", color=FIG["grid"], lw=0.6)
        ax2.set_axisbelow(True)
        for s in ax2.spines.values():
            s.set_color(FIG["grid"])
        ax2.set_title("the BEST angle's error, per estimator  "
                      "(a floor, not a typical result)",
                      fontsize=9, color=FIG["text"], pad=6)

    fig.suptitle(f"top-{TOP_N} summary across estimators   --   {side.upper()}"
                 f"\n{MODE_LABEL[mode]}",
                 fontsize=11, color=FIG["text"], y=0.995, va="bottom")
    fig.patch.set_facecolor(FIG["surface"])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FIG["surface"])
    plt.close(fig)
    return path


class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, t):
        self._stream.write(t); self._fh.write(t); self._fh.flush(); return len(t)

    def flush(self):
        self._stream.flush(); self._fh.flush()

    def __getattr__(self, n):
        return getattr(self._stream, n)


def split_tag(tag):
    """(variant, root, filter, backend) from a sim tag.

    The sim tags are shaped <variant>_<root>_transformer_sim, so the trailing
    "_sim" is stripped first and the rest parses exactly as the real side's does.
    """
    base = tag[:-4] if tag.endswith("_sim") else tag
    backend = "other"
    for b in ("transformer", "diffusion"):
        if base.endswith("_" + b):
            base, backend = base[: -len(b) - 1], b
            break
    filt = ""
    m = re.search(r"_vsm_(?P<f>.+)$", base)
    if m:
        filt, base = m.group("f"), base[: m.start()]
    root = ""
    m = re.match(r"^(?P<v>.+?)_(?P<r>from_.+)$", base)
    if m:
        base, root = m.group("v"), m.group("r")
    return base, root, filt, backend


def pretty_filter(f):
    """Only the real side carries velocity filters, but the shared figure
    header calls this on both, so the sim side answers for its own tags."""
    return "unsmoothed" if not f or f == "none" else f


def display_domain(d):
    m = DOMAIN_RE.match(str(d))
    if not m:
        return str(d), ""
    return (OBJECT_DISPLAY.get(m.group("obj").lower(), m.group("obj").title()),
            SURFACE_DISPLAY.get(m.group("surf").lower(), m.group("surf").title()))


def rank_top(df, tag, n, mode):
    """The n best angles per configuration, under one of two ranking rules.

    mode="perquantity"
        Mass ranked by mass NRMSE and mu by mu NRMSE, independently, so the two
        quantities generally name DIFFERENT angles.

    mode="mean"
        One ranking per configuration on the mean of the two NRMSEs, so a single
        set of angles serves both -- the rule sim_estimator_polar_top3.csv uses.
        An angle can make the cut on the strength of one quantity while the
        other is far off.

    Both write the value they sorted on into rank_score_pct, so a row says what
    put it where without the reader having to know which file they opened.
    """
    v, root, filt, backend = split_tag(tag)
    out, obj_of = [], {}

    def row(r, dom, q, est_c, gt_c, nr_c, n_c, rank, score):
        a_sim = float(r["push_angle_deg"])
        a_real = -a_sim if ANGLE_SIGN_FLIP else a_sim
        if dom not in obj_of:
            obj_of[dom] = display_domain(dom)
        obj, surf = obj_of[dom]
        return dict(
            side="sim", tag=tag, variant=v, root=root,
            filter=filt or "none",
            filter_label="unsmoothed" if not filt else filt,
            backend=backend, rank_mode=mode,
            domain=str(dom), object=obj, surface=surf,
            quantity=q, rank=rank,
            gt=round(float(r[gt_c]), 6), est=round(float(r[est_c]), 6),
            err=round(float(r[est_c]) - float(r[gt_c]), 6),
            nrmse_pct=round(float(r[nr_c]), 3),
            rank_score_pct=round(float(score), 3),
            n=int(r[n_c]) if n_c in df.columns and np.isfinite(r[n_c]) else -1,
            push_angle_deg=a_sim,
            angle_real_deg=a_real, angle_sim_deg=a_sim)

    if mode == "perquantity":
        for q, est_c, gt_c, nr_c, n_c, _dec in QUANTITIES:
            if nr_c not in df.columns:
                continue
            sub = df[np.isfinite(df[nr_c].astype(float))]
            for dom, g in sub.groupby("domain"):
                g = g.sort_values(nr_c, kind="mergesort")
                for rank, (_, r) in enumerate(g.head(n).iterrows(), start=1):
                    out.append(row(r, dom, q, est_c, gt_c, nr_c, n_c, rank,
                                   r[nr_c]))
        return pd.DataFrame(out)

    # mode == "mean": score the ANGLE on both quantities, then report both at it.
    cols = [(q, e, g_, nr, nc) for q, e, g_, nr, nc, _d in QUANTITIES
            if nr in df.columns]
    if not cols:
        return pd.DataFrame(out)
    nr_cols = [c[3] for c in cols]
    for dom, g in df.groupby("domain"):
        g = g.copy()
        g["_score"] = g[nr_cols].astype(float).mean(axis=1)
        g = g[np.isfinite(g["_score"])].sort_values("_score", kind="mergesort")
        for rank, (_, r) in enumerate(g.head(n).iterrows(), start=1):
            for q, est_c, gt_c, nr_c, n_c in cols:
                out.append(row(r, dom, q, est_c, gt_c, nr_c, n_c, rank,
                               r["_score"]))
    return pd.DataFrame(out)


def check_grid_symmetry():
    g = set(round(a, 6) for a in PUSH_ANGLES_DEG)
    flipped = set(round(-a, 6) for a in PUSH_ANGLES_DEG)
    return g == flipped, sorted(g - flipped)


def main():
    if not os.path.exists(PER_ANGLE_CSV):
        raise SystemExit(
            f"per-angle table not found: {PER_ANGLE_CSV}\n"
            f"  Run plot_estimator_polar_sim.py first -- this script reads its\n"
            f"  output rather than loading the checkpoints a second time, so the\n"
            f"  top-3 file and the polar figures cannot disagree about one run.\n"
            f"  Point PER_ANGLE_CSV at it if it lives elsewhere.")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(PER_ANGLE_CSV)
    need = ["tag", "domain", "push_angle_deg"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"{PER_ANGLE_CSV} is missing {miss}; it does not look "
                         f"like plot_estimator_polar_sim.py's output.")

    print("=" * 92)
    print(f" TOP-{TOP_N} PUSH DIRECTIONS PER OBJECT  --  SIMULATION")
    print("=" * 92)
    print(f"  per-angle: {PER_ANGLE_CSV}")
    print(f"  rows     : {len(df)}   tags: {df['tag'].nunique()}   "
          f"configurations: {df['domain'].nunique()}")
    for col, label in (("heldout", "held-out"), ("mass_tol", "mass tol"),
                       ("mu_tol", "mu tol"), ("m_range", "m range"),
                       ("mu_range", "mu range")):
        if col in df.columns and df[col].notna().any():
            print(f"  {label:<9}: {df[col].iloc[0]}")

    if VARIANTS.strip():
        want = set(VARIANTS.split())
        before = df["tag"].nunique()
        df = df[df["variant"].isin(want)] if "variant" in df.columns else df
        print(f"  VARIANTS : keeping {df['tag'].nunique()} of {before} tag(s)")
    if DOMAINS.strip():
        want = DOMAINS.split()
        bad = [d for d in want if d not in set(df["domain"].astype(str))]
        if bad:
            raise SystemExit(f"configuration(s) not in the per-angle table: {bad}")
        df = df[df["domain"].astype(str).isin(want)]
    if df.empty:
        raise SystemExit("nothing left after filtering")

    print(f"  ranking  : per quantity, by that quantity's own NRMSE")
    print(f"             (NOT the mean-of-both score "
          f"sim_estimator_polar_top3.csv uses, so the two will differ)")

    ok_sym, odd = check_grid_symmetry()
    if ANGLE_SIGN_FLIP:
        print(f"  convention: real = -sim, so angle_real_deg = -push_angle_deg")
        if ok_sym:
            print(f"             the grid is symmetric about 0, so every angle has "
                  f"a counterpart in the other frame")
        else:
            print(f"  [WARNING] the grid is NOT symmetric about 0 ({odd} have no "
                  f"mirror), so some rows have no counterpart to join against.")
    else:
        print(f"  convention: ANGLE_SIGN_FLIP=0, angles are used as-is on both "
              f"sides")

    print("\n  written:")
    per_mode = {}
    for mode in MODES:
        sfx = MODE_SUFFIX[mode]
        all_rows = []
        for tag, g in df.groupby("tag"):
            top = rank_top(g, tag, TOP_N, mode)
            if top.empty:
                print(f"  [skip] {tag}: no usable cell")
                continue
            top = top.sort_values(["domain", "quantity", "rank"],
                                  kind="mergesort").reset_index(drop=True)
            out = os.path.join(OUT_DIR, f"top{TOP_N}_sim_{sfx}_{tag}.csv")
            top.to_csv(out, index=False)
            print(f"  {out}")
            if FIGS != "off":
                png = os.path.join(OUT_DIR,
                                   f"top{TOP_N}_hist_sim_{sfx}_{tag}.png")
                if plot_angle_hist(top, tag, png, "sim", mode):
                    print(f"  {png}")
            if FIGS != "off" and FIG_STYLE in ("simple", "both"):
                png = os.path.join(OUT_DIR,
                                   f"top{TOP_N}_angles_sim_{sfx}_{tag}.png")
                if plot_angles_simple(top, tag, png, "sim", mode):
                    print(f"  {png}")
            if FIGS != "off" and FIG_STYLE in ("detailed", "both"):
                png = os.path.join(OUT_DIR, f"top{TOP_N}_sim_{sfx}_{tag}.png")
                if plot_rank_map(top, tag, png, "sim", mode):
                    print(f"  {png}")
            all_rows.append(top)

        if not all_rows:
            raise SystemExit(f"[{mode}] no tag produced rows")
        combined = pd.concat(all_rows, ignore_index=True)
        comb_path = os.path.join(OUT_DIR, f"top{TOP_N}_sim_{sfx}_all.csv")
        combined.to_csv(comb_path, index=False)
        print(f"  {comb_path}")

        simple_path = os.path.join(OUT_DIR, f"top{TOP_N}_angles_sim_{sfx}.csv")
        simple = write_simple_angles(combined, simple_path, "sim", mode)
        print(f"  {simple_path}")
        if FIGS != "off" and FIG_STYLE in ("detailed", "both"):
            sp = os.path.join(OUT_DIR, f"top{TOP_N}_sim_{sfx}_summary.png")
            if plot_summary(combined, sp, "sim", mode):
                print(f"  {sp}")
        per_mode[mode] = (combined, simple)

    # Both rankings in one file, so they can be compared without a join.
    if len(per_mode) > 1:
        both = pd.concat([c for c, _s in per_mode.values()], ignore_index=True)
        bp = os.path.join(OUT_DIR, f"top{TOP_N}_sim_bothmodes_all.csv")
        both.to_csv(bp, index=False)
        print(f"  {bp}")

    for mode, (combined, simple) in per_mode.items():
        print_mode_tables(combined, simple, "sim", mode)

    print("\n" + "=" * 92)
    print(" HOW TO JOIN THIS WITH THE REAL FILE")
    print("=" * 92)
    print("  Both sides emit the same columns. For the SAME GEOMETRY:")
    print("      pd.merge(real, sim, on=['domain', 'quantity', 'angle_real_deg'],")
    print("               suffixes=('_real', '_sim'))")
    print("  For WHICH ANGLE EACH SIDE LIKED BEST, which is a different question:")
    print("      pd.merge(real, sim, on=['domain', 'quantity', 'rank'], ...)")
    print("\n  Before reading a gap as sim-to-real: the simulation object is a CUBE")
    print("  at the real object's mass and friction, not its shape, and each sim")
    print("  cell averages many pushes where the robot had one. A difference here")
    print("  mixes all three.")
    print("=" * 92)


if __name__ == "__main__":
    if REPORT == "off":
        main()
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        rp = os.path.join(OUT_DIR, f"top{TOP_N}_sim_report.txt")
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