"""Regenerate Picture1: per-space-group MAE (3 models) over test-set material counts.

Two panels sharing one x axis (space group 1-230):
  top    : Crystal-X (red) vs ALIGNN (green)
  bottom : Crystal-X (red) vs CGCNN  (blue)
Gray bars = number of materials per space group in the full 6,848-crystal test set.
Both vertical axes are log, and both panels use identical scales so they can be
compared directly. Crystal-system brackets are drawn under the bottom panel only.
"""
import csv
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

BASE = Path("/ibex/user/shoaibsa/Crystal-X/baselines/training")
OUT_PNG = BASE / "results/picture1_spacegroup_mae_new.png"
OUT_PDF = BASE / "results/picture1_spacegroup_mae_new.pdf"

RED, BLUE, GREEN = "#d62728", "#1f5fbf", "#0b8a2f"
BAR = "#8f8f89"
INK, INK2, GRID, AXIS = "#0b0b0b", "#4a4a48", "#e3e3de", "#b9b9b3"

SYSTEMS = [(1, 2, "Triclinic"), (3, 15, "Monoclinic"), (16, 74, "Orthorhombic"),
           (75, 142, "Tetragonal"), (143, 167, "Trigonal"), (168, 194, "Hexagonal"),
           (195, 230, "Cubic")]

MAE_LIM = (1.6e-3, 3.2)     # identical on both panels; spans the full data range
CNT_LIM = (1, 1200)

# ---------------------------------------------------------------- data
sg = {r[0]: int(r[1]) for r in list(csv.reader(open(BASE / "data/mp2018_test_spacegroups.csv")))[1:]}
FILES = {
    "Crystal-X": BASE / "results/crystalx_bandgap/crystalx_mp2018_bandgap_test_results.csv",
    "CGCNN": BASE / "results/cgcnn_bandgap/mp2018_bandgap_test_results.csv",
    "ALIGNN": BASE / "results/alignn_bandgap/prediction_results_test_set.csv",
}
err = {}
for k, f in FILES.items():
    err[k] = {r[0].strip(): abs(float(r[1]) - float(r[2])) for r in list(csv.reader(open(f)))[1:]}

counts = np.zeros(231)
for g in sg.values():
    counts[g] += 1

mae = {}
for k in FILES:
    per = collections.defaultdict(list)
    for mid, g in sg.items():
        if mid in err[k]:
            per[g].append(err[k][mid])
    arr = np.full(231, np.nan)
    for g, v in per.items():
        arr[g] = sum(v) / len(v)
    mae[k] = arr

x = np.arange(231)
overall = {k: sum(err[k].values()) / len(err[k]) for k in FILES}
print("overall test MAE:", {k: round(v, 4) for k, v in overall.items()})

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, dpi=300)
fig.patch.set_facecolor("white")

PANELS = [("Crystal-X", "ALIGNN", RED, GREEN), ("Crystal-X", "CGCNN", RED, BLUE)]

for ax, (m1, m2, c1, c2) in zip(axes, PANELS):
    ax.set_facecolor("white")

    # counts on the right axis (log)
    axc = ax.twinx()
    # Crystal-system bands go on THIS axis, beneath the bars. They must not live on
    # `ax`: ax sits above axc (so the MAE lines draw over the bars), and an opaque
    # axvspan on ax would hide every bar inside the shaded systems.
    for i, (lo, hi, _) in enumerate(SYSTEMS):
        if i % 2 == 0:
            axc.axvspan(lo - 0.5, hi + 0.5, color="#f4f4f1", lw=0, zorder=0)
    axc.bar(x, counts, width=0.9, color=BAR, lw=0, zorder=2, alpha=1.0)
    axc.set_yscale("log")
    axc.set_ylim(*CNT_LIM)
    axc.set_ylabel("Number of materials", color=INK2, fontsize=15)
    axc.tick_params(colors=INK2, labelsize=13)
    axc.spines["top"].set_visible(False)
    for s in ("left", "right", "bottom"):
        axc.spines[s].set_color(AXIS)

    # MAE lines on the left axis (log). Plot only the occupied space groups so the
    # line runs unbroken across empty ones instead of fragmenting at every gap.
    for m, c, mk in [(m1, c1, "o"), (m2, c2, "^")]:
        occ = ~np.isnan(mae[m])
        ax.plot(x[occ], mae[m][occ], color=c, lw=1.1, alpha=0.85, zorder=3,
                marker=mk, ms=3.6, mew=0, label=m)
    ax.set_yscale("log")
    ax.set_ylim(*MAE_LIM)
    ax.set_zorder(axc.get_zorder() + 1)   # lines above bars
    ax.patch.set_visible(False)
    ax.set_ylabel("Average MAE (eV)", color=INK, fontsize=15)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=13)
    ax.spines["top"].set_visible(False)
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_color(AXIS)

axes[0].set_xlim(0, 231)

# One shared legend above the panels: an in-panel box would sit on top of the tallest
# count bars (space groups 1-2, the largest in the set) and hide them. Red appears in
# both panels, green only in the top, blue only in the bottom, so this stays unambiguous.
fig.legend(
    handles=[
        Line2D([], [], color=RED, marker="o", ms=5, lw=1.4, label="Crystal-X"),
        Line2D([], [], color=GREEN, marker="^", ms=5, lw=1.4, label="ALIGNN  (top panel)"),
        Line2D([], [], color=BLUE, marker="^", ms=5, lw=1.4, label="CGCNN  (bottom panel)"),
        plt.Rectangle((0, 0), 1, 1, fc=BAR, ec="none", label="Number of materials (test set)"),
    ],
    loc="upper center", bbox_to_anchor=(0.5, 0.963), ncol=4, frameon=False,
    fontsize=11, labelcolor=INK, handletextpad=0.6, columnspacing=2.2,
)
axes[1].set_xticks([1, 15, 50, 74, 100, 142, 167, 194, 230])
axes[1].set_xlabel("Space Groups", color=INK, fontsize=19, labelpad=64)

# ---- crystal-system brackets under the bottom panel only ----
ax = axes[1]
tr = blended_transform_factory(ax.transData, ax.transAxes)
# All brackets sit on one tier. Triclinic (2 groups) and Monoclinic (13) are far
# narrower than their names, so those two labels are set vertically under their own
# bracket -- keeps every label inside its range and off the left edge.
Y = -0.155
for lo, hi, name in SYSTEMS:
    ax.plot([lo, hi], [Y, Y], transform=tr, color="#4a6b7c", lw=1.1, clip_on=False)
    for xe in (lo, hi):
        ax.plot([xe, xe], [Y, Y + 0.018], transform=tr, color="#4a6b7c", lw=1.1, clip_on=False)
    rot = 90 if (hi - lo) < 20 else 0
    ax.text((lo + hi) / 2, Y - 0.03, name, transform=tr, ha="center", va="top",
            rotation=rot, fontsize=13, color=INK, clip_on=False)

fig.suptitle("Average MAE and number of materials per space group — MP 2018 band gap "
             "(6,848-crystal test set)", x=0.5, y=0.998, fontsize=18, color=INK)
fig.tight_layout(rect=[0, 0.0, 1, 0.945])
fig.subplots_adjust(hspace=0.07)
fig.savefig(OUT_PNG, facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_PDF, facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT_PNG)
print("wrote", OUT_PDF)
