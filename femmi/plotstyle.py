"""
femmi/plotstyle.py
One professional, paper-ready Matplotlib style for every FEMMI figure: a white
background, accessible sans-serif fonts, and colorblind-safe, print-friendly
colormaps. Call `use_paper_style()` once at the top of a plotting script.

    from femmi.plotstyle import use_paper_style, SEQ_CMAP, DIV_CMAP, PALETTE
    use_paper_style()

Colormaps:
  SEQ_CMAP = 'viridis'  -- sequential (mass maps): perceptually uniform,
                           colorblind-safe, and legible in greyscale print.
  DIV_CMAP = 'RdBu_r'   -- diverging (residuals, E/B): zero-centred.
PALETTE is the Wong (2011) colorblind-safe qualitative set for line plots.
"""

from __future__ import annotations

SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"

# Wong 2011 colorblind-safe qualitative palette.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

_RC = {
    # white, clean background
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    # accessible, consistent fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "dejavusans",
    # restrained, publication-style axes
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
    "image.cmap": SEQ_CMAP,
}


def use_paper_style():
    """Apply the FEMMI paper style to the global Matplotlib rcParams and set the
    default color cycle to the colorblind-safe palette. Idempotent."""
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update(_RC)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
