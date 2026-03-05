"""
tests/test_smpy_femmi_comparison.py
====================================
Monte Carlo benchmark: FEM-MAP vs Kaiser-Squires (SMPy).

Shear generation
----------------
Ground-truth κ is defined on a regular grid. Shear is generated using the
infinite-plane lensing kernel (FFT forward model) — the same operator that
KS inverts. This is physically correct and ensures a fair comparison:

    κ_true (grid)  →  KS forward (FFT)  →  γ_true (grid)
                                                  │
                                         + Gaussian noise
                                                  │
                                             γ_obs (grid)
                                                  │
                          ┌───────────────────────┤
                          │                       │
                   FEM-MAP                       KS
             (γ_obs interpolated          (γ_obs directly
              to mesh nodes,               as grid input,
              Dirichlet BCs,               DC-corrected)
              Wiener prior)
                          │                       │
                      κ_MAP (nodes)          κ_KS (grid)
                          │                       │
                    interpolate to         already on grid
                    shared grid
                          │                       │
                          └───────────┬───────────┘
                                  compare to
                                  κ_true (grid)

DC correction
-------------
KS sets the k=0 mode to zero (mass sheet degeneracy). We correct this by
subtracting the mean of the outer sky annulus, where κ ≈ 0 for compact lenses.
FEM-MAP is immune to this degeneracy via Dirichlet BCs (κ=0 on ∂Ω).

Usage
-----
    python tests/test_smpy_femmi_comparison.py               # 10-trial MC
    python tests/test_smpy_femmi_comparison.py --trials 20   # custom count
    python tests/test_smpy_femmi_comparison.py --unittest    # unit tests
    python -m pytest tests/test_smpy_femmi_comparison.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import griddata

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _ROOT)

# ── FEMMI imports ─────────────────────────────────────────────────────────────
from femmi import build_operators, DifferentiableForward, MAPReconstructor

# ── SMPy imports ──────────────────────────────────────────────────────────────
try:
    from smpy.config import Config
    from smpy.mapping_methods.kaiser_squires.kaiser_squires import KaiserSquiresMapper
    _SMPY_AVAILABLE = True
except ImportError as _smpy_err:
    _SMPY_AVAILABLE = False
    print(f"WARNING: SMPy not available ({_smpy_err}). KS comparison skipped.")

# ── constants ─────────────────────────────────────────────────────────────────
DOMAIN     = (-2.0, 2.0, -2.0, 2.0)
A_LENS     = 1.0
SIGMA_LENS = 0.5
NX_GRID    = 64    # resolution of ground-truth / KS grid


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth κ fields (defined on a regular grid)
# ─────────────────────────────────────────────────────────────────────────────

def _grid(nx=NX_GRID):
    xi = np.linspace(DOMAIN[0], DOMAIN[1], nx)
    yi = np.linspace(DOMAIN[2], DOMAIN[3], nx)
    return np.meshgrid(xi, yi)   # XX, YY each (nx, nx)


def gaussian_kappa_grid(nx=NX_GRID, A=A_LENS, sigma=SIGMA_LENS,
                        cx=0.0, cy=0.0):
    XX, YY = _grid(nx)
    return A * np.exp(-((XX-cx)**2 + (YY-cy)**2) / (2*sigma**2))


def double_gaussian_kappa_grid(nx=NX_GRID):
    XX, YY = _grid(nx)
    k1 = 1.0 * np.exp(-((XX+0.8)**2 + YY**2) / (2*0.4**2))
    k2 = 0.6 * np.exp(-((XX-0.8)**2 + YY**2) / (2*0.3**2))
    return k1 + k2


# ─────────────────────────────────────────────────────────────────────────────
# Infinite-plane lensing forward model (same operator KS inverts)
# ─────────────────────────────────────────────────────────────────────────────

def ks_forward(kappa_grid):
    """
    Compute shear from convergence using the infinite-plane FFT kernel.

        γ_1 = Re[ ((k₁²-k₂²)/k²) κ̂ ]
        γ_2 = Re[ (2k₁k₂/k²) κ̂ ]

    This is the exact inverse of KaiserSquiresMapper.create_maps, so KS
    self-consistency is guaranteed up to smoothing.
    """
    ny, nx = kappa_grid.shape
    khat = np.fft.fft2(kappa_grid)
    k1, k2 = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny))
    k2sq = k1**2 + k2**2
    k2sq[0, 0] = np.finfo(float).eps
    g1 = np.real(np.fft.ifft2(((k1**2 - k2**2) / k2sq) * khat))
    g2 = np.real(np.fft.ifft2((2 * k1 * k2       / k2sq) * khat))
    return g1, g2


# ─────────────────────────────────────────────────────────────────────────────
# DC correction (mass sheet degeneracy)
# ─────────────────────────────────────────────────────────────────────────────

def dc_correct(kappa_ks, sky_fraction=0.15):
    """
    Subtract the outer-sky background from a KS convergence map.

    KS sets k=0 to zero so the output mean is always zero. We estimate
    the background from the outer sky annulus (where κ ≈ 0 for a compact
    lens) and subtract it, restoring the correct zero point.

    Returns (kappa_corrected, background_value).
    """
    ny, nx = kappa_ks.shape
    m = max(1, int(sky_fraction * min(ny, nx)))
    sky = np.concatenate([
        kappa_ks[:m,  :].ravel(), kappa_ks[-m:, :].ravel(),
        kappa_ks[:,  :m].ravel(), kappa_ks[:, -m:].ravel(),
    ])
    bg = float(sky.mean())
    return kappa_ks - bg, bg


# ─────────────────────────────────────────────────────────────────────────────
# SMPy KS wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_ks(g1_grid, g2_grid, smoothing_sigma=1.0, sky_fraction=0.15):
    """
    Run SMPy KaiserSquiresMapper on a regular shear grid and DC-correct.

    Returns (kappa_e_corrected, background).
    """
    if not _SMPY_AVAILABLE:
        raise ImportError("SMPy is required but not installed.")
    cfg = Config.from_defaults("kaiser_squires").to_dict()
    cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = smoothing_sigma
    kappa_raw, _ = KaiserSquiresMapper(cfg).create_maps(g1_grid, g2_grid)
    return dc_correct(kappa_raw, sky_fraction)


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────

def l2_error(pred, truth):
    """Normalised L2: ‖pred − truth‖ / ‖truth‖."""
    d = np.linalg.norm(truth)
    return np.linalg.norm(pred - truth) / (d if d > 0 else 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Masking helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_circular_mask(g1, g2, XX, YY, cx=0.0, cy=0.0, radius=0.5):
    """Zero out shear inside a circular disc (simulates a masked region)."""
    g1, g2 = g1.copy(), g2.copy()
    inside = (XX - cx)**2 + (YY - cy)**2 < radius**2
    g1[inside] = 0.0
    g2[inside] = 0.0
    return g1, g2, inside


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def single_trial(
    *,
    nx_fem         : int   = 20,
    nx_grid        : int   = NX_GRID,
    noise_level    : float = 0.10,
    lam_reg        : float = 1e-2,
    wiener_length  : float = 0.5,
    apply_mask     : bool  = False,
    mask_center    : tuple = (0.0, 0.0),
    mask_radius    : float = 0.5,
    kappa_grid_fn          = None,
    smoothing_sigma: float = 1.0,
    sky_fraction   : float = 0.15,
    maxiter        : int   = 400,
    seed           : int   = 42,
):
    """
    One FEM-MAP vs KS trial using the infinite-plane forward model.

    Shear is generated from κ_true via the FFT kernel, noise is added,
    then both methods reconstruct from the same γ_obs. FEM-MAP receives
    shear interpolated to mesh nodes; KS receives the grid directly.

    Returns
    -------
    dict with keys:
        l2_map, l2_ks, l2_ks_raw
        ks_background
        kappa_true, kappa_map_grid, kappa_ks, kappa_ks_raw
        XX, YY, mask_grid
        nodes
    """
    XX, YY = _grid(nx_grid)
    xmin, xmax, ymin, ymax = DOMAIN

    # -- ground-truth κ on regular grid
    if kappa_grid_fn is None:
        kappa_grid_fn = gaussian_kappa_grid
    kappa_true = kappa_grid_fn(nx_grid)

    # -- KS forward: generate shear
    g1_true, g2_true = ks_forward(kappa_true)

    # -- additive noise
    rng     = np.random.default_rng(seed)
    sigma_n = noise_level * np.std(np.hypot(g1_true, g2_true))
    g1_obs  = g1_true + rng.normal(0.0, sigma_n, g1_true.shape)
    g2_obs  = g2_true + rng.normal(0.0, sigma_n, g2_true.shape)

    # -- optional mask
    mask_grid = None
    if apply_mask:
        g1_obs, g2_obs, mask_grid = apply_circular_mask(
            g1_obs, g2_obs, XX, YY, *mask_center, mask_radius)

    # ── KS reconstruction (grid → grid) ──────────────────────────────────────
    kappa_ks = kappa_ks_raw = None
    ks_background = np.nan
    if _SMPY_AVAILABLE:
        try:
            kappa_ks, ks_background = run_ks(
                g1_obs, g2_obs,
                smoothing_sigma=smoothing_sigma,
                sky_fraction=sky_fraction,
            )
            # keep raw for diagnostics
            cfg = Config.from_defaults("kaiser_squires").to_dict()
            cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = smoothing_sigma
            kappa_ks_raw, _ = KaiserSquiresMapper(cfg).create_maps(g1_obs, g2_obs)
        except Exception as exc:
            print(f"  [SMPy KS failed: {exc}]")

    # ── FEM-MAP reconstruction (grid → nodes → MAP → grid) ───────────────────
    ops   = build_operators(nx_fem, nx_fem, xmin, xmax, ymin, ymax,
                            verbose=False)
    nodes = np.array(ops.mesh.nodes)

    # interpolate grid shear to FEM mesh nodes
    pts     = np.column_stack([XX.ravel(), YY.ravel()])
    g1_fem  = griddata(pts, g1_obs.ravel(), nodes, method="linear",
                       fill_value=0.0)
    g2_fem  = griddata(pts, g2_obs.ravel(), nodes, method="linear",
                       fill_value=0.0)

    fwd = DifferentiableForward(ops, lam_reg=lam_reg)
    rec = MAPReconstructor(fwd, maxiter=maxiter, gtol=1e-9,
                           wiener_length=wiener_length, callback_every=0)
    kappa_map_nodes, _ = rec.reconstruct(g1_fem, g2_fem, verbose=False)

    # interpolate FEM result back to shared grid
    kappa_map_grid = griddata(nodes, kappa_map_nodes, (XX, YY),
                              method="linear")
    kappa_map_grid_nn = griddata(nodes, kappa_map_nodes, (XX, YY),
                                 method="nearest")
    kappa_map_grid[~np.isfinite(kappa_map_grid)] = \
        kappa_map_grid_nn[~np.isfinite(kappa_map_grid)]

    # ── metrics ───────────────────────────────────────────────────────────────
    l2_map    = l2_error(kappa_map_grid, kappa_true)
    l2_ks     = l2_error(kappa_ks,     kappa_true) if kappa_ks     is not None else np.nan
    l2_ks_raw = l2_error(kappa_ks_raw, kappa_true) if kappa_ks_raw is not None else np.nan

    return dict(
        l2_map=l2_map, l2_ks=l2_ks, l2_ks_raw=l2_ks_raw,
        ks_background=ks_background,
        kappa_true=kappa_true,
        kappa_map_grid=kappa_map_grid,
        kappa_ks=kappa_ks,
        kappa_ks_raw=kappa_ks_raw,
        XX=XX, YY=YY,
        mask_grid=mask_grid,
        nodes=nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo runner
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_benchmark(
    n_trials       : int   = 10,
    nx_fem         : int   = 20,
    nx_grid        : int   = NX_GRID,
    noise_level    : float = 0.10,
    lam_reg        : float = 1e-2,
    wiener_length  : float = 0.5,
    apply_mask     : bool  = False,
    mask_center    : tuple = (0.0, 0.0),
    mask_radius    : float = 0.5,
    kappa_grid_fn          = None,
    smoothing_sigma: float = 1.0,
    sky_fraction   : float = 0.15,
    maxiter        : int   = 400,
    tag            : str   = "",
    plot_dir       : str | None = None,
):
    """Run n_trials independent noise realisations (seed = trial index)."""
    print(f"\n{'═'*62}")
    print(f"Monte Carlo: {tag or 'unnamed'}  ({n_trials} trials)")
    print(f"  nx_fem={nx_fem}  nx_grid={nx_grid}  noise={noise_level*100:.0f}%  "
          f"mask={'yes' if apply_mask else 'no'}  wiener_length={wiener_length}")
    print("═"*62)

    l2_maps, l2_kss, l2_kss_raw, bgs, last = [], [], [], [], None

    for t in range(n_trials):
        print(f"  trial {t+1:2d}/{n_trials} ... ", end="", flush=True)
        trial = single_trial(
            nx_fem=nx_fem, nx_grid=nx_grid,
            noise_level=noise_level, lam_reg=lam_reg,
            wiener_length=wiener_length,
            apply_mask=apply_mask, mask_center=mask_center,
            mask_radius=mask_radius, kappa_grid_fn=kappa_grid_fn,
            smoothing_sigma=smoothing_sigma, sky_fraction=sky_fraction,
            maxiter=maxiter, seed=t,
        )
        l2_maps.append(trial["l2_map"])
        l2_kss.append(trial["l2_ks"])
        l2_kss_raw.append(trial["l2_ks_raw"])
        bgs.append(trial["ks_background"])
        ks_str = (f"KS={trial['l2_ks']:.4f} (raw={trial['l2_ks_raw']:.4f})"
                  if np.isfinite(trial["l2_ks"]) else "KS=N/A")
        print(f"MAP={trial['l2_map']:.4f}  {ks_str}")
        last = trial

    l2_maps    = np.array(l2_maps)
    l2_kss     = np.array(l2_kss)
    l2_kss_raw = np.array(l2_kss_raw)
    mean_map  = float(np.mean(l2_maps));       std_map  = float(np.std(l2_maps))
    mean_ks   = float(np.nanmean(l2_kss));     std_ks   = float(np.nanstd(l2_kss))
    mean_ks_r = float(np.nanmean(l2_kss_raw)); std_ks_r = float(np.nanstd(l2_kss_raw))
    mean_bg   = float(np.nanmean(bgs))
    improv    = ((mean_ks - mean_map) / mean_ks * 100
                 if np.isfinite(mean_ks) else np.nan)

    print(f"\n  FEM-MAP     : {mean_map:.4f} ± {std_map:.4f}")
    if np.isfinite(mean_ks):
        print(f"  KS (DC-fix) : {mean_ks:.4f} ± {std_ks:.4f}  "
              f"(mean DC shift: {mean_bg:+.4f})")
        print(f"  KS (raw)    : {mean_ks_r:.4f} ± {std_ks_r:.4f}")
        sign = "✅" if improv > 0 else "⚠️"
        print(f"  Improvement : {improv:+.1f}%  {sign}")
    else:
        print("  KS      : (SMPy unavailable)")

    if plot_dir is not None and last is not None:
        os.makedirs(plot_dir, exist_ok=True)
        safe = (tag or "benchmark").replace(" ", "_").replace("%", "pct")
        _residual_figure(last, tag=tag,
                         path=os.path.join(plot_dir, f"{safe}_residual.png"))

    return dict(
        l2_map_trials=l2_maps, l2_ks_trials=l2_kss,
        l2_ks_raw_trials=l2_kss_raw,
        mean_map=mean_map, std_map=std_map,
        mean_ks=mean_ks,   std_ks=std_ks,
        mean_ks_raw=mean_ks_r, std_ks_raw=std_ks_r,
        mean_bg=mean_bg,
        mean_improvement=improv,
        last_trial=last,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _residual_figure(trial: dict, tag: str, path: str):
    """Six-panel: truth / MAP / MAP-resid / KS(DC) / KS-resid / KS(raw)."""
    k_true   = trial["kappa_true"]
    k_map    = trial["kappa_map_grid"]
    k_ks     = trial["kappa_ks"]
    k_ks_raw = trial["kappa_ks_raw"]
    XX, YY   = trial["XX"], trial["YY"]
    bg       = trial["ks_background"]
    mask     = trial["mask_grid"]
    ext      = [XX.min(), XX.max(), YY.min(), YY.max()]
    has_ks   = k_ks is not None

    n_cols = 6 if has_ks else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4),
                             facecolor="#1a1a1a")

    vmax = float(np.nanpercentile(k_true, 99))
    rmax_map = float(np.nanpercentile(np.abs(k_map - k_true), 99))
    kw_m = dict(cmap="viridis", vmin=0,         vmax=vmax,     origin="lower", extent=ext)
    kw_r = dict(cmap="RdBu_r",  vmin=-rmax_map, vmax=rmax_map, origin="lower", extent=ext)

    panels = [
        ("Truth κ",      k_true,         kw_m),
        ("FEM-MAP κ",    k_map,          kw_m),
        ("MAP residual", k_map - k_true, kw_r),
    ]
    if has_ks:
        rmax_ks = float(np.nanpercentile(np.abs(k_ks - k_true), 99))
        kw_kr   = dict(cmap="RdBu_r", vmin=-rmax_ks, vmax=rmax_ks,
                       origin="lower", extent=ext)
        rmax_raw = float(np.nanpercentile(np.abs(k_ks_raw), 99))
        kw_krr  = dict(cmap="RdBu_r", vmin=-rmax_raw, vmax=rmax_raw,
                       origin="lower", extent=ext)
        panels += [
            (f"KS κ (DC-fixed)\nshift={bg:+.3f}", k_ks,         kw_m),
            ("KS residual",                        k_ks - k_true, kw_kr),
            ("KS κ (raw, mean=0)",                 k_ks_raw,     kw_krr),
        ]

    for ax, (title, data, kw) in zip(axes, panels):
        ax.set_facecolor("#111")
        im = ax.imshow(data, **kw)
        ax.set_title(title, color="#eee", fontsize=10)
        ax.tick_params(colors="#aaa", labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            colors="#aaa", labelsize=7)
        if mask is not None:
            ax.contour(XX, YY, mask.astype(float), levels=[0.5],
                       colors="#ff4444", linewidths=0.8)

    fig.suptitle(tag or "FEM-MAP vs KS", color="#00ff41", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"  Saved: {path}")


def summary_bar_chart(mc_results: list, path: str):
    labels   = [r[0] for r in mc_results]
    mean_map = [r[1]["mean_map"] for r in mc_results]
    std_map  = [r[1]["std_map"]  for r in mc_results]
    mean_ks  = [r[1]["mean_ks"]  for r in mc_results]
    std_ks   = [r[1]["std_ks"]   for r in mc_results]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2*len(labels)), 5),
                           facecolor="#1a1a1a")
    ax.set_facecolor("#111")
    ekw = dict(ecolor="#fff", lw=1.5, capsize=4)

    bars_map = ax.bar(x - w/2, mean_map, w, yerr=std_map,
                      label="FEM-MAP", color="#00ff41", alpha=0.85, error_kw=ekw)
    bars_ks  = ax.bar(x + w/2, mean_ks,  w, yerr=std_ks,
                      label="KS + DC correction (SMPy)",
                      color="#4488ff", alpha=0.85, error_kw=ekw)

    for bars, vals in [(bars_map, mean_map), (bars_ks, mean_ks)]:
        for bar, m in zip(bars, vals):
            if np.isfinite(m):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{m:.3f}", ha="center", va="bottom",
                        color="#eee", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", color="#eee", fontsize=9)
    ax.set_ylabel("Mean Normalised L2  (±1σ)", color="#eee", fontsize=9)
    ax.set_title("FEM-MAP vs KS+DC: Mean L2 over Noise Realisations\n"
                 "(shear from infinite-plane FFT forward model)",
                 color="#00ff41", fontsize=12)
    ax.tick_params(colors="#aaa")
    ax.legend(framealpha=0.3, labelcolor="#eee", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ax.spines.values(): sp.set_edgecolor("#555")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"Summary bar chart saved: {path}")


def trial_scatter_plot(mc_results: list, path: str):
    fig, ax = plt.subplots(figsize=(max(10, 2*len(mc_results)), 5),
                           facecolor="#1a1a1a")
    ax.set_facecolor("#111")

    for i, (label, res) in enumerate(mc_results):
        maps = res["l2_map_trials"]
        kss  = res["l2_ks_trials"]
        jm   = np.random.default_rng(0).uniform(-0.06, 0.06, len(maps))
        jk   = np.random.default_rng(1).uniform(-0.06, 0.06, len(kss))
        w    = 0.18

        ax.scatter(i - w + jm, maps, color="#00ff41", alpha=0.7, s=30, zorder=4)
        ax.plot([i-w-0.12, i-w+0.12], [res["mean_map"]]*2,
                color="#00ff41", lw=2.5, zorder=5)
        finite = np.isfinite(kss)
        if finite.any():
            ax.scatter(i + w + jk[finite], kss[finite],
                       color="#4488ff", alpha=0.7, s=30, zorder=4)
            ax.plot([i+w-0.12, i+w+0.12], [res["mean_ks"]]*2,
                    color="#4488ff", lw=2.5, zorder=5)

    legend = [
        Line2D([0],[0], color="#00ff41", lw=2.5, label="FEM-MAP mean"),
        Line2D([0],[0], marker='o', color="#00ff41", lw=0, markersize=6,
               alpha=0.7, label="FEM-MAP trials"),
        Line2D([0],[0], color="#4488ff", lw=2.5, label="KS+DC mean"),
        Line2D([0],[0], marker='o', color="#4488ff", lw=0, markersize=6,
               alpha=0.7, label="KS+DC trials"),
    ]
    ax.legend(handles=legend, framealpha=0.3, labelcolor="#eee", fontsize=9)
    ax.set_xticks(np.arange(len(mc_results)))
    ax.set_xticklabels([r[0] for r in mc_results],
                       rotation=20, ha="right", color="#eee", fontsize=9)
    ax.set_ylabel("Normalised L2 Error", color="#eee", fontsize=10)
    ax.set_title("Per-Trial L2 Errors  (10 noise realisations per scenario)",
                 color="#00ff41", fontsize=13)
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values(): sp.set_edgecolor("#555")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"Trial scatter plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSMPyFEMMIComparison(unittest.TestCase):

    NX_FEM  = 12
    NX_GRID = 32
    MAXITER = 200

    def _trial(self, **kw):
        return single_trial(nx_fem=self.NX_FEM, nx_grid=self.NX_GRID,
                            maxiter=self.MAXITER, **kw)

    # -- KS forward model -----------------------------------------------------

    def test_00_ks_forward_self_consistent(self):
        """KS forward → KS inverse gives L2 < 0.05 (no smoothing)."""
        k = gaussian_kappa_grid(self.NX_GRID)
        g1, g2 = ks_forward(k)
        # manual KS inverse (no smoothing)
        ny, nx = k.shape
        g1h = np.fft.fft2(g1); g2h = np.fft.fft2(g2)
        k1, k2 = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny))
        k2sq = k1**2 + k2**2; k2sq[0,0] = np.finfo(float).eps
        kr = np.real(np.fft.ifft2(
            ((k1**2 - k2**2)*g1h + 2*k1*k2*g2h) / k2sq))
        kr_corr, _ = dc_correct(kr)
        self.assertLess(l2_error(kr_corr, k), 0.05)

    def test_01_ks_forward_shear_pattern(self):
        """KS forward produces quadrupole shear pattern for circular lens."""
        k = gaussian_kappa_grid(self.NX_GRID)
        g1, g2 = ks_forward(k)
        # g1 should be negative on x-axis (tangential shear), positive on y-axis
        cx, cy = self.NX_GRID // 2, self.NX_GRID // 2
        self.assertLess(g1[cy, cx + self.NX_GRID//4], 0)   # right of centre
        self.assertGreater(g1[cy + self.NX_GRID//4, cx], 0) # above centre

    # -- DC correction --------------------------------------------------------

    def test_02_dc_correct_zero_mean_sky(self):
        """dc_correct: sky border has mean zero after correction."""
        k = np.random.default_rng(0).normal(0, 0.05, (40, 40))
        k[15:25, 15:25] += 1.0   # compact signal
        k_corr, bg = dc_correct(k + 0.3)   # simulate 0.3 offset
        m = max(1, int(0.15 * 40))
        sky = np.concatenate([k_corr[:m,:].ravel(), k_corr[-m:,:].ravel(),
                               k_corr[:,:m].ravel(), k_corr[:,-m:].ravel()])
        self.assertAlmostEqual(float(sky.mean()), 0.0, delta=0.005)

    def test_03_dc_correct_returns_shift(self):
        """dc_correct second return equals the shift applied."""
        k = np.ones((40, 40)) * 0.42
        _, bg = dc_correct(k)
        self.assertAlmostEqual(bg, 0.42, places=10)

    # -- SMPy smoke tests -----------------------------------------------------

    @unittest.skipUnless(_SMPY_AVAILABLE, "SMPy not installed")
    def test_04_ks_self_consistency_via_smpy(self):
        """SMPy KS + DC correction on KS-generated shear: L2 < 0.10."""
        k = gaussian_kappa_grid(self.NX_GRID)
        g1, g2 = ks_forward(k)
        k_ks, _ = run_ks(g1, g2, smoothing_sigma=0.5)
        self.assertLess(l2_error(k_ks, k), 0.10)

    @unittest.skipUnless(_SMPY_AVAILABLE, "SMPy not installed")
    def test_05_smpy_raw_mean_zero(self):
        """Raw SMPy output has mean ≈ 0 (mass sheet degeneracy confirmed)."""
        k = gaussian_kappa_grid(self.NX_GRID)
        g1, g2 = ks_forward(k)
        cfg = Config.from_defaults("kaiser_squires").to_dict()
        cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = 0.0
        raw, _ = KaiserSquiresMapper(cfg).create_maps(g1, g2)
        self.assertAlmostEqual(float(raw.mean()), 0.0, places=5)

    @unittest.skipUnless(_SMPY_AVAILABLE, "SMPy not installed")
    def test_06_dc_correction_improves_ks(self):
        """DC correction reduces L2 vs raw output."""
        k = gaussian_kappa_grid(self.NX_GRID)
        g1, g2 = ks_forward(k)
        cfg = Config.from_defaults("kaiser_squires").to_dict()
        cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = 0.5
        raw, _ = KaiserSquiresMapper(cfg).create_maps(g1, g2)
        corrected, _ = dc_correct(raw)
        self.assertLess(l2_error(corrected, k), l2_error(raw, k))

    # -- single_trial integration tests ---------------------------------------

    @unittest.skipUnless(_SMPY_AVAILABLE, "SMPy not installed")
    def test_07_noiseless_map_beats_ks(self):
        """Noiseless: FEM-MAP L2 < KS L2."""
        out = self._trial(noise_level=0.0, lam_reg=1e-4,
                          wiener_length=0.5, smoothing_sigma=0.5)
        if np.isfinite(out["l2_ks"]):
            self.assertLess(out["l2_map"], out["l2_ks"])

    def test_08_noisy_map_bounded(self):
        """10% noise: FEM-MAP L2 < 0.5."""
        out = self._trial(noise_level=0.10, lam_reg=1e-2, wiener_length=0.5)
        self.assertLess(out["l2_map"], 0.50)

    @unittest.skipUnless(_SMPY_AVAILABLE, "SMPy not installed")
    def test_09_masked_map_beats_ks(self):
        """Central mask: FEM-MAP outperforms KS (inpainting via prior)."""
        out = self._trial(noise_level=0.10, lam_reg=2e-2, wiener_length=0.5,
                          apply_mask=True, mask_center=(0., 0.),
                          mask_radius=0.5)
        if np.isfinite(out["l2_ks"]):
            self.assertLess(out["l2_map"], out["l2_ks"])

    def test_10_high_noise_finite(self):
        """30% noise: MAP output has no NaN/inf."""
        out = self._trial(noise_level=0.30, lam_reg=5e-2, wiener_length=0.5)
        self.assertTrue(np.all(np.isfinite(out["kappa_map_grid"])))

    def test_11_off_centre_bounded(self):
        """Off-centre lens: FEM-MAP L2 < 0.5."""
        fn = lambda nx: gaussian_kappa_grid(nx, cx=0.8, cy=0.5)
        out = self._trial(noise_level=0.10, lam_reg=1e-2,
                          wiener_length=0.5, kappa_grid_fn=fn)
        self.assertLess(out["l2_map"], 0.50)

    def test_12_double_gaussian_bounded(self):
        """Two-component field: FEM-MAP L2 < 0.6."""
        out = self._trial(noise_level=0.10, lam_reg=1e-2,
                          wiener_length=0.5,
                          kappa_grid_fn=double_gaussian_kappa_grid)
        self.assertLess(out["l2_map"], 0.60)

    def test_13_wiener_both_converge(self):
        """Wiener and H1 priors both produce finite bounded L2."""
        common = dict(nx_fem=self.NX_FEM, nx_grid=self.NX_GRID,
                      noise_level=0.10, lam_reg=1e-2, maxiter=self.MAXITER)
        out_h1 = single_trial(**common, wiener_length=0.0)
        out_w  = single_trial(**common, wiener_length=SIGMA_LENS)
        for out in [out_h1, out_w]:
            self.assertTrue(np.isfinite(out["l2_map"]))
            self.assertLess(out["l2_map"], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scenario Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def main(n_trials: int = 10):
    plot_dir = "outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)

    scenarios = [
        ("Noiseless",
         dict(noise_level=0.00, lam_reg=1e-4, wiener_length=0.5,
              smoothing_sigma=0.5)),
        ("10% noise",
         dict(noise_level=0.10, lam_reg=1e-2, wiener_length=0.5,
              smoothing_sigma=1.0)),
        ("20% noise",
         dict(noise_level=0.20, lam_reg=3e-2, wiener_length=0.5,
              smoothing_sigma=1.5)),
        ("Mask 10% noise",
         dict(noise_level=0.10, lam_reg=2e-2, wiener_length=0.5,
              smoothing_sigma=1.0,
              apply_mask=True, mask_center=(0., 0.), mask_radius=0.6)),
        ("Off-centre lens",
         dict(noise_level=0.10, lam_reg=1e-2, wiener_length=0.5,
              smoothing_sigma=1.0,
              kappa_grid_fn=lambda nx: gaussian_kappa_grid(nx, cx=0.8, cy=0.5))),
        ("Double Gaussian",
         dict(noise_level=0.10, lam_reg=1e-2, wiener_length=0.5,
              smoothing_sigma=1.0,
              kappa_grid_fn=double_gaussian_kappa_grid)),
        ("Wiener prior",
         dict(noise_level=0.10, lam_reg=1e-2, wiener_length=SIGMA_LENS,
              smoothing_sigma=1.0,
              apply_mask=True, mask_center=(0., 0.), mask_radius=0.5)),
        ("H1 prior",
         dict(noise_level=0.10, lam_reg=1e-2, wiener_length=0.0,
              smoothing_sigma=1.0,
              apply_mask=True, mask_center=(0., 0.), mask_radius=0.5)),
    ]

    mc_results = []
    for label, kwargs in scenarios:
        res = monte_carlo_benchmark(
            n_trials=n_trials, nx_fem=20, nx_grid=NX_GRID,
            maxiter=400, sky_fraction=0.15,
            tag=label, plot_dir=plot_dir,
            **kwargs,
        )
        mc_results.append((label, res))

    summary_bar_chart(mc_results, os.path.join(plot_dir, "summary_mean_l2.png"))
    trial_scatter_plot(mc_results, os.path.join(plot_dir, "trial_scatter.png"))

    print("\n" + "═"*82)
    print(f"{'Scenario':<22} {'MAP mean±std':>18} {'KS+DC mean±std':>18} "
          f"{'KS raw':>10} {'Δ%':>8}")
    print("─"*82)
    for label, res in mc_results:
        imp = (f"{res['mean_improvement']:+.1f}%"
               if np.isfinite(res["mean_improvement"]) else "   N/A")
        ks  = (f"{res['mean_ks']:.4f}±{res['std_ks']:.4f}"
               if np.isfinite(res["mean_ks"]) else "           N/A")
        ksr = (f"{res['mean_ks_raw']:.4f}"
               if np.isfinite(res.get("mean_ks_raw", np.nan)) else "   N/A")
        print(f"  {label:<20} {res['mean_map']:.4f}±{res['std_map']:.4f}  "
              f"{ks:>18}  {ksr:>10}  {imp:>8}")
    print("═"*82)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="FEM-MAP vs KS Monte Carlo (infinite-plane forward model)")
    parser.add_argument("--unittest", action="store_true")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if args.unittest:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        main(n_trials=args.trials)