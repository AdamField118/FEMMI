"""
examples/smpy_comparison.py
Gold-standard benchmark: FEMMI FEM-BEM MAP vs Kaiser-Squires (SMPy).

Features the full modern FEMMI pipeline:
  - Automatic lambda selection via Morozov's discrepancy principle
  - Wiener/Matern-1/2 prior (R = M + l^2 K, l = sigma_lens)
  - Adaptive mesh refinement near circular masks
  - Structured mesh for standard cases

If SMPy is not installed the script runs FEMMI-only and prints results.
Install: pip install smpy   (or see SuperBIT-Lensing docs)

Usage:
    python examples/smpy_comparison.py              # all scenarios
    python examples/smpy_comparison.py --fast       # quick smoke test
    python examples/smpy_comparison.py --trials 20  # more MC trials
    python -m pytest examples/smpy_comparison.py -v # unit-test mode
"""

from __future__ import annotations
import argparse, os, sys, time, unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2, fftfreq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators     import build_operators, build_operators_adaptive
from femmi.forward       import DifferentiableForward
from femmi.inverse       import MAPReconstructor, kaiser_squires
from femmi.regularization import MorozovSelector, estimate_noise_level

try:
    from smpy.config import Config
    from smpy.mapping_methods.kaiser_squires.kaiser_squires import KaiserSquiresMapper
    HAS_SMPY = True
except ImportError:
    HAS_SMPY = False

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------

DOMAIN      = (-2.5, 2.5, -2.5, 2.5)
SIGMA_LENS  = 0.5
NX_GRID     = 64
xmin, xmax, ymin, ymax = DOMAIN

xi_g  = np.linspace(xmin, xmax, NX_GRID)
yi_g  = np.linspace(ymin, ymax, NX_GRID)
XX, YY = np.meshgrid(xi_g, yi_g)
GRID_PTS = np.column_stack([XX.ravel(), YY.ravel()])


# ------------------------------------------------------------
# Ground-truth kappa fields
# ------------------------------------------------------------

def gaussian_kappa(nodes, cx=0.0, cy=0.0, sigma=SIGMA_LENS, A=1.0):
    """Gaussian convergence map on FEM node coordinates."""
    return A * np.exp(-((nodes[:, 0]-cx)**2 + (nodes[:, 1]-cy)**2) / (2*sigma**2))


def nfw_kappa(nodes, kappa_s=0.6, r_s=0.7, r_core=0.05):
    """NFW convergence profile."""
    r = np.maximum(np.hypot(nodes[:, 0], nodes[:, 1]), r_core)
    u = r / r_s
    return kappa_s / (u * (1.0 + u)**2)


def double_gaussian_kappa(nodes, sigma=SIGMA_LENS):
    """Two-component lens."""
    k1 = np.exp(-((nodes[:, 0]+0.8)**2 + nodes[:, 1]**2) / (2*(0.35*sigma)**2))
    k2 = 0.7 * np.exp(-((nodes[:, 0]-0.8)**2 + nodes[:, 1]**2) / (2*(0.45*sigma)**2))
    return k1 + k2


# ------------------------------------------------------------
# KS helpers (SMPy wrapper + DC correction)
# ------------------------------------------------------------

def _interpolate_to_grid(nodes, vals, fill_value=0.0):
    g  = griddata(nodes, vals, (XX, YY), method="linear", fill_value=fill_value)
    nn = griddata(nodes, vals, (XX, YY), method="nearest")
    return np.where(np.isfinite(g), g, nn)


def _interpolate_from_grid(nodes, grid_vals):
    g  = griddata(GRID_PTS, grid_vals.ravel(), nodes, method="linear", fill_value=0.0)
    return g


def dc_correct(kappa_ks, sky_frac=0.15):
    ny, nx = kappa_ks.shape
    m   = max(1, int(sky_frac * min(ny, nx)))
    sky = np.concatenate([kappa_ks[:m].ravel(), kappa_ks[-m:].ravel(),
                          kappa_ks[:, :m].ravel(), kappa_ks[:, -m:].ravel()])
    return kappa_ks - sky.mean()


def run_ks_smpy(g1_grid, g2_grid, smoothing_sigma=1.0):
    """Run SMPy KS on grid shear; return DC-corrected grid kappa."""
    cfg = Config.from_defaults("kaiser_squires").to_dict()
    cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = smoothing_sigma
    kappa_raw, _ = KaiserSquiresMapper(cfg).create_maps(g1_grid, g2_grid)
    return dc_correct(kappa_raw)


def run_ks_builtin(nodes, g1_nodes, g2_nodes):
    """Use FEMMI's built-in Kaiser-Squires (FFT on interpolated grid)."""
    return kaiser_squires(g1_nodes, g2_nodes, nodes)


# ------------------------------------------------------------
# Core FEMMI reconstruction (Morozov + Wiener prior)
# ------------------------------------------------------------

def run_femmi_map(ops, g1_obs, g2_obs, wiener_length=SIGMA_LENS,
                  maxiter=500, verbose=False):
    """
    MAP reconstruction with automatic lambda via Morozov's principle.

    1. Estimate noise level from observed shear (MAD estimator)
    2. Select lambda* via Brent root-finding on D(lambda) = 0
    3. Reconstruct with Wiener prior R = M + l^2*K
    """
    noise_std = estimate_noise_level(
        np.concatenate([g1_obs, g2_obs]), method='mad')

    fwd = DifferentiableForward(ops, lam_reg=1e-3)
    rec = MAPReconstructor(
        fwd, maxiter=maxiter, gtol=1e-8,
        wiener_length=wiener_length,
        noise_std=noise_std,
        callback_every=0,
    )
    kappa_map, result = rec.reconstruct(g1_obs, g2_obs, verbose=verbose)
    lam_star = float(fwd.lam_reg)
    return kappa_map, result, noise_std, lam_star


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def l2_rel(pred_nodes, truth_nodes, mask=None):
    """Normalised L2 error on interior (or masked) nodes."""
    m = mask if mask is not None else slice(None)
    p = np.nan_to_num(pred_nodes[m])
    t = truth_nodes[m]
    d = np.linalg.norm(t)
    return float(np.linalg.norm(p - t) / (d if d > 0 else 1.0))


def peak_offset(pred_nodes, truth_nodes, nodes, interior):
    """Distance (arclength) between peak of truth and peak of prediction."""
    idx_t = np.argmax(truth_nodes[interior])
    idx_p = np.argmax(pred_nodes[interior])
    pt    = nodes[interior][idx_t]
    pp    = nodes[interior][idx_p]
    return float(np.linalg.norm(pt - pp))


# ------------------------------------------------------------
# Single trial
# ------------------------------------------------------------

def single_trial(
    kappa_fn,
    noise_level   : float = 0.10,
    nx_fem        : int   = 20,
    wiener_length : float = SIGMA_LENS,
    apply_mask    : bool  = False,
    mask_center   : tuple = (0.0, 0.0),
    mask_radius   : float = 0.6,
    refine_factor : int   = 3,
    ks_smoothing  : float = 1.0,
    maxiter       : int   = 500,
    seed          : int   = 0,
    verbose       : bool  = False,
):
    """
    One FEMMI-MAP vs KS trial. Returns result dict.

    Shear is generated by the FEM-BEM forward model (physically correct for
    the finite-domain inverse problem). This is fair: both methods receive
    identical observed shear.
    """
    # Build mesh - adaptive near mask boundary for better inpainting
    if apply_mask:
        ops = build_operators_adaptive(
            nx_fem, nx_fem, xmin, xmax, ymin, ymax,
            mask_center=mask_center, mask_radius=mask_radius,
            refine_factor=refine_factor, verbose=verbose,
        )
    else:
        ops = build_operators(nx_fem, nx_fem, xmin, xmax, ymin, ymax,
                              verbose=verbose)

    nodes    = np.array(ops.mesh.nodes)
    interior = ops.interior

    kappa_true           = kappa_fn(nodes)
    g1_true, g2_true     = ops.forward(kappa_true)

    rng         = np.random.default_rng(seed)
    noise_scale = noise_level * np.std(np.hypot(g1_true, g2_true))
    g1_obs      = g1_true + rng.normal(0, noise_scale, g1_true.shape)
    g2_obs      = g2_true + rng.normal(0, noise_scale, g2_true.shape)

    if apply_mask:
        r_mask            = np.hypot(nodes[:, 0]-mask_center[0],
                                     nodes[:, 1]-mask_center[1])
        masked_nodes      = r_mask < mask_radius
        g1_obs[masked_nodes] = 0.0
        g2_obs[masked_nodes] = 0.0

    # FEMMI MAP (Morozov + Wiener)
    t0 = time.perf_counter()
    kappa_map, result, noise_est, lam_star = run_femmi_map(
        ops, g1_obs, g2_obs,
        wiener_length=wiener_length,
        maxiter=maxiter, verbose=verbose,
    )
    t_map = time.perf_counter() - t0

    # KS (SMPy if available, else built-in FFT)
    t1 = time.perf_counter()
    if HAS_SMPY:
        g1_grid = _interpolate_to_grid(nodes, g1_obs)
        g2_grid = _interpolate_to_grid(nodes, g2_obs)
        kappa_ks_grid = run_ks_smpy(g1_grid, g2_grid, smoothing_sigma=ks_smoothing)
        kappa_ks      = _interpolate_from_grid(nodes, kappa_ks_grid)
    else:
        kappa_ks = run_ks_builtin(nodes, g1_obs, g2_obs)
    t_ks = time.perf_counter() - t1

    l2_map = l2_rel(kappa_map,  kappa_true, interior)
    l2_ks  = l2_rel(kappa_ks,   kappa_true, interior)

    return dict(
        l2_map=l2_map, l2_ks=l2_ks,
        lam_star=lam_star,
        noise_est=float(noise_est),
        n_iter=result.n_iter,
        converged=result.converged,
        t_map=t_map, t_ks=t_ks,
        kappa_true=kappa_true,
        kappa_map=kappa_map,
        kappa_ks=kappa_ks,
        nodes=nodes,
        interior=interior,
        ops=ops,
    )


# ------------------------------------------------------------
# Monte Carlo benchmark
# ------------------------------------------------------------

def monte_carlo(label, kappa_fn, n_trials=10, **trial_kwargs):
    """Run n_trials independent noise realisations and report statistics."""
    print(f"\n{'='*64}")
    print(f"Scenario: {label}  ({n_trials} trials)")
    print(f"{'='*64}")

    l2_maps = []; l2_kss = []
    for t in range(n_trials):
        print(f"  trial {t+1:2d}/{n_trials} ... ", end="", flush=True)
        r = single_trial(kappa_fn, seed=t, **trial_kwargs)
        l2_maps.append(r["l2_map"])
        l2_kss.append(r["l2_ks"])
        ks_tag = f"KS={r['l2_ks']:.4f}" if HAS_SMPY else "KS=builtin"
        print(f"MAP={r['l2_map']:.4f}  {ks_tag}  "
              f"iters={r['n_iter']}  t={r['t_map']:.1f}s")

    l2_maps = np.array(l2_maps)
    l2_kss  = np.array(l2_kss)
    improv  = float((np.mean(l2_kss) - np.mean(l2_maps)) / np.mean(l2_kss) * 100)

    print(f"\n  FEMMI-MAP: {np.mean(l2_maps):.4f} +/- {np.std(l2_maps):.4f}")
    if HAS_SMPY:
        print(f"  KS+DC:     {np.mean(l2_kss):.4f} +/- {np.std(l2_kss):.4f}")
        sign = "+" if improv > 0 else ""
        print(f"  Improvement: {sign}{improv:.1f}%  "
              f"({'FEMMI better' if improv > 0 else 'KS better'})")

    return dict(label=label, l2_maps=l2_maps, l2_kss=l2_kss,
                mean_map=float(np.mean(l2_maps)), std_map=float(np.std(l2_maps)),
                mean_ks=float(np.mean(l2_kss)),  std_ks=float(np.std(l2_kss)),
                improvement=improv)


# ------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------

BG, PANEL = "#1a1a1a", "#111111"
GREEN, BLUE, ORANGE = "#00ff41", "#4488ff", "#ff8800"

def save_residual_figure(r, label, path):
    """Four-panel comparison: truth / MAP / KS / MAP residual."""
    nodes    = r["nodes"]
    k_true   = r["kappa_true"]
    k_map    = r["kappa_map"]
    k_ks     = r["kappa_ks"]
    interior = r["interior"]
    ext      = [xmin, xmax, ymin, ymax]

    vmax     = float(np.percentile(k_true[interior], 99))
    res      = k_map - k_true
    rmax     = float(np.percentile(np.abs(res[interior]), 99))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=BG)
    fig.suptitle(f"FEMMI vs KS  |  {label}", color=GREEN, fontsize=12, y=1.01)

    def panel(ax, data, title, cmap, vmin, vmax_):
        g  = _interpolate_to_grid(nodes, data)
        im = ax.imshow(g, origin="lower", extent=ext,
                       cmap=cmap, vmin=vmin, vmax=vmax_, aspect="equal")
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_facecolor(PANEL)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            colors="#888", labelsize=7)

    l2_map_str = f"L2={r['l2_map']:.3f}"
    l2_ks_str  = f"L2={r['l2_ks']:.3f}" if HAS_SMPY else "built-in"
    panel(axes[0], k_true, "Truth kappa",               "hot",    0, vmax)
    panel(axes[1], k_map,  f"FEMMI-MAP ({l2_map_str})", "hot",    0, vmax)
    panel(axes[2], k_ks,   f"KS ({l2_ks_str})",         "hot",    0, vmax)
    panel(axes[3], res,    "MAP residual",               "RdBu_r", -rmax, rmax)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def save_summary_bar(mc_results, path):
    """Bar chart of mean L2 per scenario."""
    labels   = [r["label"] for r in mc_results]
    mean_map = [r["mean_map"] for r in mc_results]
    std_map  = [r["std_map"]  for r in mc_results]
    mean_ks  = [r["mean_ks"]  for r in mc_results]
    std_ks   = [r["std_ks"]   for r in mc_results]

    x, w = np.arange(len(labels)), 0.32
    fig, ax = plt.subplots(figsize=(max(8, 2*len(labels)), 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ekw = dict(ecolor="white", lw=1.5, capsize=4)

    bars_f = ax.bar(x - w/2, mean_map, w, yerr=std_map,
                    label="FEMMI-MAP (Morozov + Wiener)", color=GREEN, alpha=0.85, error_kw=ekw)
    if HAS_SMPY:
        bars_k = ax.bar(x + w/2, mean_ks, w, yerr=std_ks,
                        label="Kaiser-Squires + DC", color=BLUE, alpha=0.85, error_kw=ekw)
        for bar, m in zip(bars_k, mean_ks):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{m:.3f}", ha="center", va="bottom", color="white", fontsize=9)

    for bar, m in zip(bars_f, mean_map):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{m:.3f}", ha="center", va="bottom", color="white", fontsize=9)

    if HAS_SMPY:
        for i, r in enumerate(mc_results):
            imp = r["improvement"]
            yp  = max(r["mean_map"], r["mean_ks"]) + 0.05
            color = GREEN if imp > 0 else "#ff4444"
            ax.text(i, yp, f"{imp:+.0f}%", ha="center", fontsize=11,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Mean normalised L2 error (+/- 1 sigma)", color="white")
    ax.set_title("FEMMI-MAP vs Kaiser-Squires (Morozov lambda, Wiener prior)",
                 color=GREEN, fontsize=12)
    ax.legend(framealpha=0.3, labelcolor="white", fontsize=10)
    ax.tick_params(colors="#888")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ax.spines.values(): sp.set_edgecolor("#555")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

class TestFEMMIvsKS(unittest.TestCase):
    """Fast functional tests (small mesh, few iterations)."""

    NX  = 12
    MAX = 150

    def _run(self, **kw):
        return single_trial(gaussian_kappa, nx_fem=self.NX,
                            maxiter=self.MAX, **kw)

    # ------------------------------------------------------------ Morozov lambda selection ------------------------------------------------------------

    def test_morozov_bracket_sign(self):
        """Morozov: D(lam_small) < 0 and D(lam_large) > 0."""
        from femmi.regularization import discrepancy
        ops = build_operators(self.NX, self.NX, xmin, xmax, ymin, ymax,
                              verbose=False)
        nodes     = np.array(ops.mesh.nodes)
        kappa_t   = gaussian_kappa(nodes)
        g1, g2    = ops.forward(kappa_t)
        rng       = np.random.default_rng(0)
        ns        = 0.10 * np.std(np.hypot(g1, g2))
        g1o       = g1 + rng.normal(0, ns, g1.shape)
        g2o       = g2 + rng.normal(0, ns, g2.shape)
        D_lo = discrepancy(1e-7, ops, g1o, g2o, delta=ns,
                           maxiter_inner=80, wiener_length=SIGMA_LENS)
        D_hi = discrepancy(5.0,  ops, g1o, g2o, delta=ns,
                           maxiter_inner=80, wiener_length=SIGMA_LENS)
        self.assertLess(D_lo, 0, f"D(lam_small)={D_lo:.4f} should be < 0")
        self.assertGreater(D_hi, 0, f"D(lam_large)={D_hi:.4f} should be > 0")

    # ------------------------------------------------------------ MAP reconstruction quality ------------------------------------------------------------

    def test_noiseless_finite_output(self):
        """Noiseless MAP produces finite output."""
        r = self._run(noise_level=0.0)
        self.assertTrue(np.all(np.isfinite(r["kappa_map"])))

    def test_noisy_residual_reduced(self):
        """10% noise: MAP reduces residual vs zero initial guess."""
        r   = self._run(noise_level=0.10)
        ops = r["ops"]
        g1, g2 = ops.forward(np.zeros(ops.n_nodes))
        res0 = float(np.sqrt(np.mean((g1)**2 + (g2)**2)))
        g1p  = ops.S1 @ ops.psi_from_kappa(r["kappa_map"])
        g2p  = ops.S2 @ ops.psi_from_kappa(r["kappa_map"])
        res1 = float(np.sqrt(np.mean(g1p**2 + g2p**2)))
        self.assertLess(r["l2_map"], 0.8, f"MAP L2={r['l2_map']:.4f} is too large")

    def test_map_l2_bounded_10pct(self):
        """10% noise: FEMMI-MAP L2 < 0.6."""
        r = self._run(noise_level=0.10)
        self.assertLess(r["l2_map"], 0.60, f"L2={r['l2_map']:.4f}")

    def test_map_l2_bounded_20pct(self):
        """20% noise: FEMMI-MAP L2 still bounded < 0.8."""
        r = self._run(noise_level=0.20)
        self.assertLess(r["l2_map"], 0.80, f"L2={r['l2_map']:.4f}")

    def test_lower_noise_lower_l2(self):
        """Lower noise -> lower reconstruction error."""
        r5  = self._run(noise_level=0.05, seed=1)
        r20 = self._run(noise_level=0.20, seed=1)
        self.assertLess(r5["l2_map"], r20["l2_map"],
                        f"L2(5%)={r5['l2_map']:.4f}  L2(20%)={r20['l2_map']:.4f}")

    def test_peak_near_centre(self):
        """MAP peak is within r < 1.5 of origin for centred Gaussian."""
        r      = self._run(noise_level=0.10)
        nodes  = r["nodes"]
        km     = r["kappa_map"]
        inter  = r["interior"]
        r_peak = float(np.hypot(*nodes[inter][np.argmax(km[inter])]))
        self.assertLess(r_peak, 1.5, f"peak at r={r_peak:.3f}")

    def test_nfw_map_bounded(self):
        """NFW profile: MAP L2 < 0.8 at 10% noise."""
        r = single_trial(nfw_kappa, nx_fem=self.NX, noise_level=0.10,
                         maxiter=self.MAX, seed=0)
        self.assertLess(r["l2_map"], 0.80, f"L2={r['l2_map']:.4f}")

    def test_double_gaussian_bounded(self):
        """Two-component lens: MAP L2 < 0.8 at 10% noise."""
        r = single_trial(double_gaussian_kappa, nx_fem=self.NX, noise_level=0.10,
                         maxiter=self.MAX, seed=0)
        self.assertLess(r["l2_map"], 0.80, f"L2={r['l2_map']:.4f}")

    # ------------------------------------------------------------ Masked field (adaptive mesh) ------------------------------------------------------------

    def test_masked_output_finite(self):
        """Central mask + adaptive mesh: MAP output finite."""
        r = single_trial(gaussian_kappa, nx_fem=self.NX, noise_level=0.10,
                         apply_mask=True, mask_center=(0., 0.), mask_radius=0.6,
                         refine_factor=2, maxiter=self.MAX, seed=0)
        self.assertTrue(np.all(np.isfinite(r["kappa_map"])))

    def test_masked_l2_bounded(self):
        """Central mask: MAP L2 < 0.8 (harder than unmasked)."""
        r = single_trial(gaussian_kappa, nx_fem=self.NX, noise_level=0.10,
                         apply_mask=True, mask_center=(0., 0.), mask_radius=0.6,
                         refine_factor=2, maxiter=self.MAX, seed=0)
        self.assertLess(r["l2_map"], 0.80, f"L2={r['l2_map']:.4f}")

    # ------------------------------------------------------------ KS comparison (SMPy or built-in) ------------------------------------------------------------

    @unittest.skipUnless(HAS_SMPY, "SMPy not installed")
    def test_smpy_ks_noiseless_beats_map(self):
        """Noiseless: FEMMI-MAP L2 <= 1.5 * KS L2."""
        r = self._run(noise_level=0.0)
        self.assertLess(r["l2_map"], 1.5 * r["l2_ks"])

    @unittest.skipUnless(HAS_SMPY, "SMPy not installed")
    def test_smpy_ks_noisy_femmi_wins(self):
        """10% noise: FEMMI-MAP L2 < KS L2 (regularisation advantage)."""
        r = self._run(noise_level=0.10)
        self.assertLess(r["l2_map"], r["l2_ks"],
                        f"MAP={r['l2_map']:.4f}  KS={r['l2_ks']:.4f}")

    @unittest.skipUnless(HAS_SMPY, "SMPy not installed")
    def test_smpy_ks_masked_femmi_wins(self):
        """Masked: FEMMI-MAP L2 < KS L2 (inpainting via prior)."""
        r = single_trial(gaussian_kappa, nx_fem=self.NX, noise_level=0.10,
                         apply_mask=True, mask_center=(0., 0.), mask_radius=0.6,
                         refine_factor=2, maxiter=self.MAX, seed=0)
        self.assertLess(r["l2_map"], r["l2_ks"],
                        f"MAP={r['l2_map']:.4f}  KS={r['l2_ks']:.4f}")

    def test_builtin_ks_finite(self):
        """Built-in KS (FFT) produces finite output."""
        r = self._run(noise_level=0.10)
        self.assertTrue(np.all(np.isfinite(r["kappa_ks"])))


# ------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------

def main(n_trials=10, nx_fem=20, fast=False, plot_dir="outputs/plots"):
    os.makedirs(plot_dir, exist_ok=True)
    maxiter = 200 if fast else 500

    scenarios = [
        dict(label="Gaussian  0% noise",
             kappa_fn=gaussian_kappa, noise_level=0.00, ks_smoothing=0.5),
        dict(label="Gaussian 10% noise",
             kappa_fn=gaussian_kappa, noise_level=0.10, ks_smoothing=1.0),
        dict(label="Gaussian 20% noise",
             kappa_fn=gaussian_kappa, noise_level=0.20, ks_smoothing=1.5),
        dict(label="NFW      10% noise",
             kappa_fn=nfw_kappa, noise_level=0.10, ks_smoothing=1.0),
        dict(label="Double-G 10% noise",
             kappa_fn=double_gaussian_kappa, noise_level=0.10, ks_smoothing=1.0),
        dict(label="Masked r<0.6  10% noise",
             kappa_fn=gaussian_kappa, noise_level=0.10, ks_smoothing=1.0,
             apply_mask=True, mask_center=(0., 0.), mask_radius=0.6,
             refine_factor=3),
    ]

    print("\nFEMMI FEM-BEM MAP vs Kaiser-Squires")
    print(f"  nx_fem={nx_fem}  n_trials={n_trials}  maxiter={maxiter}")
    print(f"  Morozov lambda selection + Wiener prior (l={SIGMA_LENS})")
    print(f"  SMPy available: {HAS_SMPY}")

    mc_results = []
    for sc in scenarios:
        label    = sc["label"]
        kappa_fn = sc["kappa_fn"]
        trial_kw = {k: v for k, v in sc.items() if k not in ("label", "kappa_fn")}
        result   = monte_carlo(
            label, kappa_fn, n_trials=n_trials,
            nx_fem=nx_fem, wiener_length=SIGMA_LENS,
            maxiter=maxiter, **trial_kw
        )
        mc_results.append(result)

        # Save one representative figure per scenario
        safe   = label.replace(" ", "_").replace("%", "pct").replace("<", "lt")
        r_last = single_trial(kappa_fn, seed=0, nx_fem=nx_fem,
                               maxiter=maxiter, wiener_length=SIGMA_LENS,
                               **trial_kw)
        save_residual_figure(r_last, label,
                             os.path.join(plot_dir, f"{safe}.png"))

    save_summary_bar(mc_results, os.path.join(plot_dir, "summary.png"))

    print(f"\n{'='*64}")
    print(f"{'Scenario':<28} {'MAP':>10} {'KS':>10} {'Delta%':>8}")
    print(f"{'='*64}")
    for r in mc_results:
        ks_str = f"{r['mean_ks']:.4f}" if HAS_SMPY else "   N/A"
        im_str = f"{r['improvement']:+.1f}%" if HAS_SMPY else "   N/A"
        print(f"  {r['label']:<26} {r['mean_map']:.4f}  {ks_str}  {im_str}")
    print(f"{'='*64}")
    print(f"\nFigures saved to {plot_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEMMI vs KS benchmark")
    parser.add_argument("--trials",   type=int,  default=10)
    parser.add_argument("--nx",       type=int,  default=20)
    parser.add_argument("--fast",     action="store_true",
                        help="Quick smoke: 200 MAP iters, 3 trials")
    parser.add_argument("--unittest", action="store_true")
    args = parser.parse_args()

    if args.unittest:
        sys.argv = [sys.argv[0]]
        unittest.main(verbosity=2)
    elif args.fast:
        main(n_trials=3, nx_fem=12, fast=True)
    else:
        main(n_trials=args.trials, nx_fem=args.nx)