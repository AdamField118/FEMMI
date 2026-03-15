"""
examples/generate_presentation_figures.py
==========================================
Publication figures for ISP presentation.  All outputs go to outputs/.

  fig1_convergence.png      -- shear convergence rate (coarse -> fine reference)
  fig2_noiseless.png        -- truth / MAP / KS, noiseless
  fig3_noisy.png            -- same at 10% noise
  fig4_masked.png           -- central mask: MAP inpaints, KS fails
  fig5_prior_comparison.png -- Wiener vs H1 prior on masked field
  fig6_summary_bar.png      -- summary bar chart

MAP uses the full pipeline:
  - Morozov discrepancy principle for automatic lambda selection
  - Wiener/Matern prior  R = M + l^2 K  (l = sigma_lens = 0.5)
  - Adaptive mesh refinement near circular masks

Run:
    python examples/generate_presentation_figures.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from femmi.operators      import build_operators, build_operators_adaptive
from femmi.forward        import DifferentiableForward
from femmi.inverse        import MAPReconstructor
from femmi.regularization import estimate_noise_level

try:
    from smpy.config import Config
    from smpy.mapping_methods.kaiser_squires.kaiser_squires import KaiserSquiresMapper
    HAS_SMPY = True
except ImportError:
    HAS_SMPY = False
    print("WARNING: SMPy not found -- KS panels will be blank.")

# ---- style ---------------------------------------------------------------
BG, PANEL = "#1a1a1a", "#111111"
GREEN, BLUE, ORANGE = "#00ff41", "#4488ff", "#ff8800"
TEXT, MUTED = "#eeeeee", "#aaaaaa"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor":  "#555555", "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "grid.color": "#333333", "grid.linestyle": "--", "font.size": 11,
})

DOMAIN      = (-2.5, 2.5, -2.5, 2.5)
SIGMA_LENS  = 0.5
OUT         = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)


# ---- helpers -------------------------------------------------------------

def gaussian_kappa(nodes, sigma=SIGMA_LENS, cx=0., cy=0.):
    return np.exp(-((nodes[:,0]-cx)**2 + (nodes[:,1]-cy)**2) / (2*sigma**2))


def nodes_to_grid(nodes, vals, nx=64, domain=DOMAIN):
    xmin, xmax, ymin, ymax = domain
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, nx)
    XX, YY = np.meshgrid(xi, yi)
    g  = griddata(nodes, vals, (XX, YY), method="linear")
    gn = griddata(nodes, vals, (XX, YY), method="nearest")
    g[~np.isfinite(g)] = gn[~np.isfinite(g)]
    return g, XX, YY


def run_map_morozov(ops, g1_obs, g2_obs, wiener=SIGMA_LENS, maxiter=500):
    """MAP with automatic Morozov lambda and Wiener prior."""
    noise_std = estimate_noise_level(np.concatenate([g1_obs, g2_obs]), method='mad')
    fwd = DifferentiableForward(ops, lam_reg=1e-3)
    rec = MAPReconstructor(fwd, maxiter=maxiter, gtol=1e-9,
                           wiener_length=wiener, noise_std=noise_std,
                           callback_every=0)
    kappa, _ = rec.reconstruct(g1_obs, g2_obs, verbose=False)
    return kappa, float(fwd.lam_reg), noise_std


def run_ks(nodes, g1, g2, nx=64, smoothing_sigma=1.0):
    if not HAS_SMPY:
        return None
    g1g, XX, YY = nodes_to_grid(nodes, g1, nx)
    g2g, _,  _  = nodes_to_grid(nodes, g2, nx)
    cfg = Config.from_defaults("kaiser_squires").to_dict()
    cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = smoothing_sigma
    ke, _ = KaiserSquiresMapper(cfg).create_maps(g1g, g2g)
    m   = max(1, int(0.15 * nx))
    sky = np.concatenate([ke[:m].ravel(), ke[-m:].ravel(),
                          ke[:, :m].ravel(), ke[:, -m:].ravel()])
    return ke - sky.mean()


def l2(pred_grid, truth_grid):
    p = np.nan_to_num(pred_grid); t = truth_grid
    d = np.linalg.norm(t)
    return float(np.linalg.norm(p - t) / (d if d > 0 else 1.0))


def colorbar(im, ax):
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
        colors=MUTED, labelsize=7)


def panel(ax, data, title, cmap="viridis", vmin=None, vmax=None,
          ext=None):
    if ext is None:
        ext = list(DOMAIN)
    im = ax.imshow(data, origin="lower", extent=ext,
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=5)
    ax.tick_params(labelsize=7, colors=MUTED)
    return im


def mask_circle(axes, r=0.6):
    theta = np.linspace(0, 2*3.14159, 300)
    for ax in axes:
        ax.plot(r*np.cos(theta), r*np.sin(theta),
                color="#ff4444", lw=1.5, ls="--")


def save_fig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# FIGURE 1 -- P3 Forward Convergence
# =========================================================================
print("\nFig 1: P3 forward convergence (coarse->fine reference)")

xmin, xmax, ymin, ymax = DOMAIN

print("  Building reference mesh (nx=40)...")
ops_ref   = build_operators(40, 40, xmin, xmax, ymin, ymax, verbose=False)
nodes_ref = np.array(ops_ref.mesh.nodes)
kappa_mfr_ref = gaussian_kappa(nodes_ref)
g1_ref, g2_ref = ops_ref.forward(kappa_mfr_ref)
interior_ref   = np.array(ops_ref.interior)

mesh_sizes = [4, 6, 8, 10, 14, 18, 24, 32]
h_vals, errs = [], []

for nx in mesh_sizes:
    print(f"  nx={nx:2d} ... ", end="", flush=True)
    ops_c    = build_operators(nx, nx, xmin, xmax, ymin, ymax, verbose=False)
    nodes_c  = np.array(ops_c.mesh.nodes)
    g1_c, g2_c = ops_c.forward(gaussian_kappa(nodes_c))

    g1_on = griddata(nodes_c, g1_c, nodes_ref, method="linear")
    g1_nn = griddata(nodes_c, g1_c, nodes_ref, method="nearest")
    g1_on[~np.isfinite(g1_on)] = g1_nn[~np.isfinite(g1_on)]
    g2_on = griddata(nodes_c, g2_c, nodes_ref, method="linear")
    g2_nn = griddata(nodes_c, g2_c, nodes_ref, method="nearest")
    g2_on[~np.isfinite(g2_on)] = g2_nn[~np.isfinite(g2_on)]

    num = float(np.sqrt(np.sum((g1_on[interior_ref]-g1_ref[interior_ref])**2 +
                               (g2_on[interior_ref]-g2_ref[interior_ref])**2)))
    den = float(np.sqrt(np.sum(g1_ref[interior_ref]**2 + g2_ref[interior_ref]**2)))
    err = num / (den if den > 0 else 1.0)
    h   = (xmax - xmin) / nx
    h_vals.append(h); errs.append(err)
    print(f"h={h:.3f}  L2={err:.4e}")

h_arr = np.array(h_vals); e_arr = np.array(errs)
mid   = slice(1, -1)
slope = float(np.polyfit(np.log(h_arr[mid]), np.log(e_arr[mid]), 1)[0])
print(f"  Fitted convergence rate: O(h^{slope:.2f})")

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(h_arr, e_arr, "o-", color=GREEN, lw=2.5, ms=8, zorder=5,
          label=f"P3 FEM (slope {slope:.2f})")
h_ref = np.array([h_arr.min()*0.8, h_arr.max()*1.2])
for exp, col, lab in [(3, ORANGE, "O(h3)"), (2, BLUE, "O(h2)")]:
    c = e_arr[mid][-1] / h_arr[mid][-1]**exp
    ax.loglog(h_ref, c*h_ref**exp, "--", color=col, alpha=0.75, lw=1.8, label=lab)
ax.set_xlabel("Mesh spacing h", fontsize=12)
ax.set_ylabel("Normalised L2 shear error", fontsize=11)
ax.set_title("P3 FEM-BEM Forward Convergence", color=GREEN, fontsize=13)
ax.legend(fontsize=11, framealpha=0.3)
ax.grid(True, which="both", alpha=0.4)
fig.tight_layout()
save_fig(fig, "fig1_convergence.png")


# =========================================================================
# Build nx=20 mesh -- shared across figs 2-5
# =========================================================================
print("\nBuilding nx=20 mesh...")
NX  = 20
ops = build_operators(NX, NX, xmin, xmax, ymin, ymax, verbose=False)
nodes = np.array(ops.mesh.nodes)

kappa_true      = gaussian_kappa(nodes)
g1_true, g2_true = ops.forward(kappa_true)
kt_grid, XX, YY  = nodes_to_grid(nodes, kappa_true, 64)
vmax_k = float(kt_grid.max())

rng     = np.random.default_rng(42)
noise_s = 0.10 * float(np.std(np.hypot(g1_true, g2_true)))
g1_obs  = g1_true + rng.normal(0, noise_s, g1_true.shape)
g2_obs  = g2_true + rng.normal(0, noise_s, g2_true.shape)

# Masked shear (r < 0.6)
MASK_R  = 0.6
r_nodes = np.hypot(nodes[:,0], nodes[:,1])
g1_m = g1_obs.copy(); g2_m = g2_obs.copy()
g1_m[r_nodes < MASK_R] = 0.; g2_m[r_nodes < MASK_R] = 0.


# =========================================================================
# FIGURE 2 -- Noiseless reconstruction
# =========================================================================
print("\nFig 2: Noiseless reconstruction...")
print("  Running MAP (Morozov + Wiener, noiseless)...")
km_nl, lam_nl, ns_nl = run_map_morozov(ops, g1_true, g2_true, maxiter=600)
km_nl_g, _, _ = nodes_to_grid(nodes, km_nl)
print(f"  lambda*={lam_nl:.3e}  noise_est={ns_nl:.3e}")

ks_nl = run_ks(nodes, g1_true, g2_true, smoothing_sigma=0.5)
e_map_nl = l2(km_nl_g, kt_grid)
e_ks_nl  = l2(ks_nl, kt_grid) if ks_nl is not None else float('nan')

ncols = 5 if HAS_SMPY else 3
fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
im = panel(axes[0], kt_grid,    "Truth",               vmin=0, vmax=vmax_k); colorbar(im, axes[0])
im = panel(axes[1], km_nl_g,    f"FEM-MAP L2={e_map_nl:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[1])
r  = km_nl_g - kt_grid; rmax = float(np.percentile(np.abs(r), 99))
im = panel(axes[2], r,          "MAP residual", cmap="RdBu_r", vmin=-rmax, vmax=rmax); colorbar(im, axes[2])
if HAS_SMPY and ks_nl is not None:
    im = panel(axes[3], ks_nl,  f"KS+DC  L2={e_ks_nl:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[3])
    rk = ks_nl - kt_grid; rmk = float(np.percentile(np.abs(rk), 99))
    im = panel(axes[4], rk,     "KS residual", cmap="RdBu_r", vmin=-rmk, vmax=rmk); colorbar(im, axes[4])
fig.suptitle("Noiseless (Morozov lambda, Wiener prior)", color=GREEN, fontsize=13)
fig.tight_layout()
save_fig(fig, "fig2_noiseless.png")


# =========================================================================
# FIGURE 3 -- 10% noise
# =========================================================================
print("\nFig 3: 10% noise...")
print("  Running MAP (Morozov + Wiener, 10% noise)...")
km_n, lam_n, ns_n = run_map_morozov(ops, g1_obs, g2_obs)
km_n_g, _, _ = nodes_to_grid(nodes, km_n)
print(f"  lambda*={lam_n:.3e}")

ks_n = run_ks(nodes, g1_obs, g2_obs, smoothing_sigma=1.5)
e_map_n = l2(km_n_g, kt_grid)
e_ks_n  = l2(ks_n, kt_grid) if ks_n is not None else float('nan')

fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
im = panel(axes[0], kt_grid,   "Truth",              vmin=0, vmax=vmax_k); colorbar(im, axes[0])
im = panel(axes[1], km_n_g,    f"FEM-MAP L2={e_map_n:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[1])
r  = km_n_g - kt_grid; rmax = float(np.percentile(np.abs(r), 99))
im = panel(axes[2], r,         "MAP residual", cmap="RdBu_r", vmin=-rmax, vmax=rmax); colorbar(im, axes[2])
if HAS_SMPY and ks_n is not None:
    im = panel(axes[3], ks_n,  f"KS+DC  L2={e_ks_n:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[3])
    rk = ks_n - kt_grid; rmk = float(np.percentile(np.abs(rk), 99))
    im = panel(axes[4], rk,    "KS residual", cmap="RdBu_r", vmin=-rmk, vmax=rmk); colorbar(im, axes[4])
fig.suptitle("10% noise (Morozov lambda, Wiener prior)", color=GREEN, fontsize=13)
fig.tight_layout()
save_fig(fig, "fig3_noisy.png")


# =========================================================================
# FIGURE 4 -- Masked field (adaptive mesh)
# =========================================================================
print("\nFig 4: Masked field  (r < 0.6), adaptive mesh...")
print("  Building adaptive mesh...")
ops_a   = build_operators_adaptive(NX, NX, xmin, xmax, ymin, ymax,
                                    mask_center=(0., 0.), mask_radius=MASK_R,
                                    refine_factor=3, verbose=False)
nodes_a = np.array(ops_a.mesh.nodes)

# Regenerate masked shear on adaptive mesh
kappa_a   = gaussian_kappa(nodes_a)
g1_ta, g2_ta = ops_a.forward(kappa_a)
noise_sa  = 0.10 * float(np.std(np.hypot(g1_ta, g2_ta)))
g1_ma = g1_ta + rng.normal(0, noise_sa, g1_ta.shape)
g2_ma = g2_ta + rng.normal(0, noise_sa, g2_ta.shape)
r_a = np.hypot(nodes_a[:,0], nodes_a[:,1])
g1_ma[r_a < MASK_R] = 0.; g2_ma[r_a < MASK_R] = 0.

print("  Running MAP (Morozov + Wiener, masked + adaptive)...")
km_mask, lam_mask, ns_mask = run_map_morozov(ops_a, g1_ma, g2_ma)
km_mask_g, _, _ = nodes_to_grid(nodes_a, km_mask)
kt_a_g, _, _    = nodes_to_grid(nodes_a, kappa_a)
print(f"  lambda*={lam_mask:.3e}")

ks_mask = run_ks(nodes, g1_m, g2_m, smoothing_sigma=1.5)
e_map_mask = l2(km_mask_g, kt_a_g)
e_ks_mask  = l2(ks_mask, kt_grid) if ks_mask is not None else float('nan')

fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
mask_circle(axes)
im = panel(axes[0], kt_a_g,     "Truth",                 vmin=0, vmax=vmax_k); colorbar(im, axes[0])
im = panel(axes[1], km_mask_g,  f"FEM-MAP L2={e_map_mask:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[1])
r  = km_mask_g - kt_a_g; rmax = float(np.percentile(np.abs(r), 99))
im = panel(axes[2], r,          "MAP residual", cmap="RdBu_r", vmin=-rmax, vmax=rmax); colorbar(im, axes[2])
if HAS_SMPY and ks_mask is not None:
    im = panel(axes[3], ks_mask, f"KS+DC  L2={e_ks_mask:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[3])
    rk = ks_mask - kt_grid; rmk = float(np.percentile(np.abs(rk), 99))
    im = panel(axes[4], rk,     "KS residual", cmap="RdBu_r", vmin=-rmk, vmax=rmk); colorbar(im, axes[4])
fig.suptitle("Masked field r<0.6 -- adaptive mesh, MAP inpaints via Wiener prior",
             color=GREEN, fontsize=13)
fig.tight_layout()
save_fig(fig, "fig4_masked.png")


# =========================================================================
# FIGURE 5 -- Wiener vs H1 prior comparison
# =========================================================================
print("\nFig 5: Wiener vs H1 prior (masked, adaptive mesh)...")
print("  Running MAP with H1 prior...")

# H1 prior: wiener_length=0 uses R=K instead of R=M+l^2*K
noise_std_a = estimate_noise_level(np.concatenate([g1_ma, g2_ma]), method='mad')
fwd_h1 = DifferentiableForward(ops_a, lam_reg=1e-3)
rec_h1 = MAPReconstructor(fwd_h1, maxiter=500, gtol=1e-9,
                           wiener_length=0.0, noise_std=noise_std_a,
                           callback_every=0)
km_h1, _ = rec_h1.reconstruct(g1_ma, g2_ma, verbose=False)
km_h1_g, _, _ = nodes_to_grid(nodes_a, km_h1)
lam_h1 = float(fwd_h1.lam_reg)
print(f"  H1 lambda*={lam_h1:.3e}")

e_h1     = l2(km_h1_g,   kt_a_g)
e_wiener = l2(km_mask_g, kt_a_g)
improv   = (e_h1 - e_wiener) / e_h1 * 100
print(f"  H1 L2={e_h1:.4f}   Wiener L2={e_wiener:.4f}   improvement={improv:+.1f}%")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
mask_circle(axes)
im = panel(axes[0], kt_a_g,    "Truth",                     vmin=0, vmax=vmax_k); colorbar(im, axes[0])
im = panel(axes[1], km_h1_g,   f"H1 prior  L2={e_h1:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[1])
im = panel(axes[2], km_mask_g, f"Wiener prior  L2={e_wiener:.3f}", vmin=0, vmax=vmax_k); colorbar(im, axes[2])
diff = km_mask_g - km_h1_g; dmax = float(np.percentile(np.abs(diff), 99))
im = panel(axes[3], diff, f"Wiener - H1  ({improv:+.1f}% L2 improvement)",
           cmap="RdBu_r", vmin=-dmax, vmax=dmax); colorbar(im, axes[3])
fig.suptitle("Wiener prior (R=M+l^2*K) vs H1 prior (R=K)  -- masked field",
             color=GREEN, fontsize=13)
fig.tight_layout()
save_fig(fig, "fig5_prior_comparison.png")


# =========================================================================
# FIGURE 6 -- Summary bar chart
# =========================================================================
print("\nFig 6: Summary bar chart...")
labels   = ["Noiseless", "10% noise", "10%+Mask"]
map_errs = [e_map_nl,   e_map_n,   e_map_mask]
ks_errs  = [e_ks_nl,    e_ks_n,    e_ks_mask]

fig, ax = plt.subplots(figsize=(8, 5))
x, w = np.arange(len(labels)), 0.32

bars_map = ax.bar(x - w/2, map_errs, w, label="FEM-MAP (Morozov+Wiener)",
                  color=GREEN, alpha=0.85)
if HAS_SMPY:
    bars_ks = ax.bar(x + w/2, ks_errs, w, label="KS + DC",
                     color=BLUE, alpha=0.85)
    for bar, v in zip(bars_ks, ks_errs):
        if np.isfinite(v):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                    f"{v:.3f}", ha="center", va="bottom", color=TEXT, fontsize=11)
    for i, (me, ke) in enumerate(zip(map_errs, ks_errs)):
        if np.isfinite(ke) and ke > 0:
            yp = max(me, ke) + 0.06
            impv = (ke - me) / ke * 100
            ax.text(i, yp, f"{impv:+.0f}%", ha="center", fontsize=12,
                    color=GREEN if impv > 0 else "#ff4444", fontweight="bold")

for bar, v in zip(bars_map, map_errs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
            f"{v:.3f}", ha="center", va="bottom", color=TEXT, fontsize=11)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel("Normalised L2 error", fontsize=11)
ax.set_title("FEM-MAP vs Kaiser-Squires", color=GREEN, fontsize=13)
ax.legend(fontsize=11, framealpha=0.3)
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
save_fig(fig, "fig6_summary_bar.png")


# =========================================================================
# Summary
# =========================================================================
print(f"""
All figures saved to outputs/

fig1_convergence.png  -- P3 convergence O(h^{slope:.1f})
fig2_noiseless.png    -- MAP L2={e_map_nl:.3f}   KS L2={e_ks_nl:.4f}
fig3_noisy.png        -- MAP L2={e_map_n:.3f}   KS L2={e_ks_n:.4f}
fig4_masked.png       -- MAP L2={e_map_mask:.3f}   KS L2={e_ks_mask:.4f}  (adaptive mesh)
fig5_prior_comp.png   -- H1 L2={e_h1:.3f}   Wiener L2={e_wiener:.3f}  ({improv:+.1f}%)
fig6_summary_bar.png  -- clean one-slide summary
""")