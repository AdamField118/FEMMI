"""
examples/generate_figures.py
============================
Preliminary-results figures for FEMMI vs Kaiser-Squires (SMPy).

Kaiser-Squires is run via SMPy's KaiserSquiresMapper, the same pipeline
used by the SuperBIT collaboration and others in the weak-lensing community.
Shear is binned to a 64x64 grid before being passed to SMPy, matching
standard practice.

Outputs (saved to outputs/):
  fig_reconstruction.png  -- truth / FEMMI-MAP / MAP residual / KS+DC / KS residual
                             Gaussian lens, 10% noise, single realisation seed=42
  fig_masked.png          -- masked field (r < 0.6): MAP inpaints, KS collapses
  fig_convergence.png     -- P3 forward-operator shear convergence O(h^2)

Run from the project root:
    python examples/generate_figures.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from femmi.operators      import build_operators, build_operators_adaptive
from femmi.forward        import DifferentiableForward
from femmi.inverse        import MAPReconstructor
from femmi.regularization import estimate_noise_level

from smpy.config import Config
from smpy.mapping_methods.kaiser_squires.kaiser_squires import KaiserSquiresMapper

# ── style ─────────────────────────────────────────────────────────────────────
BG, PANEL   = "#1a1a1a", "#111111"
GREEN, BLUE, ORANGE = "#00ff41", "#4488ff", "#ff8800"
TEXT, MUTED = "#eeeeee", "#aaaaaa"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor":  "#555555", "axes.labelcolor": TEXT,
    "xtick.color": MUTED,    "ytick.color": MUTED,
    "text.color":  TEXT,     "grid.color":  "#333333",
    "grid.linestyle": "--",  "font.size": 11,
})

DOMAIN     = (-2.5, 2.5, -2.5, 2.5)
SIGMA_LENS = 0.5
NX_GRID    = 64
xmin, xmax, ymin, ymax = DOMAIN

xi_g = np.linspace(xmin, xmax, NX_GRID)
yi_g = np.linspace(ymin, ymax, NX_GRID)
XX, YY   = np.meshgrid(xi_g, yi_g)
GRID_PTS = np.column_stack([XX.ravel(), YY.ravel()])

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def gaussian_kappa(nodes, sigma=SIGMA_LENS):
    return np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2 * sigma**2))


def nodes_to_grid(nodes, vals):
    """Interpolate node-valued field to the shared 64x64 comparison grid."""
    g  = griddata(nodes, vals, (XX, YY), method="linear")
    gn = griddata(nodes, vals, (XX, YY), method="nearest")
    return np.where(np.isfinite(g), g, gn)


def shear_nodes_to_grid(nodes, g1, g2):
    """Bin shear from FEM node positions onto the 64x64 grid for SMPy."""
    g1g = griddata(nodes, g1, (XX, YY), method="linear", fill_value=0.0)
    g2g = griddata(nodes, g2, (XX, YY), method="linear", fill_value=0.0)
    g1n = griddata(nodes, g1, (XX, YY), method="nearest")
    g2n = griddata(nodes, g2, (XX, YY), method="nearest")
    g1g = np.where(np.isfinite(g1g), g1g, g1n)
    g2g = np.where(np.isfinite(g2g), g2g, g2n)
    return g1g, g2g


def dc_correct(kappa_ks, sky_frac=0.15):
    """Subtract mean of outer sky annulus (standard DC offset correction)."""
    ny, nx = kappa_ks.shape
    m   = max(1, int(sky_frac * min(ny, nx)))
    sky = np.concatenate([kappa_ks[:m].ravel(),  kappa_ks[-m:].ravel(),
                          kappa_ks[:, :m].ravel(), kappa_ks[:, -m:].ravel()])
    return kappa_ks - sky.mean()


def run_ks_smpy(g1_grid, g2_grid, smoothing_sigma=1.5):
    """Run SMPy Kaiser-Squires and apply DC offset correction."""
    cfg = Config.from_defaults("kaiser_squires").to_dict()
    cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = smoothing_sigma
    kappa_raw, _ = KaiserSquiresMapper(cfg).create_maps(g1_grid, g2_grid)
    return dc_correct(kappa_raw)


def run_map(ops, g1_obs, g2_obs, wiener=SIGMA_LENS, maxiter=500):
    """MAP reconstruction with automatic lambda via Morozov's principle."""
    noise_std = estimate_noise_level(np.concatenate([g1_obs, g2_obs]), method="mad")
    fwd = DifferentiableForward(ops, lam_reg=1e-3)
    rec = MAPReconstructor(fwd, maxiter=maxiter, gtol=1e-9,
                           wiener_length=wiener, noise_std=noise_std,
                           callback_every=0)
    kappa, result = rec.reconstruct(g1_obs, g2_obs, verbose=False)
    return kappa, float(fwd.lam_reg), float(noise_std), result


def l2_rel(pred_grid, truth_grid):
    d = np.linalg.norm(truth_grid)
    return float(np.linalg.norm(np.nan_to_num(pred_grid) - truth_grid)
                 / (d if d > 0 else 1.0))


def panel(ax, data, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(data, origin="lower", extent=[xmin, xmax, ymin, ymax],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=5)
    ax.tick_params(labelsize=7, colors=MUTED)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
        colors=MUTED, labelsize=7)
    return im


def save_fig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path}")


# ── shared mesh and synthetic data ────────────────────────────────────────────
print("Building 20x20 P3 mesh...")
NX    = 20
ops   = build_operators(NX, NX, xmin, xmax, ymin, ymax, verbose=False)
nodes = np.array(ops.mesh.nodes)
print(f"  {ops.n_nodes} nodes  N_b={ops.bnd_mesh.n_boundary_dofs}")

kappa_true       = gaussian_kappa(nodes)
g1_true, g2_true = ops.forward(kappa_true)
kt_grid          = nodes_to_grid(nodes, kappa_true)
vmax_k           = float(kt_grid.max())

rng     = np.random.default_rng(42)
noise_s = 0.10 * float(np.std(np.hypot(g1_true, g2_true)))
g1_obs  = g1_true + rng.normal(0, noise_s, g1_true.shape)
g2_obs  = g2_true + rng.normal(0, noise_s, g2_true.shape)
print(f"  noise_std = {noise_s:.4f}  (10% of mean shear amplitude)")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 -- reconstruction at 10% noise (seed=42 single realisation)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig 1: reconstruction at 10% noise (seed=42)...")

t0 = time.perf_counter()
km, lam, ns, result = run_map(ops, g1_obs, g2_obs)
print(f"  MAP: lambda*={lam:.3e}  {result.n_iter} iters  "
      f"converged={result.converged}  ({time.perf_counter()-t0:.1f}s)")

km_grid          = nodes_to_grid(nodes, km)
g1g_obs, g2g_obs = shear_nodes_to_grid(nodes, g1_obs, g2_obs)
ks_grid          = run_ks_smpy(g1g_obs, g2g_obs, smoothing_sigma=1.5)

e_map  = l2_rel(km_grid, kt_grid)
e_ks   = l2_rel(ks_grid, kt_grid)
improv = (e_ks - e_map) / e_ks * 100
print(f"  FEMMI-MAP L2 = {e_map:.3f}")
print(f"  KS+DC     L2 = {e_ks:.3f}")
print(f"  improvement  = {improv:+.1f}%")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
panel(axes[0], kt_grid, r"Truth $\kappa$",             vmin=0, vmax=vmax_k)
panel(axes[1], km_grid, f"FEMMI-MAP  L2={e_map:.3f}",  vmin=0, vmax=vmax_k)
r_map  = km_grid - kt_grid
r_max  = float(np.percentile(np.abs(r_map), 99))
panel(axes[2], r_map,   "MAP residual", cmap="RdBu_r", vmin=-r_max, vmax=r_max)
panel(axes[3], ks_grid, f"KS+DC  L2={e_ks:.3f}",      vmin=0, vmax=vmax_k)
r_ks   = ks_grid - kt_grid
r_kmax = float(np.percentile(np.abs(r_ks), 99))
panel(axes[4], r_ks,    "KS+DC residual", cmap="RdBu_r", vmin=-r_kmax, vmax=r_kmax)

fig.suptitle(
    r"FEMMI vs Kaiser-Squires (SMPy)  |  Gaussian $\kappa$ ($\sigma=0.5$), "
    r"10% noise, seed=42  |  $20\times20$ P3 mesh, $\lambda^*$ via Morozov",
    color=GREEN, fontsize=10)
fig.tight_layout()
save_fig(fig, "fig_reconstruction.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 -- masked field (r < 0.6), adaptive mesh
# ══════════════════════════════════════════════════════════════════════════════
MASK_R = 0.6
print(f"\nFig 2: masked field (r < {MASK_R}), adaptive mesh...")

t0      = time.perf_counter()
ops_a   = build_operators_adaptive(NX, NX, xmin, xmax, ymin, ymax,
                                    mask_center=(0., 0.), mask_radius=MASK_R,
                                    refine_factor=3, verbose=False)
nodes_a = np.array(ops_a.mesh.nodes)
print(f"  adaptive mesh: {ops_a.n_nodes} nodes ({time.perf_counter()-t0:.1f}s)")

kappa_a  = gaussian_kappa(nodes_a)
g1a, g2a = ops_a.forward(kappa_a)
ns_a     = 0.10 * float(np.std(np.hypot(g1a, g2a)))
g1a_obs  = g1a + rng.normal(0, ns_a, g1a.shape)
g2a_obs  = g2a + rng.normal(0, ns_a, g2a.shape)
r_a      = np.hypot(nodes_a[:, 0], nodes_a[:, 1])
g1a_obs[r_a < MASK_R] = 0.
g2a_obs[r_a < MASK_R] = 0.

t0 = time.perf_counter()
km_a, lam_a, _, result_a = run_map(ops_a, g1a_obs, g2a_obs)
print(f"  MAP: lambda*={lam_a:.3e}  {result_a.n_iter} iters  "
      f"({time.perf_counter()-t0:.1f}s)")

km_ag       = nodes_to_grid(nodes_a, km_a)
kt_ag       = nodes_to_grid(nodes_a, kappa_a)
g1ag, g2ag  = shear_nodes_to_grid(nodes_a, g1a_obs, g2a_obs)
ks_ag       = run_ks_smpy(g1ag, g2ag, smoothing_sigma=1.5)

e_map_a  = l2_rel(km_ag, kt_ag)
e_ks_a   = l2_rel(ks_ag, kt_ag)
improv_a = (e_ks_a - e_map_a) / e_ks_a * 100
print(f"  FEMMI-MAP L2 = {e_map_a:.3f}")
print(f"  KS+DC     L2 = {e_ks_a:.3f}")
print(f"  improvement  = {improv_a:+.1f}%")

theta   = np.linspace(0, 2 * np.pi, 300)
cx_circ = MASK_R * np.cos(theta)
cy_circ = MASK_R * np.sin(theta)
vmax_a  = float(kt_ag.max())

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax in axes:
    ax.plot(cx_circ, cy_circ, color="#ff4444", lw=1.2, ls="--")
panel(axes[0], kt_ag,  r"Truth $\kappa$",               vmin=0, vmax=vmax_a)
panel(axes[1], km_ag,  f"FEMMI-MAP  L2={e_map_a:.3f}",  vmin=0, vmax=vmax_a)
ra   = km_ag - kt_ag
rma  = float(np.percentile(np.abs(ra), 99))
panel(axes[2], ra,     "MAP residual", cmap="RdBu_r", vmin=-rma, vmax=rma)
panel(axes[3], ks_ag,  f"KS+DC  L2={e_ks_a:.3f}",     vmin=0, vmax=vmax_a)
rka  = ks_ag - kt_ag
rmka = float(np.percentile(np.abs(rka), 99))
panel(axes[4], rka,    "KS+DC residual", cmap="RdBu_r", vmin=-rmka, vmax=rmka)

fig.suptitle(
    rf"Masked reconstruction ($r < {MASK_R}$, SMPy KS+DC)  |  "
    r"Mater\'{e}n prior inpaints via $\kappa$ correlation structure  "
    r"|  adaptive mesh, $\lambda^*$ via Morozov",
    color=GREEN, fontsize=10)
fig.tight_layout()
save_fig(fig, "fig_masked.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 -- forward-operator shear convergence
# ══════════════════════════════════════════════════════════════════════════════
#
# Measures ||gamma_h - gamma_ref|| as mesh is refined for fixed kappa_true.
# sigma=1.5 ensures sigma/h >= 1 on all test meshes (asymptotic regime).
# Restricted to deep interior nodes (|x|,|y| < 1.5): shear operators are
# deliberately zeroed on boundary nodes, so those are excluded from comparison.
#
print("\nFig 3: forward-operator shear convergence...")
SIGMA_FIG = 1.5
DEEP_LIM  = 1.5
NX_REF    = 40

print(f"  building reference mesh (nx={NX_REF}, sigma={SIGMA_FIG})...")
ops_ref   = build_operators(NX_REF, NX_REF, xmin, xmax, ymin, ymax, verbose=False)
nodes_ref = np.array(ops_ref.mesh.nodes)
kappa_ref = np.exp(-(nodes_ref[:, 0]**2 + nodes_ref[:, 1]**2) / (2 * SIGMA_FIG**2))
g1_ref, g2_ref = ops_ref.forward(kappa_ref)

int_ref  = np.array(ops_ref.interior, dtype=bool)
xr, yr   = nodes_ref[:, 0], nodes_ref[:, 1]
deep_ref = int_ref & (np.abs(xr) < DEEP_LIM) & (np.abs(yr) < DEEP_LIM)
print(f"  reference: {ops_ref.n_nodes} nodes, {deep_ref.sum()} deep-interior")

mesh_sizes = [8, 10, 14, 18, 24, 32]
h_vals, errs = [], []

for nx in mesh_sizes:
    sigma_over_h = SIGMA_FIG / ((xmax - xmin) / nx)
    print(f"  nx={nx:2d}  sigma/h={sigma_over_h:.1f} ...", end="", flush=True)
    ops_c   = build_operators(nx, nx, xmin, xmax, ymin, ymax, verbose=False)
    nodes_c = np.array(ops_c.mesh.nodes)
    g1_c, g2_c = ops_c.forward(
        np.exp(-(nodes_c[:, 0]**2 + nodes_c[:, 1]**2) / (2 * SIGMA_FIG**2)))

    int_c   = np.array(ops_c.interior, dtype=bool)
    xc, yc  = nodes_c[:, 0], nodes_c[:, 1]
    deep_c  = int_c & (np.abs(xc) < DEEP_LIM) & (np.abs(yc) < DEEP_LIM)

    def interp(src_v):
        g  = griddata(nodes_ref[deep_ref], src_v, nodes_c[deep_c], method="linear")
        nn = griddata(nodes_ref[deep_ref], src_v, nodes_c[deep_c], method="nearest")
        return np.where(np.isfinite(g), g, nn)

    g1_on = interp(g1_ref[deep_ref])
    g2_on = interp(g2_ref[deep_ref])

    num = float(np.sqrt(np.mean((g1_c[deep_c] - g1_on)**2 +
                                (g2_c[deep_c] - g2_on)**2)))
    den = float(np.sqrt(np.mean(g1_on**2 + g2_on**2)))
    err = num / (den if den > 0 else 1.0)
    h   = (xmax - xmin) / nx
    h_vals.append(h); errs.append(err)
    print(f"  h={h:.3f}  L2={err:.4e}")

h_arr = np.array(h_vals)
e_arr = np.array(errs)
mid   = slice(1, -1)
slope = float(np.polyfit(np.log(h_arr[mid]), np.log(e_arr[mid]), 1)[0])
print(f"  fitted gamma convergence rate: O(h^{slope:.2f})  (theory O(h^2))")

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(h_arr, e_arr, "o-", color=GREEN, lw=2.5, ms=8, zorder=5,
          label=rf"P3 FEM-BEM  $\gamma$  (fitted slope {slope:.2f})")
h_ref = np.array([h_arr.min() * 0.75, h_arr.max() * 1.25])
c2    = e_arr[mid][-1] / h_arr[mid][-1]**2
ax.loglog(h_ref, c2 * h_ref**2, "--", color=ORANGE, alpha=0.8, lw=2,
          label=r"$O(h^2)$ -- P3 theory")
ax.set_xlabel(r"Mesh spacing $h = 5/n_x$", fontsize=12)
ax.set_ylabel(r"Normalised $L^2$ shear error (deep interior)", fontsize=11)
ax.set_title(
    rf"P3 FEM-BEM Forward Convergence  ($\sigma={SIGMA_FIG}$, ref $n_x={NX_REF}$)",
    color=GREEN, fontsize=12)
ax.legend(fontsize=11, framealpha=0.3)
ax.grid(True, which="both", alpha=0.4)
fig.tight_layout()
save_fig(fig, "fig_convergence.png")


# ── summary ───────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
Summary
{'='*60}
fig_reconstruction.png  (10% noise, seed=42)
  FEMMI-MAP L2 = {e_map:.3f}
  KS+DC     L2 = {e_ks:.3f}
  improvement  = {improv:+.1f}%

fig_masked.png  (r < {MASK_R}, adaptive mesh)
  FEMMI-MAP L2 = {e_map_a:.3f}
  KS+DC     L2 = {e_ks_a:.3f}
  improvement  = {improv_a:+.1f}%

fig_convergence.png
  fitted rate  = O(h^{slope:.2f})  (theory O(h^2))
{'='*60}
""")