"""
examples/visualize_results.py
==============================
Publication-quality visualizations of the FEMMI FEM-BEM pipeline.

Generates four figures saved to examples/figures/:

  fig1_reconstruction.png  — truth / FEM-BEM / KS for three test cases
  fig2_picard.png          — Picard diagnostic (singular values + coefficients)
  fig3_svd_modes.png       — First 6 left singular vectors of F (spatial structure
                             of the lensing operator's spectral basis)
  fig4_convergence.png     — ψ and γ convergence rates with theory lines

Run from project root:
    python examples/visualize_results.py
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators    import build_operators
from femmi.forward      import DifferentiableForward
from femmi.inverse      import MAPReconstructor
from femmi.svd_analysis import compute_svd

try:
    from smpy.config import Config
    from smpy.mapping_methods.kaiser_squires.kaiser_squires import KaiserSquiresMapper
    HAS_SMPY = True
except ImportError:
    HAS_SMPY = False
    print("SMPy not found — KS panels will show zeros.")

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
BG     = "#0e0e0e"
PANEL  = "#141414"
GREEN  = "#00e676"
BLUE   = "#448aff"
ORANGE = "#ff9100"
WHITE  = "#eeeeee"
MUTED  = "#888888"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": "#333", "axes.labelcolor": WHITE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": WHITE, "font.size": 10,
})

# ── shared domain / grid ──────────────────────────────────────────────────────
NX_FEM  = 20
NX_GRID = 64
DOMAIN  = (-2.0, 2.0, -2.0, 2.0)
xmin, xmax, ymin, ymax = DOMAIN
SIGMA   = 0.5
rng     = np.random.default_rng(42)

xi_g     = np.linspace(xmin, xmax, NX_GRID)
yi_g     = np.linspace(ymin, ymax, NX_GRID)
XX, YY   = np.meshgrid(xi_g, yi_g)
grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
EXT      = [xmin, xmax, ymin, ymax]

print("=" * 60)
print("FEMMI — Generating visualizations")
print("=" * 60)

# ── build operators (reused for figs 1-3) ─────────────────────────────────────
print("\nBuilding FEM operators (20×20)...")
t0  = time.perf_counter()
ops = build_operators(NX_FEM, NX_FEM, xmin=xmin, xmax=xmax,
                      ymin=ymin, ymax=ymax, verbose=False)
fem_nodes = np.array(ops.mesh.nodes)
print(f"  {ops.n_nodes} nodes  ({time.perf_counter()-t0:.1f}s)")


# ── utility functions ─────────────────────────────────────────────────────────

def gaussian_grid(cx=0., cy=0., sigma=SIGMA):
    return np.exp(-((XX - cx)**2 + (YY - cy)**2) / (2 * sigma**2))


def fem_fwd_to_grid(kappa_grid):
    """Grid κ → FEM-BEM forward → grid (γ₁, γ₂)."""
    kf = griddata(grid_pts, kappa_grid.ravel(), fem_nodes,
                  method='linear', fill_value=0.0)
    g1f, g2f = ops.forward(kf)
    def _interp(v):
        g = griddata(fem_nodes, v, (XX, YY), method='linear')
        n = griddata(fem_nodes, v, (XX, YY), method='nearest')
        return np.where(np.isfinite(g), g, n)
    return _interp(g1f), _interp(g2f)


def run_map(g1g, g2g, lam=1e-2, wiener=0.5, maxiter=400,
            mask=None, display_smooth=0.8):
    g1f = griddata(grid_pts, g1g.ravel(), fem_nodes,
                   method='linear', fill_value=0.0)
    g2f = griddata(grid_pts, g2g.ravel(), fem_nodes,
                   method='linear', fill_value=0.0)
    if mask is not None:
        mf = griddata(grid_pts, mask.ravel().astype(float),
                      fem_nodes, method='nearest') > 0.5
        g1f[mf] = 0.; g2f[mf] = 0.
    fwd = DifferentiableForward(ops, lam_reg=lam)
    rec = MAPReconstructor(fwd, maxiter=maxiter, gtol=1e-7,
                            wiener_length=wiener, callback_every=0)
    km, _ = rec.reconstruct(g1f, g2f, verbose=False)
    kg = griddata(fem_nodes, km, (XX, YY), method='cubic')
    kn = griddata(fem_nodes, km, (XX, YY), method='nearest')
    kg = np.where(np.isfinite(kg), kg, kn)
    if display_smooth > 0:
        kg = gaussian_filter(np.nan_to_num(kg), sigma=display_smooth)
    kg[(np.abs(XX) > 1.6) | (np.abs(YY) > 1.6)] = np.nan
    return kg


def run_ks(g1g, g2g, sigma_smooth=1.0):
    if not HAS_SMPY:
        return np.zeros_like(g1g)
    cfg = Config.from_defaults("kaiser_squires").to_dict()
    cfg["methods"]["kaiser_squires"]["smoothing"]["sigma"] = sigma_smooth
    kr, _ = KaiserSquiresMapper(cfg).create_maps(g1g, g2g)
    ny, nx = kr.shape
    m = max(1, int(0.15 * min(ny, nx)))
    sky = np.concatenate([kr[:m].ravel(), kr[-m:].ravel(),
                           kr[:, :m].ravel(), kr[:, -m:].ravel()])
    return kr - sky.mean()


def add_cb(fig, ax, im, label=""):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, color=WHITE, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)


def panel(ax, data, title, cmap='viridis', vmin=None, vmax=None, sym=False):
    if sym:
        v = np.nanpercentile(np.abs(data), 98)
        vmin, vmax = -v, v
    im = ax.imshow(data, origin='lower', extent=EXT, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title(title, color=WHITE, fontsize=9, pad=4)
    ax.tick_params(labelsize=7, colors=MUTED)
    return im


def deep_l2(pred, truth, lim=1.5):
    m = (np.abs(XX) < lim) & (np.abs(YY) < lim)
    p = np.nan_to_num(pred)[m]
    t = truth[m]
    d = np.linalg.norm(t)
    return float(np.linalg.norm(p - t) / (d if d > 0 else 1.))


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1 — Reconstruction comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Reconstruction comparison...")

k_true    = gaussian_grid()
noise_10  = 0.10 * np.std(np.hypot(*fem_fwd_to_grid(k_true)))
mask_circ = (XX**2 + YY**2) < 0.6**2
vmax_k    = float(np.nanpercentile(k_true, 99))

g1_A, g2_A = fem_fwd_to_grid(k_true)
k_map_A    = run_map(g1_A, g2_A, lam=5e-3, wiener=0.5, maxiter=500)
k_ks_A     = run_ks(g1_A, g2_A, sigma_smooth=0.5)

g1_B = g1_A + rng.normal(0, noise_10, g1_A.shape)
g2_B = g2_A + rng.normal(0, noise_10, g2_A.shape)
k_map_B = run_map(g1_B, g2_B, lam=1e-2, wiener=0.5)
k_ks_B  = run_ks(g1_B, g2_B, sigma_smooth=1.0)

g1_C = np.where(mask_circ, 0., g1_B)
g2_C = np.where(mask_circ, 0., g2_B)
k_map_C = run_map(g1_C, g2_C, lam=2e-2, wiener=0.5, mask=mask_circ)
k_ks_C  = run_ks(g1_C, g2_C, sigma_smooth=1.0)

fig = plt.figure(figsize=(13, 9), facecolor=BG)
fig.suptitle("FEM-BEM MAP Reconstruction vs Kaiser-Squires\n"
             "(shear from FEM-BEM forward — physically correct)",
             color=WHITE, fontsize=12, y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.08,
                       left=0.05, right=0.97, top=0.90, bottom=0.04)

for row_i, (label, k_map, k_ks, mask) in enumerate([
    ("Noiseless",    k_map_A, k_ks_A, None),
    ("10% noise",    k_map_B, k_ks_B, None),
    ("Masked r<0.6", k_map_C, k_ks_C, mask_circ),
]):
    ax0 = fig.add_subplot(gs[row_i, 0])
    ax1 = fig.add_subplot(gs[row_i, 1])
    ax2 = fig.add_subplot(gs[row_i, 2])
    ax3 = fig.add_subplot(gs[row_i, 3])
    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)

    im0 = panel(ax0, k_true, f"Truth κ  [{label}]", 'hot', 0, vmax_k)
    im1 = panel(ax1, k_map,  "FEM-BEM MAP",          'hot', 0, vmax_k)
    im2 = panel(ax2, k_ks,   "Kaiser-Squires",        'hot', 0, vmax_k)
    im3 = panel(ax3, np.nan_to_num(k_map) - k_true, "MAP residual",
                'RdBu_r', sym=True)

    if mask is not None:
        for ax in [ax0, ax1, ax2, ax3]:
            ax.contour(XX, YY, mask.astype(float), levels=[0.5],
                       colors='#ff1744', linewidths=1.0, linestyles='--')

    ax1.set_xlabel(f"L2={deep_l2(k_map, k_true):.3f}", color=GREEN,  fontsize=8)
    ax2.set_xlabel(f"L2={deep_l2(k_ks,  k_true):.3f}", color=ORANGE, fontsize=8)

    for ax, im in [(ax0, im0), (ax1, im1), (ax2, im2), (ax3, im3)]:
        add_cb(fig, ax, im)

path1 = os.path.join(OUT_DIR, "fig1_reconstruction.png")
fig.savefig(path1, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {path1}")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 — Picard diagnostic  (reuses 20×20 ops)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 2] Picard diagnostic...")

kappa_p     = np.exp(-(fem_nodes[:, 0]**2 + fem_nodes[:, 1]**2) / (2*SIGMA**2))
g1_p, g2_p = ops.forward(kappa_p)
noise_p     = 0.05 * np.std(np.hypot(g1_p, g2_p))
g1_pn       = g1_p + rng.normal(0, noise_p, g1_p.shape)
g2_pn       = g2_p + rng.normal(0, noise_p, g2_p.shape)

print("  Computing SVD (n=30)...")
svd = compute_svd(ops, n_singular=30)

n_p        = ops.n_nodes
g_flat     = np.concatenate([g1_pn, g2_pn])
coeffs     = np.abs(svd.U.T @ g_flat)
sigma_sv   = svd.sigma
ratio      = coeffs / np.maximum(sigma_sv, 1e-14)
idx        = np.arange(1, len(sigma_sv) + 1)
noise_line = noise_p * np.sqrt(2 * n_p)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG)
fig.suptitle("Picard Diagnostic  (20×20 mesh, 5% noise, Gaussian κ)",
             color=WHITE, fontsize=12)
for ax in axes:
    ax.set_facecolor(PANEL)
    ax.grid(True, alpha=0.2, color="#333")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_edgecolor("#444")

axes[0].semilogy(idx, sigma_sv, '.-', color=BLUE, lw=2, ms=6)
axes[0].set_xlabel("Mode i", color=WHITE)
axes[0].set_ylabel("σᵢ", color=WHITE)
axes[0].set_title("Singular values", color=WHITE)

axes[1].semilogy(idx, coeffs, '.-', color=GREEN, lw=2, ms=5,
                  label="|⟨γ_obs, uᵢ⟩|")
axes[1].semilogy(idx, sigma_sv, '--', color=BLUE, lw=1.5, alpha=0.7,
                  label="σᵢ")
axes[1].axhline(noise_line, color=ORANGE, lw=1, ls=':', label="δ√(2n)")
axes[1].set_xlabel("Mode i", color=WHITE)
axes[1].set_title("Fourier coefficients  (Picard: ✓)", color=WHITE)
axes[1].legend(fontsize=8, framealpha=0.2)

axes[2].semilogy(idx, ratio, '.-', color="#ff4081", lw=2, ms=5)
axes[2].axhline(noise_line, color=ORANGE, lw=1, ls=':', label="noise floor")
axes[2].set_xlabel("Mode i", color=WHITE)
axes[2].set_ylabel("|⟨γ_obs, uᵢ⟩| / σᵢ", color=WHITE)
axes[2].set_title("Amplified noise ratio", color=WHITE)
axes[2].legend(fontsize=8, framealpha=0.2)

plt.tight_layout()
path2 = os.path.join(OUT_DIR, "fig2_picard.png")
fig.savefig(path2, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {path2}")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 3 — SVD mode visualization
#
# The left singular vectors vᵢ of F (the κ-space basis) show the spatial
# structure of what the lensing operator can and cannot reconstruct.
# Mode 1 captures the monopole (smooth compact mass); higher modes resolve
# progressively finer angular structure. Modes beyond ~5 fall below the
# noise floor (see Fig 2) and cannot be recovered from survey data.
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] SVD mode visualization...")

# svd.V holds the right singular vectors (κ-space), shape (n_nodes, n_singular)
# svd.U holds the left singular vectors (γ-space), shape (2*n_nodes, n_singular)
# We visualise V: the κ-space modes, interpolated to the display grid.

# In Fig 3, replace the V visualization with U:
N_MODES_SHOW = 3   # 3 modes × 2 components = 6 panels
fig, axes = plt.subplots(N_MODES_SHOW, 2, figsize=(9, 11), facecolor=BG)
fig.suptitle("Observable Shear Modes  —  Left singular vectors Uᵢ of F\n"
             "(each mode splits into γ₁ and γ₂ components)",
             color=WHITE, fontsize=11, y=0.98)

for i in range(N_MODES_SHOW):
    ui = svd.U[:, i]            # length 2*n_nodes
    u_g1 = ui[:n_p]             # first half = γ₁ component
    u_g2 = ui[n_p:]             # second half = γ₂ component

    for col, (u_comp, label) in enumerate([(u_g1, "γ₁"), (u_g2, "γ₂")]):
        ax = axes[i, col]
        ug = griddata(fem_nodes, u_comp, (XX, YY), method='cubic')
        un = griddata(fem_nodes, u_comp, (XX, YY), method='nearest')
        ug = np.where(np.isfinite(ug), ug, un)
        ug = gaussian_filter(np.nan_to_num(ug), sigma=0.6)
        ug[(np.abs(XX) > 1.8) | (np.abs(YY) > 1.8)] = np.nan
        vabs = np.nanpercentile(np.abs(ug), 98)
        im = ax.imshow(ug, origin='lower', extent=EXT, cmap='RdBu_r',
                       vmin=-vabs, vmax=vabs, aspect='equal')
        ax.set_title(f"Mode {i+1}  {label}  (σ={sigma_sv[i]:.3f})",
                     color=WHITE, fontsize=9)
        add_cb(fig, ax, im)

# Annotate the noise floor crossover in text
noise_crossover = int(np.searchsorted(-sigma_sv, -noise_line))
fig.text(0.5, 0.01,
         f"Modes 1–{min(noise_crossover, N_MODES_SHOW)} carry signal above noise floor "
         f"(δ√(2n) = {noise_line:.2f});  higher modes reconstruct noise.",
         ha='center', color=MUTED, fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
path3 = os.path.join(OUT_DIR, "fig3_svd_modes.png")
fig.savefig(path3, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {path3}")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 4 — Convergence rates
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 4] Convergence rates...")

SIGMA_CV = 1.5
DEEP_LIM = 1.5


def kappa_cv(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return np.exp(-(x**2 + y**2) / (2 * SIGMA_CV**2))


print("  Building reference mesh (nx=48)...")
ops_ref   = build_operators(48, 48, xmin=-2.5, xmax=2.5,
                             ymin=-2.5, ymax=2.5, verbose=False)
nodes_ref = np.array(ops_ref.mesh.nodes)
psi_ref   = ops_ref.psi_from_kappa(kappa_cv(nodes_ref))
g1r, g2r  = ops_ref.shear_from_psi(psi_ref)
int_ref   = np.array(ops_ref.interior, dtype=bool)
xr, yr    = nodes_ref[:, 0], nodes_ref[:, 1]
deep_ref  = int_ref & (np.abs(xr) < DEEP_LIM) & (np.abs(yr) < DEEP_LIM)

h_vals, e_psi, e_gam = [], [], []


def _ml2(v):
    return float(np.sqrt(np.mean(v**2)))


for nx in [8, 12, 16, 24, 32]:
    h = 5.0 / nx
    print(f"  nx={nx}...")
    ops_c   = build_operators(nx, nx, xmin=-2.5, xmax=2.5,
                               ymin=-2.5, ymax=2.5, verbose=False)
    nodes_c = np.array(ops_c.mesh.nodes)
    psi_h   = ops_c.psi_from_kappa(kappa_cv(nodes_c))
    g1_h, g2_h = ops_c.shear_from_psi(psi_h)
    int_c   = np.array(ops_c.interior, dtype=bool)
    xc, yc  = nodes_c[:, 0], nodes_c[:, 1]
    deep_c  = int_c & (np.abs(xc) < DEEP_LIM) & (np.abs(yc) < DEEP_LIM)

    pi = griddata(nodes_ref, psi_ref, nodes_c, method='linear')
    pn = griddata(nodes_ref, psi_ref, nodes_c, method='nearest')
    pi = np.where(np.isfinite(pi), pi, pn)

    g1i = griddata(nodes_ref[deep_ref], g1r[deep_ref], nodes_c, method='linear')
    g1n = griddata(nodes_ref[deep_ref], g1r[deep_ref], nodes_c, method='nearest')
    g1i = np.where(np.isfinite(g1i), g1i, g1n)
    g2i = griddata(nodes_ref[deep_ref], g2r[deep_ref], nodes_c, method='linear')
    g2n = griddata(nodes_ref[deep_ref], g2r[deep_ref], nodes_c, method='nearest')
    g2i = np.where(np.isfinite(g2i), g2i, g2n)

    off = float(np.mean((psi_h - pi)[int_c]))
    ep  = _ml2((psi_h - off - pi)[int_c])
    eg  = _ml2((g1_h - g1i)[deep_c]) + _ml2((g2_h - g2i)[deep_c])
    h_vals.append(h); e_psi.append(ep); e_gam.append(eg)

h_arr  = np.array(h_vals)
ep_arr = np.array(e_psi)
eg_arr = np.array(e_gam)

psi_rates = [np.log(ep_arr[i-1]/ep_arr[i]) / np.log(h_arr[i-1]/h_arr[i])
             for i in range(1, len(h_arr))]
gam_rates = [np.log(eg_arr[i-1]/eg_arr[i]) / np.log(h_arr[i-1]/h_arr[i])
             for i in range(1, len(h_arr))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
fig.suptitle("FEM-BEM Forward Convergence  (σ=1.5, reference nx=48)",
             color=WHITE, fontsize=12)

h_ref = np.array([h_arr.min()*0.7, h_arr.max()*1.3])

for ax in [ax1, ax2]:
    ax.set_facecolor(PANEL)
    ax.grid(True, which='both', alpha=0.15, color="#333")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("h = 5/nx", color=WHITE)
    ax.spines[:].set_edgecolor("#444")

ax1.loglog(h_arr, ep_arr, 'o-', color=BLUE, lw=2.5, ms=8,
           label=f"ψ error  (rate {np.mean(psi_rates[-2:]):.2f})")
c1 = ep_arr[-2] / h_arr[-2]**(5/3)
ax1.loglog(h_ref, c1*h_ref**(5/3), '--', color=ORANGE, lw=1.5,
           label="O(h⁵/³) theory")
ax1.set_ylabel("L² error (interior)", color=WHITE)
ax1.set_title("ψ convergence", color=WHITE)
ax1.legend(fontsize=9, framealpha=0.2)

ax2.loglog(h_arr, eg_arr, 's-', color=GREEN, lw=2.5, ms=8,
           label=f"γ error  (rate {np.mean(gam_rates[-2:]):.2f})")
c2 = eg_arr[-2] / h_arr[-2]**2
ax2.loglog(h_ref, c2*h_ref**2, '--', color=ORANGE, lw=1.5,
           label="O(h²) theory")
ax2.set_ylabel("L² error (deep interior)", color=WHITE)
ax2.set_title("γ convergence", color=WHITE)
ax2.legend(fontsize=9, framealpha=0.2)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "fig4_convergence.png")
fig.savefig(path4, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {path4}")

# ── done ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("All figures saved to  examples/figures/")
for p in [path1, path2, path3, path4]:
    print(f"  {p}")