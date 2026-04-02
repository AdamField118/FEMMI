import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from femmi.operators import build_operators_circular

ops   = build_operators_circular(radius=2.5, n_boundary=252)
nodes = np.array(ops.mesh.nodes)
kappa = np.exp(-(nodes[:,0]**2 + nodes[:,1]**2) / (2*0.5**2))
psi   = ops.psi_from_kappa(kappa)
g1, g2 = ops.shear_from_psi(psi)
gmag  = np.sqrt(g1**2 + g2**2)

# gradient of psi: use finite differences on the interpolated grid
xi = np.linspace(-2.5, 2.5, 128)
XX, YY = np.meshgrid(xi, xi)

def to_grid(v):
    g = griddata(nodes, v, (XX, YY), method='linear')
    return np.where(np.isfinite(g), g, 0)

psi_g = to_grid(psi)
dx    = xi[1] - xi[0]
dpsi_x, dpsi_y = np.gradient(psi_g, dx)
grad_mag = np.sqrt(dpsi_x**2 + dpsi_y**2)

panels = [
    (to_grid(kappa), "femmi_kappa", "hot",    False),
    (psi_g,          "femmi_psi",   "viridis", False),
    (grad_mag,       "femmi_grad",  "plasma",  False),
    (to_grid(g1),    "femmi_g1",    "RdBu_r",  True),
    (to_grid(g2),    "femmi_g2",    "RdBu_r",  True),
    (to_grid(gmag),  "femmi_gmag",  "plasma",  False),
]

for data, name, cmap, sym in panels:
    fig, ax = plt.subplots(figsize=(3, 3))
    vmax = np.nanpercentile(np.abs(data), 99)
    vmin = -vmax if sym else 0
    ax.imshow(data, origin='lower', extent=[-2.5,2.5,-2.5,2.5],
              cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.axis('off')
    fig.savefig(f"{name}.png", dpi=150, bbox_inches='tight',
                facecolor='#f5f5f5')
    plt.close()
