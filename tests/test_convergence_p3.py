"""
tests/test_convergence_p3.py
============================
P3 convergence study: validates O(h⁴) convergence for potential ψ.

Run from project root:
    bash ./run.sh tests/test_convergence_p3.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from femmi.p3_mesh_generator import generate_p3_structured_mesh
from femmi.p3_assembly        import solve_poisson_p3


# ============================================================================
# Manufactured Solutions
# ============================================================================

class SinusoidalSolution:
    """ψ = sin(πx)sin(πy)  on [0,1]²  →  κ = -π²sin(πx)sin(πy)"""
    def psi(self, x, y):   return np.sin(np.pi*x) * np.sin(np.pi*y)
    def kappa(self, x, y): return -np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
    def name(self):        return "sin(πx)sin(πy)"


class PolynomialSolution:
    """ψ = (x²-1)(y²-1)  on [-1,1]²  →  κ = x² + y² - 2"""
    def psi(self, x, y):   return (x**2 - 1) * (y**2 - 1)
    def kappa(self, x, y): return x**2 + y**2 - 2
    def name(self):        return "(x²-1)(y²-1)"


class BiquadraticSolution:
    """ψ = x(1-x)y(1-y)  on [0,1]²  →  κ = -y(1-y) - x(1-x)"""
    def psi(self, x, y):   return x*(1-x)*y*(1-y)
    def kappa(self, x, y): return -y*(1-y) - x*(1-x)
    def name(self):        return "x(1-x)y(1-y)"


# ============================================================================
# Convergence study
# ============================================================================

def convergence_study(solution, mesh_sizes, domain=(0, 1, 0, 1)):
    xmin, xmax, ymin, ymax = domain
    print(f"\n{'='*60}")
    print(f"P3 convergence: {solution.name()}")
    print(f"Domain: [{xmin},{xmax}]×[{ymin},{ymax}]")
    print(f"{'='*60}")

    h_vals, L2_vals, Linf_vals, n_dofs = [], [], [], []

    for nx, ny in mesh_sizes:
        mesh  = generate_p3_structured_mesh(nx, ny, xmin, xmax, ymin, ymax)
        nodes = np.array(mesh.nodes)
        kappa = solution.kappa(nodes[:,0], nodes[:,1])
        psi   = solve_poisson_p3(mesh, kappa)
        psi_ex = solution.psi(nodes[:,0], nodes[:,1])

        diff = psi - psi_ex
        L2   = np.sqrt(np.mean(diff**2))
        Linf = np.max(np.abs(diff))
        h    = max((xmax-xmin)/nx, (ymax-ymin)/ny)

        h_vals.append(h); L2_vals.append(L2)
        Linf_vals.append(Linf); n_dofs.append(len(nodes))

    print(f"\n{'h':>10} {'DOFs':>8} {'L² Error':>12} {'Rate':>6} {'L∞ Error':>12} {'Rate':>6}")
    print("-"*60)
    for i in range(len(mesh_sizes)):
        if i == 0:
            print(f"{h_vals[i]:10.4f} {n_dofs[i]:8d} {L2_vals[i]:12.3e}   --   {Linf_vals[i]:12.3e}   --")
        else:
            r_L2   = np.log(L2_vals[i-1]   / L2_vals[i])   / np.log(h_vals[i-1] / h_vals[i])
            r_Linf = np.log(Linf_vals[i-1] / Linf_vals[i]) / np.log(h_vals[i-1] / h_vals[i])
            print(f"{h_vals[i]:10.4f} {n_dofs[i]:8d} {L2_vals[i]:12.3e} {r_L2:6.2f} {Linf_vals[i]:12.3e} {r_Linf:6.2f}")

    avg_rate = np.mean([np.log(L2_vals[i-1]/L2_vals[i]) / np.log(h_vals[i-1]/h_vals[i])
                        for i in range(1, len(mesh_sizes))])
    print(f"\nAverage L² rate: {avg_rate:.2f}  (expected ~4.0 for P3)")
    status = "✅" if avg_rate > 3.5 else ("⚠️ " if avg_rate > 2.5 else "❌")
    print(f"{status} {'O(h⁴) confirmed' if avg_rate > 3.5 else 'below expected rate'}")

    return {'h': h_vals, 'L2': L2_vals, 'Linf': Linf_vals,
            'n_dofs': n_dofs, 'name': solution.name()}


# ============================================================================
# Plotting
# ============================================================================

def plot_convergence(results_list, out='p3_convergence.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors  = ['#ff4444', '#4444ff', '#44ff44']
    markers = ['o', 's', '^']

    for res, c, m in zip(results_list, colors, markers):
        h    = np.array(res['h'],    dtype=float)
        L2   = np.array(res['L2'],   dtype=float)
        Linf = np.array(res['Linf'], dtype=float)
        idx  = np.argsort(h)
        h, L2, Linf = h[idx], L2[idx], Linf[idx]

        ax1.loglog(h, L2,   marker=m, color=c, lw=2, ms=8, label=res['name'])
        ax2.loglog(h, Linf, marker=m, color=c, lw=2, ms=8, label=res['name'])

        h_ref = np.array([h[0], h[-1]])
        for ax, vals in [(ax1, L2), (ax2, Linf)]:
            C = vals[0] / h[0]**4
            ax.loglog(h_ref, C*h_ref**2, 'k--', lw=0.9, alpha=0.4)
            ax.loglog(h_ref, C*h_ref**3, 'k-.', lw=0.9, alpha=0.4)
            ax.loglog(h_ref, C*h_ref**4, 'k-',  lw=1.4, alpha=0.6, label='O(h⁴)' if res is results_list[0] and ax is ax1 else None)

    for ax, title in [(ax1, 'L² Error'), (ax2, 'L∞ Error')]:
        ax.set_xlabel('Mesh size h', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'P3 {title} Convergence', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
        ax.invert_xaxis()

    plt.tight_layout()
    save_path = Path(__file__).resolve().parent.parent / out
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    mesh_sizes = [(4,4), (8,8), (16,16), (32,32)]

    all_results = [
        convergence_study(SinusoidalSolution(),  mesh_sizes, domain=(0,1,0,1)),
        convergence_study(PolynomialSolution(),  mesh_sizes, domain=(-1,1,-1,1)),
        convergence_study(BiquadraticSolution(), mesh_sizes, domain=(0,1,0,1)),
    ]

    plot_convergence(all_results)
    print("\n✅ P3 convergence validation complete")