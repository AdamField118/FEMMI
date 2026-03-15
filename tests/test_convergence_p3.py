"""
tests/test_convergence_p3.py
P3 convergence study for the Poisson solve: validates O(h^4) L2 convergence.

Run:
    python tests/test_convergence_p3.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from femmi.mesh     import generate_p3_structured_mesh
from femmi.assembly import solve_poisson_p3


class SinusoidalSolution:
    """psi = sin(pi*x)*sin(pi*y) on [0,1]^2  =>  kappa = -pi^2 * psi"""
    def psi(self, x, y):   return np.sin(np.pi*x) * np.sin(np.pi*y)
    def kappa(self, x, y): return -np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
    def name(self):        return "sin(pi*x)*sin(pi*y)"


class PolynomialSolution:
    """psi = (x^2-1)(y^2-1) on [-1,1]^2  =>  kappa = x^2 + y^2 - 2"""
    def psi(self, x, y):   return (x**2 - 1) * (y**2 - 1)
    def kappa(self, x, y): return x**2 + y**2 - 2
    def name(self):        return "(x^2-1)(y^2-1)"


class BiquadraticSolution:
    """psi = x(1-x)y(1-y) on [0,1]^2  =>  kappa = -y(1-y) - x(1-x)"""
    def psi(self, x, y):   return x*(1-x)*y*(1-y)
    def kappa(self, x, y): return -y*(1-y) - x*(1-x)
    def name(self):        return "x(1-x)y(1-y)"


def convergence_study(solution, mesh_sizes, domain=(0, 1, 0, 1)):
    xmin, xmax, ymin, ymax = domain
    print(f"\nP3 convergence: {solution.name()}")

    h_vals, L2_vals, Linf_vals, n_dofs = [], [], [], []

    for nx, ny in mesh_sizes:
        mesh     = generate_p3_structured_mesh(nx, ny, xmin, xmax, ymin, ymax)
        nodes    = np.array(mesh.nodes)
        kappa    = solution.kappa(nodes[:,0], nodes[:,1])
        psi      = solve_poisson_p3(mesh, kappa)
        psi_ex   = solution.psi(nodes[:,0], nodes[:,1])
        diff     = psi - psi_ex

        h_vals.append(max((xmax-xmin)/nx, (ymax-ymin)/ny))
        L2_vals.append(np.sqrt(np.mean(diff**2)))
        Linf_vals.append(np.max(np.abs(diff)))
        n_dofs.append(len(nodes))

    print(f"  {'h':>10} {'DOFs':>8} {'L2 error':>12} {'rate':>6} {'Linf error':>12} {'rate':>6}")
    for i in range(len(mesh_sizes)):
        if i == 0:
            print(f"  {h_vals[i]:10.4f} {n_dofs[i]:8d} {L2_vals[i]:12.3e}    -   {Linf_vals[i]:12.3e}    -")
        else:
            r_L2   = np.log(L2_vals[i-1]   / L2_vals[i])   / np.log(h_vals[i-1] / h_vals[i])
            r_Linf = np.log(Linf_vals[i-1] / Linf_vals[i]) / np.log(h_vals[i-1] / h_vals[i])
            print(f"  {h_vals[i]:10.4f} {n_dofs[i]:8d} {L2_vals[i]:12.3e} {r_L2:6.2f} {Linf_vals[i]:12.3e} {r_Linf:6.2f}")

    avg_rate = np.mean([np.log(L2_vals[i-1]/L2_vals[i]) / np.log(h_vals[i-1]/h_vals[i])
                        for i in range(1, len(mesh_sizes))])
    ok = avg_rate > 3.5
    print(f"  avg L2 rate: {avg_rate:.2f}  (expected ~4.0)  {'PASS' if ok else 'FAIL'}")

    return {'h': h_vals, 'L2': L2_vals, 'Linf': Linf_vals, 'n_dofs': n_dofs, 'name': solution.name()}


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
        C     = L2[0] / h[0]**4
        ax1.loglog(h_ref, C*h_ref**4, 'k-',  lw=1.4, alpha=0.6,
                   label='O(h^4)' if res is results_list[0] else None)
        ax2.loglog(h_ref, (Linf[0]/h[0]**4)*h_ref**4, 'k-', lw=1.4, alpha=0.6)

    for ax, title in [(ax1, 'L2 Error'), (ax2, 'Linf Error')]:
        ax.set_xlabel('h'); ax.set_ylabel(title)
        ax.set_title(f'P3 {title} Convergence')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
        ax.invert_xaxis()

    plt.tight_layout()
    save_path = Path(__file__).resolve().parent.parent / out
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    mesh_sizes = [(4,4), (8,8), (16,16), (32,32)]

    all_results = [
        convergence_study(SinusoidalSolution(),  mesh_sizes, domain=(0,1,0,1)),
        convergence_study(PolynomialSolution(),  mesh_sizes, domain=(-1,1,-1,1)),
        convergence_study(BiquadraticSolution(), mesh_sizes, domain=(0,1,0,1)),
    ]

    plot_convergence(all_results)
    print("P3 convergence validation complete")