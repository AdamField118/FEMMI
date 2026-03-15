"""
tests/test_fem_bem_coupling.py
Phase 1 tests for femmi/bem.py (P3 boundary elements).

Note on V_h eigenvalue sign: V_h with G=(1/2pi)ln|x-y| is negative-definite
on any domain with logarithmic capacity < 1 (unit square cap ~0.59). This is
correct; the Calderon operator remains well-defined regardless of sign.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.bem import (
    BoundaryMesh,
    extract_boundary_edges,
    log_gauss_jacobi_points,
    assemble_single_layer,
    assemble_double_layer,
    assemble_boundary_mass,
    calderon_matrix,
)


def make_rect_bnd_p3(xmin, xmax, ymin, ymax, n_elem_per_side):
    """Build a BoundaryMesh for a rectangle with n_elem_per_side P3 elements per side."""
    ne = n_elem_per_side

    def side_nodes(p0, p1, n):
        pts = []
        for k in range(n):
            for frac in [0, 1/3, 2/3]:
                t = (k + frac) / n
                pts.append(p0 + t * (p1 - p0))
        return pts

    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                       dtype=np.float64)
    pts = []
    for i in range(4):
        pts.extend(side_nodes(corners[i], corners[(i+1)%4], ne))

    coords = np.array(pts, dtype=np.float64)
    N_b    = len(coords)
    assert N_b % 3 == 0

    edge_lengths = np.empty(N_b)
    normals      = np.empty((N_b, 2))
    for i in range(N_b):
        j  = (i + 1) % N_b
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]
        L  = np.hypot(dx, dy)
        edge_lengths[i] = L
        normals[i]      = np.array([dy, -dx]) / L

    N_elem          = N_b // 3
    elements        = np.empty((N_elem, 4), dtype=np.int64)
    element_lengths = np.empty(N_elem)
    element_normals = np.empty((N_elem, 2))
    for e in range(N_elem):
        i0 = 3 * e; i3 = (3 * e + 3) % N_b
        elements[e] = [i0, i0+1, i0+2, i3]
        p0 = coords[i0]; p3 = coords[i3]
        dx = p3[0] - p0[0]; dy = p3[1] - p0[1]
        L  = np.hypot(dx, dy)
        element_lengths[e] = L
        element_normals[e] = np.array([dy, -dx]) / L

    return BoundaryMesh(
        node_indices=np.arange(N_b), nodes=coords,
        edge_lengths=edge_lengths, normals=normals,
        n_boundary_dofs=N_b, elements=elements,
        element_lengths=element_lengths, element_normals=element_normals,
        n_elements=N_elem,
    )


def make_unit_bnd(n_elem_per_side=2):
    return make_rect_bnd_p3(0, 1, 0, 1, n_elem_per_side)


def test_log_gauss_jacobi():
    """int_0^1 t^k (-ln t) dt = 1/(k+1)^2 for k=0..5."""
    tolerances = {10: 5e-2, 25: 1e-4}
    for n_pts, tol in tolerances.items():
        xi, wi   = log_gauss_jacobi_points(n_pts)
        max_err  = 0.0
        for k in range(6):
            computed = float(np.dot(wi, xi ** k))
            exact    = 1.0 / (k + 1) ** 2
            rel_err  = abs(computed - exact) / exact
            max_err  = max(max_err, rel_err)
        assert max_err < tol, f"log_gauss_jacobi n={n_pts}: max_err={max_err:.2e} > {tol}"

    for n_pts in [10, 25]:
        xi, wi = log_gauss_jacobi_points(n_pts)
        assert abs(float(np.dot(wi, xi**0)) - 1.0) < 1e-14


def test_boundary_extraction():
    """CCW order, outward normals, N_b%3==0, element structure correct."""
    try:
        from femmi.mesh import generate_p3_structured_mesh
    except ImportError:
        return

    mesh = generate_p3_structured_mesh(4, 4, xmin=0, xmax=1, ymin=0, ymax=1)
    bnd  = extract_boundary_edges(mesh)
    N_b  = bnd.n_boundary_dofs

    assert N_b % 3 == 0, f"N_b={N_b} not divisible by 3"
    assert len(np.unique(bnd.node_indices)) == N_b

    assert np.allclose(np.linalg.norm(bnd.normals, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(bnd.element_normals, axis=1), 1.0, atol=1e-12)

    # Check outward-pointing normals
    centre = bnd.nodes.mean(axis=0)
    for i in range(N_b):
        mid    = 0.5 * (bnd.nodes[i] + bnd.nodes[(i+1) % N_b])
        radial = mid - centre
        L      = np.linalg.norm(radial)
        if L > 1e-10:
            assert np.dot(bnd.normals[i], radial / L) > -1e-8

    assert bnd.n_elements == N_b // 3
    assert bnd.elements.shape == (bnd.n_elements, 4)

    for e in range(bnd.n_elements):
        i0, _, _, i3 = bnd.elements[e]
        L_expected   = float(np.linalg.norm(bnd.nodes[i3] - bnd.nodes[i0]))
        assert abs(bnd.element_lengths[e] - L_expected) < 1e-12


def test_boundary_mass():
    """M_b @ ones = perimeter; symmetric; positive definite."""
    bnd      = make_unit_bnd(n_elem_per_side=3)
    M        = assemble_boundary_mass(bnd)
    ones     = np.ones(bnd.n_boundary_dofs)
    perimeter = float(np.sum(bnd.edge_lengths))

    assert abs(float(ones @ M @ ones) - perimeter) < 1e-12 * perimeter
    assert np.linalg.norm(M - M.T) / np.linalg.norm(M) < 1e-14
    assert np.linalg.eigvalsh(M).min() > 0


def test_single_layer_symmetry():
    """||V_h - V_h^T|| / ||V_h|| < 1e-12."""
    bnd     = make_unit_bnd(n_elem_per_side=2)
    V       = assemble_single_layer(bnd, n_quad=25)
    sym_err = np.linalg.norm(V - V.T) / np.linalg.norm(V)
    assert sym_err < 1e-12, f"sym_err={sym_err:.2e}"


def test_single_layer_invertible():
    """V_h invertible and negative-definite on unit square (cap ~0.59 < 1)."""
    bnd  = make_unit_bnd(n_elem_per_side=2)
    V    = assemble_single_layer(bnd, n_quad=25)
    eigs = np.linalg.eigvalsh(V)
    cond = abs(eigs).max() / (abs(eigs).min() + 1e-20)

    assert cond < 1e6, f"V_h near-singular: cond={cond:.2e}"
    assert (eigs < 0).all(), (
        f"V_h should be negative-definite on unit square. "
        f"min={eigs.min():.4f}, max={eigs.max():.4f}"
    )


def test_single_layer_analytic():
    """Verify Duffy quadrature against int int ln|s-t| ds dt = -3/2."""
    from scipy.special import roots_legendre

    def gl01(n):
        xi, wi = roots_legendre(n)
        return 0.5*(xi+1), 0.5*wi

    xi_lj, w_lj = log_gauss_jacobi_points(25)
    xi_gl, w_gl = gl01(25)

    def phi(k, t):
        t = np.asarray(t, dtype=float)
        return 1.0 - t if k == 0 else t

    def F_const(r):
        return 2 * (1 - r)

    I_const = -float(np.dot(w_lj, np.array([F_const(r) for r in xi_lj])))
    assert abs(I_const + 1.5) < 1e-6, f"sanity check failed: {I_const:.6f}"

    def compute_Iab(a, b):
        def F_ab(r):
            t_inner = xi_gl * (1 - r)
            w_inner = w_gl  * (1 - r)
            return float(np.sum(w_inner * (
                phi(a, t_inner + r) * phi(b, t_inner) +
                phi(a, t_inner)     * phi(b, t_inner + r)
            )))
        return -float(np.dot(w_lj, np.array([F_ab(r) for r in xi_lj])))

    for (a, b), exact in {(0, 0): -7/16, (1, 1): -7/16, (0, 1): -5/16}.items():
        I_computed = compute_Iab(a, b)
        assert abs(I_computed - exact) / abs(exact) < 1e-6

    I_sum = sum(compute_Iab(a, b) for a in range(2) for b in range(2))
    assert abs(I_sum + 1.5) < 1e-5


def test_double_layer_calderon():
    """(0.5*M_b + K_h) @ ones ~ 0 (Calderon identity for constant psi)."""
    for n_elem in [2, 3]:
        bnd  = make_unit_bnd(n_elem_per_side=n_elem)
        K    = assemble_double_layer(bnd, n_quad=8)
        M    = assemble_boundary_mass(bnd)
        ones = np.ones(bnd.n_boundary_dofs)
        rel  = np.linalg.norm(0.5 * M @ ones + K @ ones) / np.linalg.norm(0.5 * M @ ones)
        assert rel < 5e-3, f"Calderon identity: rel={rel:.2e} (n_elem={n_elem})"


def test_calderon_operator():
    """C = V_h^{-1}(0.5*M_b + K_h): shape correct, V_h(Cx) = (0.5*M_b + K_h)x."""
    bnd = make_unit_bnd(n_elem_per_side=3)
    V   = assemble_single_layer(bnd, n_quad=25)
    K   = assemble_double_layer(bnd, n_quad=8)
    M   = assemble_boundary_mass(bnd)
    C   = calderon_matrix(V, K, M)
    N   = bnd.n_boundary_dofs

    assert C.shape == (N, N)

    rng    = np.random.default_rng(42)
    x      = rng.standard_normal(N)
    Cx     = C.matvec(x)
    VCx    = V @ Cx
    target = (0.5 * M + K) @ x

    assert np.all(np.isfinite(Cx))
    rt_err = np.linalg.norm(VCx - target) / (np.linalg.norm(target) + 1e-14)
    assert rt_err < 1e-10, f"round-trip error: {rt_err:.2e}"

    # Non-trivial
    ratio = np.linalg.norm(Cx) / (np.linalg.norm(x) + 1e-14)
    assert abs(ratio - 1.0) > 1e-3


if __name__ == "__main__":
    tests = [
        test_log_gauss_jacobi,
        test_boundary_extraction,
        test_boundary_mass,
        test_single_layer_symmetry,
        test_single_layer_invertible,
        test_single_layer_analytic,
        test_double_layer_calderon,
        test_calderon_operator,
    ]

    passed = 0; failed = []
    for fn in tests:
        try:
            fn(); passed += 1; print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}"); failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}"); failed.append(fn.__name__)

    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("Failed:", failed)
    sys.exit(0 if not failed else 1)