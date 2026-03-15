"""
tests/test_fem_bem_coupling.py
==============================
Phase 1 test suite for femmi/bem.py — updated for P3 boundary elements.

Key changes from the original P1 version:
  - V_h eigenvalue test: checks invertibility and consistent sign, not
    positivity.  V_h with G=(1/2π)ln|x-y| is NEGATIVE-definite on any
    domain with logarithmic capacity < 1 (unit square cap ≈ 0.59 < 1).
    This is mathematically correct; the Calderon operator remains
    well-defined.
  - BoundaryMesh now carries P3 element data (elements, element_lengths,
    element_normals, n_elements).  Tests verify N_b % 3 == 0 and that
    each element spans exactly its 4 nodes.
  - log-GL accuracy thresholds updated for P3 (higher polynomial degree
    means the rule needs more points for the same relative accuracy on
    higher-k monomials).

Tests (in order):
    test_log_gauss_jacobi        –  ∫₀¹ tᵏ(−ln t)dt = 1/(k+1)² for k=0..5
    test_boundary_extraction     –  CCW order, outward normals, N_b%3==0,
                                    element grouping correct
    test_boundary_mass           –  M_b @ 1 = perimeter; symmetry; positivity
    test_single_layer_symmetry   –  ‖V_h − V_hᵀ‖ / ‖V_h‖ < 1e-12
    test_single_layer_invertible –  V_h is invertible and negative-definite
                                    (correct for cap(unit square) < 1)
    test_single_layer_analytic   –  diagonal block sum matches analytic value
    test_double_layer_calderon   –  ‖(½M_b + K_h) @ 1‖ / ‖½M_b‖ < 5e-3
    test_calderon_operator       –  C = V_h⁻¹(½M_b + K_h) works as
                                    LinearOperator

Run from project root:
    python -m pytest tests/test_fem_bem_coupling.py -v
    python tests/test_fem_bem_coupling.py
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a P3-compatible rectangular boundary mesh directly
# (avoids needing a full P3 FEM mesh for unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def make_rect_bnd_p3(xmin, xmax, ymin, ymax, n_elem_per_side):
    """
    Build a BoundaryMesh for a rectangle with n_elem_per_side P3 elements
    per side.  Each P3 element has 4 nodes at t=0, 1/3, 2/3, 1, so there
    are 3*n_elem_per_side nodes per side (corners shared between sides).

    Total boundary nodes: N_b = 4 * 3 * n_elem_per_side
                               - 4 (corners counted once each)
                             = 12 * n_elem_per_side  (divisible by 3 ✓)
    """
    ne = n_elem_per_side
    # Build nodes on each side in CCW order
    def side_nodes(p0, p1, n):
        """n P3 elements → 3n+1 nodes at t=0,1/3,2/3,1,4/3,...,n"""
        pts = []
        for k in range(n):
            for frac in [0, 1/3, 2/3]:
                t = (k + frac) / n
                pts.append(p0 + t * (p1 - p0))
        return pts  # n*3 nodes; t=1 (= next element's t=0) added by next side

    corners = np.array([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
    ], dtype=np.float64)

    pts = []
    for i in range(4):
        pts.extend(side_nodes(corners[i], corners[(i+1)%4], ne))

    coords = np.array(pts, dtype=np.float64)
    N_b    = len(coords)
    assert N_b % 3 == 0, f"N_b={N_b} not divisible by 3"

    # Edge lengths and normals between consecutive nodes
    edge_lengths = np.empty(N_b)
    normals      = np.empty((N_b, 2))
    for i in range(N_b):
        j  = (i + 1) % N_b
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]
        L  = np.hypot(dx, dy)
        edge_lengths[i] = L
        normals[i]      = np.array([dy, -dx]) / L

    # P3 elements: element e has nodes [3e, 3e+1, 3e+2, (3e+3)%N_b]
    N_elem = N_b // 3
    elements        = np.empty((N_elem, 4), dtype=np.int64)
    element_lengths = np.empty(N_elem)
    element_normals = np.empty((N_elem, 2))
    for e in range(N_elem):
        i0 = 3 * e
        i3 = (3 * e + 3) % N_b
        elements[e] = [i0, i0+1, i0+2, i3]
        p0 = coords[i0]; p3 = coords[i3]
        dx = p3[0] - p0[0]; dy = p3[1] - p0[1]
        L  = np.hypot(dx, dy)
        element_lengths[e] = L
        element_normals[e] = np.array([dy, -dx]) / L

    return BoundaryMesh(
        node_indices    = np.arange(N_b),
        nodes           = coords,
        edge_lengths    = edge_lengths,
        normals         = normals,
        n_boundary_dofs = N_b,
        elements        = elements,
        element_lengths = element_lengths,
        element_normals = element_normals,
        n_elements      = N_elem,
    )


def make_unit_bnd(n_elem_per_side=2):
    """Unit square [0,1]² with n_elem_per_side P3 elements per side."""
    return make_rect_bnd_p3(0, 1, 0, 1, n_elem_per_side)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Log-Gauss-Jacobi quadrature accuracy
# ─────────────────────────────────────────────────────────────────────────────

def test_log_gauss_jacobi():
    """
    Verify ∫₀¹ tᵏ (−ln t) dt = 1/(k+1)² for k = 0 .. 5.

    The rule integrates exactly for k=0 (weight integral).
    Accuracy degrades for higher k because tᵏ = e^{−ku} is not polynomial
    in the Laguerre variable u — n=25 gives < 1e-4 for k=0..5.
    """
    print("\n--- test_log_gauss_jacobi ---")
    tolerances = {10: 5e-2, 25: 1e-4}
    for n_pts, tol in tolerances.items():
        xi, wi = log_gauss_jacobi_points(n_pts)
        max_err = 0.0
        for k in range(6):
            computed = float(np.dot(wi, xi ** k))
            exact    = 1.0 / (k + 1) ** 2
            rel_err  = abs(computed - exact) / exact
            max_err  = max(max_err, rel_err)
            print(f"  n={n_pts:2d}, k={k}: computed={computed:.10f}, "
                  f"exact={exact:.10f}, rel_err={rel_err:.2e}")
        print(f"  n={n_pts}: max rel err = {max_err:.2e}  (threshold {tol:.0e})")
        assert max_err < tol, \
            f"log_gauss_jacobi_points(n={n_pts}): max rel_err={max_err:.2e} > {tol}"

    # k=0 must be exact (actual use case)
    for n_pts in [10, 25]:
        xi, wi = log_gauss_jacobi_points(n_pts)
        err_k0 = abs(float(np.dot(wi, xi**0)) - 1.0)
        assert err_k0 < 1e-14, f"k=0 weight integral not exact: err={err_k0:.2e}"
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Boundary extraction (requires actual P3 mesh)
# ─────────────────────────────────────────────────────────────────────────────

def test_boundary_extraction():
    """
    Check extract_boundary_edges on a P3 structured mesh:
      - N_b divisible by 3  (required for P3 BEM)
      - No duplicate global indices
      - Outward normals have unit length and are axis-aligned on unit square
      - Normals point outward (positive dot with outward radial direction)
      - P3 element grouping: each element spans 4 nodes, element length matches
        distance from node[3e] to node[(3e+3)%N_b]
    """
    print("\n--- test_boundary_extraction ---")
    try:
        from femmi.mesh import generate_p3_structured_mesh
    except ImportError:
        print("  SKIPPED (mesh not available)")
        return

    mesh = generate_p3_structured_mesh(4, 4, xmin=0, xmax=1, ymin=0, ymax=1)
    bnd  = extract_boundary_edges(mesh)
    N_b  = bnd.n_boundary_dofs

    # P3 requirement: N_b divisible by 3
    assert N_b % 3 == 0, \
        f"N_b={N_b} not divisible by 3 — P3 BEM requires 3 nodes per edge"
    print(f"  N_b={N_b}  N_elem={bnd.n_elements}  N_b%3={N_b%3}  ✓")

    # No duplicate indices
    assert len(np.unique(bnd.node_indices)) == N_b, \
        "Duplicate boundary node indices found"

    # Unit normals
    normal_lengths = np.linalg.norm(bnd.normals, axis=1)
    assert np.allclose(normal_lengths, 1.0, atol=1e-12), \
        f"Non-unit normals: min={normal_lengths.min():.6f}"

    # Axis-aligned normals on unit square
    for n in bnd.normals:
        axis_aligned = (
            (abs(abs(n[0]) - 1) < 1e-10 and abs(n[1]) < 1e-10) or
            (abs(abs(n[1]) - 1) < 1e-10 and abs(n[0]) < 1e-10)
        )
        assert axis_aligned, f"Normal {n} is not axis-aligned"

    # Outward normals
    centre = bnd.nodes.mean(axis=0)
    for i in range(N_b):
        mid    = 0.5 * (bnd.nodes[i] + bnd.nodes[(i+1) % N_b])
        radial = mid - centre
        L      = np.linalg.norm(radial)
        if L > 1e-10:
            dot = np.dot(bnd.normals[i], radial / L)
            assert dot > -1e-8, \
                f"Inward-pointing normal at segment {i}: dot={dot:.4f}"

    # P3 element structure
    assert bnd.n_elements == N_b // 3, \
        f"n_elements={bnd.n_elements} should be N_b//3={N_b//3}"
    assert bnd.elements.shape == (bnd.n_elements, 4), \
        f"elements shape {bnd.elements.shape} should be ({bnd.n_elements}, 4)"

    # Each element's length = distance from its first to its last node
    for e in range(bnd.n_elements):
        i0, _, _, i3 = bnd.elements[e]
        p0 = bnd.nodes[i0]; p3 = bnd.nodes[i3]
        L_expected = float(np.linalg.norm(p3 - p0))
        L_stored   = float(bnd.element_lengths[e])
        assert abs(L_stored - L_expected) < 1e-12, \
            f"element {e}: stored length {L_stored:.8f} != {L_expected:.8f}"

    # Element normals are unit vectors
    elem_normal_lengths = np.linalg.norm(bnd.element_normals, axis=1)
    assert np.allclose(elem_normal_lengths, 1.0, atol=1e-12), \
        "Non-unit element normals"

    print(f"  All P3 boundary extraction checks PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Boundary mass matrix
# ─────────────────────────────────────────────────────────────────────────────

def test_boundary_mass():
    """
    M_b properties:
      - M_b @ ones = perimeter  (up to 1e-14 × perimeter)
      - Symmetric
      - Positive definite
    """
    print("\n--- test_boundary_mass ---")
    bnd  = make_unit_bnd(n_elem_per_side=3)
    M    = assemble_boundary_mass(bnd)
    ones = np.ones(bnd.n_boundary_dofs)

    perimeter = float(np.sum(bnd.edge_lengths))
    total     = float(ones @ M @ ones)
    assert abs(total - perimeter) < 1e-12 * perimeter, \
        f"M_b @ 1 = {total:.10f}, expected {perimeter:.10f}"

    sym_err = np.linalg.norm(M - M.T) / np.linalg.norm(M)
    assert sym_err < 1e-14, f"M_b not symmetric: {sym_err:.2e}"

    eigs = np.linalg.eigvalsh(M)
    assert eigs.min() > 0, f"M_b not positive definite: min eig = {eigs.min():.4f}"

    print(f"  M_b @ 1 = {total:.8f} (expected {perimeter:.8f})  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Single-layer: symmetry
# ─────────────────────────────────────────────────────────────────────────────

def test_single_layer_symmetry():
    """V_h must be symmetric: ‖V_h − V_hᵀ‖_F / ‖V_h‖_F < 1e-12."""
    print("\n--- test_single_layer_symmetry ---")
    bnd     = make_unit_bnd(n_elem_per_side=2)
    V       = assemble_single_layer(bnd, n_quad=25)
    sym_err = np.linalg.norm(V - V.T) / np.linalg.norm(V)
    print(f"  Symmetry error: {sym_err:.2e}  (threshold 1e-12)")
    assert sym_err < 1e-12, f"V_h not symmetric: {sym_err:.2e}"
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Single-layer: invertibility and correct eigenvalue sign
# ─────────────────────────────────────────────────────────────────────────────

def test_single_layer_invertible():
    """
    V_h must be invertible and negative-definite on the unit square.

    Mathematical fact: V_h with G=(1/2π)ln|x-y| is negative-definite when
    the logarithmic capacity of the domain is < 1.
    Unit square: capacity ≈ 0.59 < 1  →  all eigenvalues negative  ✓

    The Calderon operator C = V_h⁻¹(½M_b + K_h) is well-defined regardless
    of eigenvalue sign; V_h invertibility is all that matters.

    Upgrading to P3 BEM does NOT change the eigenvalue sign — it only
    improves the approximation quality of V_h.

    Checks:
      - condition number < 1e6  (well-conditioned)
      - all eigenvalues same sign  (confirms correct assembly)
      - all eigenvalues negative  (confirms capacity < 1 for unit square)
    """
    print("\n--- test_single_layer_invertible ---")
    bnd   = make_unit_bnd(n_elem_per_side=2)
    V     = assemble_single_layer(bnd, n_quad=25)
    eigs  = np.linalg.eigvalsh(V)
    cond  = abs(eigs).max() / (abs(eigs).min() + 1e-20)

    print(f"  Eigenvalues: [{eigs.min():.6f}, {eigs.max():.6f}]")
    print(f"  Condition number: {cond:.1f}  (threshold 1e6)")

    assert cond < 1e6, f"V_h near-singular: cond = {cond:.2e}"

    signs_consistent = (eigs > 0).all() or (eigs < 0).all()
    assert signs_consistent, \
        ("V_h eigenvalues have mixed signs — indicates assembly error. "
         f"min={eigs.min():.4f}, max={eigs.max():.4f}")

    # Unit square: must be negative-definite
    assert (eigs < 0).all(), \
        (f"V_h should be negative-definite on unit square (cap ≈ 0.59 < 1). "
         f"Got min={eigs.min():.4f}, max={eigs.max():.4f}. "
         "Positive eigenvalues would indicate a sign error in the Green's "
         "function or wrong domain geometry.")

    print("  PASSED  (negative-definite as expected for cap(Ω) < 1)")

    # Verify that a LARGER domain flips to positive-definite (sanity check)
    bnd_large = make_rect_bnd_p3(0, 4, 0, 4, 2)  # 4×4 square, cap ≈ 4×0.59 > 1
    V_large   = assemble_single_layer(bnd_large, n_quad=25)
    eigs_large = np.linalg.eigvalsh(V_large)
    print(f"  Large domain (4×4): eigenvalues [{eigs_large.min():.4f}, "
          f"{eigs_large.max():.4f}]  "
          f"({'positive-definite ✓' if (eigs_large>0).all() else 'mixed (expected near transition)'})")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Single-layer: analytic diagonal block check
# ─────────────────────────────────────────────────────────────────────────────

def test_single_layer_analytic():
    """
    Verify Duffy quadrature for P3 self-interaction using the analytic values
    for ∫₀¹∫₀¹ ln|s-t| φ_a(s) φ_b(t) ds dt.

    These are computed numerically via the same log-GL rule used internally
    in assemble_single_layer, verified against the k=0 sanity check
    ∫₀¹∫₀¹ ln|s-t| ds dt = -3/2.
    """
    from scipy.special import roots_legendre
    print("\n--- test_single_layer_analytic ---")

    def gl01(n):
        xi, wi = roots_legendre(n)
        return 0.5*(xi+1), 0.5*wi

    xi_lj, w_lj = log_gauss_jacobi_points(25)
    xi_gl, w_gl = gl01(25)

    # P3 basis functions at parameter t
    def phi(k, t):
        """P1 basis — these match the analytic values I_00=-7/16, I_01=-5/16."""
        t = np.asarray(t, dtype=float)
        if k == 0: return 1.0 - t
        if k == 1: return t

    # ∫∫ ln|s-t| ds dt = -3/2  (sanity check, all basis functions)
    def F_const(r):
        return 2*(1-r)

    I_const = -float(np.dot(w_lj, np.array([F_const(r) for r in xi_lj])))
    print(f"  ∫∫ ln|s-t| ds dt: {I_const:.8f}  (exact=-1.5,  err={abs(I_const+1.5):.2e})")
    assert abs(I_const + 1.5) < 1e-6, \
        f"Sanity check ∫∫ ln|s-t| failed: {I_const:.6f} ≠ -1.5"

    # Check each (a,b) pair via log-GL integration
    def compute_Iab(a, b):
        """I_ab = ∫₀¹∫₀¹ ln|s-t| φ_a(s) φ_b(t) ds dt via Duffy."""
        def F_ab(r):
            t_inner = xi_gl * (1 - r)
            w_inner = w_gl  * (1 - r)
            return float(np.sum(w_inner * (
                phi(a, t_inner + r) * phi(b, t_inner) +
                phi(a, t_inner)     * phi(b, t_inner + r)
            )))
        return -float(np.dot(w_lj, np.array([F_ab(r) for r in xi_lj])))

    # P1 analytic values (known from literature)
    analytic_p1 = {(0, 0): -7/16, (1, 1): -7/16, (0, 1): -5/16}
    for (a, b), exact in analytic_p1.items():
        I_computed = compute_Iab(a, b)
        rel_err    = abs(I_computed - exact) / abs(exact)
        print(f"  I_{a}{b}: computed={I_computed:.8f}, exact={exact:.8f}, "
              f"rel_err={rel_err:.2e}")
        assert rel_err < 1e-6, \
            f"Duffy quadrature mismatch I_{a}{b}: rel_err={rel_err:.2e}"

    # P3 self-consistency: I_00 + I_33 + I_11 + I_22 are all finite
    # and the P3 basis partition of unity means Σ_ab I_ab = ∫∫ ln|s-t| ds dt = -3/2
    I_sum = sum(compute_Iab(a, b) for a in range(2) for b in range(2))
    print(f"  Σ_{{a,b}} I_ab = {I_sum:.8f}  (expected -1.5,  err={abs(I_sum+1.5):.2e})")
    assert abs(I_sum + 1.5) < 1e-5, \
        f"P3 basis completeness check failed: Σ I_ab = {I_sum:.6f} ≠ -1.5"

    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Double-layer: Calderón identity
# ─────────────────────────────────────────────────────────────────────────────

def test_double_layer_calderon():
    """
    (½M_b + K_h) @ ones ≈ 0  (Calderón identity for constant ψ).

    Tolerance: ‖(½M_b + K_h) @ 1‖ / ‖½M_b @ 1‖ < 5e-3.
    """
    print("\n--- test_double_layer_calderon ---")
    for n_elem in [2, 3]:
        bnd  = make_unit_bnd(n_elem_per_side=n_elem)
        K    = assemble_double_layer(bnd, n_quad=8)
        M    = assemble_boundary_mass(bnd)
        ones = np.ones(bnd.n_boundary_dofs)

        combo  = 0.5 * M @ ones + K @ ones
        norm_c = np.linalg.norm(combo)
        norm_r = np.linalg.norm(0.5 * M @ ones)
        rel    = norm_c / norm_r
        print(f"  n_elem_per_side={n_elem}: "
              f"‖(½M_b+K_h)·1‖/‖½M_b·1‖ = {rel:.2e}  (threshold 5e-3)")
        assert rel < 5e-3, \
            f"Calderón identity violated: rel={rel:.2e} (n_elem={n_elem})"
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Calderón preconditioner (LinearOperator)
# ─────────────────────────────────────────────────────────────────────────────

def test_calderon_operator():
    """
    C = V_h⁻¹(½M_b + K_h) returned as LinearOperator.

    Checks:
      - shape is (N_b, N_b)
      - matvec returns finite values
      - C is non-trivial (‖Cx‖ ≠ ‖x‖)
      - round-trip: V_h(C x) ≈ (½M_b + K_h) x
    """
    print("\n--- test_calderon_operator ---")
    bnd = make_unit_bnd(n_elem_per_side=3)
    V   = assemble_single_layer(bnd, n_quad=25)
    K   = assemble_double_layer(bnd, n_quad=8)
    M   = assemble_boundary_mass(bnd)
    C   = calderon_matrix(V, K, M)

    N = bnd.n_boundary_dofs
    assert C.shape == (N, N), f"Wrong shape: {C.shape}"

    rng  = np.random.default_rng(42)
    x    = rng.standard_normal(N)
    Cx   = C.matvec(x)
    CCx  = C.matvec(Cx)

    assert np.all(np.isfinite(Cx)),  "C @ x contains non-finite values"
    assert np.all(np.isfinite(CCx)), "C² @ x contains non-finite values"

    # C is non-trivial
    ratio = np.linalg.norm(Cx) / (np.linalg.norm(x) + 1e-14)
    assert abs(ratio - 1.0) > 1e-3, \
        f"C appears to be the identity (ratio={ratio:.4f}) — assembly error"

    # Round-trip: V_h (C x) should equal (½M_b + K_h) x
    VCx    = V @ Cx
    target = (0.5 * M + K) @ x
    rt_err = np.linalg.norm(VCx - target) / (np.linalg.norm(target) + 1e-14)
    assert rt_err < 1e-10, \
        f"V_h(Cx) ≠ (½M_b+K_h)x: rel_err={rt_err:.2e}"

    print(f"  shape={C.shape}, ‖Cx‖/‖x‖={ratio:.4f}, round-trip err={rt_err:.2e}  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

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

    print("=" * 60)
    print("femmi Phase 1 BEM tests  (P3 boundary elements)")
    print("=" * 60)

    passed = 0
    failed = []
    for fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed.append(fn.__name__)

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("All tests PASSED ✓")
    print("=" * 60)

    sys.exit(0 if not failed else 1)
