"""
tests/test_calderon.py
Buffa-Christiansen dual bases (femmi.calderon).

The point of the module is one property: a pairing whose conditioning does NOT
grow with the mesh. MATH.md reports that pairing V and W on the same mesh
gives a flat 2.26x improvement while cond still grows linearly in N_b, and the
textbook reason is that the discrete pairing needs a genuinely dual basis.

The first implementation here was wrong in a way that is worth a regression test:
it paired the node-indexed duals against the ELEMENT-indexed coarse constants,
producing a bidiagonal circulant with rows summing to 1, whose eigenvalues
(1 + w^k)/2 vanish at w = -1 -- exactly singular for every even n. It measured
cond ~ 1e16 and would have read as "the dual basis does not work".

Run:
    python -m pytest tests/test_calderon.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.calderon import (barycentric_refine, dual_constant_basis,
                            primal_constant_on_fine, mixed_gram, _lengths)


def circle(n, radius=1.0):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(th), np.sin(th)], 1)


def irregular(n, seed=0):
    th = np.sort(np.random.default_rng(seed).uniform(0, 2 * np.pi, n))
    return np.stack([np.cos(th), np.sin(th)], 1)


# ------------------------------------------------------------- refinement ---

def test_refinement_interleaves_originals_and_midpoints():
    """Even/odd interleaving is what makes 'the two halves of coarse element e'
    equal to 2e, 2e+1 and 'the two halves meeting at node i' equal to 2i-1, 2i.
    Appending midpoints instead would break every index in the dual basis."""
    P = circle(8)
    fine, parent = barycentric_refine(P)
    assert fine.shape == (16, 2)
    assert np.allclose(fine[0::2], P)
    assert np.allclose(fine[1::2], 0.5 * (P + np.roll(P, -1, axis=0)))
    assert np.array_equal(parent, np.repeat(np.arange(8), 2))


def test_refinement_preserves_total_length():
    """Splitting at midpoints of a POLYGON is exact -- the halves are collinear
    with the parent, so no length is created or lost."""
    P = irregular(37, seed=3)
    fine, _ = barycentric_refine(P)
    assert _lengths(fine).sum() == pytest.approx(_lengths(P).sum())


def test_refinement_validates_input():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        barycentric_refine(np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"n >= 3"):
        barycentric_refine(np.zeros((2, 2)))


# ------------------------------------------------------------ dual basis ---

def test_dual_functions_have_unit_integral():
    """The normalisation that makes the diagonal of the pairing independent of
    element size."""
    for P in (circle(24), irregular(31, seed=1)):
        fine, _ = barycentric_refine(P)
        Lf = _lengths(fine)
        D = dual_constant_basis(P)
        assert np.allclose(D @ Lf, 1.0)


def test_dual_functions_are_local():
    """Each dual touches exactly the two fine half-elements at its node. A wider
    support would not be a BC function and would fill in the pairing."""
    P = circle(20)
    D = dual_constant_basis(P)
    assert D.shape == (20, 40)
    assert np.all((np.abs(D) > 0).sum(axis=1) == 2)


# ---------------------------------------------------------------- pairing ---

@pytest.mark.parametrize("n", [16, 32, 64, 128, 256, 512])
def test_bc_pairing_conditioning_is_mesh_independent(n):
    """THE RESULT. On a uniform mesh the eigenvalues are 3/4 + (1/4)cos(theta),
    so cond is exactly 2 for every n -- no growth at all."""
    assert np.linalg.cond(mixed_gram(circle(n))) == pytest.approx(2.0, abs=1e-6)


def test_bc_diagonal_is_exactly_three_quarters_for_any_mesh():
    """The hat at a node has mean 3/4 on both half-elements touching it,
    whatever their lengths, so the diagonal carries no mesh dependence."""
    for P in (circle(17), irregular(41, seed=2)):
        assert np.allclose(np.diag(mixed_gram(P)), 0.75)


def test_bc_rows_sum_to_one():
    for P in (circle(23), irregular(29, seed=5)):
        assert np.allclose(mixed_gram(P).sum(axis=1), 1.0)


def test_bc_stays_conditioned_where_the_same_mesh_gram_does_not():
    """The comparison that motivates the whole module: a same-mesh Gram matrix of
    piecewise constants is diag(element lengths), so its conditioning is the
    length RATIO and degrades without bound as the mesh becomes non-uniform. The
    BC pairing does not notice."""
    for n in (32, 128, 512):
        P = irregular(n, seed=0)
        same = np.linalg.cond(np.diag(_lengths(P)))
        bc = np.linalg.cond(mixed_gram(P))
        assert bc < 3.0
        assert bc < same / 100.0


def test_pairing_against_coarse_constants_is_the_singular_trap():
    """Regression for the bug this module was first written with. Pairing the
    node-indexed duals against ELEMENT-indexed constants gives a circulant that
    is exactly singular for even n. This test documents WHY, by reproducing the
    broken pairing explicitly next to the correct one."""
    n = 64
    P = circle(n)
    fine, _ = barycentric_refine(P)
    Lf = _lengths(fine)
    broken = (dual_constant_basis(P) * Lf[None, :]) @ primal_constant_on_fine(P).T
    assert np.linalg.cond(broken) > 1e12          # singular to working precision
    assert np.linalg.cond(mixed_gram(P)) < 3.0    # the correct pairing is fine


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
