"""
FEMMI — Finite Element Mass Map Inversion
==========================================

GPU-accelerated P3 FEM pipeline for weak gravitational lensing.

Public API
----------

Operators (assembly + forward model):
    build_operators            -- uniform structured P3 mesh
    build_operators_adaptive   -- locally refined mesh near a mask
    build_wiener_regularizer   -- Matérn-like prior R = M + l²K
    FEMOperators               -- dataclass holding K, M, S1, S2, LU

Differentiable forward model:
    DifferentiableForward      -- JAX-traceable κ → (γ₁, γ₂) with custom VJP

MAP reconstruction:
    MAPReconstructor           -- L-BFGS solver with numpy adjoint
    kaiser_squires             -- FFT-based KS reference
    run_comparison             -- benchmark FEM-MAP vs KS

BEM (boundary element method):
    BoundaryMesh               -- ordered boundary node data
    extract_boundary_edges     -- extract ∂Ω nodes in CCW order from P3 mesh
    assemble_single_layer      -- V_h (N_b × N_b) single-layer BEM matrix
    assemble_double_layer      -- K_h (N_b × N_b) double-layer BEM matrix
    assemble_boundary_mass     -- M_b (N_b × N_b) boundary mass matrix
    assemble_bem_matrices      -- convenience: returns (V_h, K_h, M_b)
    calderon_matrix            -- C = V_h⁻¹(½M_b + K_h) as LinearOperator

Mesh generation:
    generate_p3_structured_mesh
    generate_p3_adaptive_mesh

Example
-------

    from femmi import build_operators, DifferentiableForward, MAPReconstructor
    import numpy as np

    ops = build_operators(20, 20)
    fwd = DifferentiableForward(ops, lam_reg=1e-2)
    rec = MAPReconstructor(fwd, wiener_length=0.5)
    kappa_map, result = rec.reconstruct(g1_obs, g2_obs)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 64-bit enforcement guard (MATH.md §18.3)
#
# For a 32×32 mesh, κ(A_coupled) = O(h⁻²) ≈ 1600.
# In 32-bit arithmetic: solve error ≈ κ × ε₃₂ ≈ 1600 × 6e-8 ≈ 1e-4,
# which dominates the P3 discretisation error h⁴ ≈ 6e-6.
# All FEMMI computations must use 64-bit floats.
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_64bit():
    """
    Ensure JAX (if available) operates in 64-bit mode.
    Raises RuntimeError if JAX is present but x64 cannot be enabled.
    """
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        x = jnp.array(1.0)
        if x.dtype != jnp.float64:
            raise RuntimeError(
                "FEMMI requires 64-bit JAX.  Set JAX_ENABLE_X64=1 or call "
                "jax.config.update('jax_enable_x64', True) before importing femmi."
            )
    except ImportError:
        # JAX not installed — scipy/numpy BEM modules will still work in 64-bit
        pass


_enforce_64bit()

# ─────────────────────────────────────────────────────────────────────────────
# Public imports
# ─────────────────────────────────────────────────────────────────────────────

from .bem import (
    BoundaryMesh,
    extract_boundary_edges,
    log_gauss_jacobi_points,
    assemble_single_layer,
    assemble_double_layer,
    assemble_boundary_mass,
    assemble_bem_matrices,
    calderon_matrix,
)

# JAX-dependent modules (operators, forward, inverse) are only available
# when JAX is installed.  Import them conditionally so that the BEM
# submodule remains usable in scipy-only environments.
_JAX_AVAILABLE = False
try:
    import jax as _jax  # noqa: F401 — presence check only
    _JAX_AVAILABLE = True
except ImportError:
    pass

if _JAX_AVAILABLE:
    from .operators import (
        FEMOperators,
        build_operators,
        build_operators_adaptive,
        build_wiener_regularizer,
        build_laplacian,
    )
    from .forward import DifferentiableForward
    from .inverse import (
        MAPReconstructor,
        ReconstructionResult,
        kaiser_squires,
        run_comparison,
    )
    from .mesh import (
        generate_p3_structured_mesh,
        generate_p3_adaptive_mesh,
    )

__all__ = [
    # operators
    "FEMOperators",
    "build_operators",
    "build_operators_adaptive",
    "build_wiener_regularizer",
    "build_laplacian",
    # forward
    "DifferentiableForward",
    # inverse
    "MAPReconstructor",
    "ReconstructionResult",
    "kaiser_squires",
    "run_comparison",
    # mesh
    "generate_p3_structured_mesh",
    "generate_p3_adaptive_mesh",
    # BEM
    "BoundaryMesh",
    "extract_boundary_edges",
    "log_gauss_jacobi_points",
    "assemble_single_layer",
    "assemble_double_layer",
    "assemble_boundary_mass",
    "assemble_bem_matrices",
    "calderon_matrix",
]

__version__ = "0.1.0"