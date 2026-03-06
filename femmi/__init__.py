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

from .p3_mesh_generator import (
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
]

__version__ = "0.1.0"
