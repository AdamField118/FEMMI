"""
JAX-based FEM for Weak Gravitational Lensing

Solves the lensing Poisson equation using P1 finite elements
"""

from .fem_solver import (
    Mesh,
    FEMSolution,
    solve_lensing_poisson,
    compute_errors,
    GaussianLens,
    PointMassLens,
    SISLens,
    SinusoidalLens,
    PolynomialLens,
)

from .mesh_generator import (
    generate_structured_mesh,
    generate_masked_structured_mesh,
    generate_unstructured_mesh,
    refine_mesh_uniform,
)

__version__ = "0.1.0"

__all__ = [
    # Core solver
    "Mesh",
    "FEMSolution",
    "solve_lensing_poisson",
    "compute_errors",
    
    # Lens models
    "GaussianLens",
    "PointMassLens",
    "SISLens",
    "SinusoidalLens",
    "PolynomialLens",
    
    # Mesh generation
    "generate_structured_mesh",
    "generate_masked_structured_mesh",
    "generate_unstructured_mesh",
    "refine_mesh_uniform",
]