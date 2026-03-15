"""
FEMMI - Finite Element Mass Map Inversion

P3 FEM-BEM pipeline for weak gravitational lensing mass reconstruction.
"""

import jax
jax.config.update("jax_enable_x64", True)


def _enforce_64bit():
    try:
        import jax.numpy as jnp
        x = jnp.array(1.0)
        if x.dtype != jnp.float64:
            raise RuntimeError(
                "FEMMI requires 64-bit JAX. Set JAX_ENABLE_X64=1 or call "
                "jax.config.update('jax_enable_x64', True) before importing femmi."
            )
    except ImportError:
        pass


_enforce_64bit()

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

_JAX_AVAILABLE = False
try:
    import jax as _jax
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
    "FEMOperators",
    "build_operators",
    "build_operators_adaptive",
    "build_wiener_regularizer",
    "build_laplacian",
    "DifferentiableForward",
    "MAPReconstructor",
    "ReconstructionResult",
    "kaiser_squires",
    "run_comparison",
    "generate_p3_structured_mesh",
    "generate_p3_adaptive_mesh",
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