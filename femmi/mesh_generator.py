"""
Mesh generation utilities for weak lensing FEM

Supports:
- Structured rectangular grids (GPU-friendly)
- Unstructured meshes via Triangle
- Survey footprint geometries
- Masking for structured grids

"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from .fem_solver import Mesh

try:
    import triangle as tr
    HAS_TRIANGLE = True
except ImportError:
    HAS_TRIANGLE = False
    print("Warning: triangle not available. Install with: pip install triangle")

def generate_structured_mesh(nx: int, ny: int,
                            xmin: float = 0.0, xmax: float = 1.0,
                            ymin: float = 0.0, ymax: float = 1.0,
                            return_numpy: bool = False) -> Mesh:
    """
    Generate structured triangular mesh on rectangular domain
    
    Each rectangular cell is split into 2 triangles:
        n2 --- n3
        |  \   |
        |   \  |
        n0 --- n1
    
    Lower triangle: [n0, n1, n2]
    Upper triangle: [n1, n3, n2]
    
    Args:
        nx, ny: Number of cells in x, y directions
        xmin, xmax, ymin, ymax: Domain bounds
        return_numpy: If True, return numpy arrays instead of JAX arrays
        
    Returns:
        Mesh object with structured triangulation
    """
    # Create node grid
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    xx, yy = np.meshgrid(x, y)
    
    nodes = np.column_stack([xx.ravel(), yy.ravel()])  # (n_nodes, 2)
    
    # Create element connectivity
    elements = []
    for i in range(ny):
        for j in range(nx):
            # Node indices (row-major ordering)
            n0 = i * (nx + 1) + j
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            
            # Two triangles per cell
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])
    
    elements = np.array(elements, dtype=np.int32)
    
    # Find boundary nodes (edges of domain)
    n_nodes = nodes.shape[0]
    boundary_mask = (
        (nodes[:, 0] <= xmin + 1e-10) |  # Left edge
        (nodes[:, 0] >= xmax - 1e-10) |  # Right edge
        (nodes[:, 1] <= ymin + 1e-10) |  # Bottom edge
        (nodes[:, 1] >= ymax - 1e-10)    # Top edge
    )
    boundary = np.where(boundary_mask)[0].astype(np.int32)
    
    # Convert to JAX arrays unless requested otherwise
    if not return_numpy:
        nodes = jnp.array(nodes)
        elements = jnp.array(elements)
        boundary = jnp.array(boundary)
    
    return Mesh(nodes=nodes, elements=elements, boundary=boundary)


def generate_masked_structured_mesh(nx: int, ny: int,
                                   mask_func,
                                   xmin: float = 0.0, xmax: float = 1.0,
                                   ymin: float = 0.0, ymax: float = 1.0) -> Mesh:
    """
    Generate structured mesh with masked regions
    
    GPU-friendly approach: keep regular grid structure, mark masked nodes
    as boundary (enforces \psi = 0 there)
    
    Args:
        nx, ny: Grid resolution
        mask_func: function(x, y) -> bool (True = valid, False = masked)
        xmin, xmax, ymin, ymax: Domain bounds
        
    Returns:
        Mesh with masked nodes added to boundary
    """
    mesh = generate_structured_mesh(nx, ny, xmin, xmax, ymin, ymax, return_numpy=True)
    
    # Evaluate mask at all nodes
    nodes_valid = np.array([
        mask_func(x, y) for x, y in mesh.nodes
    ])
    
    # Add masked nodes to boundary
    masked_nodes = np.where(~nodes_valid)[0]
    boundary_extended = np.concatenate([mesh.boundary, masked_nodes])
    boundary_extended = np.unique(boundary_extended).astype(np.int32)
    
    return Mesh(
        nodes=jnp.array(mesh.nodes),
        elements=jnp.array(mesh.elements),
        boundary=jnp.array(boundary_extended)
    )


def generate_unstructured_mesh(domain: str = 'unit_square',
                              max_area: float = 0.01,
                              vertices: Optional[np.ndarray] = None,
                              segments: Optional[np.ndarray] = None) -> Mesh:
    """
    Generate unstructured mesh using Triangle library
    
    Args:
        domain: 'unit_square', 'circle', or 'custom'
        max_area: Maximum triangle area (controls resolution)
        vertices: (n_verts, 2) for custom domain
        segments: (n_segs, 2) boundary edges for custom domain
        
    Returns:
        Mesh object
    """
    if not HAS_TRIANGLE:
        raise RuntimeError("Triangle library not available. Install: pip install triangle")
    
    # Define domain
    if domain == 'unit_square':
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    
    elif domain == 'circle':
        # Approximate circle with polygon
        n_points = 64
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        vertices = np.column_stack([np.cos(theta), np.sin(theta)])
        segments = np.column_stack([
            np.arange(n_points),
            np.roll(np.arange(n_points), -1)
        ])
    
    elif domain == 'custom':
        if vertices is None or segments is None:
            raise ValueError("Must provide vertices and segments for custom domain")
    
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Triangulate
    A = dict(vertices=vertices, segments=segments)
    B = tr.triangulate(A, f'pq30a{max_area}')
    
    nodes = B['vertices']
    elements = B['triangles']  # Triangle returns 0-indexed
    
    # Find boundary nodes
    boundary = find_boundary_nodes(nodes, segments, vertices)
    
    return Mesh(
        nodes=jnp.array(nodes),
        elements=jnp.array(elements, dtype=np.int32),
        boundary=jnp.array(boundary, dtype=np.int32)
    )


def find_boundary_nodes(nodes: np.ndarray, 
                       segments: np.ndarray,
                       original_vertices: np.ndarray,
                       tol: float = 1e-8) -> np.ndarray:
    """
    Find nodes on boundary segments
    
    Args:
        nodes: All mesh nodes
        segments: Boundary segments (references original_vertices)
        original_vertices: Original boundary vertices
        tol: Distance tolerance
        
    Returns:
        Indices of boundary nodes
    """
    boundary_nodes = []
    
    for seg in segments:
        v1 = original_vertices[seg[0]]
        v2 = original_vertices[seg[1]]
        
        # Find nodes close to this segment
        for i, node in enumerate(nodes):
            # Distance from point to line segment
            dist = point_to_segment_distance(node, v1, v2)
            if dist < tol:
                boundary_nodes.append(i)
    
    return np.unique(boundary_nodes)


def point_to_segment_distance(p: np.ndarray, 
                              a: np.ndarray, 
                              b: np.ndarray) -> float:
    """
    Distance from point p to line segment ab
    """
    ab = b - a
    ap = p - a
    
    # Project p onto line ab
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-14)
    t = np.clip(t, 0, 1)  # Clamp to segment
    
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def refine_mesh_uniform(mesh: Mesh) -> Mesh:
    """
    Uniformly refine triangular mesh (split each triangle into 4)
    
    Args:
        mesh: Input mesh
        
    Returns:
        Refined mesh
    """
    nodes = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    
    new_nodes = [nodes]
    new_elements = []
    edge_map = {}  # (i, j) -> midpoint node index
    
    next_node_idx = len(nodes)
    
    for elem in elements:
        n0, n1, n2 = elem
        
        # Get/create midpoint nodes
        def get_midpoint(i, j):
            nonlocal next_node_idx
            edge = tuple(sorted([i, j]))
            if edge not in edge_map:
                mid = (nodes[i] + nodes[j]) / 2
                new_nodes.append([mid])
                edge_map[edge] = next_node_idx
                next_node_idx += 1
            return edge_map[edge]
        
        m01 = get_midpoint(n0, n1)
        m12 = get_midpoint(n1, n2)
        m20 = get_midpoint(n2, n0)
        
        # Four new triangles
        new_elements.extend([
            [n0, m01, m20],
            [m01, n1, m12],
            [m20, m12, n2],
            [m01, m12, m20]
        ])
    
    nodes_refined = np.vstack(new_nodes)
    elements_refined = np.array(new_elements, dtype=np.int32)
    
    # Update boundary (find nodes on original boundary edges)
    # Simple approach: mark all nodes within tol of original boundary nodes
    original_boundary_coords = nodes[mesh.boundary]
    boundary_refined = []
    
    for i, node in enumerate(nodes_refined):
        for b_coord in original_boundary_coords:
            if np.linalg.norm(node - b_coord) < 1e-10:
                boundary_refined.append(i)
                break
    
    # Also check if new nodes lie on boundary edges
    # (This is approximate - proper implementation would track boundary edges)
    boundary_refined = np.unique(boundary_refined).astype(np.int32)
    
    return Mesh(
        nodes=jnp.array(nodes_refined),
        elements=jnp.array(elements_refined),
        boundary=jnp.array(boundary_refined)
    )


# ============================================================================
# Survey-specific geometries
# ============================================================================

def create_circular_survey(radius: float = 1.0, 
                          max_area: float = 0.01) -> Mesh:
    """
    Create circular survey footprint
    
    Args:
        radius: Survey radius
        max_area: Triangle max area
        
    Returns:
        Mesh for circular region
    """
    return generate_unstructured_mesh(domain='circle', max_area=max_area)


def create_rectangular_survey_with_holes(nx: int, ny: int,
                                        hole_centers: list,
                                        hole_radii: list) -> Mesh:
    """
    Create rectangular survey with circular masked regions (e.g., bright stars)
    
    Args:
        nx, ny: Grid resolution
        hole_centers: List of (x, y) tuples for hole centers
        hole_radii: List of radii for each hole
        
    Returns:
        Mesh with masked regions
    """
    def mask_func(x, y):
        # Valid if not in any hole
        for center, radius in zip(hole_centers, hole_radii):
            dx = x - center[0]
            dy = y - center[1]
            if dx**2 + dy**2 < radius**2:
                return False
        return True
    
    return generate_masked_structured_mesh(nx, ny, mask_func)
