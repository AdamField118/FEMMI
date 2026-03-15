"""
Mesh generation for P3 cubic triangular elements.

Two generators:
    generate_p3_structured_mesh  -- uniform structured grid
    generate_p3_adaptive_mesh    -- locally refined near a circular mask boundary

Both produce 10-node P3 elements with node ordering:
    [v0, v1, v2,  n3, n4,  n5, n6,  n7, n8,  n9]
     vertices     e01       e12       e20    centroid
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from .types import Mesh


def _elevate_to_p3(vertices: np.ndarray,
                   p1_elements: np.ndarray,
                   xmin: float, xmax: float,
                   ymin: float, ymax: float) -> Mesh:
    """
    Elevate a P1 triangulation to P3 by inserting edge and interior nodes.

    For each unique edge (va, vb) inserts nodes at t=1/3 and t=2/3.
    For each triangle inserts one interior node at the centroid.
    Triangles must be CCW-oriented before calling.
    """
    n_verts    = len(vertices)
    nodes_list = list(vertices)
    next_idx   = n_verts
    edge_dict  = {}

    def get_edge_nodes(va: int, vb: int) -> Tuple[int, int]:
        nonlocal next_idx
        key = (min(va, vb), max(va, vb))
        if key not in edge_dict:
            pi, pj = vertices[key[0]], vertices[key[1]]
            edge_dict[key] = [next_idx, next_idx + 1]
            nodes_list.append((2.0 * pi + pj) / 3.0)
            nodes_list.append((pi + 2.0 * pj) / 3.0)
            next_idx += 2
        i0, i1 = edge_dict[key]
        return (i0, i1) if va < vb else (i1, i0)

    p3_elements = []
    for v0, v1, v2 in p1_elements:
        n3, n4 = get_edge_nodes(v0, v1)
        n5, n6 = get_edge_nodes(v1, v2)
        n7, n8 = get_edge_nodes(v2, v0)
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        n9 = next_idx
        nodes_list.append(centroid)
        next_idx += 1
        p3_elements.append([v0, v1, v2, n3, n4, n5, n6, n7, n8, n9])

    nodes_arr = np.array(nodes_list, dtype=np.float64)
    elems_arr = np.array(p3_elements, dtype=np.int32)

    tol = 1e-10
    bnd = np.where(
        (nodes_arr[:, 0] <= xmin + tol) | (nodes_arr[:, 0] >= xmax - tol) |
        (nodes_arr[:, 1] <= ymin + tol) | (nodes_arr[:, 1] >= ymax - tol)
    )[0].astype(np.int32)

    return Mesh(nodes=nodes_arr, elements=elems_arr, boundary=bnd)


def generate_p3_structured_mesh(nx: int, ny: int,
                                 xmin: float = 0.0, xmax: float = 1.0,
                                 ymin: float = 0.0, ymax: float = 1.0,
                                 return_numpy: bool = False) -> Mesh:
    r"""
    Generate a P3 structured triangular mesh on a rectangular domain.

    Each rectangular cell is split into two triangles. Edge nodes are
    inserted at t=1/3 and t=2/3 along each edge; one centroid node per
    triangle gives 10 DOFs per element.

    Node ordering per triangle:
        v2
        /\
       /  \
    8 /    \ 6
     /      \
  9 /        \ 7
   /          \
  v0---3---4---v1

    Args:
        nx, ny       : number of cells in x and y directions
        xmin..ymax   : domain bounds
        return_numpy : if True, return plain numpy arrays instead of JAX arrays
    """
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    xx, yy = np.meshgrid(x, y)

    n_vertices   = (nx + 1) * (ny + 1)
    vertex_nodes = np.column_stack([xx.ravel(), yy.ravel()])

    p1_elements = []
    for i in range(ny):
        for j in range(nx):
            n0 = i * (nx + 1) + j
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            p1_elements.append([n0, n1, n2])
            p1_elements.append([n1, n3, n2])
    p1_elements = np.array(p1_elements, dtype=np.int32)

    edge_nodes    = {}
    nodes_list    = list(vertex_nodes)
    next_node_idx = n_vertices

    def get_edge_nodes(v1: int, v2: int) -> Tuple[int, int]:
        nonlocal next_node_idx
        edge_key = tuple(sorted([v1, v2]))
        if edge_key not in edge_nodes:
            i, j   = edge_key
            pi, pj = vertex_nodes[i], vertex_nodes[j]
            nodes_list.append((2.0 * pi + 1.0 * pj) / 3.0)
            nodes_list.append((1.0 * pi + 2.0 * pj) / 3.0)
            edge_nodes[edge_key] = [next_node_idx, next_node_idx + 1]
            next_node_idx += 2
        i0, i1 = edge_nodes[edge_key]
        return (i0, i1) if v1 < v2 else (i1, i0)

    p3_elements = []
    for v0, v1, v2 in p1_elements:
        n3, n4 = get_edge_nodes(v0, v1)
        n5, n6 = get_edge_nodes(v1, v2)
        n7, n8 = get_edge_nodes(v2, v0)
        centroid = (vertex_nodes[v0] + vertex_nodes[v1] + vertex_nodes[v2]) / 3.0
        n9 = next_node_idx
        nodes_list.append(centroid)
        next_node_idx += 1
        p3_elements.append([v0, v1, v2, n3, n4, n5, n6, n7, n8, n9])

    nodes_array = np.array(nodes_list, dtype=np.float64)
    p3_elements = np.array(p3_elements, dtype=np.int32)

    tol = 1e-10
    bnd_mask = (
        (nodes_array[:, 0] <= xmin + tol) | (nodes_array[:, 0] >= xmax - tol) |
        (nodes_array[:, 1] <= ymin + tol) | (nodes_array[:, 1] >= ymax - tol)
    )
    boundary = np.where(bnd_mask)[0].astype(np.int32)

    if not return_numpy:
        nodes_array = jnp.array(nodes_array)
        p3_elements = jnp.array(p3_elements)
        boundary    = jnp.array(boundary)

    return Mesh(nodes=nodes_array, elements=p3_elements, boundary=boundary)


def generate_p3_adaptive_mesh(nx: int, ny: int,
                               xmin: float = -2.5, xmax: float = 2.5,
                               ymin: float = -2.5, ymax: float = 2.5,
                               mask_center: Tuple[float, float] = (0.0, 0.0),
                               mask_radius: float = 0.5,
                               refine_factor: int = 3,
                               verbose: bool = True) -> Mesh:
    """
    Generate a P3 mesh with local refinement near a circular mask boundary.

    Builds an adaptively-spaced point cloud, Delaunay-triangulates it, then
    elevates P1->P3. Coarse spacing h=domain/nx is used everywhere except
    the annular band |r - r_mask| < 2*h_coarse, which uses h/refine_factor.

    Args:
        nx, ny         : background grid resolution
        xmin..ymax     : domain bounds
        mask_center    : (cx, cy) centre of the circular mask
        mask_radius    : radius of the circular mask
        refine_factor  : mesh density multiplier near the mask boundary
        verbose        : print mesh statistics
    """
    h_coarse = min((xmax - xmin) / nx, (ymax - ymin) / ny)
    h_fine   = h_coarse / refine_factor
    buffer   = 2.0 * h_coarse
    cx, cy   = mask_center

    if verbose:
        print(f"Generating adaptive P3 mesh: {nx}x{ny} background, "
              f"x{refine_factor} near mask (r={mask_radius:.2f})")
        print(f"  h_coarse={h_coarse:.4f}  h_fine={h_fine:.4f}  "
              f"annular buffer=+-{buffer:.4f}")

    xi_c = np.arange(xmin, xmax + h_coarse / 2, h_coarse)
    yi_c = np.arange(ymin, ymax + h_coarse / 2, h_coarse)
    XX_c, YY_c = np.meshgrid(xi_c, yi_c)
    r_c = np.sqrt((XX_c - cx) ** 2 + (YY_c - cy) ** 2)
    far = np.abs(r_c - mask_radius) >= buffer
    coarse_pts = np.column_stack([XX_c[far], YY_c[far]])

    xi_f = np.arange(xmin, xmax + h_fine / 2, h_fine)
    yi_f = np.arange(ymin, ymax + h_fine / 2, h_fine)
    XX_f, YY_f = np.meshgrid(xi_f, yi_f)
    r_f = np.sqrt((XX_f - cx) ** 2 + (YY_f - cy) ** 2)
    near = np.abs(r_f - mask_radius) < buffer
    fine_pts = np.column_stack([XX_f[near], YY_f[near]])

    tol     = 1e-10
    all_pts = np.vstack([coarse_pts, fine_pts])
    in_dom  = (
        (all_pts[:, 0] >= xmin - tol) & (all_pts[:, 0] <= xmax + tol) &
        (all_pts[:, 1] >= ymin - tol) & (all_pts[:, 1] <= ymax + tol)
    )
    vertices = all_pts[in_dom]

    if verbose:
        print(f"  Point cloud: {len(coarse_pts)} coarse + {len(fine_pts)} fine "
              f"= {len(vertices)} vertices")

    tri       = Delaunay(vertices)
    simplices = tri.simplices.copy()

    # Enforce CCW orientation and drop degenerate triangles
    v0p = vertices[simplices[:, 0]]
    v1p = vertices[simplices[:, 1]]
    v2p = vertices[simplices[:, 2]]
    cross = ((v1p[:, 0] - v0p[:, 0]) * (v2p[:, 1] - v0p[:, 1]) -
             (v1p[:, 1] - v0p[:, 1]) * (v2p[:, 0] - v0p[:, 0]))
    simplices[cross < 0] = simplices[cross < 0][:, [0, 2, 1]]
    simplices = simplices[np.abs(cross) > 1e-15]

    if verbose:
        print(f"  P1 triangles: {len(simplices)}")

    mesh = _elevate_to_p3(vertices, simplices, xmin, xmax, ymin, ymax)

    if verbose:
        nd  = np.array(mesh.nodes)
        el  = np.array(mesh.elements)
        bnd = np.array(mesh.boundary)
        print(f"  P3 mesh: {len(nd)} nodes, {len(el)} elements, "
              f"{len(bnd)} boundary nodes")

    return mesh


def visualize_p3_mesh(mesh, filename: str = 'p3_mesh_structure.png',
                      show_nodes: bool = True):
    """Save a plot of the mesh showing vertex, edge, and interior nodes."""
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)

    vertex_counts = np.zeros(len(nodes), dtype=int)
    for elem in elements:
        for i in range(3):
            vertex_counts[elem[i]] += 1
    is_vertex   = vertex_counts > 1
    is_interior = np.zeros(len(nodes), dtype=bool)
    for elem in elements:
        is_interior[elem[9]] = True
    is_edge = ~is_vertex & ~is_interior

    fig, ax = plt.subplots(figsize=(12, 10))
    for elem in elements:
        tri = np.array([nodes[elem[i]] for i in range(3)])
        ax.add_patch(Polygon(tri, fill=False, edgecolor='#444',
                             linewidth=1.2, alpha=0.5))
    if show_nodes:
        for pts, c, m, lbl in [
            (nodes[is_vertex],   '#ff4444', 'o', 'Vertices'),
            (nodes[is_edge],     '#4444ff', 's', 'Edge nodes'),
            (nodes[is_interior], '#44ff44', '^', 'Interior nodes'),
        ]:
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=c, s=60, marker=m,
                           edgecolors='white', lw=1, label=lbl, zorder=5)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('P3 Mesh Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_single_p3_element(mesh, elem_idx: int = 0,
                                 filename: str = 'p3_element_detail.png'):
    """Save a labeled diagram of a single P3 element showing all 10 DOF locations."""
    nodes = np.array(mesh.nodes)
    elem  = np.array(mesh.elements[elem_idx])
    ec    = nodes[elem]
    labels  = ['v0', 'v1', 'v2', 'n3(1/3)', 'n4(2/3)',
               'n5', 'n6', 'n7', 'n8', 'n9(int)']
    colors  = ['#ff4444'] * 3 + ['#4444ff'] * 6 + ['#44ff44']
    markers = ['o'] * 3 + ['s'] * 6 + ['^']
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(Polygon(ec[:3], fill=True, facecolor='#f0f0f0',
                         edgecolor='black', lw=2.5, alpha=0.3))
    for i, (pt, lbl, c, m) in enumerate(zip(ec, labels, colors, markers)):
        ax.scatter(*pt, c=c, s=150, marker=m, edgecolors='white', lw=2, zorder=5)
        ax.annotate(f'{i}: {lbl}', pt + 0.04, fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'P3 Element {elem_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()