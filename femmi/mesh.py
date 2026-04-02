"""
femmi/mesh.py
P3 triangular mesh generation.

generate_p3_structured_mesh: uniform structured grid
generate_p3_adaptive_mesh:   locally refined near a circular mask boundary

Both produce 10-node P3 elements with node ordering:
    [v0, v1, v2, n3, n4, n5, n6, n7, n8, n9]
     vertices    e01    e12    e20   centroid
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from .types import Mesh


def _elevate_to_p3(vertices, p1_elements, xmin, xmax, ymin, ymax):
    """Elevate a P1 triangulation to P3 by inserting edge and interior nodes."""
    nodes_list = list(vertices)
    next_idx   = len(vertices)
    edge_dict  = {}

    def get_edge_nodes(va, vb):
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
        n9 = next_idx
        nodes_list.append((vertices[v0] + vertices[v1] + vertices[v2]) / 3.0)
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


def generate_p3_structured_mesh(nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                                 return_numpy=False):
    """
    Generate a P3 structured triangular mesh on a rectangular domain.

    Each cell is split into two triangles; edge nodes at t=1/3, t=2/3 and
    one centroid node per triangle give 10 DOFs per element.
    """
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    xx, yy = np.meshgrid(x, y)

    vertex_nodes = np.column_stack([xx.ravel(), yy.ravel()])
    n_vertices   = len(vertex_nodes)

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

    def get_edge_nodes(v1, v2):
        nonlocal next_node_idx
        key = tuple(sorted([v1, v2]))
        if key not in edge_nodes:
            i, j = key
            pi, pj = vertex_nodes[i], vertex_nodes[j]
            nodes_list.append((2.0 * pi + 1.0 * pj) / 3.0)
            nodes_list.append((1.0 * pi + 2.0 * pj) / 3.0)
            edge_nodes[key] = [next_node_idx, next_node_idx + 1]
            next_node_idx += 2
        i0, i1 = edge_nodes[key]
        return (i0, i1) if v1 < v2 else (i1, i0)

    p3_elements = []
    for v0, v1, v2 in p1_elements:
        n3, n4 = get_edge_nodes(v0, v1)
        n5, n6 = get_edge_nodes(v1, v2)
        n7, n8 = get_edge_nodes(v2, v0)
        n9 = next_node_idx
        nodes_list.append((vertex_nodes[v0] + vertex_nodes[v1] + vertex_nodes[v2]) / 3.0)
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


def generate_p3_adaptive_mesh(nx, ny, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
                               mask_center=(0.0, 0.0), mask_radius=0.5,
                               refine_factor=3, verbose=True):
    """
    Generate a P3 mesh with local refinement near a circular mask boundary.

    Coarse spacing h = domain/nx everywhere except the annular band
    |r - r_mask| < 2*h_coarse, which uses h/refine_factor.
    """
    h_coarse = min((xmax - xmin) / nx, (ymax - ymin) / ny)
    h_fine   = h_coarse / refine_factor
    buffer   = 2.0 * h_coarse
    cx, cy   = mask_center

    if verbose:
        print(f"Adaptive mesh: {nx}x{ny} background, x{refine_factor} near mask "
              f"(r={mask_radius:.2f})  h_coarse={h_coarse:.4f}  h_fine={h_fine:.4f}")

    xi_c = np.arange(xmin, xmax + h_coarse / 2, h_coarse)
    yi_c = np.arange(ymin, ymax + h_coarse / 2, h_coarse)
    XX_c, YY_c = np.meshgrid(xi_c, yi_c)
    r_c = np.sqrt((XX_c - cx)**2 + (YY_c - cy)**2)
    coarse_pts = np.column_stack([XX_c[np.abs(r_c - mask_radius) >= buffer],
                                  YY_c[np.abs(r_c - mask_radius) >= buffer]])

    xi_f = np.arange(xmin, xmax + h_fine / 2, h_fine)
    yi_f = np.arange(ymin, ymax + h_fine / 2, h_fine)
    XX_f, YY_f = np.meshgrid(xi_f, yi_f)
    r_f = np.sqrt((XX_f - cx)**2 + (YY_f - cy)**2)
    fine_pts = np.column_stack([XX_f[np.abs(r_f - mask_radius) < buffer],
                                YY_f[np.abs(r_f - mask_radius) < buffer]])

    tol     = 1e-10
    all_pts = np.vstack([coarse_pts, fine_pts])
    in_dom  = (
        (all_pts[:, 0] >= xmin - tol) & (all_pts[:, 0] <= xmax + tol) &
        (all_pts[:, 1] >= ymin - tol) & (all_pts[:, 1] <= ymax + tol)
    )
    vertices = all_pts[in_dom]

    if verbose:
        print(f"  {len(vertices)} vertices total")

    tri       = Delaunay(vertices)
    simplices = tri.simplices.copy()

    v0p = vertices[simplices[:, 0]]
    v1p = vertices[simplices[:, 1]]
    v2p = vertices[simplices[:, 2]]
    cross = ((v1p[:, 0] - v0p[:, 0]) * (v2p[:, 1] - v0p[:, 1]) -
             (v1p[:, 1] - v0p[:, 1]) * (v2p[:, 0] - v0p[:, 0]))
    simplices[cross < 0] = simplices[cross < 0][:, [0, 2, 1]]
    simplices = simplices[np.abs(cross) > 1e-15]

    mesh = _elevate_to_p3(vertices, simplices, xmin, xmax, ymin, ymax)

    if verbose:
        nd  = np.array(mesh.nodes)
        el  = np.array(mesh.elements)
        bnd = np.array(mesh.boundary)
        print(f"  P3: {len(nd)} nodes, {len(el)} elements, {len(bnd)} boundary nodes")

    return mesh

def _elevate_to_p3_circular(vertices, p1_elements, is_bnd_vertex):
    nodes_list = list(vertices)
    next_idx   = len(vertices)
    edge_dict  = {}
    bnd_set    = set(int(i) for i in np.where(is_bnd_vertex)[0])

    def get_edge_nodes(va, vb):
        nonlocal next_idx
        key = (min(va, vb), max(va, vb))
        if key not in edge_dict:
            pi, pj = vertices[key[0]], vertices[key[1]]
            i0, i1 = next_idx, next_idx + 1
            nodes_list.append((2.0 * pi + pj) / 3.0)
            nodes_list.append((pi + 2.0 * pj) / 3.0)
            edge_dict[key] = (i0, i1)
            next_idx += 2
            if is_bnd_vertex[key[0]] and is_bnd_vertex[key[1]]:
                bnd_set.add(i0)
                bnd_set.add(i1)
        i0, i1 = edge_dict[key]
        return (i0, i1) if va < vb else (i1, i0)

    p3_elements = []
    for v0, v1, v2 in p1_elements:
        n3, n4 = get_edge_nodes(v0, v1)
        n5, n6 = get_edge_nodes(v1, v2)
        n7, n8 = get_edge_nodes(v2, v0)
        n9 = next_idx
        nodes_list.append((vertices[v0] + vertices[v1] + vertices[v2]) / 3.0)
        next_idx += 1
        p3_elements.append([v0, v1, v2, n3, n4, n5, n6, n7, n8, n9])

    nodes_arr = np.array(nodes_list, dtype=np.float64)
    bnd       = np.array(sorted(bnd_set), dtype=np.int32)
    return Mesh(nodes=nodes_arr, elements=np.array(p3_elements, dtype=np.int32), boundary=bnd)


def generate_p3_circular_mesh(radius=2.5, n_boundary=60, n_rings=None, center=(0.0, 0.0), verbose=True):
    cx, cy = center
    if n_rings is None:
        n_rings = max(3, n_boundary // 6)

    bnd_angles = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    bnd_pts    = np.column_stack([cx + radius * np.cos(bnd_angles), cy + radius * np.sin(bnd_angles)])

    int_pts = [[cx, cy]]
    for k in range(1, n_rings):
        r_k = radius * k / n_rings
        n_k = max(6, round(n_boundary * k / n_rings))
        ang_k = np.linspace(0.0, 2.0 * np.pi, n_k, endpoint=False)
        for a in ang_k:
            int_pts.append([cx + r_k * np.cos(a), cy + r_k * np.sin(a)])

    int_pts = np.array(int_pts, dtype=np.float64)
    n_int   = len(int_pts)
    all_pts = np.vstack([int_pts, bnd_pts])

    is_bnd_vertex = np.zeros(len(all_pts), dtype=bool)
    is_bnd_vertex[n_int:] = True

    tri       = Delaunay(all_pts)
    simplices = tri.simplices.copy()

    cents   = all_pts[simplices].mean(axis=1) - np.array([cx, cy])
    in_disk = (cents ** 2).sum(axis=1) <= (radius * 1.001) ** 2
    simplices = simplices[in_disk]

    v0p   = all_pts[simplices[:, 0]]
    v1p   = all_pts[simplices[:, 1]]
    v2p   = all_pts[simplices[:, 2]]
    cross = (v1p[:, 0] - v0p[:, 0]) * (v2p[:, 1] - v0p[:, 1]) - (v1p[:, 1] - v0p[:, 1]) * (v2p[:, 0] - v0p[:, 0])
    simplices[cross < 0] = simplices[cross < 0][:, [0, 2, 1]]
    simplices = simplices[np.abs(cross) > 1e-15]

    if verbose:
        print(f"  Circular mesh: R={radius:.2f}  n_bnd={n_boundary}  n_rings={n_rings}  {len(all_pts)} vertices")

    mesh = _elevate_to_p3_circular(all_pts, simplices, is_bnd_vertex)

    if verbose:
        nd, el, bnd = np.array(mesh.nodes), np.array(mesh.elements), np.array(mesh.boundary)
        print(f"  P3: {len(nd)} nodes  {len(el)} elements  {len(bnd)} boundary DOFs")

    return mesh

def visualize_p3_mesh(mesh, filename='p3_mesh_structure.png', show_nodes=True):
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
        ax.add_patch(Polygon(tri, fill=False, edgecolor='#444', linewidth=1.2, alpha=0.5))
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
    ax.set_title('P3 Mesh Structure')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()