"""
tests/test_catalog_mesh.py
Tests for the catalog-native P3 mesh (femmi.mesh.generate_p3_catalog_mesh):
boundary topology divisible by 3, compatibility with the circular BEM
extractor, exact galaxy->node mapping, dedup, near-boundary voids, and
non-circular footprints.

Run:
    python -m pytest tests/test_catalog_mesh.py -v
    python tests/test_catalog_mesh.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.mesh import generate_p3_catalog_mesh, min_triangle_angle
from femmi.bem  import extract_boundary_edges_circular


def _check(cm, x, y):
    nd  = np.array(cm.mesh.nodes)
    bnd = np.array(cm.mesh.boundary)
    cx, cy = cm.center

    assert len(bnd) % 3 == 0

    bm = extract_boundary_edges_circular(cm.mesh, center=cm.center, radius=cm.radius)
    assert bm.n_boundary_dofs == len(bnd)
    assert bm.n_elements == len(bnd) // 3

    # closed, continuous element loop
    assert bm.elements[-1, 3] == bm.elements[0, 0]
    for e in range(bm.n_elements - 1):
        assert bm.elements[e, 3] == bm.elements[e + 1, 0]

    # outward unit normals
    assert np.allclose(np.linalg.norm(bm.element_normals, axis=1), 1.0, atol=1e-10)
    mids = 0.5 * (bm.nodes[bm.elements[:, 0]] + bm.nodes[bm.elements[:, 3]])
    radial = mids - np.array([cx, cy])
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-30
    assert np.all(np.sum(bm.element_normals * radial, axis=1) > 0)

    # exact galaxy -> node mapping
    xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    assert np.allclose(nd[cm.galaxy_nodes], xy[cm.source_index], atol=1e-12)
    assert len(set(cm.galaxy_nodes.tolist()) & set(bnd.tolist())) == 0

    # all boundary nodes sit near the ring radius
    rb = np.hypot(nd[bnd, 0] - cx, nd[bnd, 1] - cy)
    assert rb.min() > 0.9 * cm.radius


def test_uniform_disk():
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 800)
    rr = 2.0 * np.sqrt(rng.uniform(0, 1, 800))
    x, y = rr * np.cos(th), rr * np.sin(th)
    cm = generate_p3_catalog_mesh(x, y, n_boundary=96, verbose=False)
    _check(cm, x, y)


def test_cluster_like_auto_center():
    rng = np.random.default_rng(1)
    g = rng.normal(0, 0.6, (1200, 2)) + np.array([1.0, -0.5])
    cm = generate_p3_catalog_mesh(g[:, 0], g[:, 1], n_boundary=120, verbose=False)
    _check(cm, g[:, 0], g[:, 1])
    # auto center near the cloud mean
    assert abs(cm.center[0] - 1.0) < 0.2 and abs(cm.center[1] + 0.5) < 0.2


def test_dedup():
    rng = np.random.default_rng(2)
    base = rng.uniform(-2, 2, (500, 2))
    dup = np.vstack([base, base[:50] + 1e-4 * rng.standard_normal((50, 2))])
    cm = generate_p3_catalog_mesh(dup[:, 0], dup[:, 1], n_boundary=96,
                                  dedup_radius=1e-2, verbose=False)
    _check(cm, dup[:, 0], dup[:, 1])
    assert len(cm.galaxy_nodes) <= 500  # near-duplicates merged


def test_near_boundary_voids():
    rng = np.random.default_rng(3)
    core = rng.normal(0, 0.5, (600, 2))
    sparse = rng.uniform(-2.5, 2.5, (40, 2))
    xy = np.vstack([core, sparse])
    cm = generate_p3_catalog_mesh(xy[:, 0], xy[:, 1], n_boundary=96,
                                  guard_ring=True, verbose=False)
    _check(cm, xy[:, 0], xy[:, 1])
    assert cm.n_guard > 0


def test_rectangular_footprint():
    rng = np.random.default_rng(4)
    x = rng.uniform(-3, 3, 700)
    y = rng.uniform(-1.2, 1.2, 700)
    cm = generate_p3_catalog_mesh(x, y, n_boundary=120, verbose=False)
    _check(cm, x, y)


def test_n_boundary_forced_multiple_of_3():
    rng = np.random.default_rng(5)
    g = rng.normal(0, 0.6, (800, 2))
    cm = generate_p3_catalog_mesh(g[:, 0], g[:, 1], n_boundary=100, verbose=False)
    assert len(np.array(cm.mesh.boundary)) % 3 == 0
    _check(cm, g[:, 0], g[:, 1])


def test_min_triangle_angle_positive():
    rng = np.random.default_rng(6)
    g = rng.normal(0, 0.6, (400, 2))
    cm = generate_p3_catalog_mesh(g[:, 0], g[:, 1], n_boundary=96, verbose=False)
    ang = min_triangle_angle(cm.mesh)
    assert ang > 0.0  # sanity; raw catalogs give slivers, this just guards finiteness


if __name__ == "__main__":
    tests = [
        test_uniform_disk,
        test_cluster_like_auto_center,
        test_dedup,
        test_near_boundary_voids,
        test_rectangular_footprint,
        test_n_boundary_forced_multiple_of_3,
        test_min_triangle_angle_positive,
    ]
    passed, failed = 0, []
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
