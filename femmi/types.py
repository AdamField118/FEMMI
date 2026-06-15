from typing import NamedTuple, Optional
import numpy as np


class Mesh(NamedTuple):
    nodes: np.ndarray
    elements: np.ndarray
    boundary: np.ndarray
    
class CatalogMesh(NamedTuple):
    """
    A catalog-native P3 mesh plus the bookkeeping needed to attach observations.

    mesh         : Mesh (FEM nodes, P3 elements, boundary node indices)
    galaxy_nodes : node index of each surviving galaxy (interior nodes)
    source_index : original catalog row of each surviving galaxy
    center       : (cx, cy) of the boundary ring, in the flat (x, y) frame
    radius       : boundary ring radius
    n_guard      : number of guard-ring points added to protect the boundary

    Attach observed shear to nodes with:
        g1n = np.zeros(ops.n_nodes); g1n[cm.galaxy_nodes] = flat.g1[cm.source_index]
        g2n = np.zeros(ops.n_nodes); g2n[cm.galaxy_nodes] = flat.g2[cm.source_index]
    """
    mesh         : object
    galaxy_nodes : np.ndarray
    source_index : np.ndarray
    center       : tuple
    radius       : float
    n_guard      : int = 0