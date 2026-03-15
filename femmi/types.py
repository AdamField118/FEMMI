# femmi/types.py
from typing import NamedTuple
import numpy as np

class Mesh(NamedTuple):
    nodes: np.ndarray
    elements: np.ndarray
    boundary: np.ndarray
