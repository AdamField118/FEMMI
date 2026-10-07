"""Unit-balanced sparse LU with an explicitly matched transpose action."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class EquilibratedLU:
    """Factor D A D; return A^-1 b = D (D A D)^-1 D b.

    C1 coefficients mix values and physical derivatives. Their stiffness
    diagonals can span many orders of magnitude. Congruence scaling improves
    float64 solves without changing the matrix equation or gauge. This remains
    a SuperLU direct solve; transpose solves apply the same scaling on both sides.
    """
    def __init__(self,A):
        A=sp.csc_matrix(A,dtype=float)
        diag=np.abs(A.diagonal())
        if not np.all(np.isfinite(diag)) or np.any(diag==0):
            raise ValueError('equilibration requires finite nonzero diagonal')
        self.scale=1/np.sqrt(diag)
        D=sp.diags(self.scale)
        self.factor=splu((D@A@D).tocsc())

    def solve(self,rhs,trans='N'):
        rhs=np.asarray(rhs,float)
        d=self.scale if rhs.ndim==1 else self.scale[:,None]
        return d*self.factor.solve(d*rhs,trans=trans)
