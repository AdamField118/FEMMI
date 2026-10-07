"""Converged quadratic MAP solves with the exact prior as a preconditioner.

Minimize ||sqrt(W)(F k-d)||² + lambda k^T R k. The objective and
normalization agree with MAPReconstructor/C1MAPReconstructor. This solver is
for positive-definite quadratic priors only; it does not replace score sampling
or nonquadratic L-BFGS. Failed solves raise instead of entering calibration.
"""
import numpy as np
import scipy.sparse.linalg as spla
from inspect import signature

_CG_TOL = "rtol" if "rtol" in signature(spla.cg).parameters else "tol"


def _cg(A,b,*,rtol,**kwargs):
    """Respect the project's SciPy >=1.10 support (tol was renamed rtol)."""
    return spla.cg(A,b,**{_CG_TOL:rtol},**kwargs)


class QuadraticMAP:
    def __init__(self, M, K, S1, S2, lu, zero_nodes, weight):
        self.M, self.K = M, K
        self.S1, self.S2 = S1, S2
        self.lu = lu
        self.zero_nodes = np.atleast_1d(zero_nodes)
        from .observations import observation_weights
        self.weight = observation_weights(S1.shape[0], weight)
        self._length = None

    def forward(self, k):
        rhs = -2.*(self.M @ k)
        rhs[self.zero_nodes] = 0.
        psi = self.lu.solve(rhs)
        return self.S1 @ psi, self.S2 @ psi

    def transpose(self, y1, y2):
        adj = self.lu.solve(self.S1.T @ y1+self.S2.T @ y2, trans='T')
        adj[self.zero_nodes] = 0.
        return -2.*(self.M.T @ adj)

    def solve(self, g1, g2, lam, length, x0=None, rtol=1e-8, maxiter=2000):
        from .observations import prepare_observations
        g1,g2,w = prepare_observations(g1,g2,len(self.weight),self.weight)
        if not np.isfinite(lam) or lam<=0 or not np.isfinite(length) or length<0:
            raise ValueError('quadratic MAP requires lambda>0 and length>=0')
        # R=M at length zero is proper; legacy length=0 H1-only remains a
        # distinct option in the old MAP APIs and is not this calibration family.
        if length != self._length:
            self.R = (self.M+length**2*self.K).tocsc()
            self.Rlu = spla.splu(self.R)
            self._length = length
        n = self.M.shape[0]
        def action(k):
            a,b = self.forward(k)
            return self.transpose(w*a,w*b)+lam*(self.R@k)
        H = spla.LinearOperator((n,n),matvec=action,dtype=float)
        P = spla.LinearOperator((n,n),matvec=self.Rlu.solve,dtype=float)
        rhs = self.transpose(w*g1,w*g2)
        count = [0]
        def callback(_):
            count[0] += 1
        k,code = _cg(H,rhs,x0=x0,M=P,rtol=rtol,atol=0.,maxiter=maxiter,callback=callback)
        rhs_norm=max(np.linalg.norm(rhs),1e-300)
        r=rhs-action(k);residual=np.linalg.norm(r)/rhs_norm
        # Recursive CG residuals can drift on ill-conditioned C1 systems.
        # Refine using freshly evaluated residuals and retain only improvements.
        refinements=0
        for _ in range(3):
            if code or residual<=10*rtol:break
            delta,correction_code=_cg(H,r,M=P,rtol=1e-3,atol=0.,maxiter=maxiter)
            if correction_code:break
            trial=k+delta;rr=rhs-action(trial);actual=np.linalg.norm(rr)/rhs_norm
            refinements+=1
            if actual<residual:k,r,residual=trial,rr,actual
        # A stricter internal target leaves room for measured float64 residual
        # drift. The ACCEPTANCE tolerance is explicit and recorded, not CG's
        # success flag. It is 1e-6 at the default internal rtol=1e-8.
        acceptance=max(100*rtol,1e-10)
        if code or not np.all(np.isfinite(k)) or residual>acceptance:
            raise RuntimeError(f'quadratic MAP did not converge: info={code}, residual={residual:.3g}')
        a,b = self.forward(k)
        loss = np.dot(w*(a-g1),a-g1)+np.dot(w*(b-g2),b-g2)+lam*np.dot(k,self.R@k)
        return k,dict(converged=True,iterations=count[0],relative_residual=float(residual),
                      objective=float(loss),residual_tolerance=float(acceptance),refinements=refinements,solver='prior-preconditioned-cg')
