"""Joint E/B Gaussian MAP on one finite-element geometry.

With J(a,b)=(-b,a), predict d=F e+J F b. Separate rotated-shear
fits omit the cross blocks of this inverse problem. Joint MAP includes them,
but cannot identify ambiguous modes independently of the assumed priors.
"""
from dataclasses import dataclass, asdict
import json
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, splu
from .quadratic import _cg
from .observations import prepare_observations


@dataclass
class JointMassMap:
    """Joint posterior mode; E/B maps are conditional on both priors.

    ``predicted_g1/g2`` are the summed prediction, not either component alone.
    ``evaluate`` returns (E, B), each NaN outside the computational mesh.
    """
    e_coefficients: np.ndarray
    b_coefficients: np.ndarray
    observed_g1: np.ndarray
    observed_g2: np.ndarray
    predicted_g1: np.ndarray
    predicted_g2: np.ndarray
    diagnostics: dict
    mapper: object

    @property
    def kappa_e(self):
        return self.e_coefficients[self.mapper.value_indices]

    @property
    def kappa_b(self):
        return self.b_coefficients[self.mapper.value_indices]

    def evaluate(self, points):
        return (self.mapper.evaluate(self.e_coefficients, points),
                self.mapper.evaluate(self.b_coefficients, points))

    def save(self, path):
        c = self.mapper.catalogue
        np.savez_compressed(path, x=c.x, y=c.y, weight=c.weight,
            g1=self.observed_g1, g2=self.observed_g2,
            kappa_e=self.kappa_e, kappa_b=self.kappa_b,
            e_coefficients=self.e_coefficients, b_coefficients=self.b_coefficients,
            predicted_g1=self.predicted_g1, predicted_g2=self.predicted_g2,
            vertices=self.mapper.vertices, triangles=self.mapper.triangles,
            metadata=json.dumps(dict(config=asdict(self.mapper.config),
                                     diagnostics=self.diagnostics)))


def joint_map(mapper, g1=None, g2=None, *, lam_e=None, lam_b=None,
              length_e=None, length_b=None):
    """Solve both fields simultaneously with positive quadratic priors.

    Minimize ||sqrt(W)(F e + J F b - d)||^2 + lam_e e.T R_e e
    + lam_b b.T R_b b, R=M+length^2 K. Defaults use the mapper prior
    for both fields. Changing their ratio changes ambiguous-mode allocation.
    Rotation is inside every normal-operator action; no dense F is formed.
    """
    s, cfg = mapper.solver, mapper.config
    if (g1 is None) != (g2 is None):
        raise ValueError('supply both shear components or neither')
    a, b = (mapper.catalogue.g1, mapper.catalogue.g2) if g1 is None else (g1, g2)
    a, b, w = prepare_observations(a, b, len(s.weight), s.weight)
    le, lb = (cfg.lam if v is None else float(v) for v in (lam_e, lam_b))
    xe, xb = (cfg.length if v is None else float(v) for v in (length_e, length_b))
    if not all(np.isfinite(v) for v in (le, lb, xe, xb)) or min(le, lb) <= 0 or min(xe, xb) < 0:
        raise ValueError('joint MAP requires positive lambdas and nonnegative lengths')
    start = time.perf_counter()
    n = mapper.dofs
    re = (s.M + xe**2 * s.K).tocsc()
    rb = re if xb == xe else (s.M + xb**2 * s.K).tocsc()
    pe = splu(re)
    pb = pe if rb is re else splu(rb)

    def forward(z):
        e1, e2 = s.forward(z[:n])
        b1, b2 = s.forward(z[n:])
        return e1-b2, e2+b1

    def adjoint(u, v):
        return np.r_[s.transpose(u, v), s.transpose(v, -u)]

    def action(z):
        u, v = forward(z)
        return adjoint(w*u, w*v) + np.r_[le*(re@z[:n]), lb*(rb@z[n:])]

    h = LinearOperator((2*n, 2*n), matvec=action, dtype=float)
    p = LinearOperator(h.shape, matvec=lambda z: np.r_[pe.solve(z[:n])/le,
                                                       pb.solve(z[n:])/lb], dtype=float)
    rhs = adjoint(w*a, w*b)
    count = [0]
    def callback(_):
        count[0] += 1
    z, code = _cg(h, rhs, M=p, rtol=cfg.rtol, atol=0., maxiter=cfg.maxiter, callback=callback)
    scale = max(np.linalg.norm(rhs), np.finfo(float).tiny)
    r = rhs-action(z)
    residual = np.linalg.norm(r)/scale
    refinements = 0
    for _ in range(3):
        if code or residual <= 10*cfg.rtol:
            break
        delta, status = _cg(h, r, M=p, rtol=1e-3, atol=0., maxiter=cfg.maxiter, callback=callback)
        if status:
            break
        trial = z+delta
        rr = rhs-action(trial)
        actual = np.linalg.norm(rr)/scale
        refinements += 1
        if actual < residual:
            z, r, residual = trial, rr, actual
    if code or not np.all(np.isfinite(z)) or residual > cfg.residual_tolerance:
        raise RuntimeError(f'joint E/B MAP did not converge: info={code}, residual={residual:.3g}')
    u, v = forward(z)
    data_loss = float(np.dot(w, (u-a)**2+(v-b)**2))
    penalty = float(le*z[:n]@(re@z[:n])+lb*z[n:]@(rb@z[n:]))
    # A global check alone can hide a poorly solved, small-amplitude B block.
    # Record block gradients in common RHS units (well-defined for zero B).
    info = dict(converged=True, solver='joint-prior-preconditioned-cg',
        iterations=count[0], refinements=refinements, relative_residual=float(residual),
        block_residuals=[float(np.linalg.norm(rr)/scale) for rr in (r[:n], r[n:])],
        residual_tolerance=cfg.residual_tolerance, objective=data_loss+penalty,
        weighted_shear_residual=data_loss, lam_e=le, lam_b=lb, length_e=xe, length_b=xb,
        solve_seconds=time.perf_counter()-start,
        interpretation='joint E/B posterior mode, conditional on both priors; not pure E/B')
    return JointMassMap(z[:n], z[n:], a.copy(), b.copy(), u, v, info, mapper)


def sampled_eb_overlap(mapper, *, rtol=1e-9, max_entries=2_000_000):
    """Small-problem rank diagnostic in the weighted observation space.

    Form sqrt(W) F, excluding zero-weight sources. A full row rank means ANY
    sampled shear can be represented as E (and also as B); exact removal of all
    represented E modes would then remove all data. This is a discrete sampling
    result, not a continuum pure-mode decomposition. Dense storage is capped.
    The numerical rank depends on rtol; repeat at nearby tolerances.
    """
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('rtol must be in (0,1)')
    s = mapper.solver
    keep = s.weight > 0
    rows = 2*int(keep.sum())
    if rows*mapper.dofs > max_entries:
        raise ValueError('dense overlap diagnostic exceeds max_entries')
    f = np.empty((rows, mapper.dofs))
    sw = np.sqrt(s.weight[keep])
    unit = np.zeros(mapper.dofs)
    for j in range(mapper.dofs):
        unit[j] = 1.
        a, b = s.forward(unit)
        f[:, j] = np.r_[sw*a[keep], sw*b[keep]]
        unit[j] = 0.
    u, singular, _ = np.linalg.svd(f, full_matrices=False)
    rank = int(np.count_nonzero(singular > rtol*singular[0])) if singular.size else 0
    q = u[:, :rank]
    half = rows//2
    jq = np.concatenate([-q[half:], q[:half]])
    cosines = np.linalg.svd(q.T@jq, compute_uv=False)
    return dict(observation_dimension=rows, e_rank=rank, b_rank=rank,
        full_row_rank=bool(rank == rows), rank_rtol=rtol,
        singular_values=singular.tolist(), principal_cosines=cosines.tolist(),
        interpretation='weighted sampled forward-space overlap; not continuum pure-mode count')
