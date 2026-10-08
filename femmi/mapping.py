"""Catalogue-native mass mapping with explicit configuration and reusable solves.

Coordinates and correlation lengths are in arcminutes on an East/North tangent
plane. Inputs are shear (not reduced shear) for one effective source plane.
Weights are used as supplied; there is no implicit noise or weight rescaling.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
from .io import FlatCatalog
from .quadratic import QuadraticMAP


@dataclass(frozen=True)
class MapperConfig:
    method: str
    lam: float
    length: float
    radius: float
    center: tuple = (0., 0.)
    rtol: float = 1e-8
    residual_tolerance: float = 1e-6
    maxiter: int = 2000
    observable: str = 'shear'
    source_plane: str = 'effective'
    boundary_padding: float = 1.12
    boundary_nodes: int | None = None

    def __post_init__(self):
        if self.method not in ('p3', 'argyris', 'hct'):
            raise ValueError('method must be p3, argyris or hct')
        for name in ('lam', 'radius', 'rtol', 'residual_tolerance'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.rtol > self.residual_tolerance or self.residual_tolerance >= 1:
            raise ValueError('require rtol <= residual_tolerance < 1')
        if not np.isfinite(self.length) or self.length < 0:
            raise ValueError('length must be finite and nonnegative')
        if np.shape(self.center) != (2,) or not np.all(np.isfinite(self.center)):
            raise ValueError('center must contain two finite coordinates')
        object.__setattr__(self, 'center', tuple(self.center))
        if not isinstance(self.maxiter, int) or isinstance(self.maxiter, bool) or self.maxiter < 1:
            raise ValueError('maxiter must be a positive integer')
        if not np.isfinite(self.boundary_padding) or self.boundary_padding <= 1:
            raise ValueError('boundary_padding must exceed one')
        if self.boundary_nodes is not None and (not isinstance(self.boundary_nodes,int) or isinstance(self.boundary_nodes,bool) or self.boundary_nodes<12):
            raise ValueError('boundary_nodes must be an integer >=12')
        if self.observable != 'shear' or self.source_plane != 'effective':
            raise ValueError('only shear at one effective source plane is supported')

    @classmethod
    def from_file(cls, path):
        import yaml
        return cls(**yaml.safe_load(Path(path).read_text()))

    def save(self, path):
        import yaml
        d = asdict(self); d['center'] = list(self.center)
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False))


def validated_catalogue(catalogue):
    """Snapshot inputs; reject implicit selection, nonfinite data and duplicates."""
    if getattr(catalogue, 'units', 'arcmin') != 'arcmin':
        raise ValueError('convert catalogue coordinates to arcmin first')
    arrays = [np.array(getattr(catalogue, key), dtype=np.float64, copy=True)
              for key in ('x', 'y', 'g1', 'g2', 'weight')]
    if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays) or len(arrays[0]) < 3:
        raise ValueError('catalogue arrays must have matching lengths >=3')
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('select finite observations before mapping')
    if np.any(arrays[-1] < 0) or not np.any(arrays[-1] > 0):
        raise ValueError('weights must be nonnegative with positive total')
    xy = np.column_stack(arrays[:2])
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('combine duplicate positions explicitly before mapping')
    for a in arrays:
        a.flags.writeable = False
    return FlatCatalog(*arrays, center=getattr(catalogue, 'center', (0., 0.)),
                       name=getattr(catalogue, 'name', ''),z=getattr(catalogue,'z',None),
                       meta=dict(getattr(catalogue,'meta',{})),row_index=getattr(catalogue,'row_index',None),
                       object_id=getattr(catalogue,'object_id',None))


@dataclass
class MassMap:
    kappa: np.ndarray
    coefficients: np.ndarray
    observed_g1: np.ndarray
    observed_g2: np.ndarray
    predicted_g1: np.ndarray
    predicted_g2: np.ndarray
    diagnostics: dict
    config: MapperConfig
    mapper: 'FEMMapper'

    def evaluate(self, points):
        """Evaluate the actual FE field; return NaN outside the mesh."""
        return self.mapper.evaluate(self.coefficients, points)

    def save(self, path):
        """Portable arrays and JSON metadata, loadable with allow_pickle=False."""
        c = self.mapper.catalogue
        np.savez_compressed(path, x=c.x, y=c.y, weight=c.weight,
            g1=self.observed_g1, g2=self.observed_g2, kappa=self.kappa, coefficients=self.coefficients,
            predicted_g1=self.predicted_g1, predicted_g2=self.predicted_g2,
            vertices=self.mapper.vertices, triangles=self.mapper.triangles,
            metadata=json.dumps(dict(config=asdict(self.config), diagnostics=self.diagnostics,
                catalogue=dict(units=c.units, tangent_point_degrees=c.center, name=c.name))))


class FEMMapper:
    """One fixed geometry/factorization; multiple shear realizations and priors.

    Inspired by SMPy's mapper/config/result separation. No plotting, truth,
    calibration, or filesystem writes occur during construction or fitting.
    """
    def __init__(self, catalogue, config):
        self.config = config
        self.catalogue = c = validated_catalogue(catalogue)
        self.kind = config.method
        xy = np.column_stack([c.x, c.y])
        if np.any(np.linalg.norm(xy-np.asarray(config.center), axis=1) > config.radius):
            raise ValueError('catalogue extends beyond the configured field radius')
        start = time.perf_counter()
        nb = config.boundary_nodes or max(18, 3*int(np.ceil(2*np.sqrt(len(c.x))/3)))
        radius = config.boundary_padding*config.radius
        if self.kind == 'p3':
            from .operators import build_operators_catalog
            ops, cm = build_operators_catalog(c.x,c.y,center=config.center,radius=radius,
                n_boundary=nb,dedup_radius=0.,guard_ring=False,verbose=False)
            if not np.array_equal(cm.source_index,np.arange(len(c.x))):
                raise ValueError('mesh changed catalogue selection')
            idx = cm.galaxy_nodes
            S1,S2 = ops.S1[idx],ops.S2[idx]
            lu,zero = ops.A_coupled_lu,ops._rhs_zero_nodes()
            self.value_indices = idx
            self.vertices = np.asarray(ops.mesh.nodes)
            self.elements = np.asarray(ops.mesh.elements)
            self.triangles = self.elements[:,:3]
        else:
            from .elements import C1Space,catalog_triangulation
            from .c1_coupling import C1CoupledOperators
            from .c1_inverse import shear_operators
            v,t,ring,idx = catalog_triangulation(c.x,c.y,radius=radius,
                center=config.center,n_boundary=nb,dedup=0.)
            if np.any(idx<0) or len(np.unique(idx)) != len(c.x):
                raise ValueError('mesh changed catalogue selection')
            self.space = C1Space(v,t,self.kind)
            ops = C1CoupledOperators(self.space,degree=5 if self.kind=='argyris' else 3)
            S1,S2 = (s[idx] for s in shear_operators(self.space))
            lu,zero = ops.A_lu,[ops.idx_gauge]
            self.value_indices = idx*self.space.n_vert_dofs
            self.vertices,self.triangles = v,t
        self.solver = QuadraticMAP(ops.M,ops.K,S1,S2,lu,zero,c.weight)
        self.dofs = ops.M.shape[0]
        self.setup_seconds = time.perf_counter()-start
        self._finder = None

    def reconstruct(self, g1=None, g2=None, *, lam=None, length=None):
        """Solve weighted quadratic MAP, reusing the mesh and coupled LU.

        lambda multiplies M + length**2 K in the same units as supplied weights.
        Multiply lambda by the same factor when uniformly rescaling weights.
        """
        c, cfg = self.catalogue, self.config
        if (g1 is None) != (g2 is None):
            raise ValueError('supply both shear components or neither')
        a,b = (c.g1,c.g2) if g1 is None else (g1,g2)
        lam = cfg.lam if lam is None else lam
        length = cfg.length if length is None else length
        start = time.perf_counter()
        k, info = self.solver.solve(a,b,lam,length,rtol=cfg.rtol,maxiter=cfg.maxiter,
                                   residual_tolerance=cfg.residual_tolerance)
        p,q = self.solver.forward(k)
        info.update(lam_used=float(lam),wiener_length=float(length),
                    setup_seconds=self.setup_seconds,solve_seconds=time.perf_counter()-start,
                    weighted_shear_residual=float(np.dot(c.weight,(p-a)**2+(q-b)**2)))
        from dataclasses import replace
        return MassMap(k[self.value_indices],k,np.array(a,copy=True),np.array(b,copy=True),p,q,info,replace(cfg,lam=lam,length=length),self)

    def diagnose_b(self, g1=None, g2=None, *, e_fit=None, b_fit=None):
        """Split the rotated-shear fit into predicted E leakage and residual.

        This is conditional on the fitted E model. Neither the residual map nor
        its norm is a pure-B estimator or a calibrated significance statistic.
        Supplied fits must use this mapper, the same observations and prior.
        """
        from .diagnostics import b_response
        return b_response(self,g1,g2,e_fit=e_fit,b_fit=b_fit)

    def evaluate(self, coefficients, points):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (self.dofs,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be a finite vector of mapper.dofs entries")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError('points must be finite with shape (n,2)')
        if self._finder is None:
            from matplotlib.tri import Triangulation
            self._finder = Triangulation(self.vertices[:,0],self.vertices[:,1],self.triangles).get_trifinder()
        cells = self._finder(points[:,0],points[:,1])
        values = np.full(len(points),np.nan)
        for t in np.unique(cells[cells>=0]):
            select = cells==t
            if self.kind != 'p3':
                values[select] = self.space.eval_value(coefficients,t,points[select])
            else:
                # Cubic Lagrange basis in mesh.py's vertex/edge/centroid order.
                v = self.vertices[self.triangles[t]]
                uv = np.linalg.solve((v[1:]-v[0]).T,(points[select]-v[0]).T).T
                b,c = uv.T; a = 1-b-c
                phi = np.array([a*(3*a-1)*(3*a-2)/2,b*(3*b-1)*(3*b-2)/2,
                    c*(3*c-1)*(3*c-2)/2,4.5*a*b*(3*a-1),4.5*a*b*(3*b-1),
                    4.5*b*c*(3*b-1),4.5*b*c*(3*c-1),4.5*c*a*(3*c-1),
                    4.5*c*a*(3*a-1),27*a*b*c]).T
                values[select] = phi @ coefficients[self.elements[t]]
        return values

    def select_regularization(self, lambdas, lengths, *, folds=3, seed=0):
        """Choose prior by held-out shear prediction, with fixed geometry.

        Training rows alone enter each likelihood. Validation positions may be
        mesh vertices but their shear never enters the training fit. Returns all
        scores; boundary winners are flagged and require a wider caller grid.
        """
        from .quadratic import QuadraticMAP
        positive = np.flatnonzero(self.catalogue.weight>0)
        if not isinstance(folds,int) or not 2<=folds<=len(positive):
            raise ValueError('folds must be between 2 and the positive-weight count')
        if not len(lambdas) or not len(lengths):
            raise ValueError('supply nonempty lambda and length grids')
        blocks = np.array_split(np.random.default_rng(seed).permutation(positive),folds)
        s = self.solver; c = self.catalogue
        scores = []
        for lam in lambdas:
            for length in lengths:
                error=0.; weight=0.
                for held in blocks:
                    w=c.weight.copy(); w[held]=0
                    train=QuadraticMAP(s.M,s.K,s.S1,s.S2,s.lu,s.zero_nodes,w)
                    k,_=train.solve(c.g1,c.g2,lam,length,rtol=self.config.rtol,
                        maxiter=self.config.maxiter,residual_tolerance=self.config.residual_tolerance)
                    a,b=train.forward(k)
                    error+=np.dot(c.weight[held],(a[held]-c.g1[held])**2+(b[held]-c.g2[held])**2)
                    weight+=c.weight[held].sum()
                scores.append(dict(lam=float(lam),length=float(length),score=float(error/(2*weight))))
        best=min(scores,key=lambda row:row['score'])
        edge=(best['lam'] in (min(lambdas),max(lambdas)) or best['length']==max(lengths)
              or (best['length']==min(lengths) and best['length']>0))
        return dict(best=best,candidates=scores,boundary_unresolved=edge,folds=folds,seed=seed,
                    criterion='held-out weighted shear MSE; no convergence truth')


def map_catalogue(catalogue, config):
    """Convenience entry point; use FEMMapper directly to reuse the setup."""
    return FEMMapper(catalogue, config).reconstruct()
