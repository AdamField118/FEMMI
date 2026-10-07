"""
femmi/io.py
Catalog ingestion for real weak-lensing shear data (FITS) and the
celestial -> flat-sky transform that FEMMI's solver operates in.

Pipeline:
    read_fits_catalog(path)        -> ShearCatalog   (RA/Dec, shear, weights)
    catalog.to_tangent_plane(...)  -> FlatCatalog     (x/y arcmin, rotated shear)
    FlatCatalog feeds the catalog-native mesh builder (Phase 2).

Conventions (read before trusting any result):
  - Tangent-plane axes are (xi, eta) = (East, North) in the gnomonic (TAN)
    projection, output in arcmin by default. xi increases with RA, eta with Dec.
  - Shear is spin-2: g = g1 + i*g2 transforms as g -> g * exp(-2i*phi) under a
    coordinate rotation by phi. A parity flip x -> -x sends g2 -> -g2.
  - Input g1/g2 are assumed defined on the equatorial (East/North) axes. The
    gnomonic projection induces a small position-dependent frame rotation away
    from the tangent point; to_tangent_plane applies it exactly. For an HST-size
    field it is < 0.1 deg, but it is applied so the tool is correct at any scale.
  - The absolute sign of g2 relative to the sky depends on how the survey
    defined ellipticity. Validate once per survey against the tangential-shear
    pattern of a known cluster (see tangential_cross_shear) and set
    `flip_g2` accordingly. The default assumes the IAU/standard equatorial
    convention with no flip.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

ARCMIN_PER_RAD = 180.0 / np.pi * 60.0
DEFAULT_SHAPE_NOISE = 0.26  # per-component intrinsic ellipticity dispersion


# ---------------------------------------------------------------------------
# Gnomonic (tangent-plane) projection
# ---------------------------------------------------------------------------

def gnomonic_project(ra, dec, ra0, dec0):
    """
    Project (ra, dec) onto the tangent plane at (ra0, dec0).

    All angles in radians. Returns (xi, eta) in radians, with xi East
    (increasing RA) and eta North (increasing Dec).
    """
    ra   = np.asarray(ra,  dtype=np.float64)
    dec  = np.asarray(dec, dtype=np.float64)
    dra  = ra - ra0
    sin_d, cos_d   = np.sin(dec),  np.cos(dec)
    sin_d0, cos_d0 = np.sin(dec0), np.cos(dec0)
    cos_dra, sin_dra = np.cos(dra), np.sin(dra)

    cosc = sin_d0 * sin_d + cos_d0 * cos_d * cos_dra
    if np.any(cosc <= 0) or not np.all(np.isfinite(cosc)):
        raise ValueError("coordinates lie outside the tangent-plane hemisphere")
    xi   = cos_d * sin_dra / cosc
    eta  = (cos_d0 * sin_d - sin_d0 * cos_d * cos_dra) / cosc
    return xi, eta


def gnomonic_deproject(xi, eta, ra0, dec0):
    """Inverse of gnomonic_project. xi, eta in radians -> (ra, dec) in radians."""
    xi  = np.asarray(xi,  dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    rho = np.hypot(xi, eta)
    c   = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)
    sin_d0, cos_d0 = np.sin(dec0), np.cos(dec0)

    safe = rho > 1e-300
    sin_dec = cos_c * sin_d0 + np.where(safe, eta, 0.0) * sin_c * cos_d0 / np.where(safe, rho, 1.0)
    dec = np.where(safe, np.arcsin(np.clip(sin_dec, -1.0, 1.0)), dec0)
    ra = ra0 + np.where(
        safe,
        np.arctan2(xi * sin_c, rho * cos_d0 * cos_c - eta * sin_d0 * sin_c),
        0.0,
    )
    return ra, dec


def projection_rotation(ra, dec, ra0, dec0, eps=1e-7):
    """
    Angle (radians) between celestial North at (ra, dec) and the +eta axis
    of the tangent plane, i.e. the local rotation the gnomonic map applies
    to the equatorial frame. Zero at the tangent point.

    Computed by finite difference so the sign is unambiguous.
    """
    ra  = np.asarray(ra,  dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    xi0, eta0 = gnomonic_project(ra, dec, ra0, dec0)
    xiN, etaN = gnomonic_project(ra, dec + eps, ra0, dec0)
    dxi, deta = xiN - xi0, etaN - eta0
    # North vector in tangent plane; angle measured from +eta toward +xi.
    return np.arctan2(dxi, deta)


# ---------------------------------------------------------------------------
# Spin-2 shear transforms
# ---------------------------------------------------------------------------

def rotate_shear(g1, g2, angle):
    """
    Rotate shear into a frame rotated by `angle` (radians): g -> g*exp(-2i*angle).
    Returns (g1', g2').
    """
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)
    c, s = np.cos(2.0 * angle), np.sin(2.0 * angle)
    return g1 * c + g2 * s, -g1 * s + g2 * c


def tangential_cross_shear(g1, g2, x, y, center=(0.0, 0.0)):
    """
    Decompose shear into tangential (g_t) and cross (g_x) components about
    `center`. g_t > 0 for tangential alignment around an overdensity; g_x is
    the 45-deg / B-mode-like component used as a null test.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    phi   = np.arctan2(y - center[1], x - center[0])
    c, s  = np.cos(2.0 * phi), np.sin(2.0 * phi)
    g_t   = -(g1 * c + g2 * s)
    g_x   =  (g1 * s - g2 * c)
    return g_t, g_x


# ---------------------------------------------------------------------------
# Catalog containers
# ---------------------------------------------------------------------------

@dataclass
class FlatCatalog:
    """Galaxy catalog projected into the flat (x, y) lensing frame."""
    x      : np.ndarray              # arcmin (or chosen unit), East
    y      : np.ndarray              # arcmin, North
    g1     : np.ndarray              # shear on the (x, y) axes
    g2     : np.ndarray
    weight : np.ndarray
    z      : Optional[np.ndarray] = None
    center : tuple = (0.0, 0.0)      # (ra0, dec0) in degrees
    units  : str = "arcmin"
    name   : str = ""
    meta   : dict = field(default_factory=dict)
    row_index: Optional[np.ndarray] = None
    object_id: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.x)

    def radius(self):
        return np.hypot(self.x, self.y)

    def select(self, mask):
        """Return a new FlatCatalog keeping only galaxies where mask is True."""
        z = None if self.z is None else self.z[mask]
        return FlatCatalog(self.x[mask], self.y[mask], self.g1[mask], self.g2[mask],
                           self.weight[mask], z=z, center=self.center,
                           units=self.units, name=self.name, meta=dict(self.meta),
                           row_index=None if self.row_index is None else self.row_index[mask],
                           object_id=None if self.object_id is None else self.object_id[mask])

    def mask_core(self, r_inner):
        """Drop galaxies within r_inner of the centre (strong-lensing regime)."""
        return self.select(self.radius() >= r_inner)


@dataclass
class ShearCatalog:
    """
    Per-galaxy weak-lensing shear catalog in celestial coordinates.

    g1, g2 are calibrated shear estimates (metacal response already applied if
    it was present at read time). weight is inverse-variance per galaxy.
    """
    ra     : np.ndarray              # degrees
    dec    : np.ndarray              # degrees
    g1     : np.ndarray
    g2     : np.ndarray
    weight : np.ndarray
    z      : Optional[np.ndarray] = None
    name   : str = ""
    meta   : dict = field(default_factory=dict)
    row_index: Optional[np.ndarray] = None
    object_id: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.ra)

    @classmethod
    def from_arrays(cls, ra, dec, g1, g2, weight=None, z=None, name=""):
        ra  = np.asarray(ra,  dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)
        g1  = np.asarray(g1,  dtype=np.float64)
        g2  = np.asarray(g2,  dtype=np.float64)
        if weight is None:
            weight = np.ones_like(ra)
        else:
            weight = np.asarray(weight, dtype=np.float64)
        if z is not None:
            z = np.asarray(z, dtype=np.float64)
        return cls(ra=ra, dec=dec, g1=g1, g2=g2, weight=weight, z=z, name=name)

    def center(self):
        """Weighted mean (ra0, dec0) in degrees, a reasonable default tangent point."""
        ra, dec = np.deg2rad(self.ra), np.deg2rad(self.dec)
        xyz = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        v = np.average(xyz, axis=1, weights=self.weight)
        if np.linalg.norm(v)<1e-12: raise ValueError('catalogue has no unique sky centre')
        return float(np.rad2deg(np.arctan2(v[1],v[0]))%360), float(np.rad2deg(np.arctan2(v[2],np.hypot(*v[:2]))))

    def to_tangent_plane(self, center=None, units="arcmin", flip_g2=False):
        """
        Project to the flat lensing frame and rotate shear onto the (x, y) axes.

        center : (ra0, dec0) in degrees. Defaults to the catalog's weighted mean.
        units  : 'arcmin' (default), 'deg', or 'rad'.
        flip_g2: apply a parity flip g2 -> -g2 (set per survey convention).
        """
        if center is None:
            center = self.center()
        ra0  = np.deg2rad(center[0])
        dec0 = np.deg2rad(center[1])
        ra_r = np.deg2rad(self.ra)
        dec_r = np.deg2rad(self.dec)

        xi, eta = gnomonic_project(ra_r, dec_r, ra0, dec0)
        omega   = projection_rotation(ra_r, dec_r, ra0, dec0)
        g1p, g2p = rotate_shear(self.g1, -self.g2 if flip_g2 else self.g2, omega)

        scale = {"rad": 1.0, "deg": np.rad2deg(1.0), "arcmin": ARCMIN_PER_RAD}[units]
        return FlatCatalog(
            x=xi * scale, y=eta * scale, g1=g1p, g2=g2p,
            weight=self.weight.copy(), z=None if self.z is None else self.z.copy(),
            center=center, units=units, name=self.name,
            meta={**self.meta, "projection":"TAN", "flip_g2":bool(flip_g2)},
            row_index=self.row_index, object_id=self.object_id,
        )


# ---------------------------------------------------------------------------
# Metacalibration response
# ---------------------------------------------------------------------------

def apply_response(e1, e2, R):
    """
    Apply a metacal response to mean ellipticities: g = R^{-1} e.

    R may be:
      - a scalar (isotropic mean response),
      - a length-2 sequence [R11, R22] (diagonal),
      - a (2, 2) array (full mean response),
      - a (N, 2, 2) array (per-object response).
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    R  = np.asarray(R, dtype=np.float64)

    if R.ndim == 0:
        return e1 / R, e2 / R
    if R.shape == (2,):
        return e1 / R[0], e2 / R[1]
    if R.shape == (2, 2):
        Rinv = np.linalg.inv(R)
        return Rinv[0, 0] * e1 + Rinv[0, 1] * e2, Rinv[1, 0] * e1 + Rinv[1, 1] * e2
    if R.ndim == 3 and R.shape[1:] == (2, 2):
        Rinv = np.linalg.inv(R)
        g1 = Rinv[:, 0, 0] * e1 + Rinv[:, 0, 1] * e2
        g2 = Rinv[:, 1, 0] * e1 + Rinv[:, 1, 1] * e2
        return g1, g2
    raise ValueError(f"Unsupported response shape {R.shape}")


# ---------------------------------------------------------------------------
# FITS reader
# ---------------------------------------------------------------------------

# Candidate column names per logical field, lowercased. First hit wins.
_DEFAULT_COLUMNS = {
    "ra":     ["ra", "alpha_j2000", "x_world", "ra_deg", "raj2000"],
    "dec":    ["dec", "delta_j2000", "y_world", "dec_deg", "dej2000"],
    "g1":     ["g1", "e1", "gamma1", "g_1", "shear1", "g1_noshear"],
    "g2":     ["g2", "e2", "gamma2", "g_2", "shear2", "g2_noshear"],
    "weight": ["weight", "w", "lensfit_weight", "shape_weight"],
    "z":      ["z", "zphot", "z_phot", "redshift", "z_b", "zb", "zbest"],
}
# metacal response candidates
_R_COLUMNS = {
    "R11": ["r11", "r_11", "response_11"],
    "R22": ["r22", "r_22", "response_22"],
    "R12": ["r12", "r_12", "response_12"],
    "R21": ["r21", "r_21", "response_21"],
}


def _find_column(colnames_lower, candidates):
    for cand in candidates:
        if cand in colnames_lower:
            return colnames_lower[cand]
    return None


def read_fits_catalog(path, column_map=None, hdu=1, name="",
                      response=None, shape_noise=DEFAULT_SHAPE_NOISE,
                      g_cov_columns=None, *, ra_unit=None, dec_unit=None,
                      input_kind='shear', id_col=None, selection=None,
                      flag_col=None, reject_bits=0, z_col=None, z_range=None, max_shear=None,
                      coord_system='radec'):
    """
    Read a weak-lensing shear catalog from a FITS binary table.

    Parameters
    ----------
    path        : FITS file path.
    column_map  : optional dict overriding column detection, e.g.
                  {"ra": "ALPHA_J2000", "g1": "g1_noshear", ...}.
    hdu         : table HDU index (default 1).
    response    : explicit calibration (see apply_response). None leaves shear
                  unchanged, including when R columns exist. Use "auto" to
                  explicitly request response-column detection. Survey selection
                  responses must be supplied by the caller, not inferred here.
    shape_noise : per-component intrinsic dispersion, used to build weights when
                  a covariance is available and no weight column exists.
    g_cov_columns : optional (var_g1_col, var_g2_col) names for per-component
                    shear variance, used to build inverse-variance weights.

    Returns
    -------
    ShearCatalog
    """
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "read_fits_catalog requires astropy. Install with "
            "`pip install astropy`."
        ) from exc

    column_map = dict(column_map or {})
    if z_col is not None: column_map["z"] = z_col
    with fits.open(path) as hdul:
        tbl = hdul[hdu].data
        if not isinstance(hdul[hdu], fits.BinTableHDU):
            raise ValueError('select a FITS binary-table HDU')
        raw_names = list(tbl.columns.names)
        lower = {nm.lower(): nm for nm in raw_names}

        def resolve(key):
            if key in column_map:
                return column_map[key]
            return _find_column(lower, _DEFAULT_COLUMNS[key])

        col = {k: resolve(k) for k in ("ra", "dec", "g1", "g2", "weight", "z")}
        for k in ("ra", "dec", "g1", "g2"):
            if col[k] is None:
                raise KeyError(
                    f"Could not find a column for '{k}'. Available columns: "
                    f"{raw_names}. Pass column_map to override."
                )

        def numeric(name):
            values=np.array(tbl[name],dtype=np.float64)
            null=tbl.columns[name].null
            if null is not None:values[values==null]=np.nan
            if values.ndim!=1:raise ValueError(f'{name} must contain scalar values per source')
            return values

        from astropy import units as u
        def angle(key, override):
            unit=override or tbl.columns[col[key]].unit or 'deg'
            return (numeric(col[key])*u.Unit(unit)).to_value(u.deg), str(unit)
        if coord_system=='radec':
            ra,ru=angle('ra',ra_unit);dec,du=angle('dec',dec_unit)
            if np.any(np.isfinite(dec)&(np.abs(dec)>90)):raise ValueError('declination outside [-90,90] degrees')
            ra=ra%360
        elif coord_system=='pixel':
            ra=numeric(col['ra']);dec=numeric(col['dec'])
            ru=du='pixel'
        else:raise ValueError('coord_system must be radec or pixel')
        e1  = numeric(col["g1"])
        e2  = numeric(col["g2"])
        z   = numeric(col["z"]) if col["z"] else None

        if input_kind not in ('shear','ellipticity'):
            raise ValueError('input_kind must be shear or ellipticity; reduced shear is unsupported')
        if input_kind=='ellipticity' and response is None:
            raise ValueError('ellipticity input requires explicit response calibration')
        response_mode='none' if response is None else 'explicit'
        if isinstance(response,str):
            if response != 'auto':raise ValueError("response string must be 'auto'")
            response = _detect_response(tbl, lower, numeric=numeric)
            if response is None:raise ValueError('response columns were requested but not found')
            response_mode='columns'
        # weights
        if col["weight"] is not None:
            weight = numeric(col["weight"])
        elif g_cov_columns is not None:
            v1 = numeric(g_cov_columns[0])
            v2 = numeric(g_cov_columns[1])
            if not np.isfinite(shape_noise) or shape_noise<=0 or np.any(v1<0) or np.any(v2<0):
                raise ValueError('shape noise must be positive and shear variances nonnegative')
            weight = 1.0 / (shape_noise**2 + 0.5 * (v1 + v2))
        else:
            weight = np.ones_like(ra)
        keep=np.ones(len(ra),bool);reasons={}
        def cut(label,good):
            good=np.asarray(good,dtype=bool)
            if good.shape != keep.shape:raise ValueError('selection must have one boolean per row')
            reasons[label]=np.flatnonzero(~good).tolist();keep[:] &= good
        if selection is not None:cut('user_selection',selection)
        if flag_col is not None:
            flags=np.asarray(tbl[flag_col],dtype=np.int64)
            good=(flags&int(reject_bits))==0
            if tbl.columns[flag_col].null is not None:good &= flags!=tbl.columns[flag_col].null
            cut('flags',good)
        if z_range is not None:
            if len(z_range)!=2 or not np.all(np.isfinite(z_range)) or z_range[0]>z_range[1]:
                raise ValueError('z_range must be finite ordered endpoints')
            if z is None:raise ValueError('redshift cut requires a redshift column')
            cut('redshift',np.isfinite(z)&(z>=z_range[0])&(z<=z_range[1]))
        cut('nonfinite',np.isfinite(ra)&np.isfinite(dec)&np.isfinite(e1)&np.isfinite(e2)&np.isfinite(weight))
        cut('weight_or_amplitude',weight>0)
        if response is not None:
            R = np.asarray(response, dtype=float)
            selected_R = R[keep] if R.ndim == 3 else R
            if not np.all(np.isfinite(selected_R)):
                raise ValueError('nonfinite response on selected rows')
            if selected_R.ndim < 2 and np.any(selected_R == 0):
                raise ValueError('response must be nonzero')
            try:
                a, b = apply_response(e1[keep], e2[keep], selected_R)
            except np.linalg.LinAlgError as exc:
                raise ValueError('singular response on selected rows') from exc
            g1, g2 = e1.copy(), e2.copy()
            g1[keep], g2[keep] = a, b
        else:
            g1, g2 = e1, e2
        ids=np.asarray(tbl[id_col]).copy() if id_col else None
        rows=np.arange(len(ra))

    cat = ShearCatalog(ra=ra, dec=dec, g1=g1, g2=g2, weight=weight, z=z,
                       name=name or str(path),
                       meta={"resolved_columns": col, "source": str(path), "hdu":hdu,
                             "coordinate_units":dict(ra=ru,dec=du),"response":response_mode,
                             "response_values":None if response is None else np.where(np.isfinite(response),response,None).tolist(),
                             "input_kind":input_kind,"n_input":len(ra),"rejected_rows":reasons,
                             "selection":dict(flag_col=flag_col,reject_bits=reject_bits,z_range=z_range,
                                              max_shear=max_shear,id_col=id_col),
                             "weight_model":"column" if col["weight"] else "shear_covariance_plus_shape_noise" if g_cov_columns else "unit"},
                       row_index=rows,object_id=ids)
    # Retain original row identity across every selection step.
    cat=ShearCatalog(cat.ra[keep],cat.dec[keep],cat.g1[keep],cat.g2[keep],cat.weight[keep],
        z=None if z is None else z[keep],name=cat.name,meta=cat.meta,
        row_index=rows[keep],object_id=None if ids is None else ids[keep])
    cat=clean_catalog(cat,max_shear=max_shear)
    if coord_system=='pixel':
        return FlatCatalog(cat.ra,cat.dec,cat.g1,cat.g2,cat.weight,z=cat.z,units='pixel',
            name=cat.name,meta=cat.meta,row_index=cat.row_index,object_id=cat.object_id)
    return cat


def _detect_response(tbl, lower, numeric=None):
    """Return a (N,2,2) per-object response if R columns exist, else None."""
    found = {k: _find_column(lower, v) for k, v in _R_COLUMNS.items()}
    if found["R11"] is None or found["R22"] is None:
        return None
    if numeric is None: numeric=lambda name: np.array(tbl[name],dtype=np.float64)
    n = len(tbl)
    R = np.zeros((n, 2, 2), dtype=np.float64)
    R[:, 0, 0] = numeric(found["R11"])
    R[:, 1, 1] = numeric(found["R22"])
    R[:, 0, 1] = numeric(found["R12"]) if found["R12"] else 0.0
    R[:, 1, 0] = numeric(found["R21"]) if found["R21"] else 0.0
    return R


def clean_catalog(cat, max_shear=None):
    """
    Drop galaxies with non-finite or non-physical entries.

    Removes NaN/inf in position, shear, or weight; drops |g| > max_shear and
    non-positive weights.
    """
    finite = (np.isfinite(cat.ra) & np.isfinite(cat.dec) &
              np.isfinite(cat.g1) & np.isfinite(cat.g2) &
              np.isfinite(cat.weight))
    physical = cat.weight > 0
    if max_shear is not None:
        if not np.isfinite(max_shear) or max_shear<=0:raise ValueError('max_shear must be positive')
        physical &= np.hypot(cat.g1,cat.g2)<=max_shear
    keep = finite & physical
    z = None if cat.z is None else cat.z[keep]
    rows=np.arange(cat.n) if cat.row_index is None else cat.row_index
    rejected=dict(cat.meta.get('rejected_rows',{}))
    rejected['nonfinite']=sorted(set(rejected.get('nonfinite',[])+rows[~finite].tolist()))
    rejected['weight_or_amplitude']=sorted(set(rejected.get('weight_or_amplitude',[])+rows[~physical].tolist()))
    return ShearCatalog(
        ra=cat.ra[keep], dec=cat.dec[keep], g1=cat.g1[keep], g2=cat.g2[keep],
        weight=cat.weight[keep], z=z, name=cat.name,
        meta={**cat.meta,"rejected_rows":rejected,
              "n_dropped":int(cat.meta.get('n_input',cat.n)-keep.sum())},
        row_index=rows[keep],object_id=None if cat.object_id is None else cat.object_id[keep],
    )
