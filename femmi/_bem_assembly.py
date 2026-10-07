"""Shared float64 assembly for straight P3 and degree-general boundary edges.

Geometry is prepared once per call; immutable reference quadrature is cached.
The NumPy and optional serial Numba paths use the same rules and contractions.
Curved, hypersingular and ACA operators retain their separate implementations.
"""
from functools import lru_cache
import os
import numpy as np


@lru_cache(maxsize=32)
def _reference(degree, n_quad, p3, single):
    from .bem import _gauss_legendre, log_gauss_jacobi_points, _p3_boundary_basis
    from .bem_hp import boundary_basis
    basis = _p3_boundary_basis if p3 else lambda x: boundary_basis(degree, x)
    xi, w = _gauss_legendre(n_quad)
    phi = basis(xi)
    # Duffy self block = L^2/(2*pi) * (log(L)*A + B). This factors
    # geometry out of the EXISTING GL/log-Laguerre sums; no new quadrature.
    A = np.zeros((degree+1, degree+1))
    B = np.zeros_like(A)
    if single:
        v_log, w_log = log_gauss_jacobi_points(n_quad)
        tau_gl = basis((xi[:, None]*(1.-xi[None, :])).ravel()).reshape(n_quad,n_quad,-1)
        tau_log = basis((xi[:, None]*(1.-v_log[None, :])).ravel()).reshape(n_quad,n_quad,-1)
        for q in range(n_quad):
            for r in range(n_quad):
                gl = xi[q]*w[q]*w[r]*(np.outer(phi[q],tau_gl[q,r])+np.outer(tau_gl[q,r],phi[q]))
                lj = xi[q]*w[q]*w_log[r]*(np.outer(phi[q],tau_log[q,r])+np.outer(tau_log[q,r],phi[q]))
                A += gl
                B += np.log(xi[q])*gl-lj
    for value in (xi,w,phi,A,B):
        value.setflags(write=False)
    return xi,w,phi,A,B


def _numpy_assemble(elems, points, lengths, normals, w, phi, diagonal, n_dofs, single):
    nd = elems.shape[1]
    result = np.zeros((n_dofs,n_dofs))
    for s in range(len(elems)):
        diff = points[s][None,:,None,:]-points[:,None,:,:]
        r2 = np.sum(diff**2,axis=-1)
        if single:
            kernel = np.where(r2>1e-30,np.log(np.maximum(r2,1e-300))/(4.*np.pi),0.)
        else:
            r2 = np.where(r2<1e-28,np.inf,r2)
            kernel = np.sum(diff*normals[:,None,None,:],axis=-1)/(2.*np.pi*r2)
        kernel *= lengths[s]*lengths[:,None,None]*w[None,:,None]*w[None,None,:]
        kernel[s] = 0.
        blocks = phi.T @ (kernel @ phi)
        blocks[s] = diagonal[s]
        for a in range(nd):
            for b in range(nd):
                np.add.at(result,(elems[s,a],elems[:,b]),blocks[:,a,b])
    return result


def _loop_assemble(elems, points, lengths, normals, w, phi, diagonal, n_dofs, single):
    """Serial element-pair contraction; no fastmath or parallel scatter races."""
    ne, nd = elems.shape
    nq = len(w)
    result = np.zeros((n_dofs,n_dofs),dtype=np.float64)
    for s in range(ne):
        for t in range(ne):
            if s == t:
                for a in range(nd):
                    for b in range(nd):
                        result[elems[s,a],elems[t,b]] += diagonal[s,a,b]
                continue
            # Contract source basis first, then target basis: O(q^2*d+q*d^2)
            # instead of evaluating the kernel repeatedly for all d^2 entries.
            tmp = np.zeros((nq,nd),dtype=np.float64)
            for q in range(nq):
                for r in range(nq):
                    dx = points[s,q,0]-points[t,r,0]
                    dy = points[s,q,1]-points[t,r,1]
                    r2 = dx*dx+dy*dy
                    value = 0.
                    if single:
                        if r2>1e-30:
                            value = np.log(max(r2,1e-300))/(4.*np.pi)
                    elif r2>=1e-28:
                        value = (dx*normals[t,0]+dy*normals[t,1])/(2.*np.pi*r2)
                    value *= lengths[s]*lengths[t]*w[q]*w[r]
                    for b in range(nd):
                        tmp[q,b] += value*phi[r,b]
            for a in range(nd):
                for b in range(nd):
                    value = 0.
                    for q in range(nq):
                        value += phi[q,a]*tmp[q,b]
                    result[elems[s,a],elems[t,b]] += value
    return result


@lru_cache(maxsize=1)
def _numba_kernel():
    try:
        from numba import njit
    except ImportError:
        return None
    return njit(cache=True,fastmath=False)(_loop_assemble)


def assemble_straight(bnd, degree, n_quad, *, single, p3=False, backend=None):
    """Assemble V or K with auto/NumPy/Numba selection.

    FEMMI_BEM_BACKEND controls existing high-level estimator calls. Explicit
    'numba' requires the speed extra; 'auto' uses NumPy when Numba is absent.
    Compilation/runtime errors are deliberately not hidden by a fallback.
    """
    backend = (backend or os.environ.get('FEMMI_BEM_BACKEND','auto')).lower()
    if backend not in ('auto','numpy','numba'):
        raise ValueError('BEM backend must be auto, numpy or numba')
    if degree<1 or n_quad<1:
        raise ValueError('degree and n_quad must be positive')
    elems = np.ascontiguousarray(bnd.elements,dtype=np.int64)
    lengths = np.ascontiguousarray(bnd.element_lengths,dtype=np.float64)
    normals = np.ascontiguousarray(bnd.element_normals,dtype=np.float64)
    nodes = np.asarray(bnd.nodes,dtype=np.float64)
    if (elems.shape != (bnd.n_elements,degree+1) or len(elems)==0
            or elems.min()<0 or elems.max()>=bnd.n_boundary_dofs
            or nodes.shape != (bnd.n_boundary_dofs,2)
            or lengths.shape != (len(elems),) or normals.shape != (len(elems),2)
            or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(normals))
            or not np.all(np.isfinite(lengths)&(lengths>0))):
        raise ValueError('invalid straight boundary mesh')
    xi,w,phi,A,B = _reference(degree,n_quad,p3,single)
    p0,p1 = nodes[elems[:,0]],nodes[elems[:,-1]]
    points = np.ascontiguousarray(p0[:,None,:]+xi[None,:,None]*(p1-p0)[:,None,:])
    diagonal = lengths[:,None,None]**2/(2.*np.pi)*(np.log(lengths)[:,None,None]*A+B)
    kernel = None if backend=='numpy' else _numba_kernel()
    if backend=='numba' and kernel is None:
        raise ImportError("Numba backend requires pip install 'femmi[speed]'")
    if kernel is None:
        kernel = _numpy_assemble
    result = kernel(elems,points,lengths,normals,w,phi,diagonal,bnd.n_boundary_dofs,single)
    return .5*(result+result.T) if single else result
