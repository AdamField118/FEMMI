"""Fourier cross-coherence of registered maps and sample propagation.

Inputs must share pixels, physical units for pixel size, footprint and masking.
This diagnostic does not register X-ray data or correct a window's mode mixing.
It is infrastructure for real-cluster validation, not a completed validation.
"""
import numpy as np


def map_coherence(kappa, tracer, pixel_size, bins=12, window=None):
    """Re(P_cross)/sqrt(P_kappa P_tracer) in annuli of cycles/unit length.

    Subtract weighted means, then apply the same real nonnegative window to
    both images. Invalid pixels must be excluded explicitly by zero window.
    Empty/zero-power annuli are NaN, not evidence of absent coherence.
    """
    a,b=np.asarray(kappa,float),np.asarray(tracer,float)
    if a.shape!=b.shape or a.ndim!=2 or min(a.shape)<4:
        raise ValueError('registered 2-D maps of matching shape required')
    if not np.isfinite(pixel_size) or pixel_size<=0:raise ValueError('positive pixel_size required')
    w=np.ones(a.shape) if window is None else np.asarray(window,float)
    if w.shape!=a.shape or not np.all(np.isfinite(w)) or np.any(w<0) or not np.any(w>0):
        raise ValueError('finite nonnegative shared window required')
    good=w>0
    if not np.all(np.isfinite(a[good])) or not np.all(np.isfinite(b[good])):
        raise ValueError('nonfinite pixels inside the selected footprint')
    def transform(x):
        mean=np.sum(x[good]*w[good])/w.sum()
        return np.fft.fft2(np.where(good,np.where(good,x,0)-mean,0)*w)
    A,B=transform(a),transform(b)
    qx=np.fft.fftfreq(a.shape[1],pixel_size)
    qy=np.fft.fftfreq(a.shape[0],pixel_size)
    q=np.hypot(qy[:,None],qx[None,:])
    if np.isscalar(bins):
        if int(bins)<1:raise ValueError('positive bin count required')
        edges=np.geomspace(q[q>0].min()*.999,q.max()*1.001,int(bins)+1)
    else:edges=np.asarray(bins,float)
    if len(edges)<2 or np.any(np.diff(edges)<=0) or edges[0]<=0:raise ValueError('positive increasing frequency bins required')
    coherence=[];count=[];frequency=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(q>=lo)&(q<hi);count.append(int(m.sum()))
        frequency.append(float(q[m].mean()) if m.any() else np.sqrt(lo*hi))
        norm=np.sqrt(np.sum(abs(A[m])**2)*np.sum(abs(B[m])**2))
        coherence.append(float(np.clip(np.real(np.vdot(A[m],B[m]))/norm,-1,1)) if norm>0 else np.nan)
    return dict(q=np.asarray(frequency),coherence=np.asarray(coherence),count=np.asarray(count),edges=edges)


def coherence_length(spectrum,threshold=.9):
    """Largest-scale-connected threshold crossing, with explicit censoring.

    Length is the midpoint of the wavelength bracket where coherence first
    falls below threshold as frequency increases. It is not an independent
    Fourier resolution measurement. Empty annuli are skipped, retaining their
    gap in the crossing bracket; populated zero-power annuli are undefined.
    """
    if not -1<threshold<1:raise ValueError('threshold must be in (-1,1)')
    q=np.asarray(spectrum['q']);c=np.asarray(spectrum['coherence'])
    valid=np.asarray(spectrum['count'])>0;q,c=q[valid],c[valid]
    if not len(q) or not np.all(np.isfinite(c)):
        return dict(length=None,status='undefined',bracket=None)
    if c[0]<threshold:return dict(length=None,status='above-field-scale',bracket=[float(1/q[0]),None])
    below=np.flatnonzero(c<threshold)
    if not len(below):return dict(length=None,status='below-resolution',bracket=[0.,float(1/q[-1])])
    j=below[0];lo,hi=1/q[j],1/q[j-1]
    return dict(length=float((lo+hi)/2),status='crossing',bracket=[float(lo),float(hi)])


def coherence_samples(samples,tracer,pixel_size,**kwargs):
    """Propagate supplied kappa samples without a linearized Fisher transform.

    Quantiles are conditional on uncensored crossings; the returned counts
    must accompany them. No sampler calibration is implied by this function.
    """
    threshold=kwargs.pop('threshold',.9)
    results=[coherence_length(map_coherence(k,tracer,pixel_size,**kwargs),threshold) for k in samples]
    if not results:raise ValueError('at least one sample required')
    lengths=[r['length'] for r in results if r['status']=='crossing']
    return dict(samples=results,n_samples=len(results),n_crossings=len(lengths),
        conditional_quantiles=np.quantile(lengths,[.16,.5,.84]).tolist() if lengths else None)
