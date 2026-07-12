"""
femmi/sampling.py
Posterior sampling for FEMMI mass reconstruction -- uncertainty quantification on
top of the MAP -- built on the DIFFERENTIABLE FEM-BEM forward. Two methods, one
entry point `sample_posterior`:

  method='rto'  (perturb-and-MAP / Randomize-Then-Optimize): for a LINEAR forward
      and a GAUSSIAN (Wiener) prior the posterior is exactly Gaussian,
          A = F^T W F / sigma_n^2 + 2 lambda R ,   b = F^T W gamma / sigma_n^2 ,
          p(kappa) = N(A^{-1} b, A^{-1}) ,
      and an EXACT independent sample is  kappa = A^{-1}(b + eta),  eta ~ N(0, A),
      i.e. one linear solve per sample on perturbed data. No step tuning, no
      mixing -- robust by construction. Each solve is matrix-free CG using the
      exact forward and its adjoint (so it fully exploits the differentiable
      operator); the prior perturbation uses a dense Cholesky of R (fine for the
      moderate meshes used for UQ).

  method='langevin' (score-based, mass-preconditioned ULA): the general path for
      NON-Gaussian priors -- in particular the learned NeuralScorePrior -- which
      have no closed-form posterior. Integrates
          kappa <- kappa - step P grad U + sqrt(2 step T) P^{1/2} xi
      with U = ||F kappa - gamma||^2_w/(2 sigma_n^2) + lambda phi(kappa), the
      mass-matrix metric P, and a warm start at the MAP. This is the mode that
      needs only the prior SCORE, so it is what pairs with score matching
      (Song & Ermon 2019; Remy et al. 2020). It is an approximate sampler on
      ill-conditioned problems -- treat its spread as indicative.

`sample_posterior(..., method='auto')` picks 'rto' for the Gaussian/Wiener prior
and 'langevin' for any other prior.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from dataclasses import dataclass

from .inverse import MAPReconstructor
from .priors import WienerPrior


@dataclass
class PosteriorSamples:
    mean   : np.ndarray            # posterior mean kappa (per node)
    std    : np.ndarray            # per-node posterior std (uncertainty map)
    samples: np.ndarray            # (n_kept, n_nodes) retained samples
    method : str
    map_kappa: np.ndarray          # the MAP (posterior mode)
    info   : dict


# --------------------------------------------------------------------------- #
# forward / adjoint helpers (matrix-free, via the differentiable operator)
# --------------------------------------------------------------------------- #
def _make_FtWF(ops, w, sigma2, R, two_lam):
    M, S1, S2 = ops.M, ops.S1, ops.S2

    def Fv(v):
        psi = ops._solve_psi(-2.0 * M @ v)
        return S1 @ psi, S2 @ psi

    def FtW(y1, y2):
        yy1, yy2 = (y1, y2) if w is None else (w * y1, w * y2)
        adj = ops._solve_adjoint(S1.T @ yy1 + S2.T @ yy2)
        return -2.0 * (M.T @ adj)

    def A(v):
        g1, g2 = Fv(v)
        return FtW(g1, g2) / sigma2 + two_lam * (R @ v)

    return Fv, FtW, A


def _rto_sample(ops, g1, g2, noise_std, wiener_length, lam, w, n_samples,
                cg_tol, seed, verbose):
    """Exact Gaussian posterior sampling by perturb-and-MAP."""
    n = ops.n_nodes
    sigma2 = noise_std**2
    two_lam = 2.0 * lam
    R = WienerPrior(ops, wiener_length).R
    Rd = R.toarray() if hasattr(R, "toarray") else np.asarray(R)
    if verbose:
        print(f"  RTO: dense Cholesky of R ({n}x{n}) ...")
    Lc = sla.cholesky(Rd + 1e-12 * np.eye(n), lower=True)   # R = Lc Lc^T

    Fv, FtW, A = _make_FtWF(ops, w, sigma2, R, two_lam)
    Aop = spla.LinearOperator((n, n), matvec=A)
    b = FtW(g1, g2) / sigma2

    def solve(rhs, x0=None):
        x, _ = spla.cg(Aop, rhs, rtol=cg_tol, maxiter=500,
                       x0=x0)
        return x

    k_map = solve(b)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, n))
    for s in range(n_samples):
        e1 = rng.standard_normal(len(g1)); e2 = rng.standard_normal(len(g2))
        z = rng.standard_normal(n)
        eta = FtW(e1, e2) / noise_std + np.sqrt(two_lam) * (Lc @ z)   # eta ~ N(0,A)
        samples[s] = solve(b + eta, x0=k_map)
        if verbose and (s + 1) % max(1, n_samples // 5) == 0:
            print(f"  RTO sample {s + 1}/{n_samples}")
    return k_map, samples


def _langevin_sample(fwd, g1, g2, noise_std, prior, lam, wiener_length, w,
                     n_steps, burnin, thin, step, temperature, warm_start,
                     maxiter_map, seed, verbose):
    """Mass-preconditioned unadjusted Langevin (works with any prior score)."""
    ops = fwd.ops
    M, S1, S2 = ops.M, ops.S1, ops.S2
    inv2s2 = 1.0 / (2.0 * noise_std**2)
    if prior is None:
        R = WienerPrior(ops, wiener_length).R

    def grad_U(k):
        psi = ops._solve_psi(-2.0 * M @ k)
        r1, r2 = S1 @ psi - g1, S2 @ psi - g2
        wr1, wr2 = (r1, r2) if w is None else (w * r1, w * r2)
        g = (-4.0 * (M.T @ ops._solve_adjoint(S1.T @ wr1 + S2.T @ wr2))) * inv2s2
        return g + (lam * (2.0 * (R @ k)) if prior is None else lam * prior.value_grad(k)[1])

    rec = MAPReconstructor(fwd, wiener_length=wiener_length, data_weight=w,
                           prior=prior, maxiter=maxiter_map, callback_every=0)
    k_map, _ = rec.reconstruct(g1, g2, verbose=False)
    k = k_map.copy() if warm_start else np.zeros(ops.n_nodes)

    m = np.asarray(M @ np.ones(ops.n_nodes)).ravel()
    P = 1.0 / np.maximum(m, 1e-12); sP = np.sqrt(P)
    if step is None:
        # top eigenvalue of P^{1/2} H P^{1/2} by power iteration
        rng0 = np.random.default_rng(seed); v = rng0.standard_normal(ops.n_nodes)
        v /= np.linalg.norm(v); g0 = grad_U(k_map); L = 1.0
        for _ in range(15):
            Hv = (grad_U(k_map + 1e-4 * sP * v) - g0) / 1e-4
            wv = sP * Hv; L = float(np.linalg.norm(wv))
            if L < 1e-30:
                break
            v = wv / L
        step = float(0.4 / (temperature * L + 1e-30))

    if verbose:
        print(f"  Langevin (mass-preconditioned): step={step:.2e}  T={temperature}  "
              f"steps={n_steps} (burnin {burnin}, thin {thin})")
    rng = np.random.default_rng(seed)
    noise_scale = np.sqrt(2.0 * step * temperature)
    kept = []
    for t in range(n_steps):
        k = k - step * P * grad_U(k) + noise_scale * sP * rng.standard_normal(ops.n_nodes)
        if t >= burnin and (t - burnin) % thin == 0:
            kept.append(k.copy())
    return k_map, np.array(kept), step


def sample_posterior(fwd, gamma1_obs, gamma2_obs, noise_std, prior=None, lam=None,
                     wiener_length=0.0, data_weight=None, method="auto",
                     n_samples=200, temperature=1.0, cg_tol=1e-6,
                     n_steps=2000, burnin=400, thin=5, step=None, warm_start=True,
                     maxiter_map=200, seed=0, verbose=True):
    """Sample the kappa posterior and return mean, std (UQ), and samples.

    noise_std : per-component shear noise sigma_n (sets the likelihood scale).
    prior     : femmi.priors.Prior or None -> Wiener. method='auto' uses exact RTO
                for the Gaussian/Wiener prior and Langevin for any other prior.
    """
    if lam is not None:
        fwd.lam_reg = lam
    lam = fwd.lam_reg
    w = None if data_weight is None else np.asarray(data_weight, float)
    g1 = np.asarray(gamma1_obs, float); g2 = np.asarray(gamma2_obs, float)

    is_gaussian = prior is None or isinstance(prior, WienerPrior)
    if method == "auto":
        method = "rto" if is_gaussian else "langevin"
    if method == "rto" and not is_gaussian:
        raise ValueError("method='rto' is exact only for the Gaussian/Wiener prior; "
                         "use method='langevin' for non-Gaussian priors.")

    if verbose:
        pname = prior.name if prior is not None else (
            f"Wiener(l={wiener_length})" if wiener_length > 0 else "H1")
        print(f"sample_posterior[{method}]: prior={pname}  lambda={lam:.2e}  "
              f"sigma_n={noise_std:.3e}  T={temperature}")

    if method == "rto":
        wl = prior.wiener_length if isinstance(prior, WienerPrior) else wiener_length
        k_map, samples = _rto_sample(fwd.ops, g1, g2, float(noise_std), wl, lam, w,
                                     n_samples, cg_tol, seed, verbose)
        info = dict(n_samples=n_samples)
    else:
        k_map, samples, step = _langevin_sample(
            fwd, g1, g2, float(noise_std), prior, lam, wiener_length, w,
            n_steps, burnin, thin, step, temperature, warm_start, maxiter_map,
            seed, verbose)
        info = dict(step=step, n_steps=n_steps, burnin=burnin, thin=thin)

    if temperature != 1.0 and method == "rto":
        samples = k_map + np.sqrt(temperature) * (samples - k_map)
    return PosteriorSamples(mean=samples.mean(0), std=samples.std(0),
                            samples=samples, method=method, map_kappa=k_map, info=info)


# backward-compatible thin wrapper
def langevin_sample(fwd, gamma1_obs, gamma2_obs, noise_std, **kw):
    kw.pop("method", None)
    return sample_posterior(fwd, gamma1_obs, gamma2_obs, noise_std,
                            method="langevin", **kw)
