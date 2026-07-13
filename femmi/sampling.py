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

  method='annealed_hmc' (tempered HMC, Remy et al. 2020 eq. 4; Song & Ermon 2019):
      the paper's sampler and the default for NON-Gaussian priors. Anneals the
      noise level sigma from sigma_max down to sigma_min; at each level the target
      is tempered (the likelihood via sigma_eff^2 = sigma_n^2 + sigma^2, and the
      prior via the noise-conditional score net r_theta(., sigma)), so modes merge
      at high sigma and separate as it cools. Each level runs HMC with a Metropolis
      correction that uses the exact quadratic likelihood difference plus a
      score line-integral estimate of the prior log-density difference (the
      num_delta_logp trick). Annealing is what tightens the mixing that a single
      temperature Langevin cannot.

  method='langevin' (mass-preconditioned ULA): the simpler single-temperature
      score-based sampler; kept as a lightweight fallback. Mixes slowly on
      ill-conditioned problems -- prefer 'annealed_hmc'.

`sample_posterior(..., method='auto')` picks 'rto' for the Gaussian/Wiener prior
and 'annealed_hmc' for any other prior.
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


def _map_warmstart(fwd, g1, g2, prior, wiener_length, w, sn2, lam, maxiter):
    """MAP used as the sampler warm-start and (for HMC/Langevin) point estimate.

    The sampler's `lam` is in the noise-normalised convention (data term
    ||F k - y||^2 / (2 sigma_n^2)); MAPReconstructor minimises the un-normalised
    ||F k - y||^2 + lam_map * phi. The two share the SAME mode iff
    lam_map = 2 sigma_n^2 lam, so convert before the solve (otherwise a properly
    large sampler lam over-smooths the warm-start MAP)."""
    saved = fwd.lam_reg
    fwd.lam_reg = 2.0 * sn2 * lam
    try:
        rec = MAPReconstructor(fwd, wiener_length=wiener_length, data_weight=w,
                               prior=prior, maxiter=maxiter, callback_every=0)
        k_map, _ = rec.reconstruct(g1, g2, verbose=False)
    finally:
        fwd.lam_reg = saved
    return k_map


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

    k_map = _map_warmstart(fwd, g1, g2, prior, wiener_length, w, noise_std**2, lam, maxiter_map)
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


def _annealed_hmc(fwd, g1, g2, noise_std, prior, lam, wiener_length, w,
                  n_levels, steps_per_level, sigma_max, sigma_min, n_leapfrog,
                  leap_frac, n_delta_logp, n_chains, keep_final, maxiter_map,
                  seed, verbose):
    """Tempered / annealed HMC (Remy et al. 2020 eq. 4; Song & Ermon 2019).

    Target at annealing level sigma:  U_sigma(kappa) =
        ||F kappa - y||^2_w / (2 (sigma_n^2 + sigma^2))  -  lambda log p_sigma(kappa).
    sigma^2 is the inverse temperature: high sigma broadens both the likelihood
    (via sigma_eff) and the prior (the score net is noise-conditional, so it
    supplies the correctly-tempered prior score at each level), merging modes;
    annealing sigma_max -> sigma_min lands on the true posterior. Each level runs
    HMC; for a score-only prior the Metropolis correction uses the exact
    (quadratic) likelihood difference plus a line-integral estimate of the prior
    log-density difference (the num_delta_logp trick)."""
    ops = fwd.ops
    M, S1, S2 = ops.M, ops.S1, ops.S2
    sn2 = float(noise_std)**2

    has_score = hasattr(prior, "score")               # NeuralScorePrior / ScorePrior
    if not has_score:
        R = WienerPrior(ops, wiener_length).R
        def pscore(k, s):
            return -2.0 * (R @ k)                     # raw grad log p (sigma-indep. Gaussian)
        def neg_logp(k):
            return float(k @ (R @ k))                 # exact prior value (x lambda later)
    else:
        def pscore(k, s):
            return np.asarray(prior.score(k, s), float)

    def lik(k, seff2):
        psi = ops._solve_psi(-2.0 * M @ k)
        r1, r2 = S1 @ psi - g1, S2 @ psi - g2
        wr1, wr2 = (r1, r2) if w is None else (w * r1, w * r2)
        value = 0.5 * float(np.dot(wr1, r1) + np.dot(wr2, r2)) / seff2
        adj = ops._solve_adjoint(S1.T @ wr1 + S2.T @ wr2)
        grad = (-2.0 * (M.T @ adj)) / seff2
        return value, grad

    def gradU(k, s, seff2):
        lv, lg = lik(k, seff2)
        return lg - lam * pscore(k, s), lv

    k_map = _map_warmstart(fwd, g1, g2, prior, wiener_length, w, sn2, lam, maxiter_map)

    def hess_topeig(k0, s, seff2, crng, n_iter=8, epsv=1e-4):
        """Largest eigenvalue of the FULL level Hessian (data + prior) near k0, so
        the leapfrog step is stable whether the likelihood or the prior dominates."""
        g0 = gradU(k0, s, seff2)[0]
        v = crng.standard_normal(ops.n_nodes); v /= np.linalg.norm(v); Lm = 1.0
        for _ in range(n_iter):
            Hv = (gradU(k0 + epsv * v, s, seff2)[0] - g0) / epsv
            Lm = float(np.linalg.norm(Hv))
            if Lm < 1e-30:
                break
            v = Hv / Lm
        return Lm

    def hmc_step(k, s, seff2, eps, crng):
        p0 = crng.standard_normal(ops.n_nodes); k0 = k.copy()
        g, lv0 = gradU(k, s, seff2)
        p = p0 - 0.5 * eps * g
        for l in range(n_leapfrog):
            k = k + eps * p
            g, lv = gradU(k, s, seff2)
            if l != n_leapfrog - 1:
                p = p - eps * g
        p = p - 0.5 * eps * g
        dK = 0.5 * (np.dot(p, p) - np.dot(p0, p0))
        dlik = lv - lv0
        if not has_score:
            dprior = lam * (neg_logp(k) - neg_logp(k0))
        else:                                         # score line integral for delta(-log p)
            dk = k - k0; acc = 0.0
            for j in range(n_delta_logp):
                t = (j + 0.5) / n_delta_logp
                acc += float(np.dot(pscore(k0 + t * dk, s), dk))
            dprior = -lam * acc / n_delta_logp
        accepted = np.log(crng.random() + 1e-30) < -(dK + dlik + dprior)
        return (k, 1) if accepted else (k0, 0)

    # Independent annealed chains: each chain anneals sigma_max -> sigma_min and
    # contributes samples from the coldest level. Independence between chains
    # (not steps within one chain) is what gives correct posterior variance.
    sigmas = np.geomspace(sigma_max, sigma_min, n_levels)
    samples, n_acc, n_prop = [], 0, 0
    thin = max(1, keep_final // 2)
    for c in range(n_chains):
        crng = np.random.default_rng(seed + 1000 + c)
        k = k_map + sigma_max * crng.standard_normal(ops.n_nodes)   # broad start
        for li, s in enumerate(sigmas):
            seff2 = sn2 + s**2
            eps = leap_frac / np.sqrt(hess_topeig(k, s, seff2, crng) + 1e-30)
            n_here = steps_per_level + (keep_final * thin if li == n_levels - 1 else 0)
            for it in range(n_here):
                k, a = hmc_step(k, s, seff2, eps, crng)
                n_acc += a; n_prop += 1
                if li == n_levels - 1 and it >= steps_per_level and (it - steps_per_level) % thin == 0:
                    samples.append(k.copy())
        if verbose and (c + 1) % max(1, n_chains // 5) == 0:
            print(f"  annealed chain {c+1}/{n_chains}  accept={n_acc/max(1,n_prop):.2f}")
    return k_map, np.array(samples), dict(accept=n_acc / max(1, n_prop),
                                          sigmas=[float(x) for x in sigmas],
                                          n_levels=n_levels, n_chains=n_chains)


def _auto_lam(ops, g1, g2, noise_std, prior, wiener_length, w, verbose):
    """Principled default for `lam` when the caller passes lam=None.

    Gaussian/Wiener prior: run the SAME Morozov discrepancy selection the MAP
    reconstructor uses to get lam_MAP (data term ||F k - y||^2 + lam_MAP k^T R k),
    then convert to the sampler's noise-normalised convention
        lam_RTO = lam_MAP / (2 sigma_n^2)
    (the two posteriors have the same mode iff this holds). This makes RTO / HMC
    reproduce the MAP reconstruction instead of being ~10^3x under-regularised.

    Score prior (neural): the network already encodes a properly normalised
    log-prior, so the Bayesian coefficient is 1.0. Any other non-Gaussian prior
    (TV / sparse / maxent) likewise defaults to 1.0.
    """
    is_gaussian = prior is None or isinstance(prior, WienerPrior)
    if not is_gaussian:
        return 1.0
    from .regularization import MorozovSelector
    wl = prior.wiener_length if isinstance(prior, WienerPrior) else wiener_length
    sel = MorozovSelector(ops, noise_std=noise_std, wiener_length=wl,
                          data_weight=w, verbose=False)
    lam_map = float(sel.select(g1, g2))
    lam = lam_map / (2.0 * noise_std ** 2)
    if verbose:
        print(f"  auto-lam: Morozov lam_MAP={lam_map:.3e} -> lam={lam:.3e} "
              f"(= lam_MAP / 2 sigma_n^2)")
    return lam


def sample_posterior(fwd, gamma1_obs, gamma2_obs, noise_std, prior=None, lam=None,
                     wiener_length=0.0, data_weight=None, method="auto",
                     n_samples=200, temperature=1.0, cg_tol=1e-6,
                     n_steps=2000, burnin=400, thin=5, step=None, warm_start=True,
                     maxiter_map=200, seed=0, verbose=True,
                     n_levels=10, steps_per_level=15, sigma_max=1.0, sigma_min=0.02,
                     n_leapfrog=5, leap_frac=0.5, n_delta_logp=4,
                     n_chains=40, keep_final=4):
    """Sample the kappa posterior and return mean, std (UQ), and samples.

    noise_std : per-component shear noise sigma_n (sets the likelihood scale).
    prior     : femmi.priors.Prior or None -> Wiener. method='auto' uses exact RTO
                for the Gaussian/Wiener prior and annealed HMC for any other prior.
    lam       : prior precision in the PROPER Bayesian posterior, whose data term
                is ||F kappa - gamma||^2 / (2 sigma_n^2). This is NOT the same
                convention as MAPReconstructor (whose data term is un-normalised):
                for the same reconstruction, lam here ~ lam_MAP / (2 sigma_n^2),
                i.e. numerically much larger. All three sampler methods share this
                convention. Leave lam=None (recommended) to auto-calibrate it --
                Morozov for the Wiener prior, 1.0 for the neural score prior; a raw
                small lam leaves the posterior noise-dominated (see _auto_lam).
    """
    w = None if data_weight is None else np.asarray(data_weight, float)
    g1 = np.asarray(gamma1_obs, float); g2 = np.asarray(gamma2_obs, float)

    # Auto-calibrate lam when the caller doesn't set one. Left uncalibrated, the
    # sampler's raw lam is trivially wrong: the data term carries weight 1/sigma_n^2
    # (~10^2-10^3), so a small lam leaves the posterior wildly under-regularised
    # (noise-dominated, L2 > 1). See _auto_lam.
    if lam is None:
        lam = _auto_lam(fwd.ops, g1, g2, float(noise_std), prior, wiener_length, w, verbose)
    fwd.lam_reg = lam

    is_gaussian = prior is None or isinstance(prior, WienerPrior)
    if method == "auto":
        method = "rto" if is_gaussian else "annealed_hmc"
    if method == "rto" and not is_gaussian:
        raise ValueError("method='rto' is exact only for the Gaussian/Wiener prior; "
                         "use method='langevin' or 'annealed_hmc' for non-Gaussian priors.")

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
    elif method == "annealed_hmc":
        k_map, samples, info = _annealed_hmc(
            fwd, g1, g2, float(noise_std), prior, lam, wiener_length, w,
            n_levels, steps_per_level, sigma_max, sigma_min, n_leapfrog,
            leap_frac, n_delta_logp, n_chains, keep_final, maxiter_map, seed, verbose)
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
