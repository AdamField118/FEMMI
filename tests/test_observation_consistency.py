"""Independent regression checks for the discrete likelihood and its transpose."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from femmi.operators import build_operators, build_operators_dirichlet
from femmi.forward import DifferentiableForward
from femmi.inverse import MAPReconstructor
from femmi.observations import prepare_observations, weighted_rms, noise_scale
from femmi.regularization import MorozovSelector, discrepancy
from femmi.sampling import _make_FtWF, sample_posterior


@pytest.fixture(scope='module', params=['bem', 'dirichlet'])
def ops(request):
    factory = build_operators if request.param == 'bem' else build_operators_dirichlet
    return factory(3, 3, -2., 2., -2., 2., verbose=False)


def dense_forward(ops):
    return np.column_stack([np.concatenate(ops.forward(e)) for e in np.eye(ops.n_nodes)])


def test_transpose_numpy_jax_and_hessian(ops):
    n = ops.n_nodes
    rng = np.random.default_rng(11)
    k, v = rng.normal(size=(2, n))
    y = rng.normal(size=2*n)
    F = dense_forward(ops)
    np.testing.assert_allclose(ops.adjoint_rhs(y[:n], y[n:]), F.T @ y, atol=2e-11)
    fwd = DifferentiableForward(ops, lam_reg=0.03)
    np.testing.assert_allclose(np.concatenate(fwd.gamma_from_kappa(jnp.array(k))), F @ k, atol=2e-11)
    _, pullback = jax.vjp(fwd.gamma_from_kappa, jnp.array(k))
    np.testing.assert_allclose(pullback((jnp.array(y[:n]), jnp.array(y[n:])))[0], F.T @ y, atol=2e-11)
    eps = 1e-5
    gp = np.asarray(fwd.grad_fn(jnp.array(k+eps*v), jnp.array(y[:n]), jnp.array(y[n:]))[1])
    gm = np.asarray(fwd.grad_fn(jnp.array(k-eps*v), jnp.array(y[:n]), jnp.array(y[n:]))[1])
    np.testing.assert_allclose(fwd.hvp(k, v, y[:n], y[n:]), (gp-gm)/(2*eps), atol=2e-8)


def test_weighted_objective_and_sampling_precision(ops):
    n = ops.n_nodes
    rng = np.random.default_rng(21)
    k = rng.normal(size=n)
    y = rng.normal(size=2*n)
    w = np.linspace(.2, 3., n); w[::4] = 0
    F = dense_forward(ops); W = np.tile(w, 2)
    rec = MAPReconstructor(DifferentiableForward(ops, .2), data_weight=w, wiener_length=.5)
    obj, _ = rec._make_obj_and_grad(y[:n], y[n:])
    loss, grad = obj(k)
    residual = F @ k - y
    np.testing.assert_allclose(loss, residual @ (W*residual) + .2*k @ (rec._R @ k))
    np.testing.assert_allclose(grad, 2*F.T @ (W*residual) + .4*(rec._R @ k), atol=2e-10)
    _, _, action = _make_FtWF(ops, w, .09, rec._R, .4)
    np.testing.assert_allclose(action(k), F.T @ (W*(F@k))/.09 + .4*(rec._R@k), atol=2e-9)


def test_mask_is_missing_in_fit_selection_and_diagnostics(ops, monkeypatch):
    n = ops.n_nodes
    rng = np.random.default_rng(31)
    g1, g2 = rng.normal(scale=.1, size=(2,n))
    mask = np.arange(n) % 3 == 0
    w = np.linspace(.2, 2, n); w[mask] = 0
    g1[mask] = np.nan; g2[mask] = 1e100
    calls = []
    def select(self, a, b, noise_std=None):
        calls.append((a.copy(), b.copy(), self.data_weight.copy()))
        return .1
    monkeypatch.setattr(MorozovSelector, 'select', select)
    rec = MAPReconstructor(DifferentiableForward(ops), data_weight=np.linspace(.2,2,n),
                           noise_std=.1, maxiter=150, wiener_length=.5, callback_every=0)
    k1, _ = rec.reconstruct(g1, g2, mask=mask, verbose=False)
    ref = MAPReconstructor(DifferentiableForward(ops), data_weight=w,
                           noise_std=.1, maxiter=150, wiener_length=.5, callback_every=0)
    k2, _ = ref.reconstruct(g1, g2, verbose=False)
    np.testing.assert_allclose(k1,k2)
    for a,b,weight in calls:
        assert np.all(a[mask] == 0) and np.all(b[mask] == 0)
        np.testing.assert_array_equal(weight,w)
    assert np.all(rec.data_weight > 0), 'mask must not persist on reusable reconstructor'
    d1, _, _ = rec.bmode_diagnostics(g1,g2,mask=mask,verbose=False)
    d2, _, _ = ref.bmode_diagnostics(g1,g2,verbose=False)
    assert d1.delta_noise == pytest.approx(d2.delta_noise)
    assert d1.delta_mad == pytest.approx(d2.delta_mad)


def test_morozov_noise_estimate_ignores_inactive_values(ops, monkeypatch):
    n = ops.n_nodes
    w = np.ones(n); w[::2] = 0; w[1::4] = 4
    y = np.linspace(-.2,.3,n); y[w==0] = np.nan
    expected = noise_scale(y,y,w)
    seen = []
    def D(self, lam, a, b, delta):
        seen.append(delta)
        return lam-1
    monkeypatch.setattr(MorozovSelector, '_D', D)
    sel=MorozovSelector(ops,data_weight=w,verbose=False)
    assert sel.select(y,y) == pytest.approx(1)
    np.testing.assert_allclose(seen,expected)
    seen.clear(); sel.select(y,y,noise_std=0)
    assert set(seen)=={0}


@pytest.mark.parametrize('weight', [[1,-1], [1,np.nan], [0,0], [1]])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError):
        prepare_observations([1,2],[1,2],2,weight)


def test_rto_covariance_matches_dense_posterior(ops):
    # A small physical mesh and strong proper prior permit a well-resolved
    # Monte Carlo covariance check, including non-binary and zero weights.
    n=ops.n_nodes; F=dense_forward(ops)
    w=np.linspace(.1,4,n); w[::3]=0
    sn=.3; lam=4.; ell=.7
    R=(ops.M+ell**2*ops.K).toarray()
    A=F.T @ (np.tile(w,2)[:,None]*F)/sn**2 + 2*lam*R
    rng=np.random.default_rng(43); y=rng.normal(scale=.1,size=2*n)
    exact_mean=np.linalg.solve(A,F.T@(np.tile(w,2)*y)/sn**2)
    ps=sample_posterior(DifferentiableForward(ops),y[:n],y[n:],sn,lam=lam,
                        data_weight=w,wiener_length=ell,n_samples=1200,
                        cg_tol=1e-10,seed=44,verbose=False)
    np.testing.assert_allclose(ps.map_kappa,exact_mean,atol=1e-8)
    probes=rng.normal(size=(n,4)); probes/=np.linalg.norm(probes,axis=0)
    target=np.diag(probes.T @ np.linalg.solve(A,probes))
    observed=np.var(ps.samples@probes,axis=0,ddof=1)
    np.testing.assert_allclose(observed,target,rtol=.15)


def test_hmc_uses_supplied_energy_prior(ops):
    from femmi.priors import WienerPrior
    n=ops.n_nodes
    rng=np.random.default_rng(46); y=rng.normal(scale=.1,size=(2,n))
    kw=dict(noise_std=.2,lam=2.,method='annealed_hmc',n_levels=2,
            steps_per_level=3,n_leapfrog=2,n_chains=2,keep_final=1,
            maxiter_map=80,sigma_max=.2,sigma_min=.05,seed=9,verbose=False)
    explicit=sample_posterior(DifferentiableForward(ops),*y,
                               prior=WienerPrior(ops,.7),wiener_length=0.,**kw)
    builtin=sample_posterior(DifferentiableForward(ops),*y,wiener_length=.7,**kw)
    np.testing.assert_allclose(explicit.samples,builtin.samples,atol=1e-11)


def test_score_only_prior_rejected_by_map_but_sampler_remains_available(ops):
    from femmi.priors import ScorePrior
    prior=ScorePrior(lambda k: -k)
    with pytest.raises(ValueError,match='consistent prior value'):
        MAPReconstructor(DifferentiableForward(ops),prior=prior)
    z=np.zeros(ops.n_nodes)
    result=sample_posterior(DifferentiableForward(ops),z,z,.1,
                            prior=prior,lam=1.,method='langevin',
                            n_steps=10,burnin=2,thin=2,step=1e-7,verbose=False)
    assert np.all(np.isfinite(result.samples))
    assert result.info['map_kind']=='initialization'


def test_posterior_mask_matches_zero_weights(ops):
    n=ops.n_nodes; mask=np.arange(n)%3==0
    w=np.linspace(.2,2,n)
    y=np.linspace(-.1,.1,n); y[mask]=np.nan
    opts=dict(noise_std=.1,lam=2.,wiener_length=.5,n_samples=3,seed=7,verbose=False)
    masked=sample_posterior(DifferentiableForward(ops),y,y,data_weight=w,mask=mask,**opts)
    w[mask]=0
    weighted=sample_posterior(DifferentiableForward(ops),y,y,data_weight=w,**opts)
    np.testing.assert_array_equal(masked.samples,weighted.samples)


def test_failed_posterior_linear_solve_is_not_returned_as_a_sample(ops,monkeypatch):
    import femmi.sampling as sampling
    monkeypatch.setattr(sampling.spla,'cg',lambda A,b,**kw:(np.zeros_like(b),1))
    z=np.zeros(ops.n_nodes)
    with pytest.raises(RuntimeError,match='CG failed'):
        sample_posterior(DifferentiableForward(ops),z,z,.1,lam=1.,
                         wiener_length=.5,n_samples=1,verbose=False)
