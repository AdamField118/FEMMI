"""
femmi/neural_prior/train.py
Denoising Score Matching training for the FEMMI neural prior, plus checkpoint
save/load. Self-contained: trains on synthetic non-Gaussian maps (data.py), no
external dataset.

DSM loss (Remy et al. 2020 eq. 3; Vincent 2011):
    x' = x + sigma * u,   u ~ N(0, I),   sigma ~ log-uniform[sigma_min, sigma_max]
    L  = E || u + sigma * r_theta(x', sigma) ||^2
At the optimum r_theta(x', sigma) = grad_x log p_sigma(x'), the score we want.

Training tracks a HELD-OUT validation DSM loss (fixed maps + fixed noise, so it is
comparable across steps) and does EARLY STOPPING with patience: it keeps the
best-validation params and stops once validation has not improved for `patience`
evaluations. This avoids the wasted compute and mild overfitting of running a
fixed huge step budget past convergence.
"""

from __future__ import annotations
import os
import re
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization

from .denoiser import ScoreUNet, init_params
from .data import batch_generator, lognormal_kappa_maps

_CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def default_ckpt_path(n_pix, base=16):
    return os.path.join(_CKPT_DIR, f"score_unet_p{n_pix}_b{base}.msgpack")


def parse_ckpt_arch(path):
    """Recover (n_pix, base) from a checkpoint filename score_unet_p{N}_b{B}.msgpack.
    Returns (None, None) if the name does not encode them."""
    m = re.search(r"_p(\d+)_b(\d+)", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _fixed_validation_set(n_pix, n_val=64, seed=1234):
    """A held-out set of maps + FIXED noise/sigma, so the val loss is comparable
    across training steps (unlike a fresh-noise train loss)."""
    x = lognormal_kappa_maps(n_val, n_pix, seed=seed)[..., None]
    rng = np.random.default_rng(seed + 1)
    sigma = np.exp(rng.uniform(np.log(0.02), np.log(1.0), n_val)).astype(np.float32)
    u = rng.standard_normal(x.shape).astype(np.float32)
    return jnp.asarray(x), jnp.asarray(sigma), jnp.asarray(u)


def train_score_model(n_pix=32, base=16, steps=8000, batch=32, lr=2e-4,
                      sigma_min=0.02, sigma_max=1.0, seed=0, verbose=True,
                      save_path=None, patience=8, val_every=250, min_steps=1000):
    """Train the score U-Net by DSM with validation-based early stopping.

    patience   : stop after this many validation checks without improvement.
    val_every  : evaluate validation loss every this many steps.
    min_steps  : never stop before this many steps (let it warm up).
    `steps` is an upper bound; training usually stops earlier at the best
    validation loss, whose params are what get saved.
    Returns (model, params).
    """
    model, params = init_params(jax.random.PRNGKey(seed), n_pix, base=base)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    gen = batch_generator(n_pix, batch=batch, seed=seed + 1)
    log_smin, log_smax = np.log(sigma_min), np.log(sigma_max)

    def loss_fn(params, x, sigma, u):
        r = model.apply(params, x + sigma[:, None, None, None] * u, sigma)
        return jnp.mean((u + sigma[:, None, None, None] * r) ** 2)

    @jax.jit
    def step(params, opt_state, x, sigma, u):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, sigma, u)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    val_x, val_sigma, val_u = _fixed_validation_set(n_pix)
    val_loss = jax.jit(loss_fn)

    key = jax.random.PRNGKey(seed + 2)
    ema = None
    best_val = np.inf
    best_params = params
    best_step = 0
    bad = 0
    for it in range(steps):
        x = jnp.asarray(next(gen))
        key, k1, k2 = jax.random.split(key, 3)
        sigma = jnp.exp(jax.random.uniform(k1, (x.shape[0],), minval=log_smin, maxval=log_smax))
        u = jax.random.normal(k2, x.shape)
        params, opt_state, loss = step(params, opt_state, x, sigma, u)
        ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)

        if (it + 1) % val_every == 0 or it == steps - 1:
            vl = float(val_loss(params, val_x, val_sigma, val_u))
            improved = vl < best_val - 1e-4
            if improved:
                best_val, best_params, best_step, bad = vl, params, it + 1, 0
            else:
                bad += 1
            if verbose:
                flag = " *" if improved else ""
                print(f"  DSM step {it+1:5d}/{steps}  train(ema)={ema:.4f}  "
                      f"val={vl:.4f}  best={best_val:.4f}@{best_step}{flag}")
            if bad >= patience and (it + 1) >= min_steps:
                if verbose:
                    print(f"  early stop: no val improvement for {patience} checks "
                          f"(best {best_val:.4f} @ step {best_step})")
                break

    params = best_params                              # restore best-validation params
    path = save_path or default_ckpt_path(n_pix, base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes({"params": params,
                                        "cfg": {"n_pix": n_pix, "base": base}}))
    if verbose:
        print(f"  saved best score checkpoint (val={best_val:.4f}) -> {os.path.relpath(path)}")
    return model, params


def load_score_model(n_pix=32, base=16, path=None):
    """Load a trained score model; returns (model, params) or (None, None).

    If `path` is given, the architecture is read from its filename
    (score_unet_p{N}_b{B}.msgpack) so you only need to reference the file -- the
    n_pix/base arguments are used only as a fallback."""
    path = path or default_ckpt_path(n_pix, base)
    if not os.path.exists(path):
        return None, None
    fn_pix, fn_base = parse_ckpt_arch(path)
    n_pix = fn_pix if fn_pix is not None else n_pix
    base = fn_base if fn_base is not None else base
    model, params0 = init_params(jax.random.PRNGKey(0), n_pix, base=base)
    with open(path, "rb") as f:
        target = serialization.from_bytes({"params": params0,
                                           "cfg": {"n_pix": n_pix, "base": base}},
                                          f.read())
    return model, target["params"]


def get_or_train(n_pix=32, base=16, steps=8000, verbose=True, path=None):
    """Load the cached score model, or train (and cache) one if absent.
    This is what makes the neural prior 'one flag away' -- first use trains a
    small default model on synthetic data; later uses load the checkpoint. When
    `path` names an existing checkpoint, its architecture is auto-detected."""
    model, params = load_score_model(n_pix, base, path)
    if model is not None:
        if verbose:
            fn_pix, fn_base = parse_ckpt_arch(path or default_ckpt_path(n_pix, base))
            print(f"  loaded cached score model "
                  f"(n_pix={fn_pix or n_pix}, base={fn_base or base})")
        return model, params
    if verbose:
        print(f"  no cached score model -- training a default one "
              f"(n_pix={n_pix}); one-time, early-stopped on validation ...")
    return train_score_model(n_pix=n_pix, base=base, steps=steps, verbose=verbose,
                             save_path=path)
