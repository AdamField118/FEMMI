"""
femmi/neural_prior/train.py
Denoising Score Matching training for the FEMMI neural prior, plus checkpoint
save/load. Self-contained: trains on synthetic non-Gaussian maps (data.py), no
external dataset.

DSM loss (Remy et al. 2020 eq. 3; Vincent 2011):
    x' = x + sigma * u,   u ~ N(0, I),   sigma ~ log-uniform[sigma_min, sigma_max]
    L  = E || u + sigma * r_theta(x', sigma) ||^2
At the optimum r_theta(x', sigma) = grad_x log p_sigma(x'), the score we want.
"""

from __future__ import annotations
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization

from .denoiser import ScoreUNet, init_params
from .data import batch_generator

_CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def default_ckpt_path(n_pix, base=16):
    return os.path.join(_CKPT_DIR, f"score_unet_p{n_pix}_b{base}.msgpack")


def train_score_model(n_pix=32, base=16, steps=600, batch=32, lr=2e-4,
                      sigma_min=0.02, sigma_max=1.0, seed=0, verbose=True,
                      save_path=None):
    """Train the score U-Net by DSM on synthetic maps. Returns (model, params).
    Saves an msgpack checkpoint (default path under checkpoints/)."""
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

    key = jax.random.PRNGKey(seed + 2)
    ema = None
    for it in range(steps):
        x = jnp.asarray(next(gen))
        key, k1, k2 = jax.random.split(key, 3)
        sigma = jnp.exp(jax.random.uniform(k1, (x.shape[0],), minval=log_smin, maxval=log_smax))
        u = jax.random.normal(k2, x.shape)
        params, opt_state, loss = step(params, opt_state, x, sigma, u)
        ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
        if verbose and (it % 250 == 0 or it == steps - 1):
            print(f"  DSM step {it:4d}/{steps}  loss(ema)={ema:.4f}")

    path = save_path or default_ckpt_path(n_pix, base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes({"params": params,
                                        "cfg": {"n_pix": n_pix, "base": base}}))
    if verbose:
        print(f"  saved score checkpoint -> {os.path.relpath(path)}")
    return model, params


def load_score_model(n_pix=32, base=16, path=None):
    """Load a trained score model; returns (model, params) or (None, None)."""
    path = path or default_ckpt_path(n_pix, base)
    if not os.path.exists(path):
        return None, None
    model, params0 = init_params(jax.random.PRNGKey(0), n_pix, base=base)
    with open(path, "rb") as f:
        target = serialization.from_bytes({"params": params0,
                                           "cfg": {"n_pix": n_pix, "base": base}},
                                          f.read())
    return model, target["params"]


def get_or_train(n_pix=32, base=16, steps=600, verbose=True, path=None):
    """Load the cached score model, or train (and cache) one if absent.
    This is what makes the neural prior 'one flag away' -- first use trains a
    small default model on synthetic data; later uses load the checkpoint."""
    model, params = load_score_model(n_pix, base, path)
    if model is not None:
        if verbose:
            print(f"  loaded cached score model (n_pix={n_pix}, base={base})")
        return model, params
    if verbose:
        print(f"  no cached score model -- training a default one "
              f"(n_pix={n_pix}, steps={steps}); one-time, ~minutes on CPU ...")
    return train_score_model(n_pix=n_pix, base=base, steps=steps, verbose=verbose,
                             save_path=path)
