"""Mean-field variational inference for metric weights."""

import jax
import optax
from jax import numpy as jnp
from jax import random


def _neg_elbo(params, key, logprob_fn, num_samples):
    """Negative ELBO for mean-field Gaussian in theta-space.

    Args:
        params: dict with ``"mu"`` (p,) and ``"log_sigma"`` (p,).
        key: JAX random key.
        logprob_fn: callable theta -> scalar log-posterior.
        num_samples: number of MC samples for the expectation.

    Returns:
        Scalar ``-ELBO``.
    """
    mu = params["mu"]
    log_sigma = params["log_sigma"]
    sigma = jnp.exp(log_sigma)
    p = mu.shape[0]

    # Reparameterization trick
    eps = random.normal(key, shape=(num_samples, p))
    theta = mu[None, :] + sigma[None, :] * eps  # (num_samples, p)

    # E_q[log p(theta | data)]
    log_probs = jax.vmap(logprob_fn)(theta)  # (num_samples,)
    expected_logprob = jnp.mean(log_probs)

    # Entropy of mean-field Gaussian: sum(log sigma_i) + p/2 * log(2 pi e)
    entropy = jnp.sum(log_sigma) + 0.5 * p * jnp.log(2.0 * jnp.pi * jnp.e)

    elbo = expected_logprob + entropy
    return -elbo


def run_vi(key, logprob_fn, p, num_steps=5000, lr=1e-2, num_elbo_samples=8):
    """Fit a mean-field Gaussian approximation via ELBO maximization.

    Args:
        key: JAX random key.
        logprob_fn: callable theta -> scalar log-posterior.
        p: dimension of theta.
        num_steps: number of optimization steps.
        lr: Adam learning rate.
        num_elbo_samples: MC samples per ELBO estimate.

    Returns:
        mu: (p,) variational mean.
        log_sigma: (p,) log variational std.
        elbo_history: (num_steps,) ELBO values (positive = better).
    """
    params = {"mu": jnp.zeros(p), "log_sigma": -jnp.ones(p)}
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step(carry, step_key):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_neg_elbo)(
            params, step_key, logprob_fn, num_elbo_samples
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    keys = random.split(key, num_steps)
    (params, _), neg_elbo_history = jax.lax.scan(step, (params, opt_state), keys)

    mu = params["mu"]
    log_sigma = params["log_sigma"]
    elbo_history = -neg_elbo_history  # positive ELBO

    return mu, log_sigma, elbo_history


def sample_vi(key, mu, log_sigma, num_samples):
    """Draw samples from the fitted variational posterior.

    Args:
        key: JAX random key.
        mu: (p,) variational mean.
        log_sigma: (p,) log variational std.
        num_samples: number of samples.

    Returns:
        theta_samples: (num_samples, p).
        w_samples: (num_samples, p) on the simplex.
    """
    p = mu.shape[0]
    sigma = jnp.exp(log_sigma)
    eps = random.normal(key, shape=(num_samples, p))
    theta_samples = mu[None, :] + sigma[None, :] * eps
    w_samples = jax.vmap(jax.nn.softmax)(theta_samples)
    return theta_samples, w_samples
