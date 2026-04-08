"""MCMC and variational inference runners — prior-agnostic, pure functions."""

import blackjax
import jax
import optax
from jax import numpy as jnp
from jax import random
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NUTS
# ---------------------------------------------------------------------------


def run_nuts(
    key,
    logprob_fn,
    init_position,
    num_samples,
    num_warmup,
    num_chains,
    step_size=1e-3,
    target_acceptance_rate=0.8,
):
    """Run multi-chain NUTS via BlackJAX.

    Args:
        key: JAX random key.
        logprob_fn: callable ``position -> scalar``.
        init_position: (dim,) initial point.
        num_samples: posterior draws per chain.
        num_warmup: warmup iterations.
        num_chains: number of MCMC chains.
        step_size: initial NUTS step size.
        target_acceptance_rate: NUTS target acceptance.

    Returns:
        ``(raw_samples, acceptance_rates)``
            raw_samples: ``(num_samples, num_chains, dim)``
            acceptance_rates: ``(num_samples, num_chains)``
    """
    dim = init_position.shape[0]

    key, *init_keys = random.split(key, num_chains + 1)
    init_positions = jnp.stack([
        init_position + 0.1 * random.normal(k, shape=(dim,)) for k in init_keys
    ])

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        num_warmup,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=step_size,
    )

    keys_warmup = random.split(key, num_chains)

    all_samples = []
    all_acceptance = []

    def _make_step_fn(kernel):
        def one_step(state, key):
            new_state, info = kernel.step(key, state)
            return new_state, (new_state.position, info.acceptance_rate)

        return one_step

    for chain_idx in range(num_chains):
        (state, params), _ = warmup.run(keys_warmup[chain_idx], init_positions[chain_idx])
        kernel = blackjax.nuts(logprob_fn, **params)
        one_step = _make_step_fn(kernel)

        key, subkey = random.split(key)
        sample_keys = random.split(subkey, num_samples)
        _, (samples_chain, acc_chain) = jax.lax.scan(one_step, state, sample_keys)

        all_samples.append(samples_chain)
        all_acceptance.append(acc_chain)

    raw_samples = jnp.stack(all_samples, axis=1)
    acceptance_rates = jnp.stack(all_acceptance, axis=1)
    return raw_samples, acceptance_rates


# ---------------------------------------------------------------------------
# Variational inference
# ---------------------------------------------------------------------------


def _neg_elbo(params, key, logprob_fn, num_samples):
    """Negative ELBO for mean-field Gaussian via reparameterization."""
    mu = params["mu"]
    log_sigma = params["log_sigma"]
    sigma = jnp.exp(log_sigma)
    p = mu.shape[0]

    eps = random.normal(key, shape=(num_samples, p))
    theta = mu[None, :] + sigma[None, :] * eps

    expected_logprob = jnp.mean(jax.vmap(logprob_fn)(theta))
    entropy = jnp.sum(log_sigma) + 0.5 * p * jnp.log(2.0 * jnp.pi * jnp.e)
    return -(expected_logprob + entropy)


def run_vi(key, logprob_fn, dim, num_steps=5000, lr=1e-2, num_elbo_samples=8):
    """Fit a mean-field Gaussian approximation via ELBO maximization.

    Args:
        key: JAX random key.
        logprob_fn: callable ``theta -> scalar`` log-posterior.
        dim: dimension of theta.
        num_steps: optimization steps.
        lr: Adam learning rate.
        num_elbo_samples: MC samples per ELBO estimate.

    Returns:
        ``(mu, log_sigma, elbo_history)``
            mu: (dim,) variational mean.
            log_sigma: (dim,) log variational std.
            elbo_history: (num_steps,) ELBO values.
    """
    params = {"mu": jnp.zeros(dim), "log_sigma": -jnp.ones(dim)}
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

    return params["mu"], params["log_sigma"], -neg_elbo_history


def sample_vi(key, mu, log_sigma, num_samples):
    """Draw samples from a fitted mean-field Gaussian posterior.

    Args:
        key: JAX random key.
        mu: (dim,) variational mean.
        log_sigma: (dim,) log variational std.
        num_samples: number of samples.

    Returns:
        ``(theta_samples, w_samples)``
            theta_samples: (num_samples, dim).
            w_samples: (num_samples, dim) on the simplex via softmax.
    """
    p = mu.shape[0]
    sigma = jnp.exp(log_sigma)
    eps = random.normal(key, shape=(num_samples, p))
    theta_samples = mu[None, :] + sigma[None, :] * eps
    w_samples = jax.vmap(jax.nn.softmax)(theta_samples)
    return theta_samples, w_samples
