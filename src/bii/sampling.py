"""MCMC and variational inference runners — prior-agnostic, pure functions."""

import blackjax
import jax
import optax
from jax import numpy as jnp
from jax import random

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

    key, key_init, key_chains = random.split(key, 3)
    init_positions = init_position[None, :] + 0.1 * random.normal(
        key_init, shape=(num_chains, dim)
    )

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        num_warmup,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=step_size,
    )

    def run_one_chain(chain_key, init_pos):
        """Warm up and sample one chain; vmapped over chains below."""
        key_warm, key_sample = random.split(chain_key)
        (state, params), _ = warmup.run(key_warm, init_pos)
        kernel = blackjax.nuts(logprob_fn, **params)

        def one_step(st, k):
            new_st, info = kernel.step(k, st)
            return new_st, (new_st.position, info.acceptance_rate)

        sample_keys = random.split(key_sample, num_samples)
        _, (positions, acc) = jax.lax.scan(one_step, state, sample_keys)
        return positions, acc  # (num_samples, dim), (num_samples,)

    # Chains run in parallel via vmap (vectorised over the GPU).
    chain_keys = random.split(key_chains, num_chains)
    positions, acceptance = jax.vmap(run_one_chain)(chain_keys, init_positions)

    # vmap stacks chains on axis 0; transpose to (num_samples, num_chains, ...).
    raw_samples = jnp.swapaxes(positions, 0, 1)
    acceptance_rates = jnp.swapaxes(acceptance, 0, 1)
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


def run_map(key, logprob_fn, dim, num_steps=3000, lr=1e-2, n_restarts=1,
            init_scale=0.5):
    """Find the MAP (posterior mode) by gradient ascent on the log-posterior.

    A fast point estimate: one optimisation instead of sampling, so it scales to
    large ``dim`` where NUTS is expensive. Adam ascends ``logprob_fn`` in the
    unconstrained space, with optional random restarts (the highest-logprob mode
    wins). This is the mode of the softmax-reparameterised posterior — the
    Dirichlet Jacobian is included — so it is consistent with the NUTS/VI
    parameterisation, and it does *not* quantify uncertainty (use NUTS for that).

    Args:
        key: JAX random key (seeds the restart inits).
        logprob_fn: callable ``position -> scalar`` log-posterior.
        dim: dimension of ``position``.
        num_steps: Adam iterations per restart.
        lr: Adam learning rate.
        n_restarts: number of random-init restarts; the best mode is returned.
            Restart 0 always starts at the origin (uniform ``w``).
        init_scale: std of the Gaussian restart inits (restarts 1..).

    Returns:
        ``(map_position, logprob_history)``
            map_position: (dim,) the best mode found.
            logprob_history: (num_steps,) log-posterior of the best restart.
    """
    def optimise(init):
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(init)

        def step(carry, _):
            pos, opt_state = carry
            neg_val, grad = jax.value_and_grad(lambda z: -logprob_fn(z))(pos)
            updates, opt_state = optimizer.update(grad, opt_state, pos)
            pos = optax.apply_updates(pos, updates)
            return (pos, opt_state), -neg_val

        (pos, _), hist = jax.lax.scan(step, (init, opt_state), None, length=num_steps)
        return pos, hist

    inits = init_scale * random.normal(key, (n_restarts, dim))
    inits = inits.at[0].set(jnp.zeros(dim))          # restart 0 = uniform w
    positions, histories = jax.vmap(optimise)(inits)
    best = jnp.argmax(histories[:, -1])
    return positions[best], histories[best]


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
