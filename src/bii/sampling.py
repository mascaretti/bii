"""NUTS and MAP runners: prior-agnostic, pure functions."""

import blackjax
import jax
import optax
from jax import numpy as jnp
from jax import random

# ---------------------------------------------------------------------------
# NUTS
# ---------------------------------------------------------------------------


def _report_progress(total):
    """Host callback for the sampling scan: prints ``sampling i/total`` every 10%."""
    every = max(1, total // 10)

    def report(i):
        i = int(i) + 1
        if i % every == 0 or i == total:
            print(f"sampling {i}/{total}", flush=True)

    return report


def run_nuts(
    key,
    logprob_fn,
    init_position,
    num_samples,
    num_warmup,
    num_chains,
    step_size=1e-3,
    target_acceptance_rate=0.8,
    progress_bar=False,
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
        progress_bar: if True, announce the run and print ``sampling i/n``
            about every 10% of the draws. The step counter is the same for
            every chain, so each line prints once, not once per chain. The
            warmup runs inside BlackJAX and has no progress lines; the first
            ``sampling`` line marks its end.

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

    if progress_bar:
        print(f"NUTS: {num_chains} chains, {num_warmup} warmup steps, "
              f"then {num_samples} draws", flush=True)
    report = _report_progress(num_samples)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=step_size,
    )

    def run_one_chain(chain_key, init_pos):
        """Warm up and sample one chain; vmapped over chains below."""
        key_warm, key_sample = random.split(chain_key)
        # The number of warmup steps goes to run(): without num_steps it
        # defaults to 1000 whatever num_warmup says.
        (state, params), _ = warmup.run(key_warm, init_pos, num_steps=num_warmup)
        kernel = blackjax.nuts(logprob_fn, **params)

        def one_step(st, xs):
            i, k = xs
            new_st, info = kernel.step(k, st)
            if progress_bar:
                jax.debug.callback(report, i)
            return new_st, (new_st.position, info.acceptance_rate)

        xs = (jnp.arange(num_samples), random.split(key_sample, num_samples))
        _, (positions, acc) = jax.lax.scan(one_step, state, xs)
        return positions, acc  # (num_samples, dim), (num_samples,)

    # Chains run in parallel via vmap (vectorised over the GPU).
    chain_keys = random.split(key_chains, num_chains)
    positions, acceptance = jax.vmap(run_one_chain)(chain_keys, init_positions)

    # vmap stacks chains on axis 0; transpose to (num_samples, num_chains, ...).
    raw_samples = jnp.swapaxes(positions, 0, 1)
    acceptance_rates = jnp.swapaxes(acceptance, 0, 1)
    return raw_samples, acceptance_rates


def run_map(key, logprob_fn, dim, num_steps=3000, lr=1e-2, n_restarts=1,
            init_scale=0.5):
    """Find the MAP (posterior mode) by gradient ascent on the log-posterior.

    A fast point estimate: one optimisation instead of sampling, so it scales to
    large ``dim`` where NUTS is expensive. Adam ascends ``logprob_fn`` in the
    unconstrained space, with optional random restarts (the highest-logprob mode
    wins). This is the mode of the softmax-reparameterised posterior — the
    Dirichlet Jacobian is included — so it is consistent with the NUTS
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
