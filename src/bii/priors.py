"""Prior log-densities as composable factory functions.

Each ``make_*_logposterior`` closes over data and returns a pure
``logprob_fn(position) -> scalar`` suitable for NUTS or VI.
"""

import jax
from jax import numpy as jnp

from bii.inference import loglik_w


def make_dirichlet_logposterior(T, Z, sig, alpha, kappa=1.0, noise_model="additive",
                                triplet_weights=None, clip_s=None, pi_inclusion=None,
                                pi_prior=None, link="probit", tau_prior=None):
    """Return ``logprob_fn(position) -> scalar`` for a Dirichlet prior.

    Args:
        T: (n,) binary triplet labels.
        Z: (n, 3, p) triplet embeddings.
        sig: noise std — scalar or (p,).
        alpha: (p,) Dirichlet concentration.
        kappa: power-likelihood exponent.
        noise_model: ``"additive"`` or ``"multiplicative"``.
        triplet_weights: optional (n,) per-triplet importance weights forwarded
            to :func:`bii.inference.loglik_w`.
        clip_s: optional float; if set, clips the per-triplet ``s`` statistic
            to ``[-clip_s, clip_s]`` before the normal CDF. Bounded-influence
            (censored-probit) robustifier against saturating triplets.
        pi_inclusion: optional float in (0, 1); fixed-mixture inclusion rate
            (see :func:`bii.inference.loglik_w`). Ignored if ``pi_prior`` is
            given.
        pi_prior: optional ``(a, b)`` tuple of Beta hyperparameters that puts
            a ``Beta(a, b)`` prior on ``pi``. When set, the position vector
            grows from ``(p,)`` to ``(p + 1,)``: the last entry is
            ``logit_pi`` and ``pi = sigmoid(logit_pi)`` is sampled jointly
            with ``theta``. The Beta prior is implemented with the Jacobian
            of the logit transform absorbed into the density, so the prior
            on ``logit_pi`` is the push-forward of ``Beta(a, b)`` on ``pi``.
        link: ``"probit"`` (default) or ``"logit"``; forwarded to
            :func:`bii.inference.loglik_w`.
        tau_prior: optional ``(a, b)`` tuple of Gamma hyperparameters that
            puts a ``Gamma(a, b)`` prior on an extra relation-noise std
            ``tau`` and inflates the per-triplet variance to ``V + tau^2``
            (the calibrated statistic for weak target-source signal; see
            :func:`bii.inference.logP_log1mP_from_deltaV`). When set, the
            position vector grows to ``(p + 1,)``: the last entry is
            ``log_tau`` and ``tau = exp(log_tau)`` is sampled jointly with
            ``theta``. The Gamma prior is implemented with the Jacobian of
            the log transform absorbed, so the prior on ``log_tau`` is the
            push-forward of ``Gamma(a, b)`` on ``tau``. Mutually exclusive
            with ``pi_prior`` (raise) — compose ``tau_prior`` with a fixed
            ``pi_inclusion`` instead if both effects are needed.

    Returns:
        A callable that takes a ``(p,)`` array (no learned nuisance), or a
        ``(p + 1,)`` array (``pi_prior`` or ``tau_prior`` given) and returns
        the log-posterior up to an additive constant.
    """
    if pi_prior is not None and tau_prior is not None:
        raise NotImplementedError(
            "pi_prior and tau_prior cannot be sampled jointly yet; "
            "use tau_prior with a fixed pi_inclusion instead."
        )

    if pi_prior is None and tau_prior is None:
        def logprob_fn(theta):
            w = jax.nn.softmax(theta)
            ll = loglik_w(w, T, Z, sig, noise_model,
                          triplet_weights=triplet_weights, clip_s=clip_s,
                          pi_inclusion=pi_inclusion, link=link)
            return kappa * ll + jnp.sum(alpha * jnp.log(w + 1e-12))
        return logprob_fn

    if tau_prior is not None:
        a_t, b_t = tau_prior

        def logprob_fn_with_tau(position):
            theta = position[:-1]
            log_tau = position[-1]
            tau = jnp.exp(log_tau)
            w = jax.nn.softmax(theta)
            ll = loglik_w(w, T, Z, sig, noise_model,
                          triplet_weights=triplet_weights, clip_s=clip_s,
                          pi_inclusion=pi_inclusion, link=link, tau2=tau**2)
            # Gamma(a, b) prior on tau with log-transform Jacobian absorbed:
            #   log p(tau) = (a-1) log tau - b tau + const
            # Jacobian: d tau / d log_tau = tau, so
            #   log p(log_tau) = log p(tau) + log tau = a log tau - b tau + const
            log_tau_prior = a_t * log_tau - b_t * tau
            return (kappa * ll
                    + jnp.sum(alpha * jnp.log(w + 1e-12))
                    + log_tau_prior)

        return logprob_fn_with_tau

    a, b = pi_prior

    def logprob_fn_with_pi(position):
        theta = position[:-1]
        logit_pi = position[-1]
        # pi = sigmoid(logit_pi); log_pi and log_one_minus_pi computed in a
        # numerically stable way to keep gradients well-behaved.
        log_pi = -jax.nn.softplus(-logit_pi)         # = log sigmoid(logit_pi)
        log_one_minus_pi = -jax.nn.softplus(logit_pi)  # = log(1 - sigmoid(logit_pi))
        pi = jnp.exp(log_pi)
        w = jax.nn.softmax(theta)
        ll = loglik_w(w, T, Z, sig, noise_model,
                      triplet_weights=triplet_weights, clip_s=clip_s,
                      pi_inclusion=pi, link=link)
        # Beta(a, b) prior on pi with logit-transform Jacobian absorbed:
        #   log p(pi) = (a-1) log pi + (b-1) log(1-pi) + const
        # Jacobian: d pi / d logit_pi = pi (1 - pi), so
        #   log p(logit_pi) = log p(pi) + log pi + log(1-pi)
        #                   = a log pi + b log(1-pi) + const
        log_pi_prior = a * log_pi + b * log_one_minus_pi
        return (kappa * ll
                + jnp.sum(alpha * jnp.log(w + 1e-12))
                + log_pi_prior)

    return logprob_fn_with_pi
