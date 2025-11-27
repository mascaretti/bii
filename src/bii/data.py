# -*- coding: utf-8 -*-
"""
Lightweight data generators used across simulations.

Current toy process:
    X ~ N(0, cov_scale * I)
    eps ~ N(0, diag(sigma))
    Z = (X + eps) / sqrt(w)
This keeps the observation model aligned with the inference utilities.

Also includes a PPP-based generator for uniform points in a p-ball.
"""
from typing import Tuple
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def generate_observations(
    key: jax.Array,
    n: int,
    p: int,
    w: Array,
    sigma: Array,
    *,
    cov_scale: float = 1.0,
) -> Tuple[Array, Array]:
    """Sample latent X and observed Z for the toy model.

    Args:
        key: JAX PRNG key.
        n: number of observations.
        p: dimension.
        w: positive weights of shape (p,).
        sigma: per-dimension noise std of shape (p,).
        cov_scale: scalar factor for the latent covariance matrix.

    Returns:
        X_latent: (n, p) latent draws.
        Z_obs: (n, p) observations (X + eps) / sqrt(w).
    """
    w = jnp.asarray(w)
    sigma = jnp.asarray(sigma)
    if w.shape[-1] != p:
        raise ValueError(f"w must have shape ({p},), got {w.shape}")
    if sigma.shape[-1] != p:
        raise ValueError(f"sigma must have shape ({p},), got {sigma.shape}")
    if jnp.any(w <= 0):
        raise ValueError("w must be strictly positive.")
    if jnp.any(sigma < 0):
        raise ValueError("sigma must be non-negative.")

    key_x, key_eps = jax.random.split(key, 2)
    X_latent = jax.random.multivariate_normal(
        key_x, jnp.zeros(p), cov_scale * jnp.eye(p), shape=(n,)
    )
    eps = jax.random.multivariate_normal(
        key_eps, jnp.zeros(p), jnp.diag(sigma), shape=(n,)
    )
    Z_obs = (X_latent + eps) / jnp.sqrt(w)[None, :]
    return X_latent, Z_obs


def _sample_uniform_ball(key: jax.Array, m: int, p: int, radius: float) -> Array:
    """Sample m points uniformly in a p-dimensional ball of given radius."""
    key_dir, key_rad = jax.random.split(key)
    normals = jax.random.normal(key_dir, (m, p))
    norms = jnp.linalg.norm(normals, axis=-1, keepdims=True)
    directions = normals / jnp.maximum(norms, 1e-9)
    u = jax.random.uniform(key_rad, (m, 1))
    r = radius * jnp.power(u, 1.0 / p)
    return directions * r


def generate_ppp_observations(
    key: jax.Array,
    *,
    expected_n: int,
    p: int,
    w: Array,
    sigma: Array,
    radius: float = 1.0,
    use_poisson: bool = True,
) -> Tuple[Array, Array, int]:
    """Generate observations from a homogeneous PPP in a p-ball with noise.

    Args:
        key: JAX PRNG key.
        expected_n: mean number of PPP points (or exact n if use_poisson=False).
        p: dimension.
        w: positive weights of shape (p,).
        sigma: per-dimension noise std of shape (p,).
        radius: ball radius for PPP sampling.
        use_poisson: if True, sample N~Poisson(expected_n), else fix N=expected_n.

    Returns:
        X_latent: (N, p) latent PPP points.
        Z_obs: (N, p) observations (X + eps) / sqrt(w).
        n: realized number of points.
    """
    w = jnp.asarray(w)
    sigma = jnp.asarray(sigma)
    if w.shape[-1] != p:
        raise ValueError(f"w must have shape ({p},), got {w.shape}")
    if sigma.shape[-1] != p:
        raise ValueError(f"sigma must have shape ({p},), got {sigma.shape}")
    if jnp.any(w <= 0):
        raise ValueError("w must be strictly positive.")
    if jnp.any(sigma < 0):
        raise ValueError("sigma must be non-negative.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    key_n, key_pts, key_eps = jax.random.split(key, 3)
    if use_poisson:
        n = jax.random.poisson(key_n, lam=float(expected_n), shape=()).astype(int)
    else:
        n = int(expected_n)
    n = int(jnp.clip(n, 0, 10_000_000))  # simple guard

    X_latent = _sample_uniform_ball(key_pts, n, p, radius)
    eps = jax.random.multivariate_normal(
        key_eps, jnp.zeros(p), jnp.diag(sigma), shape=(n,)
    )
    Z_obs = (X_latent + eps) / jnp.sqrt(w)[None, :]
    return X_latent, Z_obs, n
