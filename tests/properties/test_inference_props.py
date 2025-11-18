# Property-based tests for inference helpers

import math
import numpy as np
import jax
import jax.numpy as jnp
from hypothesis import given, settings, strategies as st

from bii.inference import (
    compute_T_from_dag,
    probit_pairwise_probabilities,
    build_pairwise_data,
)

DEFAULT_SETTINGS = settings(deadline=None, max_examples=40)


def _random_pairs(rng, n_cols, K):
    all_pairs = [
        (i, j)
        for i in range(n_cols)
        for j in range(n_cols)
        if i != j
    ]
    all_pairs = np.array(all_pairs, dtype=np.int32)
    idx = rng.integers(0, len(all_pairs), size=K, endpoint=False)
    return all_pairs[idx]


@DEFAULT_SETTINGS
@given(
    n_rows=st.integers(min_value=1, max_value=3),
    n_cols=st.integers(min_value=3, max_value=6),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_T_matches_direct_comparison(n_rows, n_cols, seed):
    rng = np.random.default_rng(seed)
    distances = rng.random((n_rows, n_cols), dtype=np.float32)
    K = min(6, n_cols * (n_cols - 1))
    dag_pairs = np.stack(
        [_random_pairs(rng, n_cols, K) for _ in range(n_rows)],
        axis=0,
    )
    mask = rng.random((n_rows, K)) > 0.25

    T = np.array(
        compute_T_from_dag(
            jnp.array(distances),
            jnp.array(dag_pairs),
            jnp.array(mask),
        )
    )

    manual = distances[
        np.arange(n_rows)[:, None],
        dag_pairs[..., 0],
    ] <= distances[
        np.arange(n_rows)[:, None],
        dag_pairs[..., 1],
    ]
    manual = manual.astype(int)
    manual = np.where(mask, manual, 0)

    np.testing.assert_array_equal(T, manual)


@DEFAULT_SETTINGS
@given(
    n_rows=st.integers(min_value=1, max_value=2),
    n_cols=st.integers(min_value=3, max_value=5),
    dim=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_probit_probabilities_match_closed_form(n_rows, n_cols, dim, seed):
    rng = np.random.default_rng(seed)
    Z_rows = rng.standard_normal((n_rows, dim), dtype=np.float32)
    Z_cols = rng.standard_normal((n_cols, dim), dtype=np.float32)
    weights = rng.random(dim, dtype=np.float32)
    sigma = 0.05 + rng.random(dim, dtype=np.float32)

    K = min(5, n_cols * (n_cols - 1))
    dag_pairs = np.stack(
        [_random_pairs(rng, n_cols, K) for _ in range(n_rows)],
        axis=0,
    )
    mask = rng.random((n_rows, K)) > 0.2

    probs = np.array(
        probit_pairwise_probabilities(
            jnp.array(Z_rows),
            jnp.array(Z_cols),
            jnp.array(dag_pairs),
            jnp.array(mask),
            jnp.array(weights),
            jnp.array(sigma),
        )
    )

    noise_var = 4.0 * np.sum((weights**2) * (sigma**4))
    noise_std = math.sqrt(noise_var)
    inv = 1.0 / noise_std if noise_std > 0 else 0.0

    manual = np.zeros_like(probs)
    for r in range(n_rows):
        z0 = Z_rows[r]
        for k in range(K):
            if not mask[r, k]:
                continue
            src, dst = dag_pairs[r, k]
            zi = Z_cols[src]
            zj = Z_cols[dst]
            margin = np.sum(weights * ((zi - z0) ** 2 - (zj - z0) ** 2))
            if noise_std > 0:
                manual[r, k] = 0.5 * (1.0 + math.erf((-margin * inv) / math.sqrt(2.0)))
            else:
                manual[r, k] = 1.0 if margin <= 0 else 0.0

    np.testing.assert_allclose(probs, manual, atol=1e-5)


def test_build_pairwise_data_deterministic_property():
    key = jax.random.PRNGKey(123)
    n = 10
    p = 2
    X = jax.random.normal(key, (n, p))
    Z = X + 0.05 * jax.random.normal(key, (n, p))
    rows_idx = jnp.arange(3, dtype=jnp.int32)
    cols_idx = jnp.arange(3, n, dtype=jnp.int32)
    pair_a = build_pairwise_data(
        X[rows_idx],
        X[cols_idx],
        Z[rows_idx],
        Z[cols_idx],
        num_shells=5,
    )
    pair_b = build_pairwise_data(
        X[rows_idx],
        X[cols_idx],
        Z[rows_idx],
        Z[cols_idx],
        num_shells=5,
    )
    np.testing.assert_array_equal(pair_a.dag_pairs, pair_b.dag_pairs)
