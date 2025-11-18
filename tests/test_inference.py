# tests/test_inference.py

import math
import numpy as np
import jax.numpy as jnp

import jax

from bii.inference import (
    remap_distances_to_columns,
    compute_T_from_dag,
    probit_pairwise_probabilities,
    build_pairwise_data,
    make_loglikelihood,
)


def test_compute_T_from_dag_matches_manual():
    distances = jnp.array(
        [
            [0.1, 0.5, 0.9],
            [0.3, 0.3, 0.4],
        ],
        dtype=jnp.float32,
    )
    dag_pairs = jnp.array(
        [
            [[0, 1], [0, 2], [1, 2]],
            [[0, 1], [1, 2], [0, 2]],
        ],
        dtype=jnp.int32,
    )
    mask = jnp.array(
        [
            [True, True, True],
            [True, True, False],
        ]
    )
    T = np.array(compute_T_from_dag(distances, dag_pairs, mask))
    expected = np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=int,
    )
    assert np.array_equal(T, expected)


def test_compute_T_from_dag_handles_single_row_and_ties():
    distances = jnp.array([[0.2, 0.2, 0.4]], dtype=jnp.float32)
    dag_pairs = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    mask = jnp.array([True, True])
    T = np.array(compute_T_from_dag(distances, dag_pairs, mask))
    assert np.array_equal(T, np.array([1, 1]))


def test_remap_distances_to_columns_matches_manual_reconstruction():
    DY_sorted = jnp.array(
        [
            [[0.2, 2], [0.5, 0], [0.7, 1]],
        ],
        dtype=jnp.float32,
    )
    cols_sorted = jnp.array([[2, 0, 1]], dtype=jnp.int32)
    remapped = np.array(remap_distances_to_columns(DY_sorted, cols_sorted))
    expected = np.array([[0.5, 0.7, 0.2]], dtype=np.float32)
    np.testing.assert_allclose(remapped, expected)


def test_probit_pairwise_probabilities_matches_closed_form():
    Z_rows = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    Z_cols = jnp.array(
        [
            [1.0, 0.0],
            [0.2, 0.0],
            [-0.5, 0.0],
        ],
        dtype=jnp.float32,
    )
    dag_pairs = jnp.array([[[0, 1], [2, 1]]], dtype=jnp.int32)
    mask = jnp.array([[True, True]])
    weights = jnp.array([1.0, 0.5], dtype=jnp.float32)
    sigma = jnp.array([0.1, 0.2], dtype=jnp.float32)

    probs = np.array(
        probit_pairwise_probabilities(Z_rows, Z_cols, dag_pairs, mask, weights, sigma)
    )

    # Manual margins
    noise_var = 4.0 * (1.0**2 * 0.1**4 + 0.5**2 * 0.2**4)
    noise_std = np.sqrt(noise_var)
    V01 = (1.0**2 - 0.2**2) * 1.0  # only first dimension has spread
    margin1 = V01
    V21 = ((-0.5) ** 2 - 0.2**2) * 1.0
    margin2 = V21
    expected = np.array(
        [
            0.5 * (1.0 + math.erf((-margin1 / noise_std) / np.sqrt(2.0))),
            0.5 * (1.0 + math.erf((-margin2 / noise_std) / np.sqrt(2.0))),
        ]
    )
    np.testing.assert_allclose(probs[0], expected, atol=1e-6)


def test_probit_pairwise_probabilities_handles_deterministic_case():
    Z_rows = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    Z_cols = jnp.array([[0.5, 0.0], [0.2, 0.0]], dtype=jnp.float32)
    dag_pairs = jnp.array([[[0, 1]]], dtype=jnp.int32)
    mask = jnp.array([[True]])
    weights = jnp.array([1.0, 0.0], dtype=jnp.float32)
    sigma = jnp.array([0.0, 0.0], dtype=jnp.float32)

    probs = np.array(
        probit_pairwise_probabilities(Z_rows, Z_cols, dag_pairs, mask, weights, sigma)
    )
    # Deterministic: z1 farther than z2 ⇒ probability zero
    assert probs[0, 0] == 0.0


def _toy_pair_data():
    key = jax.random.PRNGKey(0)
    n = 6
    p = 2
    X = jax.random.normal(key, (n, p))
    Z = X + 0.1 * jax.random.normal(key, (n, p))
    rows_idx = jnp.arange(2, dtype=jnp.int32)
    cols_idx = jnp.arange(2, n, dtype=jnp.int32)
    X_rows = X[rows_idx]
    X_cols = X[cols_idx]
    Z_rows = Z[rows_idx]
    Z_cols = Z[cols_idx]
    pair_data = build_pairwise_data(
        X_rows,
        X_cols,
        Z_rows,
        Z_cols,
        num_shells=4,
        quantile_outer=0.5,
    )
    return pair_data, X_rows, X_cols, Z_rows, Z_cols


def test_build_pairwise_data_shapes():
    pair_data, *_ = _toy_pair_data()
    assert pair_data.targets.shape == pair_data.pair_mask.shape
    assert pair_data.dag_pairs.shape[:2] == pair_data.targets.shape
    assert pair_data.margin_features.shape[:-1] == pair_data.targets.shape


def test_build_pairwise_data_deterministic():
    pair_data1, *_ = _toy_pair_data()
    pair_data2, *_ = _toy_pair_data()
    np.testing.assert_array_equal(pair_data1.targets, pair_data2.targets)
    np.testing.assert_array_equal(pair_data1.dag_pairs, pair_data2.dag_pairs)


def test_loglikelihood_matches_manual_probability():
    pair_data, X_rows, X_cols, Z_rows, Z_cols = _toy_pair_data()
    sigma = jnp.array([0.1, 0.2])
    weights = jnp.array([0.5, 1.0])
    llik = make_loglikelihood(pair_data, sigma)
    loglik_value = float(llik(weights))

    P = probit_pairwise_probabilities(
        Z_rows,
        Z_cols,
        pair_data.dag_pairs,
        pair_data.pair_mask,
        weights,
        sigma,
    )
    T = pair_data.targets
    mask = pair_data.pair_mask
    manual = jnp.where(T == 1, jnp.log(P + 1e-12), jnp.log1p(-P + 1e-12))
    manual = jnp.where(mask, manual, 0.0)
    manual_value = float(manual.sum())
    assert abs(loglik_value - manual_value) < 1e-5
