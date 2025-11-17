# tests/test_radial.py
# Unit tests for radial binning (deterministic, non-flaky)

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from bii.radial import (
    make_distance,
    equal_expected_count_shells_from_Rmax,
    equal_expected_count_shells_via_lambda,
    assign_to_shells_aligned,
    sample_representatives_uniform_aligned,
    select_representatives_first_in_shell,
    select_representatives_by_rank,
    make_shell_dag_pairs,
    make_partition_indices,
    split_XZ_by_partition,
)


def _toy_data(n=20, p=2, seed=0):
    key = random.PRNGKey(seed)
    key, k1, k2, k3 = random.split(key, 4)
    X = random.multivariate_normal(k1, jnp.zeros(p), jnp.eye(p), shape=(n,))
    w = jnp.array([0.4, 1.3] + [1.0] * (p - 2))[:p]
    eps = 0.02 * random.multivariate_normal(k2, jnp.zeros(p), jnp.eye(p), shape=(n,))
    Y_cols = (X * w) + eps
    y = 0.02 * random.multivariate_normal(k3, jnp.zeros(p), jnp.eye(p))
    Y_rows = y[None, :]
    return Y_rows, Y_cols, key


def test_equal_expected_count_shells_from_Rmax_monotone_and_shape():
    S = 4
    Rmax = 1.7
    radii = equal_expected_count_shells_from_Rmax(Rmax, S, dim=2)
    assert radii.shape == (S,)
    assert jnp.all(radii[1:] > radii[:-1])
    assert radii[-1] <= Rmax + 1e-6


def test_equal_expected_count_shells_via_lambda_increasing_and_volume_steps():
    S = 5
    lam = 3.0
    target = 2.0
    d = 2
    r = equal_expected_count_shells_via_lambda(lam, target, S, dim=d, r0=0.0)
    assert r.shape == (S,)
    assert jnp.all(r[1:] > r[:-1])
    # r^d increments are (approximately) constant
    rd = r**d
    diffs = jnp.diff(rd)
    assert float(diffs.max() - diffs.min()) < 1e-6


def test_make_distance_sorted_and_indices():
    Y_rows, Y_cols, _ = _toy_data(n=7, p=2, seed=1)
    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    d_sorted = np.array(DY_sorted[0, :, 0])
    # distances are nondecreasing
    assert np.all(d_sorted[1:] >= d_sorted[:-1])
    # cols_sorted matches the integer indices stored in DY_sorted[..., 1]
    c_sorted = np.array(cols_sorted[0])
    c_from_DY = np.array(DY_sorted[0, :, 1])
    assert np.array_equal(c_sorted, c_from_DY)


def test_assign_labels_aligned_to_original_columns():
    Y_rows, Y_cols, _ = _toy_data(n=25, p=2, seed=2)
    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    d_sorted = DY_sorted[0, :, 0]
    Rmax = jnp.quantile(d_sorted, 0.9)
    S = 4
    radii = equal_expected_count_shells_from_Rmax(float(Rmax), S, dim=2)
    shell_sorted, shell_by_col = assign_to_shells_aligned(DY_sorted, cols_sorted, radii)

    # Reconstruct distances aligned to original columns and verify shell membership
    c_sorted = np.array(cols_sorted[0], dtype=int)
    d_sorted_np = np.array(d_sorted)
    d_by_col = np.empty_like(d_sorted_np)
    d_by_col[c_sorted] = d_sorted_np
    shells = np.array(shell_by_col[0], dtype=int)

    # Check label validity and interval semantics
    for j, dist in enumerate(d_by_col):
        s = shells[j]
        assert s >= -1 and s < S
        if s == -1:
            assert dist > float(radii[-1]) - 1e-10
        else:
            inner = 0.0 if s == 0 else float(radii[s - 1])
            outer = float(radii[s])
            assert (dist > inner - 1e-10) and (dist <= outer + 1e-10)


def test_sample_representatives_in_correct_shell():
    Y_rows, Y_cols, key = _toy_data(n=40, p=2, seed=3)
    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    d_sorted = DY_sorted[0, :, 0]
    Rmax = jnp.quantile(d_sorted, 0.9)
    S = 5
    radii = equal_expected_count_shells_from_Rmax(float(Rmax), S, dim=2)

    shell_sorted, shell_by_col = assign_to_shells_aligned(DY_sorted, cols_sorted, radii)
    key, sub = random.split(key)
    rep_cols, rep_dists = sample_representatives_uniform_aligned(DY_sorted, shell_sorted, radii, sub)

    # Distances by original column for validation
    c_sorted = np.array(cols_sorted[0], dtype=int)
    d_sorted_np = np.array(d_sorted)
    d_by_col = np.empty_like(d_sorted_np)
    d_by_col[c_sorted] = d_sorted_np

    reps = np.array(rep_cols[0], dtype=int)
    repd = np.array(rep_dists[0])
    for s in range(S):
        j = reps[s]
        if j >= 0:
            dist = d_by_col[j]
            inner = 0.0 if s == 0 else float(radii[s - 1])
            outer = float(radii[s])
            assert (dist > inner - 1e-10) and (dist <= outer + 1e-10)
            assert math.isfinite(repd[s])
            assert abs(repd[s] - dist) < 1e-6
        else:
            # empty shell ⇒ NaN distance
            assert np.isnan(repd[s])


def test_select_representatives_first_in_shell_deterministic():
    Y_rows, Y_cols, key = _toy_data(n=30, p=2, seed=4)
    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    d_sorted = DY_sorted[0, :, 0]
    Rmax = jnp.quantile(d_sorted, 0.8)
    S = 4
    radii = equal_expected_count_shells_from_Rmax(float(Rmax), S, dim=2)
    shell_sorted, _ = assign_to_shells_aligned(DY_sorted, cols_sorted, radii)
    rep_cols, _ = select_representatives_first_in_shell(DY_sorted, shell_sorted, S)

    # Ensure determinism across repeated calls
    rep_cols2, _ = select_representatives_first_in_shell(DY_sorted, shell_sorted, S)
    np.testing.assert_array_equal(rep_cols, rep_cols2)

    # Representatives should be valid members of their shells
    rep_cols_np = np.array(rep_cols[0], dtype=int)
    c_sorted = np.array(cols_sorted[0], dtype=int)
    d_sorted_np = np.array(d_sorted)
    d_by_col = np.empty_like(d_sorted_np)
    d_by_col[c_sorted] = d_sorted_np
    for s, col in enumerate(rep_cols_np):
        if col < 0:
            continue
        shell_label = np.array(shell_sorted[0])[np.where(c_sorted == col)][0]
        assert shell_label == s


def test_select_representatives_by_rank_picks_nearest():
    Y_rows, Y_cols, _ = _toy_data(n=15, p=2, seed=5)
    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    reps_cols, reps_dists = select_representatives_by_rank(DY_sorted, num_shells=5)

    # Shell k uses k-th nearest neighbour
    col_sorted_np = np.array(cols_sorted[0], dtype=int)
    d_sorted_np = np.array(DY_sorted[0, :, 0])
    for k in range(5):
        col = reps_cols[0, k]
        dist = reps_dists[0, k]
        assert col == col_sorted_np[k] or (k >= len(col_sorted_np) and col == -1)
        if col >= 0:
            assert abs(dist - d_sorted_np[k]) < 1e-6


def test_make_shell_dag_pairs_full_shells():
    reps = jnp.array([[3, 10, 5, 12]], dtype=jnp.int32)
    dag_pairs, mask, counts = make_shell_dag_pairs(reps)

    expected = np.array(
        [
            [3, 10],
            [3, 5],
            [3, 12],
            [5, 10],
            [10, 12],
            [5, 12],
        ],
        dtype=np.int32,
    )
    assert dag_pairs.shape == (1, expected.shape[0], 2)
    assert np.array_equal(dag_pairs[0], expected)
    assert np.all(mask)
    assert int(counts[0]) == expected.shape[0]


def test_make_shell_dag_pairs_handles_empty_shells_and_batch():
    reps = jnp.array(
        [
            [2, -1, 7, 11],
            [-1, -1, -1, -1],
            [9, 4, 1, 8],
        ],
        dtype=jnp.int32,
    )
    dag_pairs, mask, counts = make_shell_dag_pairs(reps)

    # Row 0 should only keep pairs involving shells 0, 2, and 3
    expected_row0 = np.array(
        [
            [-1, -1],
            [2, 7],
            [2, 11],
            [-1, -1],
            [-1, -1],
            [7, 11],
        ]
    )
    assert np.array_equal(dag_pairs[0], expected_row0)
    assert np.array_equal(mask[0], np.array([False, True, True, False, False, True]))
    assert int(counts[0]) == 3

    # Row 1 has no data
    assert np.all(dag_pairs[1] == -1)
    assert not np.any(mask[1])
    assert int(counts[1]) == 0

    # Row 2 should reorder columns so edges always go low -> high
    expected_row2 = np.array(
        [
            [4, 9],
            [1, 9],
            [8, 9],
            [1, 4],
            [4, 8],
            [1, 8],
        ]
    )
    assert np.array_equal(dag_pairs[2], expected_row2)
    assert np.all(mask[2])
    assert int(counts[2]) == 6


def test_make_shell_dag_pairs_accepts_1d_input():
    reps = jnp.array([7, -1, 3], dtype=jnp.int32)
    dag_pairs, mask, count = make_shell_dag_pairs(reps)

    expected_pairs = np.array(
        [
            [-1, -1],
            [3, 7],
            [-1, -1],
        ]
    )
    assert np.array_equal(dag_pairs, expected_pairs)
    assert np.array_equal(mask, np.array([False, True, False]))
    assert int(count) == 1


def test_partition_disjoint_and_cover():
    n = 31
    p = 2
    # fake data
    X = jnp.arange(n * p, dtype=jnp.float32).reshape(n, p)
    Z = X * 2.0

    key = random.PRNGKey(123)
    rows_idx, cols_idx = make_partition_indices(key, n, size_rows=12, shuffle=True)
    # disjointness
    rset = set(map(int, np.array(rows_idx)))
    cset = set(map(int, np.array(cols_idx)))
    assert rset.isdisjoint(cset)
    # coverage
    full = rset | cset
    assert full == set(range(n))

    Xr, Xc, Zr, Zc = split_XZ_by_partition(X, Z, rows_idx, cols_idx)
    assert Xr.shape[0] == len(rset)
    assert Xc.shape[0] == len(cset)
    assert Zr.shape[0] == len(rset)
    assert Zc.shape[0] == len(cset)
