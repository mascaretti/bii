# tests/properties/test_radial_props.py
# Property-based tests for radial binning (Hypothesis)

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pytest
from hypothesis import given, settings, strategies as st

from bii.radial import (
    make_distance,
    equal_expected_count_shells_from_Rmax,
    assign_to_shells_aligned,
    make_partition_indices,
)

# Keep runs moderate to avoid flakiness/timeouts in CI
DEFAULT_SETTINGS = settings(deadline=None, max_examples=60)


def _rand_points(key, n, p, scale=1.0):
    X = random.normal(key, (n, p)) * scale
    return X


@DEFAULT_SETTINGS
@given(
    n=st.integers(min_value=8, max_value=60),
    scale=st.floats(min_value=0.2, max_value=3.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_scale_invariance_of_assignment(n, scale, seed):
    # Distances and radii should scale together → shell labels unchanged
    p = 2
    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    Y_cols = _rand_points(k1, n, p, scale=1.0)
    y = random.normal(k2, (p,))
    Y_rows = y[None, :]

    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    d_sorted = DY_sorted[0, :, 0]
    Rmax = jnp.quantile(d_sorted, 0.85)
    S = 4
    radii = equal_expected_count_shells_from_Rmax(float(Rmax), S, dim=p)
    _, shell_by_col = assign_to_shells_aligned(DY_sorted, cols_sorted, radii)

    # Scale coordinates by 'scale' and radii by same factor
    Y_cols_s = Y_cols * scale
    Y_rows_s = Y_rows * scale
    DY_sorted_s, cols_sorted_s = make_distance(Y_rows_s, Y_cols_s)
    _, shell_by_col_s = assign_to_shells_aligned(
        DY_sorted_s, cols_sorted_s, radii * scale
    )

    np.testing.assert_array_equal(
        np.array(shell_by_col), np.array(shell_by_col_s)
    )


@DEFAULT_SETTINGS
@given(
    n=st.integers(min_value=10, max_value=50),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_labels_are_valid_and_remap_consistent(n, seed):
    p = 2
    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    Y_cols = _rand_points(k1, n, p, scale=1.0)
    y = random.normal(k2, (p,))
    Y_rows = y[None, :]

    DY_sorted, cols_sorted = make_distance(Y_rows, Y_cols)
    d_sorted = DY_sorted[0, :, 0]
    Rmax = jnp.quantile(d_sorted, 0.9)
    S = 5
    radii = equal_expected_count_shells_from_Rmax(float(Rmax), S, dim=p)
    shell_sorted, shell_by_col = assign_to_shells_aligned(DY_sorted, cols_sorted, radii)

    # Valid label ranges
    s_sorted = np.array(shell_sorted[0])
    s_bycol = np.array(shell_by_col[0])
    assert s_sorted.min() >= -1 and s_sorted.max() <= S - 1
    assert s_bycol.min() >= -1 and s_bycol.max() <= S - 1

    # Remapping correctness: shell_by_col is a permutation placement of shell_sorted
    c_sorted = np.array(cols_sorted[0], dtype=int)
    s_back = np.empty_like(s_sorted)
    s_back[c_sorted] = s_sorted
    np.testing.assert_array_equal(s_back, s_bycol)


@DEFAULT_SETTINGS
@given(
    n=st.integers(min_value=12, max_value=60),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_partition_disjoint_cover_property(n, seed):
    key = random.PRNGKey(seed)
    rows_idx, cols_idx = make_partition_indices(key, n, ratio_rows=0.6, shuffle=True)
    rset = set(map(int, np.array(rows_idx)))
    cset = set(map(int, np.array(cols_idx)))
    # disjoint and covering
    assert rset.isdisjoint(cset)
    assert rset | cset == set(range(n))

