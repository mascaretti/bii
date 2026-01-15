"""
Voronoi thinning utilities (JAX).

Key ideas:
- Build Voronoi cells induced by a set of anchor points (anchors are indices into X).
- "Double-thinning" (iterative pruning): repeatedly remove anchors whose Voronoi cells
  contain fewer than `min_points_per_cell` NON-anchor points, recomputing the tessellation
  after each pruning step, until stable.
- Selection inside each cell:
    * sample_k_points_per_cell: random (uniform within each cell) using RNG.
    * pick_k_nearest_in_each_cell: deterministic; pick the k nearest points to each anchor
      among points assigned to that anchor's Voronoi cell.
"""

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


class VoronoiAssignment(NamedTuple):
    anchors: jnp.ndarray          # (m,) int indices into X
    cell_of_point: jnp.ndarray    # (n,) in {-1, 0, ..., m-1}
    is_anchor: jnp.ndarray        # (n,) bool


def _assign_voronoi_cells_no_check(
    X: jnp.ndarray,
    anchors: jnp.ndarray,
) -> VoronoiAssignment:
    """
    Single Voronoi assignment pass with no validation/pruning.
    Anchors are masked out with cell id -1.
    """
    n = X.shape[0]
    m = anchors.shape[0]

    Xa = X[anchors]  # (m, d)

    diff = X[:, None, :] - Xa[None, :, :]
    dists = jnp.sum(diff**2, axis=-1)  # (n, m)

    cell_of_point = jnp.argmin(dists, axis=1)  # (n,)

    is_anchor = jnp.zeros(n, dtype=bool).at[anchors].set(True)
    cell_of_point = jnp.where(is_anchor, -1, cell_of_point)

    return VoronoiAssignment(
        anchors=anchors,
        cell_of_point=cell_of_point,
        is_anchor=is_anchor,
    )


def _counts_per_cell(cell_of_point: jnp.ndarray, m: int) -> jnp.ndarray:
    """
    Counts NON-anchor points per cell id in [0, m-1].
    """
    non_anchor = cell_of_point >= 0
    return jnp.bincount(cell_of_point[non_anchor], minlength=m)


def assign_voronoi_cells(
    X: jnp.ndarray,
    anchors: jnp.ndarray,
    min_points_per_cell: int = 2,
) -> VoronoiAssignment:
    """
    Iterative pruning ("double-thinning"):

    Repeatedly:
      1) assign points to anchors via Voronoi
      2) count non-anchor points per anchor-cell
      3) drop anchors whose cell count < min_points_per_cell

    Stops when no anchors are dropped (fixed point). This guarantees that in the final
    assignment every remaining cell has at least `min_points_per_cell` non-anchor points.

    If pruning removes all anchors, returns an assignment with m=0 and cell_of_point all -1.

    Notes:
    - This function is intended for one-time, non-jitted pipeline use. It uses host-side
      logic (device_get + Python loop).
    """
    anchors = jnp.asarray(anchors)
    n = X.shape[0]

    while True:
        m = int(anchors.shape[0])
        if m == 0:
            return VoronoiAssignment(
                anchors=anchors,
                cell_of_point=-jnp.ones((n,), dtype=jnp.int32),
                is_anchor=jnp.zeros((n,), dtype=bool),
            )

        vor = _assign_voronoi_cells_no_check(X, anchors)
        counts = _counts_per_cell(vor.cell_of_point, m)

        counts_host = jax.device_get(counts)
        keep_mask_host = counts_host >= min_points_per_cell

        if bool(jnp.all(keep_mask_host)):
            return vor

        anchors = anchors[jnp.asarray(keep_mask_host)]


def sample_k_points_per_cell(
    key: jax.Array,
    cell_of_point: jax.Array,  # (n,), values in {-1, 0, ..., m-1}
    m: int,
    k: int = 2,
) -> jax.Array:
    """
    Samples k distinct non-anchor points per cell uniformly at random.

    Precondition: for every cid in [0, m-1], there are at least k indices i with cell_of_point[i] == cid.
    Returns: indices into X with shape (m, k).
    """
    n = cell_of_point.shape[0]
    keys = jax.random.split(key, m)
    cell_ids = jnp.arange(m)

    def sample_one(key_i, cid):
        p = (cell_of_point == cid).astype(jnp.float32)  # (n,)
        return jax.random.choice(key_i, a=n, shape=(k,), replace=False, p=p)

    return jax.vmap(sample_one)(keys, cell_ids)


def pick_k_nearest_in_each_cell(
    X: jax.Array,               # (n, d)
    anchors: jax.Array,         # (m,) indices into X
    cell_of_point: jax.Array,   # (n,) in {-1, 0, ..., m-1}
    k: int = 2,
) -> jax.Array:
    """
    For each cell id cid, pick the k points in that cell with smallest distance to anchor cid
    (i.e., the k nearest neighbours of the anchor, restricted to its own Voronoi cell).

    Precondition: each cell has at least k non-anchor points.
    Returns: indices into X with shape (m, k).
    """
    m = anchors.shape[0]
    Xa = X[anchors]  # (m, d)

    # Squared distances from every point to every anchor: (n, m)
    diff = X[:, None, :] - Xa[None, :, :]
    d2 = jnp.sum(diff**2, axis=-1)  # (n, m)

    # Keep only points belonging to each cell; everything else -> +inf so it can't be selected.
    mask = (cell_of_point[:, None] == jnp.arange(m)[None, :])  # (n, m)
    masked = jnp.where(mask, d2, jnp.inf)                      # (n, m)

    # Take k smallest per anchor-cell via top_k on negatives.
    _, idx = jax.lax.top_k((-masked).T, k)  # (m, n) -> (m, k)
    return idx


def sample_one_near_one_far_in_each_cell(
    key: jax.Array,
    X: jax.Array,               # (n, d)
    anchors: jax.Array,         # (m,) indices into X
    cell_of_point: jax.Array,   # (n,) in {-1, 0, ..., m-1}
    k_near: int,                # size of "near" band
    r_far: int,                 # consider ranks [0 .. r_far-1]; "far" band is [k_near .. r_far-1]
) -> jax.Array:
    """
    For each Voronoi cell (anchor), sample:
      - 1 point uniformly from the k_near nearest points in that cell
      - 1 point uniformly from the subsequent ranks k_near .. r_far-1 in that cell

    Returns: indices into X with shape (m, 2).

    Preconditions:
      - r_far > k_near >= 1
      - each cell has at least r_far non-anchor points
    """
    m = anchors.shape[0]
    Xa = X[anchors]  # (m, d)

    # Squared distances from every point to every anchor: (n, m)
    diff = X[:, None, :] - Xa[None, :, :]
    d2 = jnp.sum(diff**2, axis=-1)  # (n, m)

    # Mask: keep only points belonging to each anchor's Voronoi cell.
    mask = (cell_of_point[:, None] == jnp.arange(m)[None, :])   # (n, m)
    masked = jnp.where(mask, d2, jnp.inf)                       # (n, m)

    # Get r_far nearest candidates per cell via top_k on negatives.
    # idx_unsorted: (m, r_far) indices into n (points)
    _, idx_unsorted = jax.lax.top_k((-masked).T, r_far)         # (-masked).T is (m, n)

    # Recover their true distances and sort to obtain actual ranks 0..r_far-1.
    cand_d2 = jnp.take_along_axis(masked.T, idx_unsorted, axis=1)   # (m, r_far)
    order = jnp.argsort(cand_d2, axis=1)                             # (m, r_far)
    idx_sorted = jnp.take_along_axis(idx_unsorted, order, axis=1)    # (m, r_far)

    # Sample one from near band [0 .. k_near-1], one from far band [k_near .. r_far-1]
    k1, k2 = jax.random.split(key, 2)
    keys1 = jax.random.split(k1, m)
    keys2 = jax.random.split(k2, m)

    near_pos = jax.vmap(lambda kk: jax.random.randint(kk, (), 0, k_near))(keys1)  # (m,)
    far_pos  = jax.vmap(lambda kk: jax.random.randint(kk, (), 0, r_far - k_near))(keys2)  # (m,)
    far_pos = far_pos + k_near

    rows = jnp.arange(m)
    near_idx = idx_sorted[rows, near_pos]
    far_idx  = idx_sorted[rows, far_pos]

    return jnp.stack([near_idx, far_idx], axis=1)