"""Triplet formation from paired observation pools — pure functions."""

import jax
from jax import numpy as jnp
from jax import random


@jax.jit
def T_from_X(X):
    """Binary labels from clean-space triplets.

    Convention: ``[:, 0]`` is the anchor.
    ``T = 1`` when column 1 is closer to the anchor than column 2.

    Args:
        X: (n, 3, p) clean embeddings.

    Returns:
        (n,) binary labels in {0, 1}.
    """
    xi, xj, xk = X[:, 1], X[:, 2], X[:, 0]
    di = jnp.sum((xi - xk) ** 2, axis=1)
    dj = jnp.sum((xj - xk) ** 2, axis=1)
    return (di <= dj).astype(jnp.float32)


def make_triplets(key, X_pool, Z_pool, n_triplets, anchor_fraction=0.1):
    """Form triplets from paired observation pools.

    Partitions the pool into anchors and destinations, then draws
    ``n_triplets`` random destination pairs per anchor.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) clean reference embeddings.
        Z_pool: (N, p_z) noisy/normalised embeddings.
        n_triplets: destination pairs per anchor.
        anchor_fraction: fraction of pool used as anchors.

    Returns:
        ``(T, X, Z, indices)``
            T: (n_total,) binary labels.
            X: (n_total, 3, p_x) clean triplets.
            Z: (n_total, 3, p_z) noisy triplets.
            indices: (n_total, 3) pool indices.
    """
    N = X_pool.shape[0]
    n_anchors = max(1, int(N * anchor_fraction))

    key, k_split, k_pairs = random.split(key, 3)

    perm = random.permutation(k_split, N)
    anchor_idx = perm[:n_anchors]
    dest_idx = perm[n_anchors:]
    n_dest = dest_idx.shape[0]

    total = n_anchors * n_triplets
    pair_keys = random.split(k_pairs, total)

    def _sample_pair(key):
        return random.choice(key, n_dest, shape=(2,), replace=False)

    pair_positions = jax.vmap(_sample_pair)(pair_keys)

    anchor_repeated = jnp.repeat(anchor_idx, n_triplets)
    indices = jnp.stack(
        [
            anchor_repeated,
            dest_idx[pair_positions[:, 0]],
            dest_idx[pair_positions[:, 1]],
        ],
        axis=1,
    )

    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices


def make_triplets_zfar(key, X_pool, Z_pool, sig, n_triplets, anchor_fraction=0.1,
                       rank_i=100, rank_j=200):
    """Form triplets with (i, j) at fixed Z-distance ranks from each anchor.

    For each anchor k, sort destinations by Mahalanobis Z-distance
        ``d2(l, k) = sum_d (z_{l,d} - z_{k,d})^2 / sigma_d^2``
    ascending, then pick (i, j) at sliding ranks ``(rank_i + t, rank_j + t)`` for
    ``t = 0..n_triplets - 1``. Labels are still computed from X-distance (Y).

    Motivation: under flat ``sigma`` on heavy-tailed data (e.g. NHANES nutrients
    with zero-inflated alcohol), random pairs occasionally place a single feature's
    rare-tail outlier in one of (i, j) but not the other. The resulting
    ``|a_d^2 - b_d^2|`` term blows the per-triplet probit statistic ``s`` past the
    sigmoid's saturating regime, and a single saturating-wrong triplet contributes
    ~``log Phi(-5) ≈ -15`` to the loglik. Z-rank-100 destinations are by
    construction close to the anchor in Z — they cannot be "anchor-and-one-pair-
    member-non-outlier vs other-pair-member-outlier" — so the outlier pairs are
    excluded from the triplet set.

    For ``sigma`` flat across features, ``d2`` reduces to plain Euclidean Z-distance
    (up to a global scale). For ``sigma`` proportional to per-feature std, each
    feature contributes equally to the ranking.

    Args:
        key: JAX random key (used to sample anchors).
        X_pool: (N, p_x) reference embeddings (used for labels via X-distance).
        Z_pool: (N, p_z) noisy/normalised embeddings (used for ranking AND loglik).
        sig: per-feature noise std — scalar or (p_z,). Same convention as in
            ``fit_bii``; only the ratio ``1 / sigma_d^2`` matters for the ranking.
        n_triplets: triplets per anchor (sliding window length).
        anchor_fraction: fraction of pool used as anchors.
        rank_i: 1-based starting Z-rank for candidate ``i`` (e.g. 100 = 100th NN).
        rank_j: 1-based starting Z-rank for candidate ``j`` (e.g. 200 = 200th NN).

    Returns:
        ``(T, X, Z, indices)`` — same shapes / semantics as :func:`make_triplets`.

    Raises:
        ValueError: if ``rank_i``/``rank_j`` violate ``1 <= rank_i < rank_j`` or
            ``rank_j + n_triplets > N``.
    """
    if not (1 <= rank_i < rank_j):
        raise ValueError(
            f"need 1 <= rank_i < rank_j; got rank_i={rank_i}, rank_j={rank_j}"
        )
    N = Z_pool.shape[0]
    if rank_j + n_triplets > N:
        raise ValueError(
            f"rank_j + n_triplets = {rank_j + n_triplets} exceeds N = {N}"
        )

    sig2 = jnp.asarray(sig).astype(Z_pool.dtype) ** 2
    n_anchors = max(1, int(N * anchor_fraction))

    key, k_anchors = random.split(key)
    perm = random.permutation(k_anchors, N)
    anchor_idx = perm[:n_anchors]

    offsets = jnp.arange(n_triplets)
    i_positions = (rank_i - 1) + offsets
    j_positions = (rank_j - 1) + offsets

    def one_anchor(k_idx):
        z_k = Z_pool[k_idx]
        diff = Z_pool - z_k[None, :]
        d2 = jnp.sum((diff * diff) / sig2[None, :], axis=1)
        d2 = d2.at[k_idx].set(jnp.inf)
        order = jnp.argsort(d2)
        i_idx = order[i_positions]
        j_idx = order[j_positions]
        k_rep = jnp.full((n_triplets,), k_idx)
        return jnp.stack([k_rep, i_idx, j_idx], axis=1)

    indices = jax.vmap(one_anchor)(anchor_idx)
    indices = indices.reshape(-1, 3)

    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices


def make_triplets_yfar(key, X_pool, Z_pool, sig, n_triplets, anchor_fraction=0.1,
                       rank_i=200, rank_j=500, balance_labels=True):
    """Form triplets with (i, j) at fixed Y-distance ranks from each anchor.

    Counterpart to :func:`make_triplets_zfar` but ranks by Euclidean X-distance
    (target/Y space) instead of Mahalanobis Z-distance. For each anchor k, sort
    pool by ``d2_X(l, k) = sum_d (x_{l,d} - x_{k,d})^2`` ascending, then pick
    ``(i, j)`` at sliding ranks ``(rank_i + t, rank_j + t)``. Labels still come
    from ``T_from_X`` (X-distance).

    Because the ranking AND labelling spaces coincide, ``T`` would be constant
    (= 1, since ``rank_i < rank_j`` ⟹ ``d_X(i, k) ≤ d_X(j, k)``). When
    ``balance_labels`` is True, columns 1 and 2 are independently swapped per
    triplet with p=0.5, giving a 50/50 label distribution.

    Symmetry note: the swap is a no-op for the per-triplet loglik (swapping
    ``(i, j)`` flips both ``T`` and ``s = (d_i^2 - d_j^2)/sqrt(V)``, leaving
    ``log Phi(±s)`` unchanged). NUTS posterior, ESS, R-hat, WAIC and
    triplet_accuracy are therefore identical with or without ``balance_labels``;
    the option only affects diagnostics that condition on label balance.

    Args:
        key: JAX random key (anchor permutation + optional column swap).
        X_pool: (N, p_x) reference embeddings (ranking AND labels).
        Z_pool: (N, p_z) noisy/normalised embeddings (loglik only).
        sig: kept for interface compatibility with ``fit_bii``'s triplet_sampler
            protocol; unused here (Y-side ranking is plain Euclidean).
        n_triplets: triplets per anchor (sliding window length).
        anchor_fraction: fraction of pool used as anchors.
        rank_i: 1-based starting Y-rank for candidate 1.
        rank_j: 1-based starting Y-rank for candidate 2.
        balance_labels: if True, swap columns 1 and 2 per triplet w.p. 0.5.

    Returns:
        ``(T, X, Z, indices)`` — same shapes / semantics as :func:`make_triplets`.

    Raises:
        ValueError: if ``rank_i``/``rank_j`` violate ``1 <= rank_i < rank_j`` or
            ``rank_j + n_triplets > N``.
    """
    del sig  # unused; kept for triplet_sampler interface compatibility

    if not (1 <= rank_i < rank_j):
        raise ValueError(
            f"need 1 <= rank_i < rank_j; got rank_i={rank_i}, rank_j={rank_j}"
        )
    N = X_pool.shape[0]
    if rank_j + n_triplets > N:
        raise ValueError(
            f"rank_j + n_triplets = {rank_j + n_triplets} exceeds N = {N}"
        )

    n_anchors = max(1, int(N * anchor_fraction))

    key, k_anchors, k_swap = random.split(key, 3)
    perm = random.permutation(k_anchors, N)
    anchor_idx = perm[:n_anchors]

    offsets = jnp.arange(n_triplets)
    i_positions = (rank_i - 1) + offsets
    j_positions = (rank_j - 1) + offsets

    def one_anchor(k_idx):
        x_k = X_pool[k_idx]
        diff = X_pool - x_k[None, :]
        d2 = jnp.sum(diff * diff, axis=1)
        d2 = d2.at[k_idx].set(jnp.inf)
        order = jnp.argsort(d2)
        i_idx = order[i_positions]
        j_idx = order[j_positions]
        k_rep = jnp.full((n_triplets,), k_idx)
        return jnp.stack([k_rep, i_idx, j_idx], axis=1)

    indices = jax.vmap(one_anchor)(anchor_idx)
    indices = indices.reshape(-1, 3)

    if balance_labels:
        swaps = random.bernoulli(k_swap, p=0.5, shape=(indices.shape[0],))
        col1 = jnp.where(swaps, indices[:, 2], indices[:, 1])
        col2 = jnp.where(swaps, indices[:, 1], indices[:, 2])
        indices = jnp.stack([indices[:, 0], col1, col2], axis=1)

    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices


def make_triplets_rank_weighted(
    key, X_pool, Z_pool, sig, n_triplets, anchor_fraction=0.1, *,
    k_max=None, target_logweight_fn=None,
):
    """Form triplets at Y-rank pairs drawn uniformly, with importance weights.

    Designed to approximate DII's rank-integrated objective via the
    decoupling ``q (broad sampling) + p_target (targeted focus)``. For each
    anchor k:

      1. Sort pool by Euclidean X-distance from k → Y-rank order.
      2. Independently draw ``r_a, r_b ~ Uniform([1, k_max])``.
      3. Pick the points at those Y-ranks.
      4. Compute importance weight ``alpha_t = p_target(r_a, r_b)`` (up to a
         self-normalising constant, since ``q`` is uniform so its density
         cancels).

    Labels come out balanced by symmetry: ``r_a`` and ``r_b`` are drawn from
    the same distribution, so ``T = 1{r_a < r_b}`` is 50/50 in expectation.

    Why uniform ``q``: empirically the informative pairs span a wide Y-rank
    range (see ``first_triplet_y_ranks.npy``: Z-rank-(10, 25) candidates have
    Y-ranks distributed nearly uniformly across ``[1, N]``). A narrow
    Geometric ``q`` truncates the support and biases the importance
    estimator. Uniform ``q`` over the full pool keeps support everywhere,
    letting any ``p_target`` reweight without producing zeros to divide.

    Args:
        key: JAX random key (anchor permutation + per-triplet rank draws).
        X_pool: (N, p_x) reference embeddings (Y-rank ordering AND labels).
        Z_pool: (N, p_z) noisy/normalised embeddings (loglik only).
        sig: kept for ``fit_bii``'s ``triplet_sampler`` interface; unused.
        n_triplets: triplets per anchor.
        anchor_fraction: fraction of pool used as anchors.
        k_max: rank truncation; defaults to ``N - 1`` (full pool minus the
            anchor). Set lower if you want to cap the range explicitly.
        target_logweight_fn: optional callable ``(r_a, r_b) -> log p_target``
            (up to a constant). When None, all weights are 1 (uniform
            sampling, unfocused). See :func:`target_yfar_bump` for a ready
            Gaussian-bump target.

    Returns:
        ``(T, X, Z, indices, weights)`` — 5-tuple. ``weights`` is self-
        normalised so ``sum(weights) == n_anchors * n_triplets``.

    Raises:
        ValueError: ``k_max`` outside ``[1, N - 1]``.
    """
    del sig  # unused; kept for triplet_sampler interface compatibility

    N = X_pool.shape[0]
    if k_max is None:
        k_max = N - 1
    if not (1 <= k_max <= N - 1):
        raise ValueError(f"k_max={k_max} must satisfy 1 <= k_max <= N-1 (N={N})")

    n_anchors = max(1, int(N * anchor_fraction))
    total = n_anchors * n_triplets

    key, k_anchors, k_ra, k_rb = random.split(key, 4)
    perm = random.permutation(k_anchors, N)
    anchor_idx = perm[:n_anchors]

    # Uniform draws over Y-rank space [1, k_max].
    r_a_all = random.randint(k_ra, (total,), minval=1, maxval=k_max + 1)
    r_b_all = random.randint(k_rb, (total,), minval=1, maxval=k_max + 1)

    if target_logweight_fn is None:
        weights = jnp.ones(total)
    else:
        from jax.scipy.special import logsumexp
        # q is uniform on [1, k_max]^2 so log_q is a constant; it drops out
        # of the self-normalisation. log_w == log_p up to that constant.
        log_p = jax.vmap(target_logweight_fn)(r_a_all, r_b_all)
        log_w = log_p - logsumexp(log_p) + jnp.log(total)
        weights = jnp.exp(log_w)

    r_a_anchor = r_a_all.reshape(n_anchors, n_triplets)
    r_b_anchor = r_b_all.reshape(n_anchors, n_triplets)
    anchor_repeated = jnp.repeat(anchor_idx, n_triplets)

    def one_anchor(k_idx, ra_arr, rb_arr):
        x_k = X_pool[k_idx]
        diff = X_pool - x_k[None, :]
        d2 = jnp.sum(diff * diff, axis=1)
        d2 = d2.at[k_idx].set(jnp.inf)
        order = jnp.argsort(d2)
        return order[ra_arr - 1], order[rb_arr - 1]

    i_per, j_per = jax.vmap(one_anchor)(anchor_idx, r_a_anchor, r_b_anchor)

    indices = jnp.stack(
        [anchor_repeated, i_per.reshape(-1), j_per.reshape(-1)],
        axis=1,
    )

    X = X_pool[indices]
    Z = Z_pool[indices]
    T = T_from_X(X)
    return T, X, Z, indices, weights


def make_triplets_random_sparse(
    key, X_pool, Z_pool, sig, n_triplets, anchor_fraction=0.1, *,
    eps=0.025, reference_w=None, oversample=10,
):
    """Random triplets, sparsified by removing saturating-`|s|` candidates.

    Draws ``n_triplets * oversample`` random pairs per anchor via
    :func:`make_triplets`, computes the per-triplet probit statistic
    ``s = delta / sqrt(V)`` under ``reference_w``, then keeps only triplets
    with ``|s| < z_{1-eps}`` (equivalently ``P in [eps, 1 - eps]``).

    Rationale: under heavy-tailed features on NHANES-like data, ~67% of
    random pairs land in the saturating regime (`|s| > 2` under uniform `w`),
    where the loglik is dominated by extreme contributions and the only
    escape is to zero the heavy-tail feature's weight (the
    "magnesium-collapse" failure mode). Filtering on a reference weight
    removes those triplets explicitly instead of capping them via
    ``clip_s``. Survivors live inside the calibrated probit regime
    regardless of NUTS exploration, *as long as `w` doesn't move too far
    from ``reference_w``*.

    Args:
        key: JAX random key.
        X_pool: (N, p_x) reference embeddings (labels via X-distance).
        Z_pool: (N, p_z) noisy/normalised embeddings (loglik).
        sig: per-feature noise std — scalar or (p_z,).
        n_triplets: target triplets per anchor (after rejection).
            With ``oversample=10`` and a survival rate of ~33% under uniform
            on NHANES-like data, the kept count comes out near
            ``n_triplets * n_anchors * 3.3 / 10``. Set ``oversample`` higher
            if you want to guarantee at least ``n_triplets`` survivors.
        anchor_fraction: fraction of pool used as anchors.
        eps: tail mass to exclude on each side. ``eps=0.025`` excludes the
            top 5% of triplets by ``|s|`` (corresponding to ``P < 0.025``
            or ``P > 0.975``).
        reference_w: (p_z,) weight vector used to evaluate ``s``. Defaults
            to uniform ``1/p``. Use the previous fit's posterior mean for
            an adaptive filter, or a fixed reference like DII weights.
        oversample: multiplier on the pre-rejection triplet count. Tune to
            ensure enough survivors at the chosen ``eps``.

    Returns:
        ``(T, X, Z, indices)`` — 4-tuple. Variable size depending on how
        many candidates survived the rejection; the count is printed.
    """
    # Local import to avoid a hard dependency from data.py to inference.py
    # at module load time.
    from bii.inference import delta_V_one_triplet
    from scipy.stats import norm

    p = Z_pool.shape[1]
    if reference_w is None:
        reference_w = jnp.full(p, 1.0 / p, dtype=jnp.float32)
    else:
        reference_w = jnp.asarray(reference_w, dtype=jnp.float32)

    threshold = float(norm.ppf(1.0 - eps))

    # Oversample candidate triplets.
    T_c, X_c, Z_c, idx_c = make_triplets(
        key, X_pool, Z_pool, n_triplets * oversample, anchor_fraction,
    )

    # Per-triplet |s| under reference_w
    sig2 = jnp.asarray(sig).astype(Z_c.dtype) ** 2
    if sig2.ndim == 0:
        sig2 = jnp.full(p, sig2, dtype=Z_c.dtype)
    zi = Z_c[:, 1]; zj = Z_c[:, 2]; zk = Z_c[:, 0]

    def one(zi_, zj_, zk_):
        return delta_V_one_triplet(zi_, zj_, zk_, reference_w, sig2, sig2, sig2)

    delta, V = jax.vmap(one)(zi, zj, zk)
    s = delta / jnp.sqrt(V + 1e-12)
    keep = jnp.abs(s) < threshold

    # Variable-size output — JAX boolean indexing returns dynamic shape.
    T_out = T_c[keep]
    X_out = X_c[keep]
    Z_out = Z_c[keep]
    idx_out = idx_c[keep]

    n_kept = int(keep.sum())
    n_cand = int(keep.size)
    print(f"  random_sparse: kept {n_kept}/{n_cand} triplets "
          f"({100 * n_kept / n_cand:.1f}%) at eps={eps}, |s|<{threshold:.2f}",
          flush=True)

    return T_out, X_out, Z_out, idx_out


def target_yfar_bump(mu_a=200.0, mu_b=500.0, sigma=100.0):
    """Gaussian bump in rank-pair space, suitable for ``target_logweight_fn``.

    Returns a callable ``(r_a, r_b) -> log p_target`` proportional to
    ``exp(- ((r_a - mu_a)**2 + (r_b - mu_b)**2) / (2 * sigma**2))``.

    Default centre ``(200, 500)`` matches the Y-far experiment's ranks; a
    wider ``sigma`` makes the target broader / closer to uniform.
    """
    inv_two_sigma2 = 1.0 / (2.0 * sigma ** 2)

    def fn(r_a, r_b):
        return -((r_a - mu_a) ** 2 + (r_b - mu_b) ** 2) * inv_two_sigma2

    return fn
