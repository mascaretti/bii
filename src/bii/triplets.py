import jax
from jax import numpy as jnp
from jax.scipy.special import log_ndtr, ndtr

def delta_V_one_triplet(zi, zj, zk, w, sig2):
    a = zi - zk
    b = zj - zk
    delta = jnp.sum(w * (a*a - b*b))
    w2_sig2 = (w*w) * sig2
    aa = jnp.sum(w2_sig2 * (a*a))
    bb = jnp.sum(w2_sig2 * (b*b))
    ab = jnp.sum(w2_sig2 * (a*b))
    tr = jnp.sum((w*w) * (sig2*sig2))
    V = 8.0 * (aa + bb - ab) + 12.0 * tr
    return delta, V

def logP_log1mP_from_deltaV(delta, V):
    s = delta / jnp.sqrt(V + 1e-12)
    logP = log_ndtr(-s)   # log Phi(-s)
    log1mP = log_ndtr(s)  # log (1 - Phi(-s))
    return logP, log1mP

def T_from_X(X, I, J, K):
    xi, xj, xk = X[I], X[J], X[K]
    di = jnp.sum((xi - xk)**2, axis=1)
    dj = jnp.sum((xj - xk)**2, axis=1)
    return (di <= dj).astype(jnp.float32)

def make_triplets_once(key, Z, sig2, frac_anchors=0.5):
    n, p = Z.shape
    m = int(jnp.floor(frac_anchors * n))
    perm = jax.random.permutation(key, n)
    anchors = perm[:m]
    dest = perm[m:]
    Zdest = Z[dest]

    def one_anchor(k):
        zk = Z[k]
        diff = Zdest - zk[None, :]
        d = jnp.sum((diff*diff) / sig2[None, :], axis=1)
        order = jnp.argsort(d)
        i = dest[order[0]]
        j = dest[order[1]]
        return i, j, k

    I, J, K = jax.vmap(one_anchor)(anchors)
    return I, J, K

def loglik_theta(theta, X, Z, I, J, K, T, sig2):
    w = jax.nn.softmax(theta)  # simplex
    zi, zj, zk = Z[I], Z[J], Z[K]

    # vectorized delta/V over triplets
    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)
    delta, V = jax.vmap(dv)(zi, zj, zk)

    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return jnp.sum(T * logP + (1.0 - T) * log1mP)

def delta_V_from_AB(A, B, w, sig2):
    # A,B: (m,p)
    delta = jnp.sum(w[None,:] * (A*A - B*B), axis=1)
    w2_sig2 = (w*w) * sig2
    aa = jnp.sum(w2_sig2[None,:] * (A*A), axis=1)
    bb = jnp.sum(w2_sig2[None,:] * (B*B), axis=1)
    # iid idealization: drop cross-term or keep it as sample A*B term if you want
    ab = jnp.sum(w2_sig2[None,:] * (A*B), axis=1)
    tr = jnp.sum((w*w) * (sig2*sig2))
    V = 8.0 * (aa + bb - ab) + 12.0 * tr
    return delta, V

def loglik_theta_iid(theta, A, B, T, sig2):
    w = jax.nn.softmax(theta)
    delta, V = delta_V_from_AB(A, B, w, sig2)
    s = delta / jnp.sqrt(V + 1e-12)
    logP = log_ndtr(-s)
    log1mP = log_ndtr(s)
    return jnp.sum(T*logP + (1-T)*log1mP)

def simulate_iid_triplets(key, m, w_star, tau=1.0, sigma=0.15):
    p = w_star.shape[0]
    sig2 = jnp.full((p,), sigma**2)

    # sample iid X points for each role separately (fresh triplet each obs)
    key_xi, key_xj, key_xk, key_eta = random.split(key, 4)
    Xi = random.normal(key_xi, (m,p)) * tau
    Xj = random.normal(key_xj, (m,p)) * tau
    Xk = random.normal(key_xk, (m,p)) * tau

    Yi = Xi / jnp.sqrt(w_star)[None,:]
    Yj = Xj / jnp.sqrt(w_star)[None,:]
    Yk = Xk / jnp.sqrt(w_star)[None,:]

    # independent noise for each point
    eta_i = random.normal(key_eta, (m,p)) * sigma
    key_eta, sub = random.split(key_eta); eta_j = random.normal(sub, (m,p)) * sigma
    key_eta, sub = random.split(key_eta); eta_k = random.normal(sub, (m,p)) * sigma

    Zi = Yi + eta_i
    Zj = Yj + eta_j
    Zk = Yk + eta_k

    # differences
    A = Zi - Zk
    B = Zj - Zk

    # T from X (deterministic, iid across obs here)
    di = jnp.sum((Xi - Xk)**2, axis=1)
    dj = jnp.sum((Xj - Xk)**2, axis=1)
    T = (di <= dj).astype(jnp.float32)

    return A, B, T, sig2

def P_from_triplets(theta_or_w, Z, I, J, K, sig2, already_w=False):
    # returns P_{ij,k}(w) for arrays I,J,K
    if already_w:
        w = theta_or_w
    else:
        w = jax.nn.softmax(theta_or_w)

    zi, zj, zk = Z[I], Z[J], Z[K]
    # vectorized delta/V
    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)
    delta, V = jax.vmap(dv)(zi, zj, zk)

    s = delta / jnp.sqrt(V + 1e-12)
    return ndtr(-s)  # Phi(-s)

def simulate_T_from_model(key, w_star, Z, I, J, K, sig2):
    key_u, _ = random.split(key)
    P = P_from_triplets(w_star, Z, I, J, K, sig2, already_w=True)
    U = random.uniform(key_u, shape=P.shape)
    return (U < P).astype(jnp.float32)

def simulate_T_from_model(key, w_star, Z, I, J, K, sig2):
    key_u, _ = random.split(key)
    P = P_from_triplets(w_star, Z, I, J, K, sig2, already_w=True)
    U = random.uniform(key_u, shape=P.shape)
    return (U < P).astype(jnp.float32)

def make_triplets_studentised_once(key, Z, sig2, frac_anchors=0.5, M=100, target_abs_s=0.5, min_abs_s=0.1):
    n, p = Z.shape
    m = int(jnp.floor(frac_anchors * n))
    perm = jax.random.permutation(key, n)
    anchors = perm[:m]
    dest = perm[m:]
    Zdest = Z[dest]

    # reference weights: proportional to Sigma^{-1}
    w0 = (1.0 / sig2)
    w0 = w0 / jnp.sum(w0)

    def one_anchor(k):
        zk = Z[k]
        diff = Zdest - zk[None, :]
        d_maha = jnp.sum((diff * diff) / sig2[None, :], axis=1)
        order = jnp.argsort(d_maha)
        cand = order[:M]  # indices into dest/Zdest

        # i = nearest in candidate pool
        i_pos = cand[0]
        i = dest[i_pos]
        a = Z[i] - zk

        # candidates for j
        Zc = Zdest[cand]
        b = Zc - zk[None, :]

        # studentised s under w0
        da = jnp.sum(w0 * (a*a))
        db = jnp.sum(w0[None, :] * (b*b), axis=1)
        delta = da - db

        w2_sig2 = (w0*w0) * sig2
        aa = jnp.sum(w2_sig2 * (a*a))
        bb = jnp.sum(w2_sig2[None, :] * (b*b), axis=1)
        ab = jnp.sum(w2_sig2[None, :] * (b * a[None, :]), axis=1)
        tr = jnp.sum((w0*w0) * (sig2*sig2))
        V = 8.0 * (aa + bb - ab) + 12.0 * tr
        s = delta / jnp.sqrt(V + 1e-12)

        # exclude i itself (would give delta=0-ish)
        s = s.at[0].set(jnp.inf)

        # enforce a minimum |s| to avoid near-ties dominated by noise,
        # then choose closest to target_abs_s
        abs_s = jnp.abs(s)
        score = jnp.where(abs_s >= min_abs_s, jnp.abs(abs_s - target_abs_s), jnp.inf)

        j_pos_in_cand = jnp.argmin(score)
        j = dest[cand[j_pos_in_cand]]
        return i, j, k

    I, J, K = jax.vmap(one_anchor)(anchors)
    return I, J, K

def make_triplets_once_in_space(key, V, frac_anchors=0.5, mode="euclid", sig2=None):
    """
    V: (n,p) array; either X or Z
    mode:
      - "euclid": uses squared Euclidean distances in V
      - "maha":   uses diagonal Mahalanobis with sig2 (required): sum (diff^2 / sig2)
    Returns I,J,K arrays of length m = floor(frac_anchors*n)
    """
    n, p = V.shape
    m = int(jnp.floor(frac_anchors * n))
    perm = jax.random.permutation(key, n)
    anchors = perm[:m]
    dest = perm[m:]
    Vdest = V[dest]

    if mode == "maha" and sig2 is None:
        raise ValueError("sig2 must be provided for mode='maha'")

    def one_anchor(k):
        vk = V[k]
        diff = Vdest - vk[None, :]
        if mode == "euclid":
            d = jnp.sum(diff * diff, axis=1)
        elif mode == "maha":
            d = jnp.sum((diff * diff) / sig2[None, :], axis=1)
        else:
            raise ValueError("mode must be 'euclid' or 'maha'")

        order = jnp.argsort(d)
        i = dest[order[0]]
        j = dest[order[1]]
        return i, j, k

    I, J, K = jax.vmap(one_anchor)(anchors)
    return I, J, K

def delta_V_batch(Z, I, J, K, w, sig2):
    zi, zj, zk = Z[I], Z[J], Z[K]   # (m,p)
    a = zi - zk
    b = zj - zk

    delta = jnp.sum(w[None, :] * (a * a - b * b), axis=1)

    w2_sig2 = (w * w) * sig2
    aa = jnp.sum(w2_sig2[None, :] * (a * a), axis=1)
    bb = jnp.sum(w2_sig2[None, :] * (b * b), axis=1)
    ab = jnp.sum(w2_sig2[None, :] * (a * b), axis=1)
    tr = jnp.sum((w * w) * (sig2 * sig2))

    V = 8.0 * (aa + bb - ab) + 12.0 * tr
    return delta, V

def P_from_w(Z, I, J, K, w, sig2):
    delta, V = delta_V_batch(Z, I, J, K, w, sig2)
    s = delta / jnp.sqrt(V + 1e-12)
    return ndtr(-s)
