import jax
from jax import numpy as jnp
from bii.data import T_from_X
import optax
from jax.scipy.special import log_ndtr

@jax.jit
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

@jax.jit
def logP_log1mP_from_deltaV(delta, V):
    s = delta / jnp.sqrt(V + 1e-12)
    logP = log_ndtr(-s)   # log Phi(-s)
    log1mP = log_ndtr(s)  # log (1 - Phi(-s))
    return logP, log1mP

@jax.jit
def loglik_theta(theta, T, Z, sig):
    sig2 = jnp.square(sig)
    w = jax.nn.softmax(theta)  # simplex
    zi, zj, zk = Z[:, 1], Z[:, 2], Z[:, 0]

    # vectorized delta/V over triplets
    def dv(zi, zj, zk):
        return delta_V_one_triplet(zi, zj, zk, w, sig2)
    delta, V = jax.vmap(dv)(zi, zj, zk)

    logP, log1mP = logP_log1mP_from_deltaV(delta, V)
    return jnp.sum(T * logP + (1.0 - T) * log1mP)

def fit(key, X, Z, sig, steps=5000, lr=1e-2):
    key_tr, _ = jax.random.split(key)
    n = X.shape[0]
    p = X.shape[2]
    T = T_from_X(X)
    theta = jnp.zeros((p,))   # uniform init
    opt = optax.adam(lr)
    opt_state = opt.init(theta)
    
    def neg_ll(th):
        return -loglik_theta(th, T, Z, sig)
    
    valgrad = jax.jit(jax.value_and_grad(neg_ll))
    
    @jax.jit
    def step(theta, opt_state):
        loss, grads = valgrad(theta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss
    
    # Store optimization trajectory
    theta_history = [theta]
    loss_history = []
    
    for _ in range(steps):
        theta, opt_state, loss = step(theta, opt_state)
        theta_history.append(theta)
        loss_history.append(loss)
    
    # Convert to arrays
    theta_history = jnp.stack(theta_history)  # (steps+1, p)
    loss_history = jnp.array(loss_history)    # (steps,)
    
    return jax.nn.softmax(theta), theta_history, loss_history
