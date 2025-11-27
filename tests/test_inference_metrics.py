import numpy as np
import jax.numpy as jnp
from bii.inference import summarize_posterior_metrics


def test_summarize_posterior_metrics_basic():
    samples = jnp.array([
        [0.4, 0.6],
        [0.45, 0.55],
        [0.5, 0.5],
        [0.55, 0.45],
        [0.6, 0.4],
    ])
    w_true = jnp.array([0.5, 0.5])
    metrics = summarize_posterior_metrics(samples, w_true, alpha=0.1, num_warmup=10, num_samples=5, total_time=1.0)
    assert metrics["rmse"] >= 0.0
    assert metrics["mciw"] > 0.0
    assert 0.0 <= metrics["ecp"] <= 1.0
    assert metrics["ess_frac"] >= 0.0
    assert metrics["time_1000_ess"] is not None
