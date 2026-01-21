import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

def overlay_powerlaw(n_list, y_med, alpha=-0.5, anchor="middle", label=None, **plot_kw):
    n = np.asarray(n_list, dtype=float)
    y = np.asarray(y_med, dtype=float)

    if anchor == "first":
        i0 = 0
    elif anchor == "last":
        i0 = -1
    else:  # "middle"
        i0 = len(n)//2

    n0, y0 = n[i0], y[i0]
    C = y0 / (n0**alpha)            # so that y_ref(n0) == y0
    y_ref = C * (n**alpha)

    if label is None:
        label = rf"reference: $C\,n^{{{alpha}}}$ (anchored at n={n0:g})"
    plt.plot(n, y_ref, linestyle="--", label=label, **plot_kw)

def simplex_l1(w_hat, w_star):
    w_star /= jnp.sum(w_star)
    w_hat /= jnp.sum(w_hat)
    return jnp.sum(jnp.abs(w_hat - w_star))

def plot_three_way_comparison(errs_iid, errs_hybrid, errs_disjoint, n_list, R, 
                               output_path=None):
    """
    Plot comparison of all three methods without overlay.
    """
    # Compute statistics
    med_iid = jnp.median(errs_iid, axis=1)
    q25_iid = jnp.quantile(errs_iid, 0.25, axis=1)
    q75_iid = jnp.quantile(errs_iid, 0.75, axis=1)
    
    med_hybrid = jnp.median(errs_hybrid, axis=1)
    q25_hybrid = jnp.quantile(errs_hybrid, 0.25, axis=1)
    q75_hybrid = jnp.quantile(errs_hybrid, 0.75, axis=1)
    
    med_disjoint = jnp.median(errs_disjoint, axis=1)
    q25_disjoint = jnp.quantile(errs_disjoint, 0.25, axis=1)
    q75_disjoint = jnp.quantile(errs_disjoint, 0.75, axis=1)
    
    plt.figure(figsize=(7.2, 4.6))
    
    # Plot all three
    plt.plot(n_list, med_iid, marker="o", label="IID (k=1)", color='C0')
    plt.fill_between(n_list, q25_iid, q75_iid, alpha=0.15, color='C0')
    
    plt.plot(n_list, med_hybrid, marker="s", label="Hybrid (k=5)", color='C1')
    plt.fill_between(n_list, q25_hybrid, q75_hybrid, alpha=0.15, color='C1')
    
    plt.plot(n_list, med_disjoint, marker="^", label="Disjoint (single dataset)", color='C2')
    plt.fill_between(n_list, q25_disjoint, q75_disjoint, alpha=0.15, color='C2')
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of triplet observations (log scale)")
    plt.ylabel("L1 error on simplex (log scale)")
    plt.title(f"Median log-L1 error - {R} MC reps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_individual_with_overlay(errs, n_list, R, label, output_path=None):
    """
    Plot a single method with power law overlay.
    """
    med = jnp.median(errs, axis=1)
    q25 = jnp.quantile(errs, 0.25, axis=1)
    q75 = jnp.quantile(errs, 0.75, axis=1)
    
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(n_list, med, marker="o", label=label)
    plt.fill_between(n_list, q25, q75, alpha=0.15)
    
    overlay_powerlaw(n_list, med, alpha=-0.5, anchor="middle", 
                     label=r"ref slope $-1/2$", color='gray')
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of triplet observations (log scale)")
    plt.ylabel("L1 error on simplex (log scale)")
    plt.title(f"Median log-L1 error - {R} MC reps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


# Add to utils.py

def plot_correlation_experiment(results, n_triplets, output_path=None):
    """
    Plot error and loss vs. fraction of triplets/dataset_size.
    
    Args:
        results: Output from run_correlation_experiment
        n_triplets: Number of triplets (for title)
        output_path: Path to save figure
    """
    sample_size = int(results['iid']['n_triplets'][0] / results['correlated']['fractions'][0])
    fractions_corr = results['correlated']['fractions']
    n_triplets = results['iid']['n_triplets']
    
    # Compute statistics
    errors_iid = results['iid']['errors']
    errors_corr = results['correlated']['errors']

    med_err_iid = jnp.median(errors_iid, axis=1)
    q25_err_iid = jnp.quantile(errors_iid, 0.25, axis=1)
    q75_err_iid = jnp.quantile(errors_iid, 0.75, axis=1)
    med_err_corr = jnp.median(errors_corr, axis=1)
    q25_err_corr = jnp.quantile(errors_corr, 0.25, axis=1)
    q75_err_corr = jnp.quantile(errors_corr, 0.75, axis=1)

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    
    # Plot 1: Error vs. Fraction
    ax1.plot(n_triplets, med_err_iid, color='C0', linewidth=2, 
                label=f'IID baseline')
    ax1.fill_between(n_triplets, q25_err_iid, q75_err_iid, 
                     alpha=0.15, color='C0')
    ax1.plot(n_triplets, med_err_corr, marker='o', color='C1', 
             label='Correlated (single dataset)')
    ax1.fill_between(n_triplets, q25_err_corr, q75_err_corr, 
                     alpha=0.15, color='C1')
    
    ax1.set_xlabel('log triplets count')
    ax1.set_ylabel('L1 error on simplex')
    ax1.set_title(f'Error vs. N. Triplets (Sample size={sample_size})')
    ax1.legend()
    plt.xscale("log")
    plt.yscale("log")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()
