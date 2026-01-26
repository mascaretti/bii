import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from bii.inference import compute_rhat

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

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Error vs. Fraction
    ax1.plot(n_triplets, med_err_iid, color='C0', linewidth=2, 
                label=f'IID baseline')
    ax1.fill_between(n_triplets, q25_err_iid, q75_err_iid, 
                     alpha=0.15, color='C0')
    overlay_powerlaw(n_triplets, med_err_iid, alpha=-0.5, anchor="middle", 
                     label=r"ref slope $-1/2$", color='gray')
    ax1.plot(n_triplets, med_err_corr, marker='o', color='C1', 
             label='Correlated (single dataset)')
    ax1.fill_between(n_triplets, q25_err_corr, q75_err_corr, 
                     alpha=0.15, color='C1')
    
    ax1.set_xlabel('log triplets count')
    ax1.set_ylabel('log L1 error on simplex')
    ax1.set_title(f'Error vs. N. Triplets (Sample size={sample_size})')
    ax1.legend()
    plt.xscale("log")
    plt.yscale("log")
    ax1.grid(True, alpha=0.3)

    shared_triplets = results['correlated']['overlaps']
    med_shared_triplets = jnp.median(shared_triplets, axis=1)
    ax2.loglog(n_triplets, med_shared_triplets)
    ax2.set_xlabel('triplets count')
    ax2.set_ylabel('reappearences')
    ax2.set_title('# of triplets with shared indices')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_posterior_marginals(w_samples, w_star=None, output_path=None, figsize=None):
    """
    Plot marginal posterior distributions for each weight component.
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
        w_star: True weights (p,) or None
        output_path: Path to save figure or None to display
        figsize: Figure size tuple
    """
    # Handle different input shapes
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    p = w_flat.shape[1]
    
    # Determine grid layout
    ncols = min(5, p)
    nrows = int(np.ceil(p / ncols))
    
    if figsize is None:
        figsize = (3.5 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if p == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(p):
        ax = axes[i]
        
        # Plot histogram
        ax.hist(w_flat[:, i], bins=50, alpha=0.7, density=True, 
                color='C0', edgecolor='black', linewidth=0.5, label='Posterior')
        
        # Add true value if provided
        if w_star is not None:
            ax.axvline(w_star[i], color='red', linestyle='--', 
                      linewidth=2.5, label=f'True: {w_star[i]:.3f}', zorder=10)
        
        # Add posterior mean
        mean_val = jnp.mean(w_flat[:, i])
        ax.axvline(mean_val, color='green', linestyle='-', 
                  linewidth=2, label=f'Mean: {mean_val:.3f}', alpha=0.8)
        
        ax.set_xlabel(f'$w_{i}$', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Component {i}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Hide extra subplots
    for i in range(p, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trace_plots(samples, w_star=None, output_path=None, figsize=None):
    """
    Plot MCMC trace plots for convergence diagnostics.
    
    Args:
        samples: (num_samples, num_chains, p)
        w_star: True weights (p,) or None
        output_path: Path to save figure
        figsize: Figure size
    """
    num_samples, num_chains, p = samples.shape
    
    # Determine grid layout
    ncols = min(5, p)
    nrows = int(np.ceil(p / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if p == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_chains))
    
    for i in range(p):
        ax = axes[i]
        
        # Plot each chain
        for chain_idx in range(num_chains):
            ax.plot(samples[:, chain_idx, i], alpha=0.6, linewidth=0.8,
                   color=colors[chain_idx], label=f'Chain {chain_idx+1}')
        
        # Add true value if provided
        if w_star is not None:
            ax.axhline(w_star[i], color='red', linestyle='--', 
                      linewidth=2, label='True', zorder=10, alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(f'$w_{i}$', fontsize=11)
        ax.set_title(f'Component {i}', fontsize=12, fontweight='bold')
        if i == 0 and num_chains <= 6:
            ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Hide extra subplots
    for i in range(p, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_posterior_summary(w_samples, w_star, output_path=None):
    """
    Plot posterior mean with credible intervals vs true values.
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
        w_star: True weights (p,)
        output_path: Path to save figure
    """
    # Flatten chains if present
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    p = w_flat.shape[1]
    
    # Compute statistics
    mean = jnp.mean(w_flat, axis=0)
    q025 = jnp.quantile(w_flat, 0.025, axis=0)
    q975 = jnp.quantile(w_flat, 0.975, axis=0)
    q05 = jnp.quantile(w_flat, 0.05, axis=0)
    q95 = jnp.quantile(w_flat, 0.95, axis=0)
    
    fig, ax = plt.subplots(figsize=(max(8, p*0.8), 5))
    
    x = np.arange(p)
    
    # Plot credible intervals
    ax.fill_between(x, q025, q975, alpha=0.3, color='C0', 
                    label='95% Credible Interval')
    ax.fill_between(x, q05, q95, alpha=0.4, color='C0', 
                    label='90% Credible Interval')
    
    # Plot posterior mean
    ax.plot(x, mean, 'o-', color='C0', linewidth=2.5, 
            markersize=10, label='Posterior mean', markeredgecolor='white', markeredgewidth=1.5)
    
    # Plot true values
    ax.plot(x, w_star, 's--', color='red', linewidth=2.5, 
            markersize=10, label='True $w^*$', markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('Component index', fontsize=12)
    ax.set_ylabel('Weight value', fontsize=12)
    ax.set_title('Posterior Estimates vs True Weights', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence_diagnostics(samples, output_path=None):
    """
    Plot R-hat convergence diagnostics over sample sizes.
    
    Args:
        samples: (num_samples, num_chains, p)
        output_path: Path to save figure
    """
    from posterior_inference import compute_rhat
    
    num_samples, num_chains, p = samples.shape
    
    # Compute R-hat for different sample sizes
    sample_sizes = np.linspace(100, num_samples, 20, dtype=int)
    sample_sizes = sample_sizes[sample_sizes >= 50]  # Minimum samples
    
    rhats = []
    for n in sample_sizes:
        rhat = compute_rhat(samples[:n, :, :])
        rhats.append(rhat)
    
    rhats = np.array(rhats)  # (num_points, p)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(p):
        ax.plot(sample_sizes, rhats[:, i], label=f'Component {i}', 
                alpha=0.8, linewidth=2, marker='o', markersize=4)
    
    ax.axhline(1.01, color='red', linestyle='--', linewidth=2.5, 
               label='Target threshold (1.01)', zorder=10)
    ax.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Number of samples', fontsize=12)
    ax.set_ylabel('$\hat{R}$', fontsize=14)
    ax.set_title('Gelman-Rubin Convergence Diagnostic', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', ncol=2 if p > 5 else 1)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim([0.98, max(1.15, np.max(rhats) * 1.05)])
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pairwise_posteriors(w_samples, w_star=None, output_path=None, max_pairs=10):
    """
    Plot pairwise posterior distributions (2D scatter/density plots).
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
        w_star: True weights (p,) or None
        output_path: Path to save figure
        max_pairs: Maximum number of pairs to plot
    """
    # Flatten chains if present
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    p = w_flat.shape[1]
    
    # Select pairs to plot
    pairs = [(i, j) for i in range(p) for j in range(i+1, p)]
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    
    num_pairs = len(pairs)
    ncols = min(5, num_pairs)
    nrows = int(np.ceil(num_pairs / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3*nrows))
    if num_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        
        # 2D histogram / hexbin
        ax.hexbin(w_flat[:, i], w_flat[:, j], gridsize=35, 
                 cmap='Blues', mincnt=1, alpha=0.8, linewidths=0.2)
        
        # True value
        if w_star is not None:
            ax.plot(w_star[i], w_star[j], 'r*', markersize=18, 
                   label='True', zorder=10, markeredgecolor='white', markeredgewidth=1)
        
        # Posterior mean
        mean_i = jnp.mean(w_flat[:, i])
        mean_j = jnp.mean(w_flat[:, j])
        ax.plot(mean_i, mean_j, 'go', markersize=12, 
               label='Mean', zorder=10, markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel(f'$w_{i}$', fontsize=11)
        ax.set_ylabel(f'$w_{j}$', fontsize=11)
        ax.set_title(f'Components ({i}, {j})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Hide extra subplots
    for idx in range(num_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_map_mle_comparison(w_samples, w_mle, w_star, output_path=None):
    """
    Plot comparison of MAP, MLE, and true values.
    
    This plot helps verify that:
    1. MAP ≈ MLE (for uniform prior, they should overlap)
    2. Distance from true value makes sense given posterior variance
    3. Issues are not due to HMC sampler
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
        w_mle: MLE estimate (p,)
        w_star: True weights (p,)
        output_path: Path to save figure
    """
    # Flatten chains if present
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    p = w_flat.shape[1]
    
    # Compute statistics
    mean = jnp.mean(w_flat, axis=0)
    median = jnp.median(w_flat, axis=0)  # MAP approximation for unimodal
    std = jnp.std(w_flat, axis=0)
    q025 = jnp.quantile(w_flat, 0.025, axis=0)
    q975 = jnp.quantile(w_flat, 0.975, axis=0)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ============================================================
    # Plot 1: Point estimates comparison
    # ============================================================
    ax = axes[0]
    x = np.arange(p)
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, w_star, width, label='True $w^*$', 
           alpha=0.8, color='red', edgecolor='darkred', linewidth=1.5)
    ax.bar(x, w_mle, width, label='MLE', 
           alpha=0.8, color='orange', edgecolor='darkorange', linewidth=1.5)
    ax.bar(x + width, median, width, label='MAP (posterior median)', 
           alpha=0.8, color='green', edgecolor='darkgreen', linewidth=1.5)
    
    # Add posterior mean as points
    ax.plot(x + width, mean, 'ko', markersize=8, label='Posterior mean',
           markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    ax.set_xlabel('Component index', fontsize=13)
    ax.set_ylabel('Weight value', fontsize=13)
    ax.set_title('Point Estimates Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # ============================================================
    # Plot 2: Deviations from true value with uncertainty
    # ============================================================
    ax = axes[1]
    
    # Compute deviations
    dev_mle = w_mle - w_star
    dev_map = median - w_star
    dev_mean = mean - w_star
    
    # Error bars showing posterior uncertainty
    ci_lower = q025 - w_star
    ci_upper = q975 - w_star
    
    # Plot deviations
    ax.plot(x, dev_mle, 'o-', markersize=10, linewidth=2.5, 
            label='MLE - True', color='orange', markeredgecolor='white', 
            markeredgewidth=1.5)
    ax.plot(x, dev_map, 's-', markersize=10, linewidth=2.5, 
            label='MAP - True', color='green', markeredgecolor='white', 
            markeredgewidth=1.5)
    
    # Add error bars for posterior uncertainty
    ax.errorbar(x, dev_mean, yerr=[dev_mean - ci_lower, ci_upper - dev_mean],
                fmt='d', markersize=8, linewidth=2, capsize=5, capthick=2,
                label='Posterior mean ± 95% CI', color='blue', alpha=0.7)
    
    # Add reference line at zero
    ax.axhline(0, color='red', linestyle='--', linewidth=2, 
               label='Perfect match', alpha=0.7)
    
    # Shade 1-sigma region
    ax.fill_between(x, -np.mean(std), np.mean(std), 
                    alpha=0.2, color='gray', label='±1 avg. posterior std')
    
    ax.set_xlabel('Component index', fontsize=13)
    ax.set_ylabel('Deviation from true value', fontsize=13)
    ax.set_title('Estimation Error Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print numerical summary
    print("\n" + "="*70)
    print("MAP vs MLE Comparison")
    print("="*70)
    print(f"\nL1 distance (MAP, MLE):  {jnp.sum(jnp.abs(median - w_mle)):.6f}")
    print(f"L1 distance (MAP, True): {jnp.sum(jnp.abs(median - w_star)):.6f}")
    print(f"L1 distance (MLE, True): {jnp.sum(jnp.abs(w_mle - w_star)):.6f}")
    print(f"\nMax absolute difference (MAP, MLE): {jnp.max(jnp.abs(median - w_mle)):.6f}")
    print(f"\nComponents where true value is outside 95% CI:")
    outside_ci = (w_star < q025) | (w_star > q975)
    if jnp.any(outside_ci):
        for i in jnp.where(outside_ci)[0]:
            print(f"  Component {i}: true={w_star[i]:.4f}, "
                  f"CI=[{q025[i]:.4f}, {q975[i]:.4f}], "
                  f"width={q975[i]-q025[i]:.4f}")
    else:
        print("  None - all true values within 95% CI")
    print("="*70 + "\n")

"""
Visualization utilities for posterior analysis.
"""

import numpy as np
from matplotlib import pyplot as plt
import jax.numpy as jnp
from pathlib import Path


def plot_posterior_marginals(w_samples, w_star=None, output_path=None, figsize=None):
    """
    Plot marginal posterior distributions for each weight component.
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
        w_star: True weights (p,) or None
        output_path: Path to save figure or None to display
        figsize: Figure size tuple
    """
    # Handle different input shapes
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    p = w_flat.shape[1]
    
    # Determine grid layout
    ncols = min(5, p)
    nrows = int(np.ceil(p / ncols))
    
    if figsize is None:
        figsize = (3.5 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if p == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(p):
        ax = axes[i]
        
        # Plot histogram
        ax.hist(w_flat[:, i], bins=50, alpha=0.7, density=True, 
                color='C0', edgecolor='black', linewidth=0.5, label='Posterior')
        
        # Add true value if provided
        if w_star is not None:
            ax.axvline(w_star[i], color='red', linestyle='--', 
                      linewidth=2.5, label=f'True: {w_star[i]:.3f}', zorder=10)
        
        # Add posterior mean
        mean_val = jnp.mean(w_flat[:, i])
        ax.axvline(mean_val, color='green', linestyle='-', 
                  linewidth=2, label=f'Mean: {mean_val:.3f}', alpha=0.8)
        
        ax.set_xlabel(f'$w_{i}$', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Component {i}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Hide extra subplots
    for i in range(p, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trace_plots(samples, w_star=None, output_path=None, figsize=None):
    """
    Plot MCMC trace plots for convergence diagnostics.
    
    Args:
        samples: (num_samples, num_chains, p)
        w_star: True weights (p,) or None
        output_path: Path to save figure
        figsize: Figure size
    """
    num_samples, num_chains, p = samples.shape
    
    # Determine grid layout
    ncols = min(5, p)
    nrows = int(np.ceil(p / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if p == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_chains))
    
    for i in range(p):
        ax = axes[i]
        
        # Plot each chain
        for chain_idx in range(num_chains):
            ax.plot(samples[:, chain_idx, i], alpha=0.6, linewidth=0.8,
                   color=colors[chain_idx], label=f'Chain {chain_idx+1}')
        
        # Add true value if provided
        if w_star is not None:
            ax.axhline(w_star[i], color='red', linestyle='--', 
                      linewidth=2, label='True', zorder=10, alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(f'$w_{i}$', fontsize=11)
        ax.set_title(f'Component {i}', fontsize=12, fontweight='bold')
        if i == 0 and num_chains <= 6:
            ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Hide extra subplots
    for i in range(p, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_posterior_summary(w_samples, w_star, output_path=None):
    """
    Plot posterior mean with credible intervals vs true values.
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
        w_star: True weights (p,)
        output_path: Path to save figure
    """
    # Flatten chains if present
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    p = w_flat.shape[1]
    
    # Compute statistics
    mean = jnp.mean(w_flat, axis=0)
    q025 = jnp.quantile(w_flat, 0.025, axis=0)
    q975 = jnp.quantile(w_flat, 0.975, axis=0)
    q05 = jnp.quantile(w_flat, 0.05, axis=0)
    q95 = jnp.quantile(w_flat, 0.95, axis=0)
    
    fig, ax = plt.subplots(figsize=(max(8, p*0.8), 5))
    
    x = np.arange(p)
    
    # Plot credible intervals
    ax.fill_between(x, q025, q975, alpha=0.3, color='C0', 
                    label='95% Credible Interval')
    ax.fill_between(x, q05, q95, alpha=0.4, color='C0', 
                    label='90% Credible Interval')
    
    # Plot posterior mean
    ax.plot(x, mean, 'o-', color='C0', linewidth=2.5, 
            markersize=10, label='Posterior mean', markeredgecolor='white', markeredgewidth=1.5)
    
    # Plot true values
    ax.plot(x, w_star, 's--', color='red', linewidth=2.5, 
            markersize=10, label='True $w^*$', markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('Component index', fontsize=12)
    ax.set_ylabel('Weight value', fontsize=12)
    ax.set_title('Posterior Estimates vs True Weights', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence_diagnostics(samples, output_path=None):
    """
    Plot R-hat convergence diagnostics over sample sizes.
    
    Args:
        samples: (num_samples, num_chains, p)
        output_path: Path to save figure
    """
    
    
    num_samples, num_chains, p = samples.shape
    
    # Compute R-hat for different sample sizes
    sample_sizes = np.linspace(100, num_samples, 20, dtype=int)
    sample_sizes = sample_sizes[sample_sizes >= 50]  # Minimum samples
    
    rhats = []
    for n in sample_sizes:
        rhat = compute_rhat(samples[:n, :, :])
        rhats.append(rhat)
    
    rhats = np.array(rhats)  # (num_points, p)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(p):
        ax.plot(sample_sizes, rhats[:, i], label=f'Component {i}', 
                alpha=0.8, linewidth=2, marker='o', markersize=4)
    
    ax.axhline(1.01, color='red', linestyle='--', linewidth=2.5, 
               label='Target threshold (1.01)', zorder=10)
    ax.axhline(1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Number of samples', fontsize=12)
    ax.set_ylabel(r'$\hat{R}$', fontsize=14)
    ax.set_title('Gelman-Rubin Convergence Diagnostic', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', ncol=2 if p > 5 else 1)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim([0.98, max(1.15, np.max(rhats) * 1.05)])
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pairwise_posteriors(w_samples, w_star=None, output_path=None, max_pairs=10):
    """
    Plot pairwise posterior distributions (2D scatter/density plots).
    
    Args:
        w_samples: (num_samples, num_chains, p) or (num_samples, p)
        w_star: True weights (p,) or None
        output_path: Path to save figure
        max_pairs: Maximum number of pairs to plot
    """
    # Flatten chains if present
    if w_samples.ndim == 3:
        w_flat = w_samples.reshape(-1, w_samples.shape[-1])
    else:
        w_flat = w_samples
    
    p = w_flat.shape[1]
    
    # Select pairs to plot
    pairs = [(i, j) for i in range(p) for j in range(i+1, p)]
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    
    num_pairs = len(pairs)
    ncols = min(5, num_pairs)
    nrows = int(np.ceil(num_pairs / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3*nrows))
    if num_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        
        # 2D histogram / hexbin
        ax.hexbin(w_flat[:, i], w_flat[:, j], gridsize=35, 
                 cmap='Blues', mincnt=1, alpha=0.8, linewidths=0.2)
        
        # True value
        if w_star is not None:
            ax.plot(w_star[i], w_star[j], 'r*', markersize=18, 
                   label='True', zorder=10, markeredgecolor='white', markeredgewidth=1)
        
        # Posterior mean
        mean_i = jnp.mean(w_flat[:, i])
        mean_j = jnp.mean(w_flat[:, j])
        ax.plot(mean_i, mean_j, 'go', markersize=12, 
               label='Mean', zorder=10, markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel(f'$w_{i}$', fontsize=11)
        ax.set_ylabel(f'$w_{j}$', fontsize=11)
        ax.set_title(f'Components ({i}, {j})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Hide extra subplots
    for idx in range(num_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
