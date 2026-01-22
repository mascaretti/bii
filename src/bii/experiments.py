from tqdm import tqdm
import jax
from jax import numpy as jnp
from jax import random
from bii.data import make_iid, make_hybrid, make_data, T_from_X, count_shared_destinations
from bii.inference import fit
from bii.utils import simplex_l1


def run_consistency_n(key, w_star, n_list, p, R=20, tau=1.0, sigma=0.5,
                      steps=5000, lr=1e-2, mode='iid', triplets_per_dataset=5):
    """
    Args:
        mode: 'iid', 'hybrid', or 'disjoint'
        triplets_per_dataset: Used only for 'hybrid' mode
    """
    assert p == w_star.shape[0]
    errs = []
    w_hats = []
    
    for n in tqdm(n_list):
        key, subkey = random.split(key)
        keys_rep = random.split(subkey, R)
        w_rep_list = []
        e_rep_list = []
        
        for krep in keys_rep:
            kdata, kfit = random.split(krep)
            
            if mode == 'iid':
                X, Z = make_iid(key=kdata, n=n, p=p, sig=sigma, tau=tau, w0=w_star)
            elif mode == 'hybrid':
                X, Z = make_hybrid(key=kdata, n_triplets=n, p=p, sig=sigma, tau=tau, 
                                  w0=w_star, triplets_per_dataset=triplets_per_dataset)
            elif mode == 'disjoint':
                X, Z = make_data(key=kdata, n_triplets=n, p=p, sig=sigma, tau=tau, 
                               w0=w_star, data_multiplier=20)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            w_hat, _, _ = fit(kfit, X, Z, sigma / jnp.sqrt(w_star), steps=steps)
            w_rep_list.append(w_hat)
            e_rep_list.append(simplex_l1(w_hat, w_star))
        
        w_rep = jnp.stack(w_rep_list)
        e_rep = jnp.array(e_rep_list)
        
        w_hats.append(w_rep)
        errs.append(e_rep)
    
    return jnp.stack(errs), jnp.stack(w_hats)

# Add to experiments.py

def run_correlation_experiment(key, w_star, n_triplets, p, dataset_size, R=20, 
                               tau=1.0, sigma=0.5, steps=5000, lr=1e-2):
    """
    Compare IID baseline vs. increasingly correlated triplets from single datasets.
    
    Args:
        n_triplets: Fixed number of triplets to extract
        dataset_sizes: List of dataset sizes to try (larger = less correlation)
        R: Number of replicates
        
    Returns:
        results: dict with keys 'iid' and 'correlated', each containing:
            - 'errors': shape (len(dataset_sizes), R)
            - 'final_losses': shape (len(dataset_sizes), R)
            - 'fractions': n_triplets / dataset_size for each dataset_size
    """
    results = {
        'iid': {'errors': [], 'final_losses': [], 'n_triplets': []},
        'correlated': {'errors': [], 'final_losses': [], 'fractions': [], 
                       'overlaps': []}  
    }


    # IID baseline (doesn't depend on dataset_size, but we'll repeat for comparison)
    print("Running IID baseline...")
    for triplets in tqdm(n_triplets, desc="IID reps"):
        key, subkey = random.split(key)
        keys_rep = random.split(subkey, R)
        
        iid_errors = []
        iid_losses = []
        
        for krep in keys_rep:
            kdata, kfit = random.split(krep)
            X, Z = make_iid(key=kdata, n=triplets, p=p, sig=sigma, tau=tau, w0=w_star)
            w_hat, _, loss_hist = fit(kfit, X, Z, sigma / jnp.sqrt(w_star), steps=steps, lr=lr)
            iid_errors.append(simplex_l1(w_hat, w_star))
            iid_losses.append(loss_hist[-1])
            
        results['iid']['errors'].append(iid_errors)
        results['iid']['final_losses'].append(iid_losses)
        results['iid']['n_triplets'].append(triplets)
        
    # Correlated triplets with varying dataset sizes
    print("\nRunning correlated experiments...")
    for triplets in tqdm(n_triplets, desc="Number of triplets"):
        corr_overlaps = []
        key, subkey = random.split(key)
        keys_rep = random.split(subkey, R)
        
        corr_errors = []
        corr_losses = []
        fraction = triplets / dataset_size
        
        for krep in keys_rep:
            kdata, kfit = random.split(krep)
            
            try:
                X, Z, indices = make_data(key=kdata, n_triplets=triplets, p=p, sig=sigma, 
                                          tau=tau, w0=w_star, sample_size=dataset_size, 
                                          data_multiplier=None, origin_ratio=0.5,
                                          split_data=True, repeat_pairs=True, return_indices=True)
                # Compute overlap
                overlap_stats = count_shared_destinations(indices, split_data=True)
                corr_overlaps.append(overlap_stats)
                w_hat, _, loss_hist = fit(kfit, X, Z, sigma / jnp.sqrt(w_star), 
                                          steps=steps, lr=lr)
                corr_errors.append(simplex_l1(w_hat, w_star))
                corr_losses.append(loss_hist[-1])
            except ValueError as e:
                # If can't extract enough triplets, use NaN
                corr_errors.append(jnp.nan)
                corr_losses.append(jnp.nan)
                corr_overlaps.append(jnp.nan)

        results['correlated']['overlaps'].append(corr_overlaps)
        results['correlated']['errors'].append(corr_errors)
        results['correlated']['final_losses'].append(corr_losses)
        results['correlated']['fractions'].append(fraction)
        
    # Convert to arrays
    results['iid']['errors'] = jnp.array(results['iid']['errors'])
    results['iid']['final_losses'] = jnp.array(results['iid']['final_losses'])
    results['iid']['n_triplets'] = jnp.array(results['iid']['n_triplets'])
    results['correlated']['overlaps'] = jnp.array(results['correlated']['overlaps'])
    results['correlated']['errors'] = jnp.array(results['correlated']['errors'])
    results['correlated']['final_losses'] = jnp.array(results['correlated']['final_losses'])
    results['correlated']['fractions'] = jnp.array(results['correlated']['fractions'])
    
    return results
