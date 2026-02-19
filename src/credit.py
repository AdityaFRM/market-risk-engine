import numpy as np
from src.simulation import simulate_normal_returns

def simulate_credit_losses(pd, exposure, lgd, sims, seed=None):
    """
    Simulate portfolio credit losses under independent defaults.

    Parameters
    ----------
    pd : np.array
        Probability of default per obligor
    exposure : np.array
        Exposure per obligor
    lgd : np.array
        Loss given default per obligor
    sims : int
        Number of simulations

    Returns
    -------
    np.array
        Simulated portfolio loss distribution
    """

    if seed is not None:
        np.random.seed(seed)

    n_obligors = len(pd)

    # Uniform random draws
    U = np.random.uniform(size=(sims, n_obligors))

    # Default indicator matrix
    defaults = U < pd

    # Loss per obligor
    losses = defaults * exposure * lgd

    # Aggregate portfolio loss
    portfolio_losses = losses.sum(axis=1)

    return portfolio_losses
    
    
from scipy.stats import norm


def simulate_correlated_credit_losses(pd, exposure, lgd, rho, sims, seed=None):
    """
    Simulate portfolio credit losses with one-factor Gaussian correlation.
    """

    if seed is not None:
        np.random.seed(seed)

    n_obligors = len(pd)

    # Common systematic factor
    F = np.random.normal(size=sims)

    # Idiosyncratic shocks
    epsilon = np.random.normal(size=(sims, n_obligors))

    # Construct correlated latent variable
    Z = (np.sqrt(rho) * F[:, None] +
         np.sqrt(1 - rho) * epsilon)

    # Convert to uniform via CDF
    U = norm.cdf(Z)

    # Default indicator
    defaults = U < pd

    # Loss calculation
    losses = defaults * exposure * lgd

    portfolio_losses = losses.sum(axis=1)

    return portfolio_losses

    # Loss per obligor
    losses = defaults * exposure * lgd

    # Aggregate portfolio loss
    portfolio_losses = losses.sum(axis=1)

    return portfolio_losses


def compute_credit_var(losses, alpha=0.99):
    """
    Compute Credit VaR from loss distribution.
    """
    return np.percentile(losses, alpha * 100)


def compute_credit_es(losses, alpha=0.99):
    """
    Compute Credit Expected Shortfall.
    """
    var = compute_credit_var(losses, alpha)
    return losses[losses > var].mean()

def compare_independent_vs_correlated(pd, exposure, lgd, rho, sims, alpha=0.99, seed=None):

    indep_losses = simulate_credit_losses(pd, exposure, lgd, sims, seed)
    corr_losses = simulate_correlated_credit_losses(pd, exposure, lgd, 
                                                    rho, sims, seed)

    indep_var = compute_credit_var(indep_losses, alpha)
    corr_var = compute_credit_var(corr_losses, alpha)

    return {
        "independent_var": indep_var,
        "correlated_var": corr_var,
        "delta": corr_var - indep_var
    }


def simulate_sector_credit_losses(pd, exposure, lgd, sector_ids, rho, sims, seed=None):
    """
    Simulate credit losses using sector-level correlation model.
    
    sector_ids : array of integers indicating sector membership
    rho : sector correlation parameter
    """

    if seed is not None:
        np.random.seed(seed)

    n_obligors = len(pd)
    unique_sectors = np.unique(sector_ids)
    n_sectors = len(unique_sectors)

    # Sector factors (one per sector per simulation)
    sector_factors = np.random.normal(size=(sims, n_sectors))

    # Idiosyncratic shocks
    epsilon = np.random.normal(size=(sims, n_obligors))

    # Map each obligor to its sector factor
    Z = np.zeros((sims, n_obligors))

    for idx, sector in enumerate(unique_sectors):
        mask = sector_ids == sector
        Z[:, mask] = (
            np.sqrt(rho) * sector_factors[:, idx][:, None] +
            np.sqrt(1 - rho) * epsilon[:, mask]
        )

    # Convert to uniform
    from scipy.stats import norm
    U = norm.cdf(Z)

    defaults = U < pd
    losses = defaults * exposure * lgd

    return losses.sum(axis=1)