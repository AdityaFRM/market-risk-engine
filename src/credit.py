import numpy as np
from scipy.stats import norm


# =========================================
# Independent Default Model
# =========================================
def simulate_credit_losses(pd, ead, lgd, sims, seed=None):
    """
    Simulate portfolio credit losses under independent defaults.
    """

    if seed is not None:
        np.random.seed(seed)

    pd = np.array(pd)
    ead = np.array(ead)
    lgd = np.array(lgd)

    n_obligors = len(pd)

    U = np.random.uniform(size=(sims, n_obligors))

    defaults = U < pd

    losses = defaults * ead * lgd

    return losses.sum(axis=1)


# =========================================
# One-Factor Gaussian Copula
# =========================================
def simulate_correlated_credit_losses(
    pd, ead, lgd, rho, sims, seed=None, F=None
):
    """
    Simulate credit losses using one-factor Gaussian copula.
    """

    if seed is not None:
        np.random.seed(seed)

    pd = np.array(pd)
    ead = np.array(ead)
    lgd = np.array(lgd)

    n_obligors = len(pd)

    # Use external systemic factor if provided
    if F is None:
        F = np.random.normal(size=sims)

    epsilon = np.random.normal(size=(sims, n_obligors))

    Z = (
        np.sqrt(rho) * F[:, None] +
        np.sqrt(1 - rho) * epsilon
    )

    U = norm.cdf(Z)

    defaults = U < pd

    losses = defaults * ead * lgd

    return losses.sum(axis=1)


# =========================================
# Risk Metrics
# =========================================
def compute_credit_var(losses, alpha=0.99):
    return np.percentile(losses, alpha * 100)


def compute_credit_es(losses, alpha=0.99):
    var = compute_credit_var(losses, alpha)
    return losses[losses >= var].mean()


# =========================================
# Comparison Utility
# =========================================
def compare_independent_vs_correlated(
    pd, ead, lgd, rho, sims, alpha=0.99, seed=None
):

    indep_losses = simulate_credit_losses(pd, ead, lgd, sims, seed)

    corr_losses = simulate_correlated_credit_losses(
        pd, ead, lgd, rho, sims, seed
    )

    indep_var = compute_credit_var(indep_losses, alpha)
    corr_var = compute_credit_var(corr_losses, alpha)

    return {
        "independent_var": indep_var,
        "correlated_var": corr_var,
        "delta": corr_var - indep_var
    }


# =========================================
# Sector Model
# =========================================
def simulate_sector_credit_losses(
    pd=None,
    lgd=None,
    ead=None,
    rho=None,
    sims=10000,
    seed=None
):
    """
    Sector-based credit model with intra-sector correlation.
    """

    if seed is not None:
        np.random.seed(seed)

    pd = np.array(pd)
    ead = np.array(ead)
    lgd = np.array(lgd)
    sector_ids = np.array(sector_ids)

    n_obligors = len(pd)
    unique_sectors = np.unique(sector_ids)
    n_sectors = len(unique_sectors)

    # Sector factors
    sector_factors = np.random.normal(size=(sims, n_sectors))

    # Idiosyncratic shocks
    epsilon = np.random.normal(size=(sims, n_obligors))

    Z = np.zeros((sims, n_obligors))

    for idx, sector in enumerate(unique_sectors):
        mask = sector_ids == sector

        Z[:, mask] = (
            np.sqrt(rho) * sector_factors[:, idx][:, None] +
            np.sqrt(1 - rho) * epsilon[:, mask]
        )

    U = norm.cdf(Z)

    defaults = U < pd

    losses = defaults * ead * lgd

    return losses.sum(axis=1)