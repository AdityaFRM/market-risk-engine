import numpy as np
from src.simulation import simulate_multivariate_returns
from src.portfolio import compute_portfolio_returns
from src.var import compute_var

def apply_volatility_stress(sigma, shock_multiplier):
    """
    Apply multiplicative stress to volatility vector.
    """
    return sigma * shock_multiplier


def apply_correlation_stress(corr_matrix, shock_factor):
    """
    correlation increase by shock_factor
    """
    stressed_corr = corr_matrix.copy()
    
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if i != j:
                 stressed_corr[i,j] = min(1.0, corr_matrix[i,j] + shock_factor)
                       
    return stressed_corr


def run_stress_scenario(mu, sigma, corr, weights,
                        vol_shock, corr_shock,
                        sims, alpha, seed=None):
    """
    Run base and stressed VaR comparison.
    """

    # --- Base covariance ---
    D = np.diag(sigma)
    cov = D @ corr @ D

    # --- Base simulation ---
    sim_returns = simulate_multivariate_returns(mu, cov, sims, seed)
    portfolio_returns = compute_portfolio_returns(sim_returns, weights)
    base_var = compute_var(portfolio_returns, alpha)

    # --- Stress inputs ---
    stressed_sigma = apply_volatility_stress(sigma, vol_shock)
    stressed_corr = apply_correlation_stress(corr, corr_shock)

    stressed_D = np.diag(stressed_sigma)
    stressed_cov = stressed_D @ stressed_corr @ stressed_D

    # --- Stressed simulation ---
    stressed_returns = simulate_multivariate_returns(mu, stressed_cov, sims, seed)
    stressed_portfolio_returns = compute_portfolio_returns(stressed_returns, weights)
    stressed_var = compute_var(stressed_portfolio_returns, alpha)

    return {
        "base_var": base_var,
        "stressed_var": stressed_var,
        "delta_var": stressed_var - base_var
    }
