import numpy as np

def simulate_fx_var(
    exposures,
    rates,
    vols,
    sims = 10000,
    seed = None
):
    """
    Simulate portfolio VaR for multicurrency portfolio
    """
    if seed is not None:
        np.random.seed(seed)
    
    exposures = np.array(exposures)
    rates = np.array(rates)
    vols = np.array(vols)
    
    returns = np.random.normal(0, vols, size = (sims, len(exposures)))
    
    pnl = exposures * rates * returns
    
    var_99 = np.percentile(pnl,1)
    
    return -var_99
    
    
def simulate_correlated_fx_var(
    exposures,
    rates,
    volatilities,
    corr_matrix,
    sims = 10000,
    seed = None
):
    """
    Simulate for correlated FX exposures
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    volatilities = np.array(volatilities)
    rates = np.array(rates)
    exposures = np.array(exposures)

    
    D = np.diag(volatilities)
    
    cov_matrix = D @ corr_matrix @ D
    
    returns = np.random.multivariate_normal(mean = np.zeros(len(exposures)), cov = cov_matrix, size = sims)
    
    PnL = (returns * exposures * rates).sum(axis = 1)
    
    var_99 = np.percentile(PnL, 1)
    
    return -var_99