import numpy as np

def compute_var(returns, alpha = 0.99):
    """
    calculate Value at Risk corresponding to
    alpha confidence level
    """   
    return np.percentile(returns, 100*(1-alpha))

def compute_es(returns, alpha = 0.99):
    """
    calculate Value at Risk corresponding to
    alpha confidence level
    """
    x = compute_var(returns, alpha) 
    return returns[returns < x].mean() 