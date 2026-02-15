import numpy as np

def simulate_normal_returns(mean, stdev, sims, seed = None):
    """
    simulate normally distributed returns to
    be calculated through returns = u+z.s
    and those returns be analysed for Var, ES
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    z = np.random.normal(size = sims)
    returns = mean + stdev*z
    
    return returns