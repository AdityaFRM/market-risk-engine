import numpy as np

def compute_portfolio_returns(asset_returns, weights):
    """
    Take portfolio weights & asset_returns output
    of simulation.py
    """
    return asset_returns @ weights