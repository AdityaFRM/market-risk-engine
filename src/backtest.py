import numpy as np

def count_var_breaches(returns, var):
    """
    count number of car breaches
    """
    breaches = returns < var
    return breaches.sum(), breaches

def breach_ratio(num_breaches, total_obs):
    """
    Compute breach ratio
    """
    return num_breaches/total_obs


from scipy.stats import chi2


def kupiec_test(num_breaches, total_obs, alpha):
    """
    Kupiec Proportion of Failures (POF) test.

    H0: The true breach probability equals (1 - alpha).
    """

    p = 1 - alpha
    x = num_breaches
    n = total_obs

    # Avoid division errors
    if x == 0 or x == n:
        return np.nan, np.nan

    likelihood_ratio = -2 * (
        (n - x) * np.log((1 - p) / (1 - x / n)) +
        x * np.log(p / (x / n))
    )

    p_value = 1 - chi2.cdf(likelihood_ratio, df=1)

    return likelihood_ratio, p_value