import numpy as np
from scipy.stats import norm


def simulate_enterprise_var(
    credit_params,
    fx_params,
    liquidity_params,
    sims=10000,
    seed=None
):
    """
    Enterprise risk simulation with systemic linkage
    (Credit + FX + Liquidity)
    """

    if seed is not None:
        np.random.seed(seed)

    # =======================
    # SYSTEMIC FACTOR
    # =======================
    F = np.random.normal(size=sims)

    # =======================
    # CREDIT RISK
    # =======================
    pd = np.array(credit_params["pd"])
    lgd = np.array(credit_params["lgd"])
    ead = np.array(credit_params["exposure"])
    rho = credit_params["rho"]

    n = len(pd)

    epsilon = np.random.normal(size=(sims, n))

    Z = np.sqrt(rho) * F[:, None] + np.sqrt(1 - rho) * epsilon
    U = norm.cdf(Z)

    defaults = U < pd

    credit_losses = (defaults * lgd * ead).sum(axis=1)

    # =======================
    # FX RISK
    # =======================
    exposures_fx = np.array(fx_params["exposures"])
    rates_fx = np.array(fx_params["rates"])
    vols_fx = np.array(fx_params["vols"])

    epsilon_fx = np.random.normal(size=(sims, len(exposures_fx)))

    # FX driven partly by systemic factor
    fx_returns = F[:, None] * vols_fx + epsilon_fx * vols_fx * 0.5

    fx_pnl = (exposures_fx * rates_fx * fx_returns).sum(axis=1)

    # =======================
    # LIQUIDITY RISK
    # =======================
    inflows = np.array(liquidity_params["inflows"])
    outflows = np.array(liquidity_params["outflows"])
    times = np.array(liquidity_params["times"])
    buffer = liquidity_params["buffer"]
    haircut = liquidity_params["haircut"]

    survival_times = []

    for f in F:
        # Higher run when systemic stress is bad
        run = np.clip(0.2 + 0.2 * (-f), 0, 0.6)

        net = (inflows * (1 - haircut)) - (outflows * (1 + run))
        cumulative = np.cumsum(net)
        available = buffer + cumulative

        negative = np.where(available < 0)[0]

        if len(negative) == 0:
            survival_times.append(np.max(times))
        else:
            survival_times.append(times[negative[0]])

    survival_times = np.array(survival_times)

    # Penalty for early failure
    liquidity_penalty = np.where(survival_times <= 3, 500, 0)

    # =======================
    # ENTERPRISE LOSS
    # =======================
    enterprise_loss = credit_losses - fx_pnl + liquidity_penalty

    var_99 = np.percentile(enterprise_loss, 99)

    return var_99, enterprise_loss