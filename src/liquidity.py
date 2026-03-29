import numpy as np
import pandas as pd


def build_cashflow_ladder(
    inflows,
    outflows,
    times,
    initial_buffer=0.0,
    inflow_haircut=0.0,
    outflow_stress=0.0
):

    df = pd.DataFrame({
        "time": times,
        "inflows": inflows,
        "outflows": outflows
    })

    df = df.sort_values("time")

    df["effective_inflows"] = df["inflows"] * (1 - inflow_haircut)

    df["stressed_outflows"] = df["outflows"] * (1 + outflow_stress)

    df["net"] = df["effective_inflows"] - df["stressed_outflows"]

    df["cumulative_net"] = df["net"].cumsum()

    df["available_liquidity"] = initial_buffer + df["cumulative_net"]

    return df


def compute_survival_horizon(ladder_df):
    negative_rows = ladder_df[ladder_df["available_liquidity"] < 0]

    if len(negative_rows) == 0:
        return None

    return negative_rows.iloc[0]["time"]


def simulate_liquidity_survival(
    inflows,
    outflows,
    times,
    initial_buffer,
    inflow_haircut,
    sims=1000,
    seed=None
):

    if seed is not None:
        np.random.seed(seed)

    survival_times = []

    for _ in range(sims):

        # Random deposit run between 10% and 40%
        outflow_stress = np.random.uniform(0.0, 0.5)

        ladder = build_cashflow_ladder(
            inflows,
            outflows,
            times,
            initial_buffer=initial_buffer,
            inflow_haircut=inflow_haircut,
            outflow_stress=outflow_stress
        )

        horizon = compute_survival_horizon(ladder)

        if horizon is None:
            survival_times.append(np.max(times))
        else:
            survival_times.append(horizon)

    return np.array(survival_times)


def simulate_liquidity_with_systemic_factor(
    inflows,
    outflows,
    times,
    initial_buffer,
    inflow_haircut,
    systemic_factors
):
    """
    Simulate survival time using pre-generated systemic factor array.
    """

    survival_times = []

    for F in systemic_factors:

        # Map systemic factor to deposit run intensity
        # Bad systemic shock (large negative F) -> higher run

        run_intensity = 0.2 + 0.2 * (-F)  # simple linear mapping

        # Clamp between 0 and 0.6
        run_intensity = np.clip(run_intensity, 0.0, 0.6)

        ladder = build_cashflow_ladder(
            inflows,
            outflows,
            times,
            initial_buffer=initial_buffer,
            inflow_haircut=inflow_haircut,
            outflow_stress=run_intensity
        )

        horizon = compute_survival_horizon(ladder)

        if horizon is None:
            survival_times.append(np.max(times))
        else:
            survival_times.append(horizon)

    return np.array(survival_times)


def compute_lcr(hqla, inflows, outflows):
    """
    Compute Liquidity Coverage Ratio (LCR)
    
    inflows, outflows: arrays over 30-day horizon
    """

    total_outflows = np.sum(outflows)
    total_inflows = np.sum(inflows)

    # Inflows capped at 75% of outflows
    capped_inflows = min(total_inflows, 0.75 * total_outflows)

    net_outflows = total_outflows - capped_inflows

    lcr = hqla / net_outflows if net_outflows > 0 else np.inf

    return lcr