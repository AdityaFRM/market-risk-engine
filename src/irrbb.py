import numpy as np


def present_value(cashflows, times, rates, rate_times):
    """
    Present value using linear interpolation on yield curve.

    cashflows : array
    times     : array (in years)
    rates     : array (yield curve rates)
    rate_times: array (maturities corresponding to rates)
    """

    # Interpolate curve to cashflow maturities
    interpolated_rates = np.interp(times, rate_times, rates)

    discount_factors = np.exp(-interpolated_rates * times)

    return np.sum(cashflows * discount_factors)


def apply_key_rate_shock(rates, shock_vector):
    """
    Apply maturity-specific rate shocks.
    """
    return rates + shock_vector


def run_irrbb_scenario(cashflows, times, rates, rate_times, shock_vector):
    """
    Run IRRBB scenario and return PV impact.
    """

    base_pv = present_value(cashflows, times, rates, rate_times)

    shocked_rates = apply_key_rate_shock(rates, shock_vector)
    shocked_pv = present_value(cashflows, times, shocked_rates, rate_times)

    return {
        "base_pv": base_pv,
        "shocked_pv": shocked_pv,
        "delta_pv": shocked_pv - base_pv
    }


def compute_dv01(cashflows, times, rates, rate_times, bump=0.0001):
    """
    Numerical DV01 using finite difference.
    """

    base_pv = present_value(cashflows, times, rates, rate_times)

    bumped_rates = rates + bump
    bumped_pv = present_value(cashflows, times, bumped_rates, rate_times)

    return bumped_pv - base_pv


def compute_dv01_analytical(cashflows, times, rates, rate_times, bump=0.0001):
    """
    Analytical DV01 under continuous compounding.
    """

    interpolated_rates = np.interp(times, rate_times, rates)

    discount_factors = np.exp(-interpolated_rates * times)

    derivative = -times * cashflows * discount_factors

    return np.sum(derivative) * bump


def compute_key_rate_dv01(cashflows, times, rates, rate_times, bump=0.0001):
    """
    Compute key rate DV01 for each curve bucket.
    """

    base_pv = present_value(cashflows, times, rates, rate_times)

    krd = []

    for i in range(len(rates)):
        bumped_rates = rates.copy()
        bumped_rates[i] += bump

        bumped_pv = present_value(cashflows, times, bumped_rates, rate_times)

        krd.append(bumped_pv - base_pv)

    return np.array(krd)


def portfolio_present_value(instruments, rates, rate_times):
    """
    Compute total portfolio PV.
    instruments: list of dicts with keys:
        - 'cashflows'
        - 'times'
    """

    total_pv = 0.0

    for inst in instruments:
        total_pv += present_value(
            inst["cashflows"],
            inst["times"],
            rates,
            rate_times
        )

    return total_pv


def portfolio_key_rate_dv01(instruments, rates, rate_times, bump=0.0001):
    """
    Compute portfolio key rate DV01 vector.
    """

    base_pv = portfolio_present_value(instruments, rates, rate_times)

    krd = []

    for i in range(len(rates)):
        bumped_rates = rates.copy()
        bumped_rates[i] += bump

        bumped_pv = portfolio_present_value(instruments, bumped_rates, rate_times)

        krd.append(bumped_pv - base_pv)

    return np.array(krd)


def compute_ear(instruments, rates, rate_times, horizon, shock_vector):
    """
    Compute Earnings-at-Risk (EaR) over a given horizon.

    horizon: time window (in years)
    """

    base_income = 0.0
    shocked_income = 0.0

    shocked_rates = rates + shock_vector

    for inst in instruments:

        cashflows = inst["cashflows"]
        times = inst["times"]

        # Only include cashflows within horizon
        mask = times <= horizon

        if np.any(mask):

            interp_base = np.interp(times[mask], rate_times, rates)
            interp_shocked = np.interp(times[mask], rate_times, shocked_rates)

            base_income += np.sum(
                cashflows[mask] * np.exp(-interp_base * times[mask])
            )

            shocked_income += np.sum(
                cashflows[mask] * np.exp(-interp_shocked * times[mask])
            )

    return {
        "base_income": base_income,
        "shocked_income": shocked_income,
        "delta_income": shocked_income - base_income
    }


def compute_convexity(cashflows, times, rates, rate_times, bump=0.0001):
    """
    Numerical convexity using central difference.
    """

    base_pv = present_value(cashflows, times, rates, rate_times)

    rates_up = rates + bump
    rates_down = rates - bump

    pv_up = present_value(cashflows, times, rates_up, rate_times)
    pv_down = present_value(cashflows, times, rates_down, rate_times)

    convexity = (pv_up - 2 * base_pv + pv_down) / (bump ** 2)

    return convexity


def generate_basel_shock_scenarios(rate_times, shock_size=0.02):
    """
    Generate simplified Basel-style IRRBB shock scenarios.

    shock_size: e.g. 0.02 = 200bps
    """

    n = len(rate_times)

    # Parallel shocks
    parallel_up = np.full(n, shock_size)
    parallel_down = np.full(n, -shock_size)

    # Steepener: short down, long up
    steepener = np.linspace(-shock_size, shock_size, n)

    # Flattener: short up, long down
    flattener = np.linspace(shock_size, -shock_size, n)

    return {
        "parallel_up": parallel_up,
        "parallel_down": parallel_down,
        "steepener": steepener,
        "flattener": flattener
    }