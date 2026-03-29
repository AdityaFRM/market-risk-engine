# Market Risk Engine

A modular Python-based quantitative risk engine implementing core market risk and model validation workflows.

## Key Features
- Monte Carlo Value-at-Risk (VaR)
- Expected Shortfall (ES)
- Backtesting (Violation Analysis / Kupiec)
- Stress Testing & Scenario Analysis
- Sensitivity Analysis

## Architecture
- src/simulation.py → scenario generation
- src/var.py → VaR & ES computation
- src/backtest.py → model validation
- src/stress.py → stress scenarios
- src/portfolio.py → portfolio construction

## Demo
See `notebooks/risk_engine_demo.ipynb` for full workflow demonstration.

## Use Case
Designed to replicate real-world risk analytics and model validation workflows used in investment banks.
