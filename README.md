# Portfolio Risk & Optimization Engine

A Python-based quantitative finance toolkit for portfolio risk assessment and optimization using Modern Portfolio Theory.

## Overview

This project downloads live market data, calculates risk metrics (VaR, CVaR, Sharpe ratio, maximum drawdown), runs Monte Carlo optimization to find optimal portfolio allocations, and performs scenario-based stress testing.

**Key Finding:** Adding a 40% bond allocation reduced simulated "Tech Meltdown" losses from –8.85% to –3.10%.

## Features

- **Risk Metrics** – VaR, CVaR, Sharpe ratio, maximum drawdown
- **Portfolio Optimization** – Monte Carlo simulation to construct Efficient Frontier
- **Benchmark Analysis** – Alpha and beta vs S&P 500
- **Scenario Testing** – Stress test portfolios against market crash scenarios
- **Interactive Dashboard** – Streamlit web interface with live data

## Installation

```bash
git clone https://github.com/nicolemas27/Portfolio-risk-optimization.git
cd portfolio-risk-optimization
pip install -r requirements.txt
```

## Usage

**Run Jupyter Notebook:**
```bash
jupyter notebook Portfolio_Risk_&_Optimization_Engine.ipynb
```

**Run Streamlit Dashboard:**
```bash
streamlit run app.py
```

**Run Tests:**
```bash
pytest tests/ -v
```

## Tech Stack

- **Data:** `yfinance`, `pandas`, `numpy`
- **Analysis:** `scipy`, `statsmodels`
- **Visualization:** `matplotlib`, `seaborn`
- **Dashboard:** `streamlit`

## Methodology

### Risk Metrics

**Value at Risk (VaR):** Maximum expected loss at 95% confidence
```
VaR = -F^(-1)(1 - α)
```

**Sharpe Ratio:** Risk-adjusted return
```
Sharpe = (R_p - R_f) / σ_p
```

### Optimization

Monte Carlo simulation generates random portfolio weights to approximate the Efficient Frontier and identify optimal allocations.

## License

MIT

## Disclaimer

For educational purposes only. Not financial advice.
