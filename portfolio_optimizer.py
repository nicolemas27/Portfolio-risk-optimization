"""
Portfolio Risk & Optimization Engine

A comprehensive quantitative finance toolkit for portfolio risk assessment,
optimization, and scenario analysis using modern portfolio theory.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class AdvancedRiskManagement:
    """
    Advanced portfolio risk management and optimization engine.
    
    This class implements modern portfolio theory concepts including:
    - Value at Risk (VaR) and Conditional VaR calculation
    - Sharpe ratio and maximum drawdown metrics
    - Monte Carlo portfolio optimization
    - Alpha/Beta benchmark analysis
    - Scenario stress testing
    
    Attributes:
        weights (np.ndarray): Portfolio asset weights (normalized to sum to 1.0)
        tickers (List[str]): Asset ticker symbols
        data (pd.DataFrame): Historical price data
        returns (pd.DataFrame): Daily returns for each asset
        port_returns (pd.Series): Portfolio-weighted daily returns
        risk_free_rate (float): Annual risk-free rate for Sharpe calculation
    
    Example:
        >>> risk = AdvancedRiskManagement(
        ...     portfolio=[0.4, 0.3, 0.3],
        ...     tickers=['AAPL', 'MSFT', 'TLT'],
        ...     start_date="2022-01-01"
        ... )
        >>> var_95 = risk.calculate_var(confidence_level=0.95)
        >>> print(f"95% VaR: {var_95:.2%}")
    """
    
    def __init__(
        self,
        portfolio: List[float],
        tickers: List[str],
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        risk_free_rate: float = 0.04
    ):
        """
        Initialize the risk management engine with portfolio parameters.
        
        Args:
            portfolio: List of asset weights (will be normalized to sum to 1.0)
            tickers: List of Yahoo Finance ticker symbols
            start_date: Historical data start date (YYYY-MM-DD format)
            end_date: Historical data end date (defaults to today)
            risk_free_rate: Annual risk-free rate (default: 4%)
        
        Raises:
            ValueError: If portfolio and tickers have different lengths
            ValueError: If any weight is negative
        """
        if len(portfolio) != len(tickers):
            raise ValueError(
                f"Portfolio weights ({len(portfolio)}) and tickers ({len(tickers)}) "
                "must have the same length"
            )
        
        if any(w < 0 for w in portfolio):
            raise ValueError("Portfolio weights cannot be negative (short selling disabled)")

        total = np.sum(portfolio)
        if total <= 0:
            raise ValueError("Total portfolio weight must be greater than zero")
        self.weights = np.array(portfolio) / total
        
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        
        # Download market data
        print(f"Downloading historical data for {', '.join(tickers)}...")
        self.data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )['Close']
        
        # Handle single ticker case (yfinance returns Series instead of DataFrame)
        if len(tickers) == 1:
            if isinstance(self.data, pd.Series):
                self.data = self.data.to_frame(name=tickers[0])
        
        # Calculate daily returns
        self.returns = self.data.pct_change().dropna()
        
        # Calculate portfolio-weighted returns
        self.port_returns = self.returns.dot(self.weights)
        
        print(f"✓ Loaded {len(self.data)} trading days from {self.data.index[0].date()} "
              f"to {self.data.index[-1].date()}")
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical simulation method.
        
        VaR estimates the maximum expected loss over a given time horizon
        at a specified confidence level. For example, a 95% VaR of -2%
        means there is a 5% chance of losing more than 2% in a single day.
        
        Formula:
            VaR_α = -F^(-1)(1 - α)
        
        where F^(-1) is the inverse CDF of the return distribution.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Value at Risk as a decimal (e.g., -0.02 for -2%)
        
        Note:
            Returns negative value representing potential loss.
            Uses empirical quantile from historical data (non-parametric).
        """
        return np.percentile(self.port_returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        CVaR measures the average loss on days when VaR is exceeded,
        providing a more conservative risk measure than VaR alone.
        
        Formula:
            CVaR_α = E[R | R ≤ -VaR_α]
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Conditional VaR as a decimal
        
        Note:
            CVaR is also known as Expected Shortfall (ES) or Average VaR.
            It captures tail risk better than VaR by averaging extreme losses.
        """
        var = self.calculate_var(confidence_level)
        return self.port_returns[self.port_returns <= var].mean()
    
    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio for the portfolio.
        
        The Sharpe ratio measures risk-adjusted return by comparing
        excess return (above risk-free rate) to volatility.
        
        Formula:
            Sharpe = (R_p - R_f) / σ_p
        
        where:
            R_p = annualized portfolio return
            R_f = risk-free rate
            σ_p = annualized volatility
        
        Returns:
            Annualized Sharpe ratio
        
        Interpretation:
            < 1.0: Suboptimal risk-adjusted returns
            1.0 - 2.0: Good performance
            > 2.0: Excellent performance
        
        Note:
            Assumes 252 trading days per year for annualization.
        """
        avg_return = self.port_returns.mean() * 252
        std_dev = self.port_returns.std() * np.sqrt(252)
        
        if std_dev == 0:
            return 0.0
        
        return (avg_return - self.risk_free_rate) / std_dev
    
    def calculate_drawdown(self) -> float:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Maximum drawdown measures the largest percentage drop from a
        historical peak to a subsequent trough, indicating worst-case
        capital erosion during the analysis period.
        
        Formula:
            MaxDD = min((W_t - W_peak) / W_peak)
        
        where W_t is the wealth index at time t.
        
        Returns:
            Maximum drawdown as a decimal (negative value)
        
        Example:
            A max drawdown of -0.25 means the portfolio lost 25% from
            its peak value at some point during the period.
        """
        wealth_index = (1 + self.port_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        return drawdown.min()
    
    def calculate_alpha_beta(
        self,
        benchmark_ticker: str = "SPY"
    ) -> Tuple[float, float]:
        """
        Calculate alpha and beta relative to a market benchmark.
        
        Beta measures systematic risk (market sensitivity), while alpha
        measures excess return beyond what the Capital Asset Pricing Model
        (CAPM) predicts.
        
        Formulas:
            β = Cov(R_p, R_m) / Var(R_m)
            α = R_p - [R_f + β(R_m - R_f)]
        
        Args:
            benchmark_ticker: Ticker symbol for benchmark (default: SPY)
        
        Returns:
            Tuple of (beta, annualized_alpha)
        
        Interpretation:
            Beta < 1: Less volatile than market
            Beta = 1: Moves with market
            Beta > 1: More volatile than market
            
            Alpha > 0: Outperformance vs benchmark
            Alpha < 0: Underperformance vs benchmark
        
        Note:
            Uses linear regression: R_p = α + βR_m + ε
        """
        # Download benchmark data for same period
        bench_data = yf.download(
            benchmark_ticker,
            start=self.data.index[0],
            end=self.data.index[-1],
            auto_adjust=True,
            progress=False
        )['Close']
        
        bench_ret = bench_data.pct_change().dropna()
        
        # Align portfolio and benchmark returns
        combined = pd.concat([self.port_returns, bench_ret], axis=1).dropna()
        combined.columns = ['Portfolio', 'Benchmark']
        
        # Linear regression: portfolio_return = alpha + beta * benchmark_return
        beta, alpha = np.polyfit(combined['Benchmark'], combined['Portfolio'], 1)
        
        # Annualize alpha (daily alpha * 252 trading days)
        ann_alpha = alpha * 252
        
        return beta, ann_alpha
    
    def optimize_portfolio(
        self,
        num_portfolios: int = 2000,
        objective: str = "sharpe"
    ) -> Tuple[pd.Series, plt.Figure]:
        """
        Find optimal portfolio weights using Monte Carlo simulation.
        
        Generates random portfolio weight combinations to approximate
        the Efficient Frontier and identify the optimal allocation based
        on specified objective (maximize Sharpe or minimize volatility).
        
        Args:
            num_portfolios: Number of random portfolios to simulate
            objective: Optimization objective ('sharpe' or 'volatility')
        
        Returns:
            Tuple of:
                - Series with optimal portfolio metrics and weights
                - Matplotlib figure showing the Efficient Frontier
        
        Raises:
            ValueError: If objective is not 'sharpe' or 'volatility'
        
        Note:
            Assumes no short-selling (all weights >= 0).
            Does not use gradient-based optimization (purely random sampling).
        """
        if objective not in ["sharpe", "volatility"]:
            raise ValueError("Objective must be 'sharpe' or 'volatility'")
        
        results = []
        
        for _ in range(num_portfolios):
            # Generate random weights that sum to 1.0
            w = np.random.random(len(self.weights))
            w /= np.sum(w)
            
            # Calculate portfolio metrics
            p_ret = self.returns.dot(w).mean() * 252  # Annualized return
            p_std = self.returns.dot(w).std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate Sharpe ratio
            if p_std > 0:
                p_sharpe = (p_ret - self.risk_free_rate) / p_std
            else:
                p_sharpe = 0.0
            
            results.append((p_ret, p_std, p_sharpe, w))
        
        # Create DataFrame of results
        res_df = pd.DataFrame(
            results,
            columns=['Return', 'Volatility', 'Sharpe', 'Weights']
        )
        
        # Find optimal portfolio based on objective
        if objective == "sharpe":
            best = res_df.iloc[res_df['Sharpe'].idxmax()]
            opt_label = "Max Sharpe"
        else:  # volatility
            best = res_df.iloc[res_df['Volatility'].idxmin()]
            opt_label = "Min Volatility"
        
        # Create Efficient Frontier visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of all portfolios
        sc = ax.scatter(
            res_df['Volatility'],
            res_df['Return'],
            c=res_df['Sharpe'],
            cmap='viridis',
            alpha=0.5,
            edgecolors='none'
        )
        
        # Highlight optimal portfolio
        ax.scatter(
            best['Volatility'],
            best['Return'],
            color='red',
            marker='*',
            s=500,
            label=opt_label,
            edgecolors='black',
            linewidths=1.5,
            zorder=5
        )
        
        # Formatting
        plt.colorbar(sc, label='Sharpe Ratio', ax=ax)
        ax.set_xlabel('Annualized Volatility (Risk)', fontsize=12)
        ax.set_ylabel('Annualized Return', fontsize=12)
        ax.set_title('Efficient Frontier - Monte Carlo Optimization', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return best, fig
    
    def find_min_volatility_weights(
        self,
        num_portfolios: int = 2000
    ) -> pd.Series:
        """
        Find portfolio weights that minimize volatility (risk).
        
        This is a convenience method that calls optimize_portfolio with
        the 'volatility' objective. Useful for conservative investors
        seeking capital preservation over maximum returns.
        
        Args:
            num_portfolios: Number of random portfolios to simulate
        
        Returns:
            Series containing optimal weights and metrics
        """
        results = []
        
        for _ in range(num_portfolios):
            w = np.random.random(len(self.weights))
            w /= np.sum(w)
            
            p_ret = self.returns.dot(w).mean() * 252
            p_std = self.returns.dot(w).std() * np.sqrt(252)
            
            results.append((p_ret, p_std, w))
        
        res_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Weights'])
        min_vol = res_df.iloc[res_df['Volatility'].idxmin()]
        
        return min_vol
    
    def scenario_test(
        self,
        shocks: Dict[str, float],
        scenario_name: str = "Custom Scenario"
    ) -> Dict[str, float]:
        """
        Perform scenario-based stress testing on the portfolio.
        
        Simulates portfolio impact under hypothetical market conditions
        by applying specified percentage changes to each asset.
        
        Args:
            shocks: Dictionary mapping ticker symbols to percentage changes
                   (e.g., {'AAPL': -0.20, 'TLT': 0.05})
            scenario_name: Descriptive name for the scenario
        
        Returns:
            Dictionary containing:
                - 'scenario_name': Name of the scenario
                - 'portfolio_impact': Weighted portfolio impact
                - 'var_95': Current 95% VaR for comparison
                - 'exceeds_var': Boolean indicating if impact exceeds VaR
        
        Example:
            >>> shocks = {'AAPL': -0.25, 'MSFT': -0.20, 'TLT': 0.08}
            >>> result = risk.scenario_test(shocks, "Tech Crash 2025")
            >>> print(f"Impact: {result['portfolio_impact']:.2%}")
        """
        # Validate that all tickers have shock values
        missing_tickers = set(self.tickers) - set(shocks.keys())
        if missing_tickers:
            raise ValueError(f"Missing shock values for tickers: {missing_tickers}")
        
        # Calculate weighted impact
        impact = sum(
            self.weights[i] * shocks[ticker]
            for i, ticker in enumerate(self.tickers)
        )
        
        # Compare to VaR
        var_95 = self.calculate_var(0.95)
        exceeds_var = abs(impact) > abs(var_95)
        
        return {
            'scenario_name': scenario_name,
            'portfolio_impact': impact,
            'var_95': var_95,
            'exceeds_var': exceeds_var,
            'individual_impacts': {
                ticker: self.weights[i] * shocks[ticker]
                for i, ticker in enumerate(self.tickers)
            }
        }
    
    def plot_analysis(
        self,
        benchmark_ticker: str = "SPY"
    ) -> plt.Figure:
        """
        Generate comprehensive performance analysis plots.
        
        Creates a two-panel visualization:
        1. Portfolio growth vs benchmark (cumulative returns)
        2. Distribution of daily returns with VaR threshold
        
        Args:
            benchmark_ticker: Ticker symbol for benchmark comparison
        
        Returns:
            Matplotlib figure object
        """
        # Download benchmark data
        benchmark_data = yf.download(
            benchmark_ticker,
            start=self.data.index[0],
            end=self.data.index[-1],
            auto_adjust=True,
            progress=False
        )['Close']
        
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        # Calculate growth indices (cumulative returns)
        benchmark_growth = (1 + benchmark_returns).cumprod()
        portfolio_growth = (1 + self.port_returns).cumprod()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: Growth comparison
        ax1.plot(
            portfolio_growth.index,
            portfolio_growth.values,
            label="Portfolio",
            color='#2E86AB',
            linewidth=2
        )
        ax1.plot(
            benchmark_growth.index,
            benchmark_growth.values,
            label=f"Benchmark ({benchmark_ticker})",
            color='#A23B72',
            linestyle='--',
            linewidth=2
        )
        ax1.set_title("Cumulative Growth: Portfolio vs Benchmark", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Date", fontsize=10)
        ax1.set_ylabel("Growth Index (Base = 1.0)", fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Return distribution with VaR
        ax2.hist(
            self.port_returns,
            bins=50,
            alpha=0.7,
            color='#F18F01',
            edgecolor='black',
            label="Daily Returns"
        )
        
        var_95 = self.calculate_var(0.95)
        ax2.axvline(
            var_95,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'95% VaR ({var_95:.2%})'
        )
        
        ax2.set_title("Distribution of Daily Returns", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Daily Return", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation(self) -> plt.Figure:
        """
        Generate asset correlation heatmap.
        
        Visualizes the correlation matrix of asset returns to assess
        diversification benefits. Low or negative correlations indicate
        better diversification potential.
        
        Returns:
            Matplotlib figure object with correlation heatmap
        
        Note:
            Correlation ranges from -1 (perfect negative) to +1 (perfect positive).
            Values near 0 indicate little linear relationship.
        """
        corr = self.returns.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title("Asset Correlation Matrix", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of portfolio metrics.
        
        Returns:
            Formatted string containing all key risk metrics
        """
        var_95 = self.calculate_var(0.95)
        cvar_95 = self.calculate_cvar(0.95)
        sharpe = self.calculate_sharpe_ratio()
        max_dd = self.calculate_drawdown()
        beta, alpha = self.calculate_alpha_beta()
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║         PORTFOLIO RISK & PERFORMANCE REPORT              ║
╠══════════════════════════════════════════════════════════╣
║ Portfolio Composition:                                   ║
"""
        
        for ticker, weight in zip(self.tickers, self.weights):
            report += f"║   {ticker:6s}: {weight:6.2%}                                        ║\n"
        
        report += f"""╠══════════════════════════════════════════════════════════╣
║ Risk Metrics:                                            ║
║   Value at Risk (95%):        {var_95:8.2%}              ║
║   Conditional VaR (95%):      {cvar_95:8.2%}              ║
║   Maximum Drawdown:           {max_dd:8.2%}              ║
╠══════════════════════════════════════════════════════════╣
║ Performance Metrics:                                     ║
║   Sharpe Ratio:               {sharpe:8.2f}              ║
║   Beta (vs SPY):              {beta:8.2f}              ║
║   Annualized Alpha:           {alpha:8.2%}              ║
╠══════════════════════════════════════════════════════════╣
║ Data Period:                                             ║
║   Start: {str(self.data.index[0].date()):12s}                                ║
║   End:   {str(self.data.index[-1].date()):12s}                                ║
║   Days:  {len(self.data):5d}                                            ║
╚══════════════════════════════════════════════════════════╝
"""
        return report


# Convenience functions for quick analysis

def quick_analysis(
    tickers: List[str],
    weights: Optional[List[float]] = None,
    start_date: str = "2022-01-01"
) -> AdvancedRiskManagement:
    """
    Perform quick portfolio analysis with default parameters.
    
    Args:
        tickers: List of ticker symbols
        weights: List of weights (defaults to equal-weighted)
        start_date: Start date for historical data
    
    Returns:
        Initialized AdvancedRiskManagement object
    """
    if weights is None:
        weights = [1.0 / len(tickers)] * len(tickers)
    
    return AdvancedRiskManagement(weights, tickers, start_date)


if __name__ == "__main__":
    # Example usage
    print("Portfolio Risk & Optimization Engine - Example Analysis\n")
    
    # Define portfolio
    tickers = ['AAPL', 'MSFT', 'TLT', 'GLD']
    weights = [0.35, 0.25, 0.30, 0.10]
    
    # Initialize
    risk = AdvancedRiskManagement(
        portfolio=weights,
        tickers=tickers,
        start_date="2022-01-01"
    )
    
    # Print comprehensive report
    print(risk.generate_report())
    
    # Scenario test
    print("\n" + "="*60)
    print("SCENARIO TEST: Tech Meltdown")
    print("="*60)
    
    scenario_result = risk.scenario_test(
        shocks={'AAPL': -0.20, 'MSFT': -0.18, 'TLT': 0.05, 'GLD': 0.08},
        scenario_name="Tech Meltdown 2026"
    )
    
    print(f"\nScenario: {scenario_result['scenario_name']}")
    print(f"Portfolio Impact: {scenario_result['portfolio_impact']:.2%}")
    print(f"Current VaR (95%): {scenario_result['var_95']:.2%}")
    print(f"Exceeds VaR: {'⚠️ YES' if scenario_result['exceeds_var'] else '✓ NO'}")