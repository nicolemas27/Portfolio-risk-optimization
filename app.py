"""
Portfolio Risk & Optimization Engine - Streamlit Dashboard

Interactive web interface for portfolio analysis, optimization, and stress testing.
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict
import warnings

# Import custom portfolio optimizer module
from portfolio_optimizer import AdvancedRiskManagement

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Portfolio Risk & Optimization Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Portfolio Risk & Optimization Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced quantitative analysis for data-driven portfolio management</p>', unsafe_allow_html=True)

# GLOSSARY EXPANDER


with st.expander("📖 Financial Metrics Glossary", expanded=False):
    st.markdown("""
    ### Risk Metrics
    
    **Value at Risk (VaR)**  
    Statistical estimate of the maximum expected loss over a given time period at a specified confidence level.  
    *Example:* 95% VaR of -2% means there's only a 5% chance of losing more than 2% on any given day.
    
    **Conditional Value at Risk (CVaR)**  
    Average loss on days when VaR threshold is exceeded. Also known as Expected Shortfall.  
    *Formula:* CVaR = E[Loss | Loss > VaR]
    
    **Sharpe Ratio**  
    Measures risk-adjusted returns by comparing excess return to volatility.  
    *Formula:* (Portfolio Return - Risk-Free Rate) / Portfolio Volatility  
    *Interpretation:* >1.0 = Good, >2.0 = Excellent
    
    **Maximum Drawdown**  
    Largest peak-to-trough decline in portfolio value during the analysis period.  
    *Use case:* Measures worst-case capital erosion.
    
    ### Benchmark Metrics
    
    **Beta (β)**  
    Measures portfolio volatility relative to the market (typically S&P 500).  
    - β < 1: Less volatile than market  
    - β = 1: Moves with market  
    - β > 1: More volatile than market
    
    **Alpha (α)**  
    Excess return beyond what the Capital Asset Pricing Model (CAPM) predicts.  
    *Interpretation:* Positive α = Outperformance relative to risk taken
    
    ### Optimization
    
    **Efficient Frontier**  
    Set of optimal portfolios offering the highest expected return for each level of risk.  
    *Method:* Monte Carlo simulation of random weight combinations.
    
    **Maximum Sharpe Portfolio**  
    Portfolio allocation that maximizes risk-adjusted returns.
    
    **Minimum Volatility Portfolio**  
    Conservative allocation that minimizes overall portfolio risk.
    """)

# ===========================
# SIDEBAR - INPUT CONTROLS
# ===========================

with st.sidebar:
    st.header("⚙️ Portfolio Configuration")
    
    # Ticker input
    st.subheader("Asset Selection")
    ticker_input = st.text_input(
        "Ticker Symbols (comma-separated)",
        value="AAPL,MSFT,TLT,GLD",
        help="Enter Yahoo Finance ticker symbols. Examples: AAPL (Apple), SPY (S&P 500 ETF), TLT (Bonds), GLD (Gold)"
    )
    
    # Parse tickers
    try:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if len(tickers) == 0:
            st.error("⚠️ Please enter at least one ticker symbol")
            st.stop()
    except Exception as e:
        st.error(f"⚠️ Error parsing tickers: {e}")
        st.stop()
    
    # Weight allocation
    st.subheader("Portfolio Weights")
    st.caption("Allocate percentage of capital to each asset. Weights will be normalized to sum to 100%.")
    
    weights = []
    default_weight = 1.0 / len(tickers)
    
    # Create weight sliders for each ticker
    for ticker in tickers:
        weight = st.slider(
            f"{ticker}",
            min_value=0.0,
            max_value=1.0,
            value=default_weight,
            step=0.05,
            key=f"weight_{ticker}",
            help=f"Allocation percentage for {ticker}"
        )
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        
        # Display normalized weights
        st.info("**Normalized Allocation:**")
        for ticker, weight in zip(tickers, weights):
            st.write(f"• {ticker}: {weight:.1%}")
    else:
        st.error("⚠️ Total weight cannot be zero. Using equal weights.")
        weights = [default_weight] * len(tickers)
    
    # Date range selection
    st.subheader("Analysis Period")
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=730),  # 2 years default
        help="Historical data start date. Longer periods provide more stable estimates."
    )
    
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        help="Historical data end date. Leave as today for most recent data."
    )
    
    # Risk-free rate
    st.subheader("Advanced Settings")
    risk_free_rate = st.number_input(
        "Risk-Free Rate (Annual)",
        min_value=0.0,
        max_value=0.10,
        value=0.04,
        step=0.005,
        format="%.3f",
        help="Annual risk-free rate for Sharpe ratio calculation (e.g., 3-month T-bill rate)"
    )
    
    # Run analysis button
    st.markdown("---")
    run_analysis = st.button(
        "🚀 Run Analysis",
        type="primary",
        help="Download market data and calculate all portfolio metrics"
    )

# ===========================
# MAIN ANALYSIS SECTION
# ===========================

if run_analysis:
    try:
        with st.spinner("⏳ Downloading market data and computing portfolio metrics..."):
            # Initialize risk management engine
            risk_engine = AdvancedRiskManagement(
                portfolio=weights,
                tickers=tickers,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                risk_free_rate=risk_free_rate
            )
            
            # Store in session state for scenario testing
            st.session_state['risk_engine'] = risk_engine
            st.session_state['tickers'] = tickers
            st.session_state['weights'] = weights
            st.session_state['analysis_complete'] = True
        
        st.success("✅ Analysis complete!")
    
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.info("💡 **Troubleshooting Tips:**\n"
                "- Verify ticker symbols are valid on Yahoo Finance\n"
                "- Check internet connection\n"
                "- Ensure date range is valid")
        st.stop()

# Display results if analysis has been run
if st.session_state.get('analysis_complete', False):
    risk_engine = st.session_state['risk_engine']
    tickers = st.session_state['tickers']
    weights = st.session_state['weights']
    
    # ===========================
    # RISK METRICS DASHBOARD
    # ===========================
    
    st.markdown("---")
    st.header("📊 Risk Metrics Dashboard")
    
    # Calculate metrics
    var_95 = risk_engine.calculate_var(confidence_level=0.95)
    cvar_95 = risk_engine.calculate_cvar(confidence_level=0.95)
    sharpe = risk_engine.calculate_sharpe_ratio()
    max_dd = risk_engine.calculate_drawdown()
    beta, alpha = risk_engine.calculate_alpha_beta()
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📉 Value at Risk (95%)",
            f"{var_95:.2%}",
            help="Maximum expected daily loss with 95% confidence"
        )
    
    with col2:
        st.metric(
            "⚠️ Conditional VaR (95%)",
            f"{cvar_95:.2%}",
            delta=f"{(cvar_95/var_95 - 1):.1%} vs VaR",
            help="Average loss when VaR is exceeded (tail risk)"
        )
    
    with col3:
        sharpe_color = "normal"
        if sharpe > 2.0:
            sharpe_color = "inverse"
        elif sharpe < 1.0:
            sharpe_color = "off"
        
        st.metric(
            "📈 Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="Excellent" if sharpe > 2.0 else ("Good" if sharpe > 1.0 else "Below Target"),
            help="Risk-adjusted return measure"
        )
    
    with col4:
        st.metric(
            "📊 Maximum Drawdown",
            f"{max_dd:.2%}",
            help="Largest peak-to-trough decline"
        )
    
    # Alpha and Beta display
    st.markdown("### 🎯 Benchmark Comparison (vs S&P 500)")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        beta_interpretation = ""
        if beta < 0.8:
            beta_interpretation = "Low Volatility (Defensive)"
        elif beta < 1.2:
            beta_interpretation = "Market-Like Volatility"
        else:
            beta_interpretation = "High Volatility (Aggressive)"
        
        st.metric(
            "Beta (β)",
            f"{beta:.2f}",
            delta=beta_interpretation
        )
    
    with col_b:
        alpha_interpretation = "Outperforming" if alpha > 0 else "Underperforming"
        st.metric(
            "Annualized Alpha (α)",
            f"{alpha:.2%}",
            delta=alpha_interpretation
        )
    
    # Interpretation box
    st.markdown(f"""
    <div class="info-box">
        <strong>💡 Interpretation:</strong><br>
        Your portfolio has a <strong>β of {beta:.2f}</strong>, indicating it is 
        <strong>{'more' if beta > 1 else 'less'} volatile</strong> than the S&P 500. 
        The <strong>α of {alpha:.2%}</strong> suggests you are 
        <strong>{'outperforming' if alpha > 0 else 'underperforming'}</strong> 
        relative to the market risk you're taking.
    </div>
    """, unsafe_allow_html=True)
    
    # ===========================
    # PERFORMANCE VISUALIZATION
    # ===========================
    
    st.markdown("---")
    st.header("📈 Performance Analysis")
    
    try:
        performance_fig = risk_engine.plot_analysis(benchmark_ticker="SPY")
        st.pyplot(performance_fig)
        
        st.caption("""
        **Left Panel:** Cumulative growth comparison between your portfolio and the S&P 500 benchmark.  
        **Right Panel:** Distribution of daily returns with 95% VaR threshold marked in red.
        """)
    except Exception as e:
        st.warning(f"⚠️ Could not generate performance plots: {e}")
    
    # ===========================
    # CORRELATION ANALYSIS
    # ===========================
    
    st.markdown("---")
    st.header("🔗 Asset Correlation Matrix")
    
    try:
        correlation_fig = risk_engine.plot_correlation()
        st.pyplot(correlation_fig)
        
        st.caption("""
        **Diversification Insight:** Values close to +1 indicate assets move together (low diversification benefit).  
        Values close to -1 indicate inverse movement (strong diversification benefit).  
        Values near 0 indicate little relationship.
        """)
    except Exception as e:
        st.warning(f"⚠️ Could not generate correlation heatmap: {e}")
    
    # ===========================
    # PORTFOLIO OPTIMIZATION
    # ===========================
    
    st.markdown("---")
    st.header("🎯 Portfolio Optimization")
    
    st.markdown("""
    Using **Monte Carlo simulation** to explore the Efficient Frontier by generating 
    thousands of random portfolio weight combinations and identifying optimal allocations.
    """)
    
    # Optimization settings
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        num_simulations = st.select_slider(
            "Number of Simulations",
            options=[500, 1000, 2000, 5000, 10000],
            value=2000,
            help="More simulations = better approximation of Efficient Frontier (but slower)"
        )
    
    with col_opt2:
        optimization_objective = st.radio(
            "Optimization Objective",
            options=["Maximum Sharpe Ratio", "Minimum Volatility"],
            help="Choose your optimization goal"
        )
    
    if st.button("⚡ Run Optimization", key="optimize_btn"):
        with st.spinner(f"Running {num_simulations:,} Monte Carlo simulations..."):
            try:
                objective = "sharpe" if optimization_objective == "Maximum Sharpe Ratio" else "volatility"
                
                best_portfolio, frontier_fig = risk_engine.optimize_portfolio(
                    num_portfolios=num_simulations,
                    objective=objective
                )
                
                st.session_state['optimization_result'] = best_portfolio
                st.session_state['frontier_fig'] = frontier_fig
                
                st.success("✅ Optimization complete!")
            
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")
    
    # Display optimization results
    if 'optimization_result' in st.session_state:
        best = st.session_state['optimization_result']
        frontier_fig = st.session_state['frontier_fig']
        
        # Plot Efficient Frontier
        st.pyplot(frontier_fig)
        
        # Display optimal weights
        st.markdown("### 🏆 Optimal Portfolio Allocation")
        
        optimal_df = pd.DataFrame({
            'Asset': tickers,
            'Current Weight': [f"{w:.1%}" for w in weights],
            'Optimal Weight': [f"{w:.1%}" for w in best['Weights']],
            'Change': [f"{(best['Weights'][i] - weights[i]):.1%}" for i in range(len(tickers))]
        })
        
        st.dataframe(optimal_df, use_container_width=True)
        
        # Performance metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Expected Annual Return", f"{best['Return']:.2%}")
        
        with col_m2:
            st.metric("Annual Volatility", f"{best['Volatility']:.2%}")
        
        with col_m3:
            st.metric("Sharpe Ratio", f"{best['Sharpe']:.2f}")
    
    # ===========================
    # TIME SERIES FORECAST
    # ===========================
    
    st.markdown("---")
    st.header("🔮 5-Day Return Forecast (ARIMA)")
    
    if st.button("📊 Generate Forecast", key="forecast_btn"):
        with st.spinner("Fitting ARIMA model to historical returns..."):
            try:
                from statsmodels.tsa.arima.model import ARIMA
                
                # Fit ARIMA(1,0,1) model
                model = ARIMA(risk_engine.port_returns, order=(1, 0, 1))
                fitted = model.fit()
                
                # Forecast next 5 days
                forecast = fitted.forecast(steps=5)
                forecast_cumulative = (1 + forecast).cumprod() - 1
                
                # Display forecast
                st.markdown("### Daily Return Predictions")
                
                forecast_cols = st.columns(5)
                for i, (col, pred) in enumerate(zip(forecast_cols, forecast)):
                    with col:
                        delta_color = "normal" if pred >= 0 else "inverse"
                        st.metric(
                            f"Day {i+1}",
                            f"{pred:.2%}",
                            delta="📈" if pred >= 0 else "📉"
                        )
                
                st.info(f"**5-Day Cumulative Return Forecast:** {forecast_cumulative.iloc[-1]:.2%}")
                
                st.caption("""
                ⚠️ **Disclaimer:** ARIMA forecasts are statistical estimates based on historical patterns. 
                Actual returns may differ significantly. Not financial advice.
                """)
                
            except ImportError:
                st.error("❌ ARIMA forecasting requires `statsmodels` package.")
                st.code("pip install statsmodels", language="bash")
            
            except Exception as e:
                st.error(f"❌ Forecast failed: {e}")
    
    # ===========================
    # SCENARIO STRESS TESTING
    # ===========================
    
    st.markdown("---")
    st.header("💥 Scenario Stress Testing")
    
    st.markdown("""
    Simulate hypothetical market events to understand your portfolio's resilience.  
    Define percentage changes for each asset under a specific scenario (e.g., market crash, sector rotation).
    """)
    
    with st.expander("🎮 Configure Custom Scenario", expanded=False):
        scenario_name = st.text_input(
            "Scenario Name",
            value="Tech Sector Crash",
            help="Descriptive name for this stress test"
        )
        
        st.markdown("**Asset-Level Shocks (%)**")
        st.caption("Enter expected percentage change for each asset. Negative = loss, Positive = gain.")
        
        shock_cols = st.columns(len(tickers))
        shocks = {}
        
        for i, (col, ticker) in enumerate(zip(shock_cols, tickers)):
            with col:
                default_shock = -0.05 if i < len(tickers)//2 else 0.03
                shocks[ticker] = st.number_input(
                    ticker,
                    min_value=-1.0,
                    max_value=1.0,
                    value=default_shock,
                    step=0.01,
                    format="%.2f",
                    key=f"shock_{ticker}"
                ) / 100  # Convert to decimal
        
        if st.button("⚡ Run Scenario Test", key="scenario_test_btn"):
            try:
                # Convert shocks back to percentage for scenario_test
                shock_dict = {k: v * 100 for k, v in shocks.items()}
                
                result = risk_engine.scenario_test(
                    shocks={k: v for k, v in shocks.items()},
                    scenario_name=scenario_name
                )
                
                st.session_state['scenario_result'] = result
            
            except Exception as e:
                st.error(f"❌ Scenario test failed: {e}")
    
    # Display scenario results
    if 'scenario_result' in st.session_state:
        result = st.session_state['scenario_result']
        
        impact = result['portfolio_impact']
        var_95_val = result['var_95']
        exceeds = result['exceeds_var']
        
        # Impact display
        if exceeds:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ HIGH RISK SCENARIO</strong><br>
                Scenario: <strong>{result['scenario_name']}</strong><br>
                Portfolio Impact: <strong style="color: #DC2626;">{impact:.2%}</strong><br>
                This exceeds your 95% VaR threshold ({var_95_val:.2%})!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ WITHIN RISK TOLERANCE</strong><br>
                Scenario: <strong>{result['scenario_name']}</strong><br>
                Portfolio Impact: <strong>{impact:.2%}</strong><br>
                This is within your 95% VaR bounds ({var_95_val:.2%}).
            </div>
            """, unsafe_allow_html=True)
        
        # Asset-level breakdown
        st.markdown("**Asset-Level Impact Breakdown:**")
        
        impact_df = pd.DataFrame([
            {
                'Asset': ticker,
                'Weight': f"{weights[i]:.1%}",
                'Shock': f"{shocks[ticker]*100:.1%}",
                'Contribution': f"{result['individual_impacts'][ticker]:.2%}"
            }
            for i, ticker in enumerate(tickers)
        ])
        
        st.dataframe(impact_df, use_container_width=True)
        
        if st.button("🗑️ Clear Scenario Results"):
            del st.session_state['scenario_result']
            st.rerun()

else:
    # Initial state message
    st.info("""
    👈 **Get Started:**
    1. Configure your portfolio in the sidebar
    2. Adjust tickers, weights, and date range
    3. Click **"Run Analysis"** to begin
    
    The engine will download live market data and compute comprehensive risk metrics,
    optimization results, and stress test scenarios.
    """)

# ===========================
# FOOTER
# ===========================

st.markdown("---")
st.caption("""
⚠️ **Important Disclaimer:**  
This application is for educational and research purposes only. It does not constitute 
financial advice. Past performance does not guarantee future results. Consult with a 
qualified financial advisor before making investment decisions.

**Data Source:** Yahoo Finance | **Framework:** Modern Portfolio Theory (Markowitz, 1952)
""")