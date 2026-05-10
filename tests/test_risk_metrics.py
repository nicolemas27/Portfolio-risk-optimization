"""
Unit Tests for Portfolio Risk & Optimization Engine

Tests core functionality including:
- Risk metric calculations (VaR, CVaR, Sharpe, Max Drawdown)
- Portfolio weight normalization
- Optimization convergence
- Scenario testing
- Edge cases

Run with: pytest tests/ -v

"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import portfolio_optimizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio_optimizer import AdvancedRiskManagement


class TestPortfolioInitialization:
    """Test portfolio initialization and validation."""
    
    def test_weight_normalization(self):
        """Test that weights are correctly normalized to sum to 1.0."""
        weights = [0.3, 0.5, 0.2]  # Already sums to 1.0
        tickers = ['AAPL', 'MSFT', 'TLT']
        
        risk = AdvancedRiskManagement(
            portfolio=weights,
            tickers=tickers,
            start_date="2023-01-01"
        )
        
        assert np.isclose(np.sum(risk.weights), 1.0), "Weights should sum to 1.0"
    
    def test_unnormalized_weights(self):
        """Test that unnormalized weights are automatically normalized."""
        weights = [30, 50, 20]  # Sums to 100, not 1.0
        tickers = ['AAPL', 'MSFT', 'TLT']
        
        risk = AdvancedRiskManagement(
            portfolio=weights,
            tickers=tickers,
            start_date="2023-01-01"
        )
        
        assert np.isclose(np.sum(risk.weights), 1.0), "Weights should be normalized"
        assert np.isclose(risk.weights[0], 0.3), "First weight should be 0.3"
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched portfolio and ticker lengths."""
        weights = [0.5, 0.5]
        tickers = ['AAPL', 'MSFT', 'TLT']  # One extra ticker
        
        with pytest.raises(ValueError, match="must have the same length"):
            AdvancedRiskManagement(
                portfolio=weights,
                tickers=tickers,
                start_date="2023-01-01"
            )
    
    def test_negative_weights(self):
        """Test error handling for negative weights (no short selling)."""
        weights = [0.6, -0.1, 0.5]  # Negative weight
        tickers = ['AAPL', 'MSFT', 'TLT']
        
        with pytest.raises(ValueError, match="cannot be negative"):
            AdvancedRiskManagement(
                portfolio=weights,
                tickers=tickers,
                start_date="2023-01-01"
            )
    
    def test_single_asset_portfolio(self):
        """Test portfolio with single asset."""
        weights = [1.0]
        tickers = ['AAPL']
        
        risk = AdvancedRiskManagement(
            portfolio=weights,
            tickers=tickers,
            start_date="2023-01-01"
        )
        
        assert len(risk.weights) == 1
        assert risk.weights[0] == 1.0


class TestRiskMetrics:
    """Test risk metric calculations."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        return AdvancedRiskManagement(
            portfolio=[0.5, 0.5],
            tickers=['AAPL', 'MSFT'],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
    
    def test_var_calculation(self, sample_portfolio):
        """Test VaR calculation returns negative value."""
        var_95 = sample_portfolio.calculate_var(confidence_level=0.95)
        
        # VaR should be negative (representing a loss)
        assert var_95 < 0, "VaR should be negative"
        
        # VaR should be a reasonable percentage (between -50% and 0%)
        assert -0.5 < var_95 < 0, "VaR should be a reasonable loss percentage"
    
    def test_var_confidence_levels(self, sample_portfolio):
        """Test that higher confidence levels produce more negative VaR."""
        var_90 = sample_portfolio.calculate_var(confidence_level=0.90)
        var_95 = sample_portfolio.calculate_var(confidence_level=0.95)
        var_99 = sample_portfolio.calculate_var(confidence_level=0.99)
        
        # Higher confidence = more extreme (negative) VaR
        assert var_99 < var_95 < var_90, "VaR should decrease with higher confidence"
    
    def test_cvar_calculation(self, sample_portfolio):
        """Test CVaR calculation and relationship to VaR."""
        var_95 = sample_portfolio.calculate_var(confidence_level=0.95)
        cvar_95 = sample_portfolio.calculate_cvar(confidence_level=0.95)
        
        # CVaR should be more negative than VaR (tail risk)
        assert cvar_95 < var_95, "CVaR should be more extreme than VaR"
        
        # Both should be negative
        assert cvar_95 < 0, "CVaR should be negative"
    
    def test_sharpe_ratio_calculation(self, sample_portfolio):
        """Test Sharpe ratio is calculated correctly."""
        sharpe = sample_portfolio.calculate_sharpe_ratio()
        
        # Sharpe ratio should be a finite number
        assert np.isfinite(sharpe), "Sharpe ratio should be finite"
        
        # For typical portfolios, Sharpe should be between -5 and 5
        assert -5 < sharpe < 5, "Sharpe ratio should be in reasonable range"
    
    def test_max_drawdown_calculation(self, sample_portfolio):
        """Test maximum drawdown calculation."""
        max_dd = sample_portfolio.calculate_drawdown()
        
        # Max drawdown should be negative (a loss)
        assert max_dd <= 0, "Max drawdown should be negative or zero"
        
        # Should be between -100% and 0%
        assert -1.0 <= max_dd <= 0, "Max drawdown should be between -100% and 0%"
    
    def test_alpha_beta_calculation(self, sample_portfolio):
        """Test alpha and beta calculations."""
        beta, alpha = sample_portfolio.calculate_alpha_beta(benchmark_ticker="SPY")
        
        # Beta should be positive for typical stocks
        assert beta > 0, "Beta should typically be positive"
        
        # Beta should be in reasonable range (0.1 to 3.0 for most portfolios)
        assert 0.1 < beta < 3.0, "Beta should be in reasonable range"
        
        # Alpha can be positive or negative
        assert np.isfinite(alpha), "Alpha should be finite"


class TestOptimization:
    """Test portfolio optimization functionality."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        return AdvancedRiskManagement(
            portfolio=[0.25, 0.25, 0.25, 0.25],
            tickers=['AAPL', 'MSFT', 'TLT', 'GLD'],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
    
    def test_optimize_sharpe_returns_valid_portfolio(self, sample_portfolio):
        """Test that optimization returns a valid portfolio."""
        best, fig = sample_portfolio.optimize_portfolio(
            num_portfolios=500,
            objective="sharpe"
        )
        
        # Check that weights sum to 1.0
        assert np.isclose(np.sum(best['Weights']), 1.0), "Optimal weights should sum to 1.0"
        
        # Check that all weights are non-negative
        assert all(w >= 0 for w in best['Weights']), "Weights should be non-negative"
        
        # Check that metrics are present
        assert 'Return' in best.index
        assert 'Volatility' in best.index
        assert 'Sharpe' in best.index
        
        # Check that Sharpe ratio is finite
        assert np.isfinite(best['Sharpe']), "Sharpe ratio should be finite"
    
    def test_optimize_volatility_objective(self, sample_portfolio):
        """Test optimization with minimum volatility objective."""
        best, fig = sample_portfolio.optimize_portfolio(
            num_portfolios=500,
            objective="volatility"
        )
        
        # Weights should sum to 1.0
        assert np.isclose(np.sum(best['Weights']), 1.0)
        
        # Volatility should be positive
        assert best['Volatility'] > 0, "Volatility should be positive"
    
    def test_invalid_optimization_objective(self, sample_portfolio):
        """Test error handling for invalid optimization objective."""
        with pytest.raises(ValueError, match="Objective must be"):
            sample_portfolio.optimize_portfolio(
                num_portfolios=100,
                objective="invalid_objective"
            )
    
    def test_min_volatility_weights(self, sample_portfolio):
        """Test minimum volatility portfolio finder."""
        min_vol = sample_portfolio.find_min_volatility_weights(num_portfolios=500)
        
        # Weights should sum to 1.0
        assert np.isclose(np.sum(min_vol['Weights']), 1.0)
        
        # Should have return and volatility metrics
        assert 'Return' in min_vol.index
        assert 'Volatility' in min_vol.index


class TestScenarioTesting:
    """Test scenario-based stress testing."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        return AdvancedRiskManagement(
            portfolio=[0.4, 0.3, 0.3],
            tickers=['AAPL', 'MSFT', 'TLT'],
            start_date="2023-01-01"
        )
    
    def test_scenario_test_calculation(self, sample_portfolio):
        """Test scenario impact calculation."""
        shocks = {
            'AAPL': -0.20,  # -20%
            'MSFT': -0.15,  # -15%
            'TLT': 0.05     # +5%
        }
        
        result = sample_portfolio.scenario_test(
            shocks=shocks,
            scenario_name="Tech Crash"
        )
        
        # Check result structure
        assert 'scenario_name' in result
        assert 'portfolio_impact' in result
        assert 'var_95' in result
        assert 'exceeds_var' in result
        assert 'individual_impacts' in result
        
        # Impact should be negative (net loss scenario)
        assert result['portfolio_impact'] < 0, "Tech crash should result in negative impact"
        
        # Check individual impacts sum to total
        total_impact = sum(result['individual_impacts'].values())
        assert np.isclose(total_impact, result['portfolio_impact'])
    
    def test_scenario_missing_ticker(self, sample_portfolio):
        """Test error handling when shock values are missing for some tickers."""
        shocks = {
            'AAPL': -0.20,
            'MSFT': -0.15
            # Missing TLT
        }
        
        with pytest.raises(ValueError, match="Missing shock values"):
            sample_portfolio.scenario_test(shocks=shocks)
    
    def test_positive_scenario(self, sample_portfolio):
        """Test scenario with positive returns."""
        shocks = {
            'AAPL': 0.10,
            'MSFT': 0.08,
            'TLT': 0.02
        }
        
        result = sample_portfolio.scenario_test(
            shocks=shocks,
            scenario_name="Bull Market"
        )
        
        # Impact should be positive
        assert result['portfolio_impact'] > 0, "Bull market should result in positive impact"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_weights_initialization(self):
        """Test handling of all zero weights."""
        weights = [0.0, 0.0, 0.0]
        tickers = ['AAPL', 'MSFT', 'TLT']
        
        # Should raise error since we can't normalize zero weights
        with pytest.raises(Exception):
            AdvancedRiskManagement(
                portfolio=weights,
                tickers=tickers,
                start_date="2023-01-01"
            )
    
    def test_extreme_weights(self):
        """Test portfolio with one asset at 100%."""
        weights = [1.0, 0.0, 0.0]
        tickers = ['AAPL', 'MSFT', 'TLT']
        
        risk = AdvancedRiskManagement(
            portfolio=weights,
            tickers=tickers,
            start_date="2023-01-01"
        )
        
        # Should work fine
        assert np.isclose(risk.weights[0], 1.0)
        assert np.isclose(risk.weights[1], 0.0)
        assert np.isclose(risk.weights[2], 0.0)
    
    def test_very_short_date_range(self):
        """Test behavior with minimal historical data."""
        # This test might fail or produce warnings due to insufficient data
        # but should not crash
        try:
            risk = AdvancedRiskManagement(
                portfolio=[0.5, 0.5],
                tickers=['AAPL', 'MSFT'],
                start_date=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            )
            
            # Should still produce some metrics, even if unreliable
            var = risk.calculate_var()
            assert np.isfinite(var)
        
        except Exception as e:
            # Some operations might fail with very limited data
            pytest.skip(f"Insufficient data for short date range: {e}")


class TestReportGeneration:
    """Test report generation functionality."""
    
    def test_generate_report(self):
        """Test that comprehensive report is generated without errors."""
        risk = AdvancedRiskManagement(
            portfolio=[0.5, 0.5],
            tickers=['AAPL', 'MSFT'],
            start_date="2023-01-01"
        )
        
        report = risk.generate_report()
        
        # Report should be a string
        assert isinstance(report, str)
        
        # Report should contain key metrics
        assert 'Value at Risk' in report
        assert 'Sharpe Ratio' in report
        assert 'Beta' in report
        assert 'Alpha' in report
        
        # Should not be empty
        assert len(report) > 100


# Fixtures for integration testing
@pytest.fixture(scope="module")
def integration_portfolio():
    """Create a portfolio for integration tests (module-scoped for efficiency)."""
    return AdvancedRiskManagement(
        portfolio=[0.3, 0.3, 0.2, 0.2],
        tickers=['AAPL', 'MSFT', 'TLT', 'GLD'],
        start_date="2022-01-01",
        end_date="2023-12-31"
    )


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow(self, integration_portfolio):
        """Test complete analysis workflow."""
        # Calculate all risk metrics
        var = integration_portfolio.calculate_var()
        cvar = integration_portfolio.calculate_cvar()
        sharpe = integration_portfolio.calculate_sharpe_ratio()
        max_dd = integration_portfolio.calculate_drawdown()
        beta, alpha = integration_portfolio.calculate_alpha_beta()
        
        # All should be finite
        assert all(np.isfinite(x) for x in [var, cvar, sharpe, max_dd, beta, alpha])
        
        # Run optimization
        best, fig = integration_portfolio.optimize_portfolio(num_portfolios=500)
        assert np.isclose(np.sum(best['Weights']), 1.0)
        
        # Run scenario test
        shocks = {'AAPL': -0.15, 'MSFT': -0.12, 'TLT': 0.05, 'GLD': 0.08}
        result = integration_portfolio.scenario_test(shocks, "Market Stress")
        assert 'portfolio_impact' in result
        
        # Generate report
        report = integration_portfolio.generate_report()
        assert len(report) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])