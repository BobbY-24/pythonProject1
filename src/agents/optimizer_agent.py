import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.optimize import minimize
from .base import BaseAgent
from utils import Action


class MeanVarianceAgent(BaseAgent):
    """Mean-variance optimization agent"""

    def __init__(self, lookback_period: int = 60, risk_aversion: float = 1.0):
        super().__init__(f"MeanVariance_{lookback_period}_{risk_aversion}")
        self.lookback_period = lookback_period
        self.risk_aversion = risk_aversion

    def decide_action(self, data: pd.DataFrame, **kwargs) -> Action:
        """Single asset decision (not primary use case for this agent)"""
        if len(data) < self.lookback_period:
            return Action.HOLD

        returns = data['Close'].pct_change().dropna()
        recent_returns = returns.iloc[-self.lookback_period:]

        # Simple momentum check
        if recent_returns.mean() > 0:
            return Action.BUY
        else:
            return Action.SELL

    def compute_weights(self, data: pd.DataFrame, symbols: list) -> Dict[str, float]:
        """Compute optimal portfolio weights using mean-variance optimization"""
        if len(data) < self.lookback_period:
            return {symbol: 1.0 / len(symbols) for symbol in symbols}

        # For demonstration, assume data contains multiple symbols
        # In practice, you'd have a multi-symbol dataframe
        returns = data['Close'].pct_change().dropna().iloc[-self.lookback_period:]

        if len(returns) < 10:  # Not enough data
            return {symbol: 1.0 / len(symbols) for symbol in symbols}

        # Calculate expected returns and covariance (simplified for single asset)
        mu = returns.mean() * 252  # Annualized
        sigma2 = returns.var() * 252  # Annualized variance

        # For multiple assets, you'd calculate covariance matrix
        # Here's a simplified version for single asset
        n_assets = len(symbols)
        expected_returns = np.full(n_assets, mu)
        cov_matrix = np.eye(n_assets) * sigma2

        # Optimize portfolio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            # Mean-variance objective
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance)

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = result.x
        else:
            optimal_weights = initial_weights

        return {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}


class RiskParityAgent(BaseAgent):
    """Risk parity portfolio optimization"""

    def __init__(self, lookback_period: int = 60):
        super().__init__(f"RiskParity_{lookback_period}")
        self.lookback_period = lookback_period

    def decide_action(self, data: pd.DataFrame, **kwargs) -> Action:
        """Single asset decision"""
        return Action.HOLD  # Risk parity focuses on portfolio allocation

    def compute_weights(self, data: pd.DataFrame, symbols: list) -> Dict[str, float]:
        """Compute risk parity weights"""
        if len(data) < self.lookback_period:
            return {symbol: 1.0 / len(symbols) for symbol in symbols}

        returns = data['Close'].pct_change().dropna().iloc[-self.lookback_period:]

        if len(returns) < 10:
            return {symbol: 1.0 / len(symbols) for symbol in symbols}

        # Calculate volatility (for single asset case)
        volatility = returns.std()

        # For multiple assets, weights would be inversely proportional to volatility
        # Simplified for single asset case
        n_assets = len(symbols)
        volatilities = np.full(n_assets, volatility)

        # Risk parity: weights inversely proportional to volatility
        inv_vol = 1.0 / volatilities
        weights = inv_vol / np.sum(inv_vol)

        return {symbol: weight for symbol, weight in zip(symbols, weights)}