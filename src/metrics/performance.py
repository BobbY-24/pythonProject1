import numpy as np
import pandas as pd

def cumulative_returns(portfolio_values: pd.Series):
    """Cumulative return over the period."""
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

def annualized_returns(portfolio_values: pd.Series, periods_per_year=252):
    """Annualized return assuming daily data."""
    total_return = cumulative_returns(portfolio_values)
    n_periods = len(portfolio_values)
    years = n_periods / periods_per_year
    return (1 + total_return) ** (1 / years) - 1

def sharpe_ratio(portfolio_values: pd.Series, risk_free_rate=0.0, periods_per_year=252):
    """Annualized Sharpe ratio."""
    returns = portfolio_values.pct_change().dropna()
    excess = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()

def max_drawdown(portfolio_values: pd.Series):
    """Maximum drawdown."""
    roll_max = portfolio_values.cummax()
    drawdown = (portfolio_values - roll_max) / roll_max
    return drawdown.min()
