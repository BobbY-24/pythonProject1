
import numpy as np
import pandas as pd
from typing import Dict, Optional
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Action(Enum):
    """Trading actions"""
    BUY = 1
    SELL = -1
    HOLD = 0

def calculate_cumulative_returns(returns: pd.Series) -> float:
    """Calculate cumulative returns from a series of daily returns"""
    return (1 + returns).prod() - 1

def calculate_annualized_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return based on daily returns"""
    mean_return = returns.mean()
    return (1 + mean_return) ** periods_per_year - 1

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio of returns"""
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else 0

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + cumulative_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Optimized RSI calculation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))
