import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .manager import PortfolioManager
from config import config

logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """Enhanced portfolio simulator with weight-based rebalancing"""

    def __init__(self,
                 initial_capital: float = config.INITIAL_BALANCE,
                 transaction_fee: float = config.TRANSACTION_FEE,
                 rebalance_frequency: str = 'monthly'):  # daily, weekly, monthly
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.rebalance_frequency = rebalance_frequency

    def run_simulation(self,
                       data: pd.DataFrame,
                       agent,  # BaseAgent type - avoiding import here
                       symbols: List[str] = None,
                       benchmark_symbol: str = None) -> Dict:
        """Run portfolio simulation with rebalancing"""

        if symbols is None:
            symbols = [config.DEFAULT_SYMBOL]

        portfolio = PortfolioManager(self.initial_capital, self.transaction_fee)
        results = []

        # Track rebalancing schedule
        last_rebalance = None

        logger.info(f"Starting simulation with {len(data)} periods")

        for i in range(len(data)):
            current_date = data.index[i] if hasattr(data.index[i], 'date') else i
            current_data = data.iloc[:i + 1]

            # Get current prices (simplified - in practice you'd have multi-asset data)
            current_prices = {symbol: data['Close'].iloc[i] for symbol in symbols}

            # Check if it's time to rebalance
            should_rebalance = self._should_rebalance(current_date, last_rebalance)

            if should_rebalance and len(current_data) > 20:  # Need some history
                # Get target weights from agent
                target_weights = agent.compute_weights(current_data, symbols)

                # Rebalance portfolio
                portfolio.rebalance(target_weights, current_prices)
                last_rebalance = current_date

                logger.info(f"Rebalanced on {current_date}: {target_weights}")

            # Record portfolio state
            portfolio_value = portfolio.get_portfolio_value(current_prices)
            current_weights = portfolio.get_weights(current_prices)

            results.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': portfolio.cash,
                'positions': portfolio.get_positions().copy(),
                'weights': current_weights.copy(),
                'prices': current_prices.copy()
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate performance metrics
        portfolio_values = results_df['portfolio_value']
        metrics = self._calculate_performance_metrics(portfolio_values, data, benchmark_symbol)

        return {
            'results_df': results_df,
            'final_value': portfolio_values.iloc[-1],
            'metrics': metrics,
            'portfolio': portfolio,
            'agent_name': agent.name
        }

    def _should_rebalance(self, current_date, last_rebalance) -> bool:
        """Determine if portfolio should be rebalanced"""
        if last_rebalance is None:
            return True

        if self.rebalance_frequency == 'daily':
            return True
        elif self.rebalance_frequency == 'weekly':
            # Rebalance every 5 trading days
            return (current_date - last_rebalance) >= 5 if isinstance(current_date, int) else True
        elif self.rebalance_frequency == 'monthly':
            # Rebalance every 20 trading days
            return (current_date - last_rebalance) >= 20 if isinstance(current_date, int) else True

        return False

    def _calculate_performance_metrics(self, portfolio_values: pd.Series,
                                       data: pd.DataFrame,
                                       benchmark_symbol: str = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = portfolio_values.pct_change().dropna()

        metrics = {
            'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1,
            'annualized_return': (1 + returns.mean()) ** config.PERIODS_PER_YEAR - 1,
            'volatility': returns.std() * np.sqrt(config.PERIODS_PER_YEAR),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'calmar_ratio': self._calculate_calmar_ratio(returns, portfolio_values),
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
        }

        return metrics

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - config.RISK_FREE_RATE / config.PERIODS_PER_YEAR
        return np.sqrt(
            config.PERIODS_PER_YEAR) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    def _calculate_calmar_ratio(self, returns: pd.Series, portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = (1 + returns.mean()) ** config.PERIODS_PER_YEAR - 1
        max_dd = abs(self._calculate_max_drawdown(portfolio_values))
        return annual_return / max_dd if max_dd != 0 else 0
