from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    avg_price: float = 0.0

    @property
    def value(self) -> float:
        return self.quantity * self.avg_price

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price


class PortfolioManager:
    """Enhanced portfolio management with rebalancing"""

    def __init__(self,
                 initial_cash: float = config.INITIAL_BALANCE,
                 transaction_fee: float = config.TRANSACTION_FEE):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.transaction_history: List[Dict] = []
        self.transaction_fee = transaction_fee

    def buy(self, symbol: str, price: float, quantity: float) -> bool:
        """Execute buy order"""
        total_cost = price * quantity * (1 + self.transaction_fee)

        if self.cash < total_cost:
            logger.warning(f"Insufficient cash for {symbol} buy order")
            return False

        self.cash -= total_cost

        if symbol in self.positions:
            # Update average price
            pos = self.positions[symbol]
            total_value = pos.quantity * pos.avg_price + price * quantity
            pos.quantity += quantity
            pos.avg_price = total_value / pos.quantity
        else:
            self.positions[symbol] = Position(symbol, quantity, price)

        self._record_transaction('BUY', symbol, price, quantity, total_cost)
        return True

    def sell(self, symbol: str, price: float, quantity: float) -> bool:
        """Execute sell order"""
        if symbol not in self.positions or self.positions[symbol].quantity < quantity:
            logger.warning(f"Insufficient position for {symbol} sell order")
            return False

        proceeds = price * quantity * (1 - self.transaction_fee)
        self.cash += proceeds
        self.positions[symbol].quantity -= quantity

        if self.positions[symbol].quantity < 1e-6:  # Close to zero
            del self.positions[symbol]

        self._record_transaction('SELL', symbol, price, quantity, -proceeds)
        return True

    def rebalance(self, target_weights: Dict[str, float], current_prices: Dict[str, float]):
        """Rebalance portfolio to target weights"""
        total_value = self.get_portfolio_value(current_prices)

        logger.info(f"Rebalancing portfolio. Total value: ${total_value:,.2f}")

        # Calculate target positions
        target_values = {symbol: weight * total_value for symbol, weight in target_weights.items()}

        # Get current positions values
        current_values = {}
        for symbol in target_weights.keys():
            if symbol in self.positions:
                current_values[symbol] = self.positions[symbol].quantity * current_prices.get(symbol, 0)
            else:
                current_values[symbol] = 0

        # Execute trades to reach target allocation
        for symbol, target_value in target_values.items():
            current_value = current_values.get(symbol, 0)
            difference = target_value - current_value

            if abs(difference) > total_value * 0.01:  # Only trade if difference > 1%
                current_price = current_prices.get(symbol)
                if current_price is None or current_price <= 0:
                    logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue

                shares_needed = difference / current_price

                if shares_needed > 0:
                    # Buy
                    self.buy(symbol, current_price, abs(shares_needed))
                    logger.info(f"Bought {shares_needed:.2f} shares of {symbol}")
                else:
                    # Sell
                    available_shares = self.positions.get(symbol, Position(symbol, 0)).quantity
                    shares_to_sell = min(abs(shares_needed), available_shares)
                    if shares_to_sell > 0:
                        self.sell(symbol, current_price, shares_to_sell)
                        logger.info(f"Sold {shares_to_sell:.2f} shares of {symbol}")

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.avg_price)
            total_value += position.quantity * current_price
        return total_value

    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return {symbol: pos.quantity for symbol, pos in self.positions.items()}

    def get_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get current portfolio weights"""
        total_value = self.get_portfolio_value(current_prices)
        if total_value == 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.avg_price)
            weights[symbol] = (position.quantity * current_price) / total_value

        # Add cash weight
        weights['CASH'] = self.cash / total_value
        return weights

    def _record_transaction(self, action: str, symbol: str, price: float,
                            quantity: float, total_amount: float):
        """Record transaction for audit trail"""
        self.transaction_history.append({
            'action': action,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'total_amount': total_amount,
            'timestamp': pd.Timestamp.now(),
            'cash_after': self.cash
        })