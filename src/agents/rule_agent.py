import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from .base import BaseAgent, SignalAgent
    from utils import Action, calculate_rsi
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.base import BaseAgent, SignalAgent
    from utils import Action, calculate_rsi


class MovingAverageCrossoverAgent(SignalAgent):
    """Moving average crossover strategy"""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(f"MACross_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window

    def decide_action(self, data: pd.DataFrame, **kwargs) -> Action:
        """Decide action based on MA crossover"""
        if len(data) < self.long_window:
            return Action.HOLD

        short_ma = data['Close'].iloc[-self.short_window:].mean()
        long_ma = data['Close'].iloc[-self.long_window:].mean()

        if short_ma > long_ma * 1.01:  # Add small threshold
            return Action.BUY
        elif short_ma < long_ma * 0.99:
            return Action.SELL
        return Action.HOLD

    def compute_weights(self, data: pd.DataFrame, symbols: list) -> Dict[str, float]:
        """Compute weights based on MA signals"""
        weights = {}
        for symbol in symbols:
            # In practice, you'd fetch data for each symbol
            action = self.decide_action(data)
            if action == Action.BUY:
                weights[symbol] = 1.0 / len(symbols)
            else:
                weights[symbol] = 0.0

        return self.validate_weights(weights)


class RSIMeanReversionAgent(SignalAgent):
    """RSI-based mean reversion strategy"""

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(f"RSI_{rsi_period}_{oversold}_{overbought}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def decide_action(self, data: pd.DataFrame, **kwargs) -> Action:
        """Decide action based on RSI levels"""
        if len(data) < self.rsi_period:
            return Action.HOLD

        rsi = calculate_rsi(data['Close'], self.rsi_period).iloc[-1]

        if rsi < self.oversold:
            return Action.BUY
        elif rsi > self.overbought:
            return Action.SELL
        return Action.HOLD

    def compute_weights(self, data: pd.DataFrame, symbols: list) -> Dict[str, float]:
        """Compute weights based on RSI signals"""
        weights = {}
        for symbol in symbols:
            action = self.decide_action(data)
            if action == Action.BUY:
                # Weight inversely proportional to RSI (more oversold = higher weight)
                rsi = calculate_rsi(data['Close'], self.rsi_period).iloc[-1]
                weights[symbol] = max(0, (self.oversold - rsi) / self.oversold)
            else:
                weights[symbol] = 0.0

        return self.validate_weights(weights)


class MomentumAgent(SignalAgent):
    """Momentum-based strategy"""

    def __init__(self, lookback_period: int = 20):
        super().__init__(f"Momentum_{lookback_period}")
        self.lookback_period = lookback_period

    def decide_action(self, data: pd.DataFrame, **kwargs) -> Action:
        """Decide action based on price momentum"""
        if len(data) < self.lookback_period:
            return Action.HOLD

        current_price = data['Close'].iloc[-1]
        past_price = data['Close'].iloc[-self.lookback_period]
        momentum = (current_price / past_price - 1)

        if momentum > 0.05:  # 5% threshold
            return Action.BUY
        elif momentum < -0.05:
            return Action.SELL
        return Action.HOLD

    def compute_weights(self, data: pd.DataFrame, symbols: list) -> Dict[str, float]:
        """Compute weights based on momentum"""
        weights = {}
        momentums = []

        for symbol in symbols:
            if len(data) >= self.lookback_period:
                current_price = data['Close'].iloc[-1]
                past_price = data['Close'].iloc[-self.lookback_period]
                momentum = (current_price / past_price - 1)
                momentums.append(max(0, momentum))  # Only positive momentum
            else:
                momentums.append(0)

        # Normalize momentums to weights
        total_momentum = sum(momentums)
        if total_momentum > 0:
            for i, symbol in enumerate(symbols):
                weights[symbol] = momentums[i] / total_momentum
        else:
            weights = {symbol: 1.0 / len(symbols) for symbol in symbols}

        return weights


# Test the module directly
if __name__ == "__main__":
    print("Testing rule agents...")

    # Create test data
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': [1000] * 100
    }, index=dates)

    # Test agents
    agents = [
        MovingAverageCrossoverAgent(10, 20),
        RSIMeanReversionAgent(),
        MomentumAgent()
    ]

    for agent in agents:
        action = agent.decide_action(data)
        weights = agent.compute_weights(data, ['AAPL', 'MSFT'])
        print(f"✅ {agent.name}: Action={action}, Weights={weights}")