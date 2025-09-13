from abc import ABC, abstractmethod
from typing import Dict, Union
import pandas as pd
import numpy as np
from utils import Action


class BaseAgent(ABC):
    """Base agent interface"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def decide_action(self, data: pd.DataFrame, **kwargs) -> Action:
        """Decide single trading action (BUY/SELL/HOLD)"""
        pass

    @abstractmethod
    def compute_weights(self, data: pd.DataFrame, symbols: list) -> Dict[str, float]:
        """Compute portfolio weights for multiple symbols"""
        pass

    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights are valid (sum to 1, non-negative)"""
        total = sum(abs(w) for w in weights.values())
        if total > 1e-6:  # Avoid division by zero
            return {k: v / total for k, v in weights.items()}
        return {k: 1.0 / len(weights) for k in weights.keys()}


class SignalAgent(BaseAgent):
    """Base class for signal-generating agents"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals for historical data"""
        signals = []
        for i in range(len(data)):
            current_data = data.iloc[:i + 1]
            if i == 0:
                signals.append(Action.HOLD.value)
            else:
                action = self.decide_action(current_data)
                signals.append(action.value)

        return pd.Series(signals, index=data.index)
