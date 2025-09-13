from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:
    """Centralized configuration"""
    # Trading
    DEFAULT_SYMBOL: str = "AAPL"
    DEFAULT_START_DATE: str = "2022-01-01"
    DEFAULT_END_DATE: str = "2023-01-01"

    # Portfolio
    INITIAL_BALANCE: float = 100_000
    TRANSACTION_FEE: float = 0.001
    SLIPPAGE: float = 0.0005

    # ML
    TRAIN_SPLIT: float = 0.7
    MODEL_DIR: str = "models"
    DATA_DIR: str = "data"

    # Performance
    RISK_FREE_RATE: float = 0.02
    PERIODS_PER_YEAR: int = 252

    # API
    API_TITLE: str = "Stock Agent API"
    API_VERSION: str = "1.0.0"


config = Config()