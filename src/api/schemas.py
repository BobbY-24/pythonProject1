from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class BacktestRequest(BaseModel):
    """Backtest request schema"""
    symbols: List[str] = Field(default=["AAPL"], description="List of symbols to trade")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")
    agent_type: str = Field(default="MovingAverageCrossoverAgent", description="Agent type")
    agent_params: Dict = Field(default_factory=dict, description="Agent parameters")
    initial_capital: float = Field(default=100000, description="Initial capital")
    rebalance_frequency: str = Field(default="monthly", description="Rebalancing frequency")

class BacktestResult(BaseModel):
    """Backtest result schema"""
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    calmar_ratio: float

class PortfolioMetrics(BaseModel):
    """Portfolio metrics schema"""
    portfolio_value: float
    cash: float
    positions: Dict[str, float]
    weights: Dict[str, float]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float

class TradeRequest(BaseModel):
    """Trade execution request"""
    symbol: str = Field(description="Symbol to trade")
    action: str = Field(description="BUY or SELL")
    quantity: float = Field(description="Number of shares")
    price: Optional[float] = Field(default=None, description="Price (market price if not specified)")

class WeightRequest(BaseModel):
    """Portfolio weight request"""
    symbols: List[str] = Field(description="List of symbols")
    agent_type: str = Field(default="MeanVarianceAgent", description="Agent type for weight computation")
    agent_params: Dict = Field(default_factory=dict, description="Agent parameters")

class WeightResponse(BaseModel):
    """Portfolio weight response"""
    weights: Dict[str, float]
    agent_name: str
    timestamp: datetime