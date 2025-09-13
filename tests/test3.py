import pandas as pd
import numpy as np
from src.data.fetcher import get_stock
from src.portfolio.manager import PortfolioManager
from src.portfolio.simulator import Simulator
from src.agents.agents import MomentumAgent
from src.metrics import performance

def test_fetcher():
    df = get_stock("AAPL", start="2022-01-01", end="2022-02-01")
    assert not df.empty
    assert "Close" in df.columns

def test_simulator_runs():
    df = get_stock("AAPL", start="2020-01-01", end="2020-06-01")
    portfolio = PortfolioManager(100_000)
    agent = MomentumAgent()
    sim = Simulator(portfolio, agent, "AAPL")

    final_val = sim.run(df)
    assert final_val > 0

def test_metrics_calculation():
    values = pd.Series([100, 105, 95, 120, 110, 130])
    assert np.isclose(performance.cumulative_returns(values), 0.3)
    assert performance.max_drawdown(values) <= 0
    assert not np.isnan(performance.sharpe_ratio(values))
