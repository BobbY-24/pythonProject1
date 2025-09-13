from src.portfolio.simulator import PortfolioSimulator
from src.agents.optimizer_agent import MeanVarianceAgent
from src.data.fetcher import DataFetcher

# Get data
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", "2020-01-01", "2023-01-01")

# Create agent and simulator
agent = MeanVarianceAgent(lookback_period=60)
simulator = PortfolioSimulator(rebalance_frequency='monthly')

# Run simulation
results = simulator.run_simulation(data, agent, symbols=["AAPL", "MSFT"])