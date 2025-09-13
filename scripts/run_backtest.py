import pandas as pd
from src.data.fetcher import load_csv
from src.ml.train_model import train_and_save, prepare_xy
from src.agents.ml_agent import MLAgent
from src.portfolio.simulator import Simulator
from src.portfolio.manager import PortfolioManager

if __name__ == "__main__":
    df = load_csv("data/AAPL.csv")
    model = train_and_save(df, "models/rf_model.joblib")

    df2, X, y, _ = prepare_xy(df)
    agent = MLAgent("models/rf_model.joblib")

    portfolio = PortfolioManager(cash=100000)
    sim = Simulator(df2, portfolio, agent)
    results = sim.run(X)

    print("Final portfolio value:", results[-1])
# Example in Backtester
portfolio = PortfolioManager(initial_cash=100_000)
for i, row in df.iterrows():
    price = row["Close"]
    action = agent.decide_action(row, i, df)
    if action == "BUY":
        portfolio.buy("AAPL", price, 10)
    elif action == "SELL":
        portfolio.sell("AAPL", price, 10)
