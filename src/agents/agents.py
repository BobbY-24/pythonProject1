import pandas as pd

class MomentumAgent:
    def __init__(self, window=5):
        self.window = window

    def decide_action(self, row, i, df: pd.DataFrame):
        if i < self.window:
            return "HOLD"
        # If current close > mean of last N closes → buy
        window_mean = df["Close"].iloc[i-self.window:i].mean()
        if row["Close"] > window_mean:
            return "BUY"
        elif row["Close"] < window_mean:
            return "SELL"
        return "HOLD"


class MovingAverageCrossoverAgent:
    def __init__(self, short=5, long=20):
        self.short = short
        self.long = long

    def decide_action(self, row, i, df: pd.DataFrame):
        if i < self.long:
            return "HOLD"
        short_ma = df["Close"].iloc[i-self.short:i].mean()
        long_ma = df["Close"].iloc[i-self.long:i].mean()
        if short_ma > long_ma:
            return "BUY"
        elif short_ma < long_ma:
            return "SELL"
        return "HOLD"
