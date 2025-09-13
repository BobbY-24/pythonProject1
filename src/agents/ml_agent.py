import joblib
import pandas as pd

class MLAgent:
    def __init__(self, model_path, buy_threshold=0.55, sell_threshold=0.45, position_size=0.1):
        data = joblib.load(model_path)
        self.pipeline = data['pipeline']
        self.feature_cols = data['feature_cols']
        self.buy_th = buy_threshold
        self.sell_th = sell_threshold
        self.position_size = position_size

    def predict_proba(self, X_row: pd.Series):
        X_df = X_row[self.feature_cols].to_frame().T
        return self.pipeline.predict_proba(X_df)[0, 1]

    def decide(self, X_row: pd.Series):
        prob = self.predict_proba(X_row)
        if prob > self.buy_th:
            return 1   # buy
        elif prob < self.sell_th:
            return -1  # sell
        return 0       # hold
