import pandas as pd

def make_targets(df: pd.DataFrame, mode='classification', threshold=0.0):
    df = df.copy()
    df['next_return'] = df['Close'].pct_change().shift(-1)
    if mode == 'classification':
        df['label'] = (df['next_return'] > threshold).astype(int)
    else:
        df['label'] = df['next_return']
    return df.dropna().reset_index(drop=True)
