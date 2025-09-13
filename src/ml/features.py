import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret1'] = df['Close'].pct_change(1)
    df['ret3'] = df['Close'].pct_change(3)
    df['ret5'] = df['Close'].pct_change(5)

    for w in (5, 10, 20, 50):
        df[f'sma_{w}'] = df['Close'].rolling(w).mean()
        df[f'ema_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()

    df['mom_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['vol_10'] = df['ret1'].rolling(10).std()

    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_width'] = (2 * df['bb_std']) / df['bb_mid']

    df['vol_change'] = df['Volume'].pct_change()
    df['vol_mean20'] = df['Volume'].rolling(20).mean()

    return df.dropna().reset_index(drop=True)
