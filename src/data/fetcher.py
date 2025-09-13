import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import config
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import config

logger = logging.getLogger(__name__)


class DataFetcher:
    """Optimized data fetching with caching"""

    def __init__(self, cache_dir: str = config.DATA_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_stock_data(self,
                       symbol: str,
                       start: str = config.DEFAULT_START_DATE,
                       end: str = config.DEFAULT_END_DATE,
                       force_refresh: bool = False) -> pd.DataFrame:
        """Get stock data with intelligent caching"""
        cache_file = self.cache_dir / f"{symbol}_{start}_{end}.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading {symbol} from cache")
            return pd.read_parquet(cache_file)

        logger.info(f"Fetching {symbol} from Yahoo Finance")
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            df = df.reset_index()

            # Flatten columns if MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]

            # Cache the data
            df.to_parquet(cache_file, index=False)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.fast_info.get("last_price")
        except Exception as e:
            logger.warning(f"Failed to get current price for {symbol}: {e}")
            return None

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data - FIXED VERSION"""
        df = df.copy()

        # Returns - vectorized
        for period in [1, 3, 5, 10]:
            df[f'ret_{period}'] = df['Close'].pct_change(period)

        # Moving averages - batch computation
        windows = [5, 10, 20, 50]
        for w in windows:
            df[f'sma_{w}'] = df['Close'].rolling(w, min_periods=1).mean()
            df[f'ema_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()

        # Momentum and volatility
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['volatility_10'] = df['ret_1'].rolling(10, min_periods=1).std()

        # Bollinger Bands
        bb_period = 20
        df['bb_mid'] = df['Close'].rolling(bb_period).mean()
        df['bb_std'] = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        # RSI
        try:
            from utils import calculate_rsi
            df['rsi'] = calculate_rsi(df['Close'])
        except ImportError:
            # Fallback RSI calculation
            df['rsi'] = self._calculate_rsi_fallback(df['Close'])

        # FIXED: Replace deprecated fillna method
        return df.ffill().fillna(0)  # Use ffill() instead of fillna(method='ffill')

    def _calculate_rsi_fallback(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Fallback RSI calculation if utils import fails"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, float('inf'))
        return 100 - (100 / (1 + rs))


# Test the module directly
if __name__ == "__main__":
    print("Testing data fetcher...")

    fetcher = DataFetcher()

    # Test with dummy data
    import numpy as np

    dummy_data = pd.DataFrame({
        'Close': 100 + np.random.randn(50).cumsum(),
        'Volume': np.random.randint(1000, 5000, 50)
    })

    enhanced_data = fetcher.add_technical_indicators(dummy_data)
    print(f"✅ Added {len(enhanced_data.columns)} technical indicators")
    print(f"Features: {list(enhanced_data.columns)}")