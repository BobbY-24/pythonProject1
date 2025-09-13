import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import config

logger = logging.getLogger(__name__)


class DataStore:
    """Data storage manager for historical data"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(config.DATA_DIR) / "trading_data.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                portfolio_value REAL,
                cash REAL,
                positions TEXT,  -- JSON string
                agent_name TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def save_stock_data(self, symbol: str, df: pd.DataFrame):
        """Save stock data to database"""
        conn = sqlite3.connect(self.db_path)
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        df_copy.to_sql('stock_data', conn, if_exists='append', index=False)
        conn.close()
        logger.info(f"Saved {len(df)} records for {symbol}")

    def load_stock_data(self, symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Load stock data from database"""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM stock_data WHERE symbol = ?"
        params = [symbol]

        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if not df.empty:
            df['Date'] = pd.to_datetime(df['date'])

        return df

    def save_portfolio_snapshot(self, portfolio_value: float, cash: float,
                                positions: dict, agent_name: str):
        """Save portfolio snapshot"""
        import json
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO portfolio_history (portfolio_value, cash, positions, agent_name)
            VALUES (?, ?, ?, ?)
        ''', (portfolio_value, cash, json.dumps(positions), agent_name))

        conn.commit()
        conn.close()