from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime

from .schemas import (
    BacktestRequest, BacktestResult, PortfolioMetrics,
    TradeRequest, WeightRequest, WeightResponse
)
from src.data.fetcher import DataFetcher
from src.portfolio.manager import PortfolioManager
from src.portfolio.simulator import PortfolioSimulator
from src.agents.rule_agent import MovingAverageCrossoverAgent, RSIMeanReversionAgent
from src.agents.ml_agent import MLAgent
from src.agents.optimizer_agent import MeanVarianceAgent, RiskParityAgent
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Advanced Trading System API with Portfolio Optimization"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_fetcher = DataFetcher()
portfolio_manager = PortfolioManager()


# Agent factory
def create_agent(agent_type: str, params: Dict = None):
    """Factory function to create agents"""
    if params is None:
        params = {}

    agents = {
        "MovingAverageCrossoverAgent": MovingAverageCrossoverAgent,
        "RSIMeanReversionAgent": RSIMeanReversionAgent,
        "MLAgent": MLAgent,
        "MeanVarianceAgent": MeanVarianceAgent,
        "RiskParityAgent": RiskParityAgent
    }

    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agents[agent_type](**params)


@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to the Advanced Trading System API!",
        "version": config.API_VERSION,
        "endpoints": {
            "backtest": "/backtest",
            "portfolio": "/portfolio",
            "trade": "/trade",
            "weights": "/weights",
            "train": "/train"
        }
    }


@app.post("/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run a backtest with specified parameters"""
    try:
        # Fetch data for all symbols (simplified - using first symbol for now)
        data = data_fetcher.get_stock_data(
            request.symbols[0],
            request.start_date,
            request.end_date
        )

        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified period")

        # Create agent
        agent = create_agent(request.agent_type, request.agent_params)

        # Run simulation
        simulator = PortfolioSimulator(
            initial_capital=request.initial_capital,
            rebalance_frequency=request.rebalance_frequency
        )

        results = simulator.run_simulation(
            data=data,
            agent=agent,
            symbols=request.symbols,
            benchmark_symbol="SPY"
        )

        # Return results
        return BacktestResult(
            final_value=results['final_value'],
            total_return=results['metrics']['total_return'],
            annualized_return=results['metrics']['annualized_return'],
            sharpe_ratio=results['metrics']['sharpe_ratio'],
            max_drawdown=results['metrics']['max_drawdown'],
            volatility=results['metrics']['volatility'],
            win_rate=results['metrics']['win_rate'],
            calmar_ratio=results['metrics']['calmar_ratio']
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio", response_model=PortfolioMetrics)
async def get_portfolio_metrics():
    """Get current portfolio metrics"""
    try:
        # Get current prices for all positions
        current_prices = {}
        for symbol in portfolio_manager.get_positions().keys():
            price = data_fetcher.get_current_price(symbol)
            if price:
                current_prices[symbol] = price

        # Calculate metrics
        portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
        weights = portfolio_manager.get_weights(current_prices)

        # Calculate simple performance metrics (would need historical data for full metrics)
        total_return = (portfolio_value / portfolio_manager.initial_cash) - 1

        return PortfolioMetrics(
            portfolio_value=portfolio_value,
            cash=portfolio_manager.cash,
            positions=portfolio_manager.get_positions(),
            weights=weights,
            total_return=total_return,
            annualized_return=0.0,  # Would need historical data
            sharpe_ratio=0.0,  # Would need historical data
            max_drawdown=0.0  # Would need historical data
        )

    except Exception as e:
        logger.error(f"Failed to get portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trade")
async def execute_trade(request: TradeRequest):
    """Execute a trade"""
    try:
        # Get current price if not provided
        if request.price is None:
            request.price = data_fetcher.get_current_price(request.symbol)
            if request.price is None:
                raise HTTPException(status_code=400, detail=f"Could not get price for {request.symbol}")

        # Execute trade
        if request.action.upper() == "BUY":
            success = portfolio_manager.buy(request.symbol, request.price, request.quantity)
        elif request.action.upper() == "SELL":
            success = portfolio_manager.sell(request.symbol, request.price, request.quantity)
        else:
            raise HTTPException(status_code=400, detail="Action must be BUY or SELL")

        if not success:
            raise HTTPException(status_code=400, detail="Trade execution failed")

        # Get updated portfolio value
        current_prices = {request.symbol: request.price}
        portfolio_value = portfolio_manager.get_portfolio_value(current_prices)

        return {
            "status": "success",
            "message": f"{request.action} {request.quantity} shares of {request.symbol}",
            "portfolio_value": portfolio_value
        }

    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/weights", response_model=WeightResponse)
async def compute_weights(request: WeightRequest):
    """Compute optimal portfolio weights"""
    try:
        # Get data for the first symbol (simplified)
        data = data_fetcher.get_stock_data(request.symbols[0])

        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")

        # Create agent
        agent = create_agent(request.agent_type, request.agent_params)

        # Compute weights
        weights = agent.compute_weights(data, request.symbols)

        return WeightResponse(
            weights=weights,
            agent_name=agent.name,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Weight computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_ml_model(symbol: str = "AAPL", start_date: str = "2020-01-01", end_date: str = "2023-01-01"):
    """Train ML model"""
    try:
        # Get training data
        data = data_fetcher.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            raise HTTPException(status_code=404, detail="No training data found")

        # Create and train ML agent
        agent = MLAgent()
        model_path = f"{config.MODEL_DIR}/{symbol}_ml_model.joblib"
        agent.train(data, model_path)

        return {
            "status": "success",
            "message": f"Model trained for {symbol}",
            "model_path": model_path,
            "training_samples": len(data)
        }

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)