import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_basic_imports():
    """Test basic imports work"""
    try:
        # Test config import
        from src.config import config
        print(f"✅ Config imported successfully: {config.DEFAULT_SYMBOL}")

        # Test utils import
        from src.utils import Action, calculate_rsi
        print(f"✅ Utils imported successfully: {Action.BUY}")

        # Test data imports
        from src.data.fetcher import DataFetcher
        from src.data.store import DataStore
        print("✅ Data modules imported successfully")

        # Test agent imports
        from src.agents.base import BaseAgent
        from src.agents.rule_agent import MovingAverageCrossoverAgent
        print("✅ Agent modules imported successfully")

        # Test portfolio imports
        from src.portfolio.manager import PortfolioManager
        from src.portfolio.simulator import PortfolioSimulator
        print("✅ Portfolio modules imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_agent_creation():
    """Test creating agents"""
    try:
        from src.agents.rule_agent import MovingAverageCrossoverAgent, RSIMeanReversionAgent

        # Create agents
        ma_agent = MovingAverageCrossoverAgent(20, 50)
        rsi_agent = RSIMeanReversionAgent()

        print(f"✅ Agents created: {ma_agent.name}, {rsi_agent.name}")
        return True

    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return False


def test_portfolio_manager():
    """Test portfolio manager"""
    try:
        from src.portfolio.manager import PortfolioManager

        pm = PortfolioManager(initial_cash=10000)
        success = pm.buy("AAPL", 150.0, 10)

        print(f"✅ Portfolio manager test: Buy success = {success}")
        print(f"   Cash remaining: ${pm.cash:.2f}")
        print(f"   Positions: {pm.get_positions()}")

        return True

    except Exception as e:
        print(f"❌ Portfolio manager error: {e}")
        return False


def test_data_fetcher():
    """Test data fetcher"""
    try:
        from src.data.fetcher import DataFetcher
        import pandas as pd

        # Create dummy data for testing
        fetcher = DataFetcher()

        # Test technical indicators on dummy data
        dummy_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500]
        })

        enhanced_data = fetcher.add_technical_indicators(dummy_data)
        print(f"✅ Data fetcher test: Added {len(enhanced_data.columns)} features")

        return True

    except Exception as e:
        print(f"❌ Data fetcher error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing fixed import structure...\n")

    tests = [
        test_basic_imports,
        test_agent_creation,
        test_portfolio_manager,
        test_data_fetcher
    ]

    results = []
    for test in tests:
        print(f"\n--- Running {test.__name__} ---")
        results.append(test())

    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("🎉 All imports working correctly!")
    else:
        print("⚠️  Some imports failed - check the errors above")
