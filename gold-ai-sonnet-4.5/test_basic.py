"""
Basic tests for Gold AI Sonnet 4.5
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock

from main import GoldAISonnet, ShieldProtocol, NebulaAssistant
from config import validate_config, get_config

class TestShieldProtocol:
    """Test Shield Protocol risk management"""

    def test_position_size_calculation(self):
        """Test position size calculation"""
        shield = ShieldProtocol()

        # Test basic calculation
        balance = 10000
        stop_loss_pips = 50
        symbol = 'XAUUSD'

        size = shield.calculate_position_size(balance, stop_loss_pips, symbol)

        # Should be reasonable lot size
        assert 0.01 <= size <= 10.0

    def test_equity_lock(self):
        """Test equity lock mechanism"""
        shield = ShieldProtocol()

        # Test normal conditions
        assert shield.check_equity_lock(10000, 10000) == False
        assert shield.is_trading_allowed() == True

        # Test lock trigger
        assert shield.check_equity_lock(8500, 10000) == True  # 15% loss
        assert shield.is_trading_allowed() == False

class TestNebulaAssistant:
    """Test AI assistant functionality"""

    @pytest.mark.asyncio
    async def test_market_analysis(self):
        """Test market condition analysis"""
        assistant = NebulaAssistant()

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='H')
        data = pd.DataFrame({
            'close': np.random.uniform(1950, 2050, 50),
            'high': np.random.uniform(1950, 2050, 50),
            'low': np.random.uniform(1950, 2050, 50),
            'open': np.random.uniform(1950, 2050, 50),
            'volume': np.random.randint(1000, 10000, 50)
        })

        analysis = await assistant.analyze_market_conditions('XAUUSD', 'H1', data)

        # Check analysis structure
        required_keys = ['trend', 'volatility', 'support_resistance', 'momentum', 'recommendation']
        for key in required_keys:
            assert key in analysis

        # Check recommendation
        assert analysis['recommendation'] in ['BUY', 'SELL', 'WAIT']

class TestConfiguration:
    """Test configuration validation"""

    def test_valid_config(self):
        """Test valid configuration"""
        config = get_config()
        assert validate_config(config) == True

    def test_invalid_trading_config(self):
        """Test invalid trading configuration"""
        config = get_config()
        config['trading']['max_positions'] = 20  # Invalid

        assert validate_config(config) == False

class TestGoldAISonnet:
    """Test main trading system"""

    @pytest.fixture
    def mock_mt5_connector(self):
        """Mock MT5 connector"""
        mock_connector = Mock()
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.get_account_info = AsyncMock(return_value={
            'balance': 10000,
            'equity': 10000
        })
        mock_connector.get_positions = AsyncMock(return_value=[])
        return mock_connector

    @pytest.mark.asyncio
    async def test_initialization(self, mock_mt5_connector):
        """Test system initialization"""
        # Mock the imports to avoid actual dependencies
        import main
        main.MT5Connector = Mock(return_value=mock_mt5_connector)

        # Create system
        trader = GoldAISonnet()

        # Mock the initialization methods
        trader.sentiment_analyzer.initialize = AsyncMock()
        trader.trend_predictor.initialize = AsyncMock()
        trader.strategy_classifier.initialize = AsyncMock()

        # Test initialization
        result = await trader.initialize()
        assert result == True

    @pytest.mark.asyncio
    async def test_system_status(self):
        """Test system status retrieval"""
        trader = GoldAISonnet()

        # Mock positions
        trader.positions = {
            1: Mock(current_profit=100.0),
            2: Mock(current_profit=-50.0)
        }

        # Mock MT5 connector
        trader.mt5_connector.get_account_info = AsyncMock(return_value={
            'balance': 10000,
            'equity': 10500
        })

        status = await trader.get_system_status()

        assert status['positions_count'] == 2
        assert status['account_balance'] == 10000
        assert status['trading_allowed'] == True

if __name__ == "__main__":
    # Run basic functionality test
    print("Running Gold AI Sonnet 4.5 basic tests...")

    # Test Shield Protocol
    shield = ShieldProtocol()
    size = shield.calculate_position_size(10000, 50, 'XAUUSD')
    print(f"✓ Position size calculation: {size}")

    # Test configuration
    config = get_config()
    if validate_config(config):
        print("✓ Configuration validation passed")
    else:
        print("✗ Configuration validation failed")

    # Test AI assistant
    async def test_ai():
        assistant = NebulaAssistant()
        data = pd.DataFrame({
            'close': [2000, 2005, 2010, 2008, 2012],
            'high': [2010, 2015, 2020, 2018, 2022],
            'low': [1990, 1995, 2000, 1998, 2002],
            'open': [2000, 2005, 2010, 2008, 2012],
            'volume': [5000, 6000, 7000, 5500, 8000]
        })

        analysis = await assistant.analyze_market_conditions('XAUUSD', 'H1', data)
        print(f"✓ AI analysis: Trend={analysis['trend']}, Recommendation={analysis['recommendation']}")

    asyncio.run(test_ai())

    print("Basic tests completed!")
