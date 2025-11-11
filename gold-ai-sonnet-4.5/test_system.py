"""
Test script for Gold AI Sonnet 4.5
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import GoldAISonnet

async def test_mt5_connection():
    """Test MT5 connection"""
    print("🔄 Testing MT5 connection...")

    trader = GoldAISonnet(symbol='XAUUSD', timeframe='H1')

    try:
        connected = await trader.initialize()
        if connected:
            print("✅ MT5 connection successful")

            # Test account info
            account_info = await trader.mt5_connector.get_account_info()
            if account_info:
                print(f"📊 Account: {account_info['login']}")
                print(f"💰 Balance: ${account_info['balance']:.2f}")
                print(f"📈 Equity: ${account_info['equity']:.2f}")
            else:
                print("⚠️  Could not retrieve account info")

            # Test market data
            market_data = await trader.mt5_connector.get_historical_data('XAUUSD', 'H1', 10)
            if market_data is not None and len(market_data) > 0:
                print(f"📈 Market data retrieved: {len(market_data)} bars")
                print(f"💹 Latest price: {market_data['close'].iloc[-1]:.5f}")
            else:
                print("⚠️  Could not retrieve market data")

            # Test AI analysis
            if len(market_data) >= 50:
                analysis = await trader.nebula_assistant.analyze_market_conditions(
                    'XAUUSD', 'H1', market_data
                )
                print(f"🤖 AI Analysis: Trend={analysis.get('trend', 'N/A')}, Signal={analysis.get('recommendation', 'N/A')}")

            await trader.stop_trading()
            return True
        else:
            print("❌ MT5 connection failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_risk_management():
    """Test risk management calculations"""
    print("\n🔄 Testing risk management...")

    from main import ShieldProtocol

    shield = ShieldProtocol()

    # Test position sizing
    account_balance = 10000.0
    stop_loss_pips = 50.0  # 50 pips stop loss

    position_size = shield.calculate_position_size(account_balance, stop_loss_pips, 'XAUUSD')
    print(f"📏 Position size for ${account_balance} with {stop_loss_pips} pip SL: {position_size} lots")

    # Test equity lock
    current_equity = 8500.0  # 15% loss
    lock_triggered = shield.check_equity_lock(current_equity, account_balance)
    print(f"🔒 Equity lock triggered: {lock_triggered}")

    print("✅ Risk management tests completed")

async def test_indicators():
    """Test technical indicators"""
    print("\n🔄 Testing technical indicators...")

    trader = GoldAISonnet()

    # Create sample data
    import pandas as pd
    import numpy as np

    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'open': 2000 + np.random.randn(100).cumsum(),
        'high': 2000 + np.random.randn(100).cumsum() + 5,
        'low': 2000 + np.random.randn(100).cumsum() - 5,
        'close': 2000 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    })

    # Test ATR calculation
    atr = trader._calculate_atr(sample_data, 14)
    print(f"📊 ATR(14): {atr:.5f}")

    # Test trend analysis
    analysis = trader.nebula_assistant._determine_trend(sample_data)
    print(f"📈 Trend analysis: {analysis}")

    print("✅ Indicator tests completed")

async def main():
    """Run all tests"""
    print("🧪 Gold AI Sonnet 4.5 - System Tests")
    print("=" * 50)

    # Test 1: MT5 Connection
    mt5_ok = await test_mt5_connection()

    # Test 2: Risk Management
    await test_risk_management()

    # Test 3: Indicators
    await test_indicators()

    print("\n" + "=" * 50)
    if mt5_ok:
        print("🎉 All tests completed successfully!")
        print("\n🚀 Ready to start trading with: python main.py")
        print("🌐 Or launch web panel with: python web_panel.py")
    else:
        print("⚠️  Some tests failed. Please check your MT5 configuration.")
        print("💡 Make sure MT5 terminal is running and credentials are correct.")

if __name__ == "__main__":
    asyncio.run(main())
