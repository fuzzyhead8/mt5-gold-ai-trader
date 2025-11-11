"""
Test script for AI integration in Gold AI Sonnet
"""

import asyncio
import pandas as pd
import numpy as np
from main import NebulaAssistant

async def test_ai_integration():
    """Test AI integration with sample market data"""
    print("🧪 Testing AI Integration")
    print("=" * 50)

    # Create AI assistant
    assistant = NebulaAssistant()

    # Create sample market data (bullish trend)
    dates = pd.date_range(pd.Timestamp.now().date(), periods=100, freq='h')
    np.random.seed(42)

    # Generate bullish trend data
    base_price = 4100
    trend = np.linspace(0, 50, 100)  # Upward trend
    noise = np.random.normal(0, 10, 100)
    close_prices = base_price + trend + noise

    # Generate OHLC data
    highs = close_prices + np.abs(np.random.normal(0, 5, 100))
    lows = close_prices - np.abs(np.random.normal(0, 5, 100))
    opens = close_prices + np.random.normal(0, 2, 100)
    volumes = np.random.randint(1000, 10000, 100)

    market_data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })

    print(f"📊 Generated {len(market_data)} bars of sample market data")
    print(f"💹 Current Price: {close_prices[-1]:.2f}")
    # Test AI analysis
    print("\n🤖 Running AI Market Analysis...")
    analysis = await assistant.analyze_market_conditions('XAUUSD', 'H1', market_data)

    print("\n📈 AI Analysis Results:")
    print(f"   Trend: {analysis.get('trend', 'N/A')}")
    print(f"   Volatility: {analysis.get('volatility', 'N/A')}")
    print(f"   Momentum: {analysis.get('momentum', 'N/A')}")
    print(f"   Recommendation: {analysis.get('recommendation', 'N/A')}")
    print(f"   Confidence: {analysis.get('confidence', 'N/A')}")
    print(f"   AI Sources: {analysis.get('ai_sources', 0)}")
    print(f"   Reasoning: {analysis.get('reasoning', 'N/A')}")

    # Test with bearish data
    print("\n📉 Testing with Bearish Market Data...")

    # Generate bearish trend data
    trend_bearish = np.linspace(0, -50, 100)  # Downward trend
    close_prices_bearish = base_price + trend_bearish + noise

    highs_bearish = close_prices_bearish + np.abs(np.random.normal(0, 5, 100))
    lows_bearish = close_prices_bearish - np.abs(np.random.normal(0, 5, 100))
    opens_bearish = close_prices_bearish + np.random.normal(0, 2, 100)

    market_data_bearish = pd.DataFrame({
        'open': opens_bearish,
        'high': highs_bearish,
        'low': lows_bearish,
        'close': close_prices_bearish,
        'volume': volumes
    })

    analysis_bearish = await assistant.analyze_market_conditions('XAUUSD', 'H1', market_data_bearish)

    print("\n📈 Bearish AI Analysis Results:")
    print(f"   Trend: {analysis_bearish.get('trend', 'N/A')}")
    print(f"   Recommendation: {analysis_bearish.get('recommendation', 'N/A')}")
    print(f"   Confidence: {analysis_bearish.get('confidence', 'N/A')}")

    # Test with insufficient data
    print("\n⚠️ Testing with Insufficient Data...")
    small_data = market_data.head(5)
    analysis_small = await assistant.analyze_market_conditions('XAUUSD', 'H1', small_data)

    print(f"   Recommendation: {analysis_small.get('recommendation', 'N/A')}")
    print(f"   Reasoning: {analysis_small.get('reasoning', 'N/A')}")

    print("\n✅ AI Integration Test Completed!")

if __name__ == "__main__":
    asyncio.run(test_ai_integration())
