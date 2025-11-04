import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from models.strategy_classifier import StrategyClassifier
from models.trend_model import TrendPredictor
from news_module.news_fetcher import NewsFetcher
from news_module.sentiment_analyzer import NewsSentimentHandler
from lot_optimizer.optimizer import LotSizeOptimizer
from mt5_connector.trade_executor import TradeExecutor
from mt5_connector.account_handler import AccountHandler
from mt5_connector.order_manager import OrderManager
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy

import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 200

# Initialize MT5
if not mt5.initialize():
    logging.error("MT5 initialization failed")
    quit()

account = AccountHandler()
acc_info = account.get_account_info()
if not acc_info:
    logging.error("Account info retrieval failed")
    quit()

# Load models
classifier = StrategyClassifier()
sentiment_handler = NewsSentimentHandler()
lot_optimizer = LotSizeOptimizer()
trade_executor = TradeExecutor(SYMBOL)
order_manager = OrderManager(SYMBOL)
trend_model = TrendPredictor()

# Load training for classifier (example data, should be persisted in real case)
classifier.train(pd.DataFrame({
    "volatility": [0.3, 0.4, 0.2, 0.5, 0.1, 1.2, 1.5, 1.0, 1.3, 1.1, 0.8, 0.9, 0.7, 0.6, 0.85],
    "volume": [50000, 60000, 40000, 70000, 30000, 300000, 350000, 250000, 320000, 280000, 150000, 180000, 120000, 160000, 140000],
    "momentum": [0.4, 0.5, 0.3, 0.6, 0.2, -0.8, -1.0, -0.7, -0.9, -0.6, 1.1, 1.2, 0.9, 1.0, 0.95],
    "sentiment_score": [0.2, 0.3, 0.1, 0.4, 0.0, 0.9, 1.0, 0.8, 0.95, 0.85, -0.6, -0.5, -0.7, -0.4, -0.55],
    "strategy": ["swing"]*5 + ["scalping"]*5 + ["day_trading"]*5
}))

# Get latest price data
data = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Compute inputs
volatility = df['close'].rolling(window=10).std().iloc[-1]
volume = df['real_volume'].iloc[-1]
momentum = df['close'].pct_change().mean()

# News + sentiment
fetcher = NewsFetcher(api_key=os.getenv('NEWS_API_KEY'))
headlines = fetcher.fetch_news()
sentiment = sentiment_handler.process_news(headlines)
sent_score = sentiment['sentiment_score']

# Predict strategy
input_df = pd.DataFrame([[volatility, volume, momentum, sent_score]], 
                        columns=['volatility', 'volume', 'momentum', 'sentiment_score'])
strategy_type = classifier.predict(input_df)[0]

# Run selected strategy
if strategy_type == 'scalping':
    strat = ScalpingStrategy(SYMBOL)
elif strategy_type == 'day_trading':
    strat = DayTradingStrategy(SYMBOL)
else:
    strat = SwingTradingStrategy(SYMBOL)

signals = strat.generate_signals(df)
latest_signal = signals['signal'].iloc[-1]

# Execute trade
if latest_signal in ['buy', 'sell']:
    price = mt5.symbol_info_tick(SYMBOL).ask if latest_signal == 'buy' else mt5.symbol_info_tick(SYMBOL).bid
    stop_loss = price - 10 if latest_signal == 'buy' else price + 10
    take_profit = price + 20 if latest_signal == 'buy' else price - 20
    lot = lot_optimizer.optimize(
        balance=acc_info['balance'],
        entry_price=price,
        stop_price=stop_loss
    )
    trade_executor.send_order(action=latest_signal, lot=lot, price=price, sl=stop_loss, tp=take_profit)

mt5.shutdown()
