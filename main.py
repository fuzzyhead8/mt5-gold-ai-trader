#!/usr/bin/env python3
"""
Simplified MT5 Gold AI Trader Main Module

This module handles the core trading loop but delegates all trading logic to individual strategies.
Clean separation of concerns for better maintainability and backtest compatibility.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import threading
import argparse
import signal
import sys
import io
import os
from datetime import datetime

from models.strategy_classifier import StrategyClassifier
from news_module.news_fetcher import NewsFetcher
from news_module.sentiment_analyzer import NewsSentimentHandler
from mt5_connector.account_handler import AccountHandler

# Import strategies
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy
from strategies.golden_scalping_simplified import GoldenScalpingStrategySimplified
from strategies.goldstorm_strategy import GoldStormStrategy
from strategies.vwap_strategy import VWAPStrategy

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class TradingBot:
    """Simplified trading bot that delegates all trading logic to strategies"""
    
    def __init__(self, symbol="XAUUSD", bars=200, manual_strategy=None):
        self.symbol = symbol
        self.bars = bars
        self.manual_strategy = manual_strategy
        self.classifier = StrategyClassifier()
        self.sentiment_handler = NewsSentimentHandler()
        self.account_handler = AccountHandler()
        self.acc_info = None
        self.connected = False
        self.stop_event = threading.Event()
        
        # Strategy instances
        self.strategies = {
            'scalping': ScalpingStrategy(symbol),
            'day_trading': DayTradingStrategy(symbol),
            'golden': GoldenScalpingStrategySimplified(symbol),
            'goldstorm': GoldStormStrategy(symbol),
            'vwap': VWAPStrategy(symbol),
            'swing': SwingTradingStrategy(symbol)
        }

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            logging.error("MT5 initialization failed")
            return False
        
        self.acc_info = self.account_handler.get_account_info()
        if not self.acc_info:
            logging.error("Account info retrieval failed")
            mt5.shutdown()
            return False
        
        self.connected = True
        logging.info("MT5 initialized and account info retrieved")
        return True

    def shutdown_mt5(self):
        """Clean shutdown of MT5 connection"""
        if self.connected:
            self.close_all_positions()
            mt5.shutdown()
            self.connected = False
            logging.info("MT5 connection closed and positions cleared")

    def close_all_positions(self):
        """Emergency close all positions"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": mt5.symbol_info_tick(self.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(self.symbol).ask,
                    "deviation": 20,
                    "magic": 999999,
                    "comment": "Emergency Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
            logging.info("All positions closed")

    def load_classifier(self):
        """Load the strategy classification model"""
        try:
            self.classifier.load_model('models/trained_classifier.joblib')
            logging.info("Classifier loaded from trained model")
        except Exception as e:
            logging.error(f"Failed to load classifier: {e}")

    def get_sentiment_score(self) -> float:
        """Get current market sentiment score"""
        try:
            fetcher = NewsFetcher(api_key=os.getenv('NEWS_API_KEY'), cache_duration_hours=4)
            
            # Try cached first to avoid API limits
            articles = fetcher.get_cached_or_fallback()
            if not articles:
                logging.info("No cached news, attempting fresh fetch...")
                articles = fetcher.fetch_news(page_size=10)
            
            sentiment_results = self.sentiment_handler.process_news(articles)
            sent_score = np.mean([item['score'] for item in sentiment_results]) if sentiment_results else 0
            
            logging.info(f"Using {len(articles)} news articles for sentiment analysis (score: {sent_score:.3f})")
            return sent_score
            
        except Exception as e:
            logging.warning(f"News processing failed: {e}")
            return 0  # Neutral sentiment as fallback

    def select_strategy(self, df: pd.DataFrame, sentiment_score: float) -> str:
        """Select trading strategy based on market conditions"""
        if self.manual_strategy:
            return self.manual_strategy
        
        # Use classifier to predict best strategy
        try:
            volatility = df['close'].rolling(window=10).std().iloc[-1]
            volume = df['tick_volume'].iloc[-1]
            momentum = df['close'].pct_change().rolling(5).mean().iloc[-1]
            
            input_df = pd.DataFrame([[volatility, volume, momentum, sentiment_score]], 
                                  columns=['volatility', 'volume', 'momentum', 'sentiment_score'])
            
            strategy_type = self.classifier.predict(input_df)[0]
            logging.info(f"AI predicted strategy: {strategy_type}")
            return strategy_type
            
        except Exception as e:
            logging.error(f"Strategy prediction failed: {e}")
            return 'goldstorm'  # Default fallback

    def get_market_data(self, timeframe: int, bars: int = None) -> pd.DataFrame:
        """Fetch market data with retry logic"""
        bars = bars or self.bars
        data = None
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries and data is None:
            try:
                data = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
                if data is None:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2 ** retry_count)
                break
            except Exception as e:
                retry_count += 1
                logging.error(f"MT5 data fetch error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)
        
        if data is None:
            raise Exception(f"Failed to fetch market data after {max_retries} attempts")
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df

    def trading_iteration(self):
        """Single trading iteration"""
        if self.stop_event.is_set():
            return
        
        try:
            # Get sentiment score
            sentiment_score = self.get_sentiment_score()
            sentiment_str = "bullish" if sentiment_score > 0 else "bearish" if sentiment_score < 0 else "neutral"
            
            # Get market data for strategy selection
            df_prediction = self.get_market_data(mt5.TIMEFRAME_M1, 100)
            
            # Select strategy
            strategy_name = self.select_strategy(df_prediction, sentiment_score)
            
            if strategy_name not in self.strategies:
                logging.error(f"Unknown strategy: {strategy_name}")
                return
            
            strategy = self.strategies[strategy_name]
            config = strategy.get_strategy_config()
            
            logging.info(f"Using {strategy_name} strategy")
            
            # Get appropriate timeframe data for the strategy
            if hasattr(strategy, 'timeframe'):
                timeframe = strategy.timeframe
            else:
                timeframe = mt5.TIMEFRAME_M15  # Default
                
            df = self.get_market_data(timeframe)
            
            # Check data sufficiency
            if len(df) < 20:
                logging.warning(f"Insufficient data: {len(df)} bars. Skipping iteration.")
                return
            
            # Execute strategy
            if hasattr(strategy, 'execute_strategy'):
                # New style strategy with built-in execution
                strategy.execute_strategy(df, sentiment_str, self.acc_info['balance'])
            else:
                # Legacy strategy - just generate signals
                signals = strategy.generate_signals(df)
                latest_signal = signals['signal'].iloc[-1]
                logging.info(f"{strategy_name} signal: {latest_signal} (legacy mode)")
            
            # Get sleep time from strategy config
            sleep_time = config.get('sleep_time', 900)  # Default 15 minutes
            
            logging.info(f"{strategy_name} iteration completed. Sleeping for {sleep_time} seconds...")
            
            # Sleep with interruption check
            start = time.time()
            while time.time() - start < sleep_time and not self.stop_event.is_set():
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"Trading iteration failed: {e}")
            time.sleep(60)  # Wait 1 minute before retry

    def trading_loop(self):
        """Main trading loop"""
        while not self.stop_event.is_set():
            try:
                self.trading_iteration()
            except Exception as e:
                logging.error(f"Unexpected error in trading loop: {e}")
                time.sleep(60)

    def run(self):
        """Main entry point"""
        def signal_handler(sig, frame):
            logging.info("SIGINT received, setting stop event")
            self.stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Initialize MT5
        if not self.initialize_mt5():
            return

        # Load classifier
        self.load_classifier()

        # Start trading thread
        trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()

        try:
            while trading_thread.is_alive() and not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Stopping... Ctrl+C received")
        finally:
            self.stop_event.set()
            self.shutdown_mt5()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean MT5 Gold AI Trader')
    parser.add_argument('symbol', nargs='?', type=str, default='XAUUSD', 
                       help='Trading symbol (default: XAUUSD)')
    parser.add_argument('bars', nargs='?', type=int, default=200, 
                       help='Number of historical bars (default: 200)')
    parser.add_argument('strategy', nargs='?', type=str, default='auto', 
                       choices=['auto', 'scalping', 'swing', 'day_trading', 'golden', 'goldstorm', 'vwap'], 
                       help='Trading strategy (default: auto)')
    
    args = parser.parse_args()
    
    bot = TradingBot(
        symbol=args.symbol, 
        bars=args.bars, 
        manual_strategy=args.strategy if args.strategy != 'auto' else None
    )
    bot.run()
