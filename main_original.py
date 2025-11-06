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
import json
from datetime import datetime
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
from strategies.golden_scalping_simplified import GoldenScalpingStrategySimplified
from strategies.goldstorm_strategy import GoldStormStrategy
from strategies.vwap_strategy import VWAPStrategy

import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class TradingBot:
    def __init__(self, symbol="XAUUSD", bars=200, manual_strategy=None):
        self.symbol = symbol
        self.bars = bars
        self.manual_strategy = manual_strategy
        self.classifier = StrategyClassifier()
        self.sentiment_handler = NewsSentimentHandler()
        self.lot_optimizer = LotSizeOptimizer()
        self.trade_executor = TradeExecutor(self.symbol)
        self.order_manager = OrderManager(self.symbol)
        self.trend_model = TrendPredictor()
        self.account_handler = AccountHandler()
        self.acc_info = None
        self.connected = False
        self.stop_event = threading.Event()
        self.position_tracker = {}  # Track open positions for monitoring

    def initialize_mt5(self):
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
        if self.connected:
            self.close_all_positions()
            mt5.shutdown()
            self.connected = False
            logging.info("MT5 connection closed and positions cleared")

    def close_all_positions(self):
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
        self.classifier.load_model('models/trained_classifier.joblib')
        logging.info("Classifier loaded from trained model")

    def validate_signal_with_sentiment(self, signal: str, sentiment: str) -> bool:
        """
        Validate if a trading signal aligns with market sentiment
        
        Args:
            signal: 'buy' or 'sell'
            sentiment: 'bullish', 'bearish', or 'neutral'
            
        Returns:
            bool: True if signal is valid for the sentiment
        """
        # Trading logic:
        # - Buy only on bullish or neutral sentiment
        # - Sell only on bearish or neutral sentiment
        # - Never buy on bearish sentiment
        # - Never sell on bullish sentiment
        
        if signal == 'buy':
            return sentiment in ['bullish', 'neutral']
        elif signal == 'sell':
            return sentiment in ['bearish', 'neutral']
        else:
            return False

    def get_strategy(self, df, sent_score):
        volatility = df['close'].rolling(window=10).std().iloc[-1]
        volume = df['tick_volume'].iloc[-1]
        momentum = df['close'].pct_change().rolling(5).mean().iloc[-1]
        input_df = pd.DataFrame([[volatility, volume, momentum, sent_score]], 
                                columns=['volatility', 'volume', 'momentum', 'sentiment_score'])
        strategy_type = self.classifier.predict(input_df)[0]
        logging.info(f"Predicted strategy: {strategy_type}")
        if strategy_type == 'scalping':
            return ScalpingStrategy(self.symbol)
        elif strategy_type == 'day_trading':
            return DayTradingStrategy(self.symbol)
        elif strategy_type == 'golden':
            return GoldenScalpingStrategySimplified(self.symbol)
        else:
            return SwingTradingStrategy(self.symbol)

    def dynamic_iteration(self):
        if self.stop_event.is_set():
            return

        # Initialize with fallback values
        sent_score = 0  # Default neutral sentiment
        error_count = 0
        max_errors = 3
        
        if self.manual_strategy:
            strategy_type = self.manual_strategy
            logging.info(f"Manual strategy: {strategy_type}")

            # Fetch sentiment with improved error handling and rate limiting
            fetcher = NewsFetcher(api_key=os.getenv('NEWS_API_KEY'), cache_duration_hours=4)  # Longer cache
            try:
                # Only fetch news if cache is expired, otherwise use cached data
                articles = fetcher.get_cached_or_fallback()
                if not articles:
                    logging.info("No cached news, attempting fresh fetch...")
                    articles = fetcher.fetch_news(page_size=10)  # Reduced page size
                
                sentiment_results = self.sentiment_handler.process_news(articles)
                sent_score = np.mean([item['score'] for item in sentiment_results]) if sentiment_results else 0
                logging.info(f"Using {len(articles)} news articles for sentiment analysis (score: {sent_score:.3f})")
            except Exception as e:
                error_count += 1
                logging.warning(f"News processing failed ({error_count}/{max_errors}): {e}")
                # Use neutral sentiment as fallback
                sent_score = 0

            # Instantiate strategy based on manual input
            if strategy_type == 'scalping':
                predicted_strat = ScalpingStrategy(self.symbol)
            elif strategy_type == 'day_trading':
                predicted_strat = DayTradingStrategy(self.symbol)
            elif strategy_type == 'golden':
                predicted_strat = GoldenScalpingStrategySimplified(self.symbol)
            elif strategy_type == 'goldstorm':
                predicted_strat = GoldStormStrategy(self.symbol)
            elif strategy_type == 'vwap':
                predicted_strat = VWAPStrategy(self.symbol)
            else:
                predicted_strat = SwingTradingStrategy(self.symbol)
        else:
            # Auto mode: Predict strategy first to determine timeframe
            # Fetch recent data for prediction with retry logic
            m1_timeframe = mt5.TIMEFRAME_M1
            data = None
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries and data is None:
                try:
                    data = mt5.copy_rates_from_pos(self.symbol, m1_timeframe, 0, min(self.bars, 100))  # Reduced bars for prediction
                    if data is None:
                        error_count += 1
                        retry_count += 1
                        logging.warning(f"MT5 data fetch failed (attempt {retry_count}/{max_retries})")
                        if retry_count < max_retries:
                            time.sleep(2 ** retry_count)  # Exponential backoff
                    break
                except Exception as e:
                    error_count += 1
                    retry_count += 1
                    logging.error(f"MT5 data fetch error (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        time.sleep(2 ** retry_count)
            
            if data is None:
                logging.error(f"Failed to fetch MT5 data after {max_retries} attempts. Skipping iteration.")
                # Wait longer before next attempt to avoid hammering failed connection
                time.sleep(30)
                return
                
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Fetch sentiment with caching priority
            try:
                fetcher = NewsFetcher(api_key=os.getenv('NEWS_API_KEY'), cache_duration_hours=4)
                # Always try cached first to avoid API limits
                articles = fetcher.get_cached_or_fallback()
                if not articles:
                    logging.info("No cached news available, attempting fresh fetch...")
                    articles = fetcher.fetch_news(page_size=10)
                
                sentiment_results = self.sentiment_handler.process_news(articles)
                sent_score = np.mean([item['score'] for item in sentiment_results]) if sentiment_results else 0
                logging.info(f"Using {len(articles)} news articles for sentiment analysis (score: {sent_score:.3f})")
            except Exception as e:
                error_count += 1
                logging.warning(f"News processing failed ({error_count}/{max_errors}): {e}")
                sent_score = 0

            predicted_strat = self.get_strategy(df, sent_score)
            strategy_type = predicted_strat.__class__.__name__.replace('Strategy', '').lower()
            logging.info(f"Predicted strategy: {strategy_type}")

        # Determine timeframe and sleep based on predicted strategy
        if strategy_type == 'scalping':
            timeframe = mt5.TIMEFRAME_M1
            sleep_time = 60  # 1 min
            sl_pips, tp_pips = 50, 80  # Increased for XAUUSD minimum requirements
            min_distance_pips = 30.0  # Much higher minimum for Gold
            strategy_name = "scalping"
        elif strategy_type == 'golden':
            timeframe = mt5.TIMEFRAME_M15  # M15 for GOLDEN FORMULA
            sleep_time = 900  # 15 min - optimized for signal quality
            sl_pips, tp_pips = 60, 120  # Optimized for GOLDEN FORMULA risk/reward
            min_distance_pips = 35.0  # Professional-grade minimum distance
            strategy_name = "golden"
        elif strategy_type == 'goldstorm':
            timeframe = mt5.TIMEFRAME_M15  # M15 for GOLDSTORM strategy
            sleep_time = 900  # 15 min - optimized for volatility analysis
            sl_pips, tp_pips = 150, 300  # Higher SL/TP for trend-following approach (2:1 RR)
            min_distance_pips = 50.0  # Higher minimum for more selective entries
            strategy_name = "goldstorm"
        elif strategy_type == 'vwap':
            timeframe = mt5.TIMEFRAME_M15  # M15 for VWAP strategy
            sleep_time = 900  # 15 min - optimized for VWAP analysis
            sl_pips, tp_pips = 80, 160  # ATR-based SL/TP as per VWAP strategy
            min_distance_pips = 40.0  # Professional minimum for VWAP entries
            strategy_name = "vwap"
        elif strategy_type == 'day_trading':
            timeframe = mt5.TIMEFRAME_M15
            sleep_time = 900  # 15 min
            sl_pips, tp_pips = 100, 200
            min_distance_pips = 40.0  # Higher minimum for Gold
            strategy_name = "day_trading"
        else:  # swing
            timeframe = mt5.TIMEFRAME_H4
            sleep_time = 3600  # 1 hour
            sl_pips, tp_pips = 80, 160  # Increased for better Gold trading
            min_distance_pips = 35.0  # Higher minimum for Gold
            strategy_name = "swing"

        logging.info(f"Using {timeframe} timeframe for {strategy_name} iteration...")

        # Fetch data on selected timeframe with retry logic
        data = None
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries and data is None:
            try:
                data = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, self.bars)
                if data is None:
                    error_count += 1
                    retry_count += 1
                    logging.warning(f"MT5 {timeframe} data fetch failed (attempt {retry_count}/{max_retries})")
                    if retry_count < max_retries:
                        time.sleep(2 ** retry_count)  # Exponential backoff
                break
            except Exception as e:
                error_count += 1
                retry_count += 1
                logging.error(f"MT5 {timeframe} data fetch error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)
        
        if data is None:
            logging.error(f"Failed to fetch {timeframe} data after {max_retries} attempts. Skipping iteration.")
            # Longer wait on persistent failures to avoid hammering
            time.sleep(60)
            return
            
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Check if we have enough data
        if len(df) < 20:
            logging.warning(f"Insufficient data received: {len(df)} bars. Skipping iteration.")
            time.sleep(30)
            return

        signals = predicted_strat.generate_signals(df)
        latest_signal = signals['signal'].iloc[-1]
        logging.info(f"{strategy_name.capitalize()} signal: {latest_signal}")

        if latest_signal in ['buy', 'sell']:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logging.error("Failed to get symbol info")
                return
            point = symbol_info.point
            digits = symbol_info.digits
            
            # Get broker's minimum stop distance
            broker_min_distance = symbol_info.trade_stops_level * point if symbol_info.trade_stops_level > 0 else 0
            # Use the larger of broker requirement or our minimum
            min_distance = max(broker_min_distance, min_distance_pips * point)
            
            logging.info(f"Broker min distance: {broker_min_distance/point:.1f} pips, Our min: {min_distance_pips} pips, Using: {min_distance/point:.1f} pips")

            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logging.error("Failed to get tick")
                return
            price = tick.ask if latest_signal == 'buy' else tick.bid
            price = round(price, digits)

            # Calculate distances ensuring they meet minimum requirements
            sl_distance = max(sl_pips * point, min_distance)
            tp_distance = max(tp_pips * point, min_distance)  # TP also needs minimum distance
            
            stop_loss = round(price - sl_distance if latest_signal == 'buy' else price + sl_distance, digits)
            take_profit = round(price + tp_distance if latest_signal == 'buy' else price - tp_distance, digits)
            
            # Validate stop distances
            actual_sl_distance = abs(stop_loss - price)
            actual_tp_distance = abs(take_profit - price)
            
            if actual_sl_distance < min_distance or actual_tp_distance < min_distance:
                logging.error(f"Stop distances too small - SL: {actual_sl_distance/point:.1f} pips, TP: {actual_tp_distance/point:.1f} pips, Min required: {min_distance/point:.1f} pips")
                return
                
            logging.info(f"Trade levels - Price: {price}, SL: {stop_loss} ({actual_sl_distance/point:.1f} pips), TP: {take_profit} ({actual_tp_distance/point:.1f} pips)")

            lot = self.lot_optimizer.optimize(
                balance=self.acc_info['balance'],
                entry_price=price,
                stop_price=stop_loss
            )
            lot = min(lot, 0.1)
            lot = round(lot, 2)

            # Create comment with strategy name
            order_comment = f"AI bot - {strategy_name}"
            
            # CRITICAL FIX: Validate signal against sentiment
            sentiment_str = "bullish" if sent_score > 0 else "bearish" if sent_score < 0 else "neutral"
            
            # Check if signal aligns with sentiment
            if not self.validate_signal_with_sentiment(latest_signal, sentiment_str):
                logging.warning(f"Signal {latest_signal} rejected due to conflicting sentiment ({sentiment_str}). "
                              f"Only buy on bullish/neutral, sell on bearish/neutral sentiment.")
                return
            
            logging.info(f"Signal {latest_signal} approved for {sentiment_str} sentiment")

            result = self.trade_executor.send_order(action=latest_signal, lot=lot, price=price, sl=stop_loss, tp=take_profit, comment=order_comment)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"{strategy_name.capitalize()} trade executed successfully: {latest_signal} {lot} lots of {self.symbol} at {price}, SL: {stop_loss}, TP: {take_profit}")

                # Enhanced trade logging with position tracking
                trade_data = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": self.symbol,
                    "action": latest_signal,
                    "entry_price": price,
                    "exit_price": None,
                    "volume": lot,
                    "profit_usd": 0.0,
                    "strategy": strategy_name,
                    "sentiment": sentiment_str,
                    "duration_minutes": 0,
                    "ticket": result.order,
                    "status": "open"
                }
                
                try:
                    log_file = "logs/trade_logs.json"
                    if os.path.exists(log_file):
                        with open(log_file, "r") as f:
                            data = json.load(f)
                        trades = data.get("trades", [])
                    else:
                        trades = []
                    trades.append(trade_data)
                    data = {"trades": trades}
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    with open(log_file, "w") as f:
                        json.dump(data, f, indent=2)
                    logging.info(f"Trade logged with ticket {result.order}")
                    
                    # Add to position tracker for monitoring
                    self.position_tracker[result.order] = {
                        'trade_data': trade_data,
                        'opening_time': datetime.now()
                    }
                    
                except Exception as e:
                    logging.error(f"Failed to log trade: {e}")
            else:
                logging.error(f"{strategy_name.capitalize()} trade failed: {result.retcode if result else 'No result'} - {result.comment if result else 'Unknown error'}")

        # Implement adaptive sleep based on error count
        base_sleep = sleep_time
        if error_count > 0:
            # Add extra delay if there were errors to prevent hammering failed services
            adaptive_sleep = base_sleep + (error_count * 30)  # Add 30s per error
            logging.warning(f"Errors detected ({error_count}), extending sleep to {adaptive_sleep}s")
            sleep_time = min(adaptive_sleep, base_sleep * 3)  # Cap at 3x normal sleep
        
        logging.info(f"{strategy_name.capitalize()} iteration completed. Sleeping for {sleep_time} seconds...")
        start = time.time()
        while time.time() - start < sleep_time and not self.stop_event.is_set():
            time.sleep(1)  # Use 1s intervals instead of 0.1s



    def dynamic_loop(self):
        while not self.stop_event.is_set():
            self.dynamic_iteration()


    def run(self):
        def signal_handler(sig, frame):
            logging.info("SIGINT received, setting stop event")
            self.stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        if not self.initialize_mt5():
            return

        self.load_classifier()

        dynamic_thread = threading.Thread(target=self.dynamic_loop, daemon=True)
        dynamic_thread.start()

        try:
            while dynamic_thread.is_alive() and not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Stopping... Ctrl+C received")
        finally:
            self.stop_event.set()
            self.shutdown_mt5()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MT5 Gold AI Trader with Dynamic Strategy Selection')
    parser.add_argument('symbol', nargs='?', type=str, default='XAUUSD', help='Trading symbol (default: XAUUSD)')
    parser.add_argument('bars', nargs='?', type=int, default=200, help='Number of historical bars to fetch (default: 200)')
    parser.add_argument('strategy', nargs='?', type=str, default='auto', choices=['auto', 'scalping', 'swing', 'day_trading', 'golden', 'goldstorm', 'vwap'], help='Trading strategy (auto for AI prediction, or specify scalping/swing/day_trading/golden/goldstorm/vwap)')
    args = parser.parse_args()

    bot = TradingBot(symbol=args.symbol, bars=args.bars, manual_strategy=args.strategy if args.strategy != 'auto' else None)
    bot.run()
