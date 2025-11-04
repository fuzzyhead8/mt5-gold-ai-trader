import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import threading
import argparse
import signal
import sys
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

import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

class TradingBot:
    def __init__(self, symbol="XAUUSD", bars=200):
        self.symbol = symbol
        self.bars = bars
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
                    "magic": 234000,
                    "comment": "Emergency Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
            logging.info("All positions closed")

    def load_classifier(self):
        self.classifier.load_model('models/trained_classifier.joblib')
        logging.info("Classifier loaded from trained model")

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
        else:
            return SwingTradingStrategy(self.symbol)

    def dynamic_iteration(self):
        if self.stop_event.is_set():
            return

        # Predict strategy first to determine timeframe
        # Fetch recent data for prediction (use M1 for quick prediction)
        m1_timeframe = mt5.TIMEFRAME_M1
        data = mt5.copy_rates_from_pos(self.symbol, m1_timeframe, 0, self.bars)
        if data is None:
            logging.error("Failed to fetch data for prediction")
            return
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        fetcher = NewsFetcher(api_key=os.getenv('NEWS_API_KEY'))
        articles = fetcher.fetch_news()
        sentiment_results = self.sentiment_handler.process_news(articles)
        sent_score = np.mean([item['score'] for item in sentiment_results]) if sentiment_results else 0

        predicted_strat = self.get_strategy(df, sent_score)
        strategy_type = predicted_strat.__class__.__name__.replace('Strategy', '').lower()
        logging.info(f"Predicted strategy: {strategy_type}")

        # Determine timeframe and sleep based on predicted strategy
        if strategy_type == 'scalping':
            timeframe = mt5.TIMEFRAME_M1
            sleep_time = 60  # 1 min
            sl_pips, tp_pips = 10, 20
            min_distance_pips = 1.0
            strategy_name = "scalping"
        elif strategy_type == 'day_trading':
            timeframe = mt5.TIMEFRAME_M15
            sleep_time = 900  # 15 min
            sl_pips, tp_pips = 100, 200
            min_distance_pips = 5.0
            strategy_name = "day_trading"
        else:  # swing
            timeframe = mt5.TIMEFRAME_M5
            sleep_time = 300  # 5 min
            sl_pips, tp_pips = 50, 125
            min_distance_pips = 2.0
            strategy_name = "swing"

        logging.info(f"Using {timeframe} timeframe for {strategy_name} iteration...")

        # Fetch data on selected timeframe
        data = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, self.bars)
        if data is None:
            logging.error(f"Failed to fetch {timeframe} data")
            return
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

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
            min_distance = symbol_info.trade_stops_level * point if symbol_info.trade_stops_level > 0 else min_distance_pips * point

            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logging.error("Failed to get tick")
                return
            price = tick.ask if latest_signal == 'buy' else tick.bid
            price = round(price, digits)

            sl_distance = max(sl_pips * point, min_distance)
            tp_distance = tp_pips * point
            stop_loss = round(price - sl_distance if latest_signal == 'buy' else price + sl_distance, digits)
            take_profit = round(price + tp_distance if latest_signal == 'buy' else price - tp_distance, digits)

            lot = self.lot_optimizer.optimize(
                balance=self.acc_info['balance'],
                entry_price=price,
                stop_price=stop_loss
            )
            lot = min(lot, 0.1)
            lot = round(lot, 2)

            result = self.trade_executor.send_order(action=latest_signal, lot=lot, price=price, sl=stop_loss, tp=take_profit)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"{strategy_name.capitalize()} trade executed successfully: {latest_signal} {lot} lots of {self.symbol} at {price}, SL: {stop_loss}, TP: {take_profit}")

                # Log trade to JSON
                sentiment_str = "bullish" if sent_score > 0 else "bearish" if sent_score < 0 else "neutral"
                trade_data = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "symbol": self.symbol,
                    "action": latest_signal,
                    "entry_price": price,
                    "exit_price": None,
                    "volume": lot,
                    "profit_usd": 0.0,
                    "strategy": strategy_name,
                    "sentiment": sentiment_str,
                    "duration_minutes": 0
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
                    logging.info("Trade logged to logs/trade_logs.json")
                except Exception as e:
                    logging.error(f"Failed to log trade: {e}")
            else:
                logging.error(f"{strategy_name.capitalize()} trade failed: {result.retcode if result else 'No result'} - {result.comment if result else 'Unknown error'}")

        logging.info(f"{strategy_name.capitalize()} iteration completed. Sleeping for {sleep_time} seconds...")
        start = time.time()
        while time.time() - start < sleep_time and not self.stop_event.is_set():
            time.sleep(0.1)



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
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Trading symbol (default: XAUUSD)')
    parser.add_argument('--bars', type=int, default=200, help='Number of historical bars to fetch (default: 200)')
    args = parser.parse_args()

    bot = TradingBot(symbol=args.symbol, bars=args.bars)
    bot.run()
