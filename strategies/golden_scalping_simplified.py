import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from strategies.base_strategy import BaseStrategy

class GoldenScalpingStrategySimplified(BaseStrategy):
    """
    GOLDEN FORMULA - SIMPLIFIED VERSION
    Removes over-optimization and focuses on robust, fundamental indicators
    Designed to reduce backtest/live trading discrepancy
    """
    
    def __init__(self, symbol):
        super().__init__(symbol)
        self.min_volume = 30  # Reduced volume filter

        
    def _calculate_rsi(self, prices, window=14):
        """Standard RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Standard MACD calculation"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _is_valid_trading_time(self, timestamp):
        """Check if current time is valid for trading"""
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            # Avoid low liquidity periods (Friday 21:00 - Sunday 21:00 UTC)
            # And very early Monday morning (00:00-03:00 UTC)
            avoid_periods = (22 <= hour <= 23) or (0 <= hour <= 3)
            return not avoid_periods
        return True
    
    def generate_signals(self, data):
        """
        SIMPLIFIED GOLDEN FORMULA: Focus on 3-4 robust indicators
        Reduces overfitting and improves live trading alignment
        Vectorized version for performance optimization
        """
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'tick_volume':
                    data[col] = 100
                else:
                    raise ValueError(f"Required column '{col}' missing from data")
        
        # Core indicators - simplified and robust (vectorized)
        data['ema_fast'] = data['close'].ewm(span=8).mean()
        data['ema_slow'] = data['close'].ewm(span=21).mean()
        data['rsi'] = self._calculate_rsi(data['close'], window=14)
        data['macd'], data['macd_signal'], data['macd_histogram'] = self._calculate_macd(data['close'])
        
        # Price momentum (simplified) and volume (vectorized)
        data['price_momentum'] = data['close'].pct_change(3)
        data['volume_avg'] = data['tick_volume'].rolling(window=20).mean()
        
        # EMA momentum (vectorized)
        data['ema_fast_rising'] = data['ema_fast'] > data['ema_fast'].shift(1)
        data['ema_fast_falling'] = data['ema_fast'] < data['ema_fast'].shift(1)
        
        # MACD momentum (vectorized)
        data['macd_hist_improving'] = data['macd_histogram'] > data['macd_histogram'].shift(1)
        data['macd_hist_deteriorating'] = data['macd_histogram'] < data['macd_histogram'].shift(1)
        
        # Trading time validity (vectorized)
        if hasattr(data.index, 'hour'):
            data['hour'] = data.index.hour
            valid_time = ~(((data['hour'] >= 22) | (data['hour'] <= 3)))
            data.drop('hour', axis=1, inplace=True)
        else:
            valid_time = pd.Series(True, index=data.index)
        
        # Skip conditions (vectorized)
        insufficient_volume = data['tick_volume'] < self.min_volume
        invalid_rsi = data['rsi'].isna()
        invalid_macd = data['macd_histogram'].isna()
        skip_mask = insufficient_volume | invalid_rsi | invalid_macd
        
        # SIMPLIFIED BUY CONDITIONS (all 4 required - vectorized AND)
        buy_trend = (data['ema_fast'] > data['ema_slow']) & data['ema_fast_rising']
        buy_rsi = (data['rsi'] > 25) & (data['rsi'] < 70)
        buy_macd = data['macd_hist_improving'] & (data['macd_histogram'] > -0.5)
        buy_momentum = (data['price_momentum'] > 0.0001) & (data['tick_volume'] > data['volume_avg'] * 0.8)
        
        buy_signal = buy_trend & buy_rsi & buy_macd & buy_momentum
        
        # SIMPLIFIED SELL CONDITIONS (all 4 required - vectorized AND)
        sell_trend = (data['ema_fast'] < data['ema_slow']) & data['ema_fast_falling']
        sell_rsi = (data['rsi'] > 30) & (data['rsi'] < 75)
        sell_macd = data['macd_hist_deteriorating'] & (data['macd_histogram'] < 0.5)
        sell_momentum = (data['price_momentum'] < -0.0001) & (data['tick_volume'] > data['volume_avg'] * 0.8)
        
        sell_signal = sell_trend & sell_rsi & sell_macd & sell_momentum
        
        # Generate signals (vectorized)
        data['signal'] = 'hold'
        valid_mask = valid_time & ~skip_mask
        
        data.loc[valid_mask & buy_signal, 'signal'] = 'buy'
        data.loc[valid_mask & sell_signal, 'signal'] = 'sell'
        
        # Fill initial signals with hold (up to the point where indicators are valid)
        first_valid_idx = max(26, data.index.get_loc(data.first_valid_index()) if hasattr(data, 'first_valid_index') else 0)
        data.iloc[:first_valid_idx, data.columns.get_loc('signal')] = 'hold'
        
        # Drop temporary columns
        temp_cols = ['ema_fast_rising', 'ema_fast_falling', 'macd_hist_improving', 
                     'macd_hist_deteriorating', 'volume_avg']
        data.drop(columns=[col for col in temp_cols if col in data.columns], inplace=True, errors='ignore')
        
        # Return essential columns only
        return_cols = ['open', 'high', 'low', 'close', 'signal', 'rsi', 'macd_histogram', 
                      'price_momentum', 'ema_fast', 'ema_slow']
        
        return data[return_cols]
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Golden Scalping Simplified strategy: generate signals and place trades if conditions met.
        
        Args:
            df: Market data DataFrame
            sentiment: Current market sentiment ('bullish', 'bearish', 'neutral')
            balance: Current account balance
            
        Returns:
            dict: Execution result with status and details
        """
        # Generate signals
        signals_df = self.generate_signals(df)
        if len(signals_df) == 0:
            self.logger.info("No data available for signal generation")
            return {"status": "no_data", "signal": "hold"}
        
        latest_signal = signals_df['signal'].iloc[-1]
        if latest_signal not in ['buy', 'sell']:
            self.logger.info(f"No actionable signal generated: {latest_signal}")
            return {"status": "no_signal", "signal": latest_signal}
        
        # Validate signal with sentiment
        if not self.validate_signal_with_sentiment(latest_signal, sentiment):
            self.logger.info(f"Signal '{latest_signal}' rejected due to sentiment '{sentiment}'")
            return {"status": "sentiment_rejected", "signal": latest_signal}
        
        # Get current market price
        current_price = self.get_market_price(latest_signal)
        if current_price is None:
            self.logger.error("Failed to get current market price")
            return {"status": "no_price", "signal": latest_signal}
        
        # Get symbol info for point and digits
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            self.logger.error(f"Failed to get symbol info for {self.symbol}")
            return {"status": "no_symbol_info", "signal": latest_signal}
        
        point = symbol_info.point
        digits = symbol_info.digits
        
        # Define SL and TP distances (50 pips SL, 100 pips TP for 1:2 RR)
        # For XAUUSD, 1 pip = 0.1, point = 0.01, so 50 pips = 5.0 price units
        sl_distance = 5.0  # 50 pips
        tp_distance = 10.0  # 100 pips
        
        # Calculate SL and TP levels
        if latest_signal == 'buy':
            stop_loss = round(current_price - sl_distance, digits)
            take_profit = round(current_price + tp_distance, digits)
        else:  # sell
            stop_loss = round(current_price + sl_distance, digits)
            take_profit = round(current_price - tp_distance, digits)
        
        # Calculate position size based on risk (2% of balance)
        lot_size = self.calculate_position_size(balance, current_price, stop_loss, risk_percent=2.0)
        
        # Validate stop distances
        if not self.validate_stop_distances(current_price, stop_loss, take_profit):
            self.logger.warning("Stop loss/take profit distances invalid")
            return {"status": "invalid_stops", "signal": latest_signal}
        
        # Execute the trade
        result = self.execute_trade(
            latest_signal, sentiment, current_price, 
            stop_loss, take_profit, lot_size, "GoldenScalpingSimplified"
        )
        
        if result:
            self.logger.info(f"Trade executed successfully: {latest_signal} at {current_price}, lot: {lot_size}")
            return {
                "status": "executed", 
                "signal": latest_signal, 
                "ticket": result.order,
                "price": current_price,
                "sl": stop_loss,
                "tp": take_profit,
                "lot_size": lot_size
            }
        else:
            self.logger.error(f"Failed to execute {latest_signal} trade")
            return {"status": "execution_failed", "signal": latest_signal}

    def get_strategy_config(self) -> dict:
        """Return strategy configuration parameters"""
        return {
            "strategy_name": "Golden Scalping Simplified",
            "symbol": self.symbol,
            "timeframe": "M5",
            "parameters": {
                "min_volume": self.min_volume,
                "ema_fast_period": 8,
                "ema_slow_period": 21,
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "price_momentum_lookback": 3,
                "volume_average_period": 20
            },
            "signal_conditions": {
                "rsi_buy_range": [25, 70],
                "rsi_sell_range": [30, 75],
                "price_momentum_threshold": 0.0001,
                "volume_multiplier": 0.8,
                "macd_histogram_threshold": 0.5
            },
            "description": "Simplified Golden Formula with 4 robust conditions per signal"
        }
