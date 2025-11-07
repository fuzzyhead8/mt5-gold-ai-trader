import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from strategies.base_strategy import BaseStrategy

class ScalpingStrategy(BaseStrategy):
    def __init__(self, symbol):
        super().__init__(symbol)
        self.min_volume = 50  # Moderate minimum tick volume 
        self.volatility_threshold = 2.5  # Dynamic volatility threshold

        
    def _calculate_rsi(self, prices, window=5):
        """Calculate RSI optimized for scalping (faster response)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices, window=12, num_std=1.8):
        """Calculate Bollinger Bands optimized for gold scalping"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band, rolling_mean
    
    def _calculate_momentum(self, prices, window=3):
        """Calculate short-term momentum"""
        return prices.pct_change(window)
    
    def _is_market_active(self, timestamp):
        """Check if market is in active trading hours (more lenient for scalping)"""
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            # Avoid very low activity hours (weekends handled elsewhere)
            # Allow most of the day except for the quietest hours
            return not (4 <= hour <= 6)  # Only avoid 4-6 AM (quietest period)
        return True  # Default to active if can't determine
    
    def generate_signals(self, data):
        """
        Enhanced scalping strategy with multiple indicators and risk management
        """
        # Ensure we have required columns
        required_cols = ['close', 'high', 'low', 'tick_volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'tick_volume':
                    data[col] = 100  # Default volume if missing
                else:
                    raise ValueError(f"Required column '{col}' missing from data")
        
        # Calculate technical indicators optimized for scalping
        data['ema_fast'] = data['close'].ewm(span=5).mean()  # Fast EMA (optimized)
        data['ema_slow'] = data['close'].ewm(span=13).mean()   # Slow EMA (Fibonacci number)
        data['rsi'] = self._calculate_rsi(data['close'], window=5)
        
        # Additional trend confirmation
        data['ema_trend'] = data['close'].ewm(span=21).mean()  # Longer trend filter
        data['trend_strength'] = (data['close'] - data['ema_trend']) / data['ema_trend']
        
        # Bollinger Bands for overbought/oversold conditions
        data['bb_upper'], data['bb_lower'], data['bb_middle'] = self._calculate_bollinger_bands(data['close'])
        
        # Volume indicators
        data['volume_ma'] = data['tick_volume'].rolling(window=10).mean()
        data['volume_ratio'] = data['tick_volume'] / data['volume_ma']
        
        # Volatility measures
        data['atr'] = (data['high'] - data['low']).rolling(window=7).mean()
        data['volatility'] = data['close'].rolling(window=10).std()
        
        # Momentum indicators
        data['momentum_3'] = self._calculate_momentum(data['close'], 3)
        data['momentum_5'] = self._calculate_momentum(data['close'], 5)
        data['price_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Price action indicators
        data['price_vs_ema'] = (data['close'] - data['ema_slow']) / data['ema_slow']
        
        # Generate signals
        signal = []
        for i in range(1, len(data)):
            current_time = data.index[i] if hasattr(data, 'index') and hasattr(data.index[i], 'hour') else None
            
            # Check if market is active
            if current_time and not self._is_market_active(current_time):
                signal.append('hold')
                continue
            
            # Current values
            ema_fast_curr = data['ema_fast'].iloc[i]
            ema_fast_prev = data['ema_fast'].iloc[i-1]
            ema_slow_curr = data['ema_slow'].iloc[i]
            ema_slow_prev = data['ema_slow'].iloc[i-1]
            rsi_curr = data['rsi'].iloc[i]
            volume_ratio = data['volume_ratio'].iloc[i]
            close_curr = data['close'].iloc[i]
            bb_upper = data['bb_upper'].iloc[i]
            bb_lower = data['bb_lower'].iloc[i]
            bb_middle = data['bb_middle'].iloc[i]
            momentum = data['momentum_3'].iloc[i]
            momentum_5 = data['momentum_5'].iloc[i]
            tick_volume = data['tick_volume'].iloc[i]
            volatility = data['volatility'].iloc[i]
            atr = data['atr'].iloc[i]
            price_pos = data['price_position'].iloc[i]
            trend_strength = data['trend_strength'].iloc[i]
            price_vs_ema = data['price_vs_ema'].iloc[i]
            
            # Skip if insufficient volume or extreme volatility (balanced filtering)
            if (tick_volume < self.min_volume or 
                pd.isna(volume_ratio) or 
                volatility > data['close'].iloc[i] * 0.001):  # 0.1% volatility limit (middle ground)
                signal.append('hold')
                continue
            
            # BUY CONDITIONS (Balanced approach - quality over quantity)
            buy_conditions = [
                # Strong trend confirmation with crossover
                ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev,
                # RSI in favorable range but not extreme
                30 < rsi_curr < 65,
                # Price near lower BB (oversold)
                price_pos < 0.4,
                # Positive momentum
                momentum > 0,
                # Above average volume
                volume_ratio > 1.0,
                # Controlled volatility
                volatility < data['volatility'].rolling(20).mean().iloc[i] * 1.3
            ]
            
            # SELL CONDITIONS (Balanced approach - quality over quantity)
            sell_conditions = [
                # Strong trend confirmation with crossover
                ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev,
                # RSI in favorable range but not extreme
                35 < rsi_curr < 70,
                # Price near upper BB (overbought)
                price_pos > 0.6,
                # Negative momentum
                momentum < 0,
                # Above average volume
                volume_ratio > 1.0,
                # Controlled volatility
                volatility < data['volatility'].rolling(20).mean().iloc[i] * 1.3
            ]
            
            # Balanced signal logic for optimal trade frequency
            if sum(buy_conditions) >= 4:  # Back to working threshold
                signal.append('buy')
            elif sum(sell_conditions) >= 4:  # Back to working threshold
                signal.append('sell')
            else:
                signal.append('hold')

        signal.insert(0, 'hold')  # no signal for first row
        data['signal'] = signal
        
        # Return enhanced data with all indicators
        return data[['close', 'signal', 'rsi', 'volume_ratio', 'volatility', 'ema_fast', 'ema_slow', 'bb_upper', 'bb_lower']]
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Scalping strategy: generate signals and place trades if conditions met.
        
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
            stop_loss, take_profit, lot_size, "Scalping"
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
            "strategy_name": "Scalping",
            "symbol": self.symbol,
            "timeframe": "M1",
            "parameters": {
                "min_volume": self.min_volume,
                "volatility_threshold": self.volatility_threshold,
                "ema_fast_period": 5,
                "ema_slow_period": 13,
                "ema_trend_period": 21,
                "rsi_period": 5,
                "bb_window": 12,
                "bb_std_dev": 1.8,
                "momentum_window": 3,
                "atr_window": 7,
                "volume_ma_window": 10,
                "volatility_window": 10
            },
            "signal_conditions": {
                "rsi_buy_range": [30, 65],
                "rsi_sell_range": [35, 70],
                "bb_position_buy_threshold": 0.4,
                "bb_position_sell_threshold": 0.6,
                "volume_ratio_threshold": 1.0,
                "conditions_required": 4
            },
            "description": "Fast scalping strategy optimized for 1-minute timeframe"
        }
