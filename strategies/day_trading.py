import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from strategies.base_strategy import BaseStrategy

class DayTradingStrategy(BaseStrategy):
    def __init__(self, symbol):
        super().__init__(symbol)
        # Parameters optimized for M15 timeframe (daily trading) - Made less conservative
        self.rsi_period = 14  # Shorter period for more responsive signals
        self.rsi_overbought = 65  # Less extreme levels for more signals
        self.rsi_oversold = 35
        self.ema_fast = 12  # For trend filtering
        self.ema_slow = 26
        self.volume_threshold = 50  # Lower minimum volume threshold


    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI with proper handling of edge cases"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value

    def _calculate_ema(self, prices, span):
        """Calculate exponential moving average"""
        return prices.ewm(span=span, adjust=False).mean()

    def _is_valid_signal_time(self, timestamp):
        """Check if time is suitable for day trading (avoid low liquidity periods)"""
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            # Active trading hours: 07:00-17:00 (European + US overlap)
            return 7 <= hour <= 17
        return True  # Default to valid if can't determine time

    def generate_signals(self, data):
        """
        Enhanced day trading strategy with RSI, trend filtering, and volume confirmation
        """
        # Ensure we have required columns
        if 'tick_volume' not in data.columns:
            data['tick_volume'] = 100  # Default volume if missing
        
        # Calculate technical indicators
        data['RSI'] = self._calculate_rsi(data['close'], self.rsi_period)
        data['EMA_fast'] = self._calculate_ema(data['close'], self.ema_fast)
        data['EMA_slow'] = self._calculate_ema(data['close'], self.ema_slow)
        
        # Volume analysis
        data['volume_ma'] = data['tick_volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['tick_volume'] / data['volume_ma']
        
        # Price momentum and volatility
        data['price_change'] = data['close'].pct_change()
        data['volatility'] = data['close'].rolling(window=20).std()
        data['atr'] = ((data['high'] - data['low']).rolling(window=14).mean())
        
        # Trend direction
        data['trend'] = np.where(data['EMA_fast'] > data['EMA_slow'], 1, -1)
        
        signals = []
        for i in range(len(data)):
            if i == 0:
                signals.append('hold')
                continue
                
            # Current values
            rsi = data['RSI'].iloc[i]
            ema_fast = data['EMA_fast'].iloc[i]
            ema_slow = data['EMA_slow'].iloc[i]
            volume_ratio = data['volume_ratio'].iloc[i]
            trend = data['trend'].iloc[i]
            volatility = data['volatility'].iloc[i]
            tick_volume = data['tick_volume'].iloc[i]
            
            # Skip if insufficient data or extreme conditions
            if (pd.isna(rsi) or pd.isna(ema_fast) or pd.isna(ema_slow) or 
                tick_volume < self.volume_threshold):
                signals.append('hold')
                continue
            
            # Note: Removed time-based filtering to allow more signals during training
                
            # Enhanced signal logic with multiple confirmations
            signal = 'hold'
            
            # More balanced signal generation with tighter controls
            
            # BUY conditions - require RSI oversold AND trending up
            if (rsi < self.rsi_oversold and trend == 1 and 
                volume_ratio > 1.0 and 
                data['EMA_fast'].iloc[i] > data['EMA_fast'].iloc[i-1]):  # EMA fast rising
                signal = 'buy'
                
            # SELL conditions - require RSI overbought AND trending down
            elif (rsi > self.rsi_overbought and trend == -1 and 
                  volume_ratio > 1.0 and 
                  data['EMA_fast'].iloc[i] < data['EMA_fast'].iloc[i-1]):  # EMA fast falling
                signal = 'sell'
                
            # Additional opportunity - RSI reversal signals
            elif rsi < 30 and trend == 1:  # Strong oversold in uptrend
                signal = 'buy'
            elif rsi > 70 and trend == -1:  # Strong overbought in downtrend
                signal = 'sell'
            
            signals.append(signal)

        data['signal'] = signals
        
        # Return enhanced dataset with indicators
        return data[['close', 'RSI', 'EMA_fast', 'EMA_slow', 'signal', 'volume_ratio', 'volatility', 'trend']]
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Day Trading strategy: generate signals and place trades if conditions met.
        
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
            stop_loss, take_profit, lot_size, "DayTrading"
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
            "strategy_name": "Day Trading",
            "symbol": self.symbol,
            "timeframe": "M15",
            "parameters": {
                "rsi_period": self.rsi_period,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold,
                "ema_fast_period": self.ema_fast,
                "ema_slow_period": self.ema_slow,
                "volume_threshold": self.volume_threshold,
                "volume_ma_window": 20,
                "volatility_window": 20,
                "atr_window": 14
            },
            "signal_conditions": {
                "rsi_extreme_levels": [30, 70],
                "volume_ratio_threshold": 1.0,
                "trend_confirmation_required": True,
                "ema_momentum_required": True
            },
            "description": "Day trading strategy optimized for 15-minute timeframe with RSI and trend filtering"
        }
