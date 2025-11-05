import numpy as np
import pandas as pd

class DayTradingStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
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
