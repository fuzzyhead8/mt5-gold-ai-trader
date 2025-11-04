import numpy as np
import pandas as pd

class DayTradingStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        # Parameters optimized for M15 timeframe (daily trading)
        self.rsi_period = 21  # Longer period for more stable signals
        self.rsi_overbought = 75  # More extreme levels
        self.rsi_oversold = 25
        self.ema_fast = 12  # For trend filtering
        self.ema_slow = 26
        self.volume_threshold = 80  # Minimum volume threshold

    def _calculate_rsi(self, prices, window=21):
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
            
            # Time-based filtering
            current_time = data.index[i] if hasattr(data, 'index') else None
            if current_time and not self._is_valid_signal_time(current_time):
                signals.append('hold')
                continue
                
            # Enhanced signal logic with multiple confirmations
            signal = 'hold'
            
            # BUY conditions (trend-following with RSI oversold confirmation)
            buy_conditions = [
                rsi < self.rsi_oversold,  # RSI oversold
                trend == 1,  # Uptrend (EMA fast > EMA slow)
                volume_ratio > 1.0,  # Above average volume
                volatility < data['volatility'].rolling(50).mean().iloc[i] * 1.5  # Not extremely volatile
            ]
            
            # SELL conditions (trend-following with RSI overbought confirmation)  
            sell_conditions = [
                rsi > self.rsi_overbought,  # RSI overbought
                trend == -1,  # Downtrend (EMA fast < EMA slow)
                volume_ratio > 1.0,  # Above average volume
                volatility < data['volatility'].rolling(50).mean().iloc[i] * 1.5  # Not extremely volatile
            ]
            
            # Require at least 3 out of 4 conditions to be met
            if sum(buy_conditions) >= 3:
                # Additional check: ensure we're not in extreme volatility
                recent_volatility = data['volatility'].iloc[max(0, i-5):i+1].mean()
                if recent_volatility < data['volatility'].iloc[i] * 1.2:
                    signal = 'buy'
            elif sum(sell_conditions) >= 3:
                # Additional check: ensure we're not in extreme volatility
                recent_volatility = data['volatility'].iloc[max(0, i-5):i+1].mean()
                if recent_volatility < data['volatility'].iloc[i] * 1.2:
                    signal = 'sell'
            
            signals.append(signal)

        data['signal'] = signals
        
        # Return enhanced dataset with indicators
        return data[['close', 'RSI', 'EMA_fast', 'EMA_slow', 'signal', 'volume_ratio', 'volatility', 'trend']]
