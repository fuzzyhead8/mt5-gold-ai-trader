import numpy as np
import pandas as pd

class ScalpingStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.min_volume = 50  # Moderate minimum tick volume 
        self.volatility_threshold = 2.5  # Dynamic volatility threshold
        
    def _calculate_rsi(self, prices, window=7):
        """Calculate RSI optimized for scalping"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices, window=10, num_std=1.5):
        """Calculate Bollinger Bands for scalping (tighter bands)"""
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
        data['ema_fast'] = data['close'].ewm(span=3).mean()  # Very fast EMA
        data['ema_slow'] = data['close'].ewm(span=8).mean()   # Slow EMA
        data['rsi'] = self._calculate_rsi(data['close'], window=7)
        
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
        data['price_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
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
            tick_volume = data['tick_volume'].iloc[i]
            volatility = data['volatility'].iloc[i]
            atr = data['atr'].iloc[i]
            price_pos = data['price_position'].iloc[i]
            
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
            
            # Require 4 out of 6 conditions for signal (balanced approach)
            if sum(buy_conditions) >= 4:
                signal.append('buy')
            elif sum(sell_conditions) >= 4:
                signal.append('sell')
            else:
                signal.append('hold')

        signal.insert(0, 'hold')  # no signal for first row
        data['signal'] = signal
        
        # Return enhanced data with all indicators
        return data[['close', 'signal', 'rsi', 'volume_ratio', 'volatility', 'ema_fast', 'ema_slow', 'bb_upper', 'bb_lower']]
