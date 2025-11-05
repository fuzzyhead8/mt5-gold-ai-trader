import numpy as np
import pandas as pd

class GoldenScalpingStrategySimplified:
    """
    GOLDEN FORMULA - SIMPLIFIED VERSION
    Removes over-optimization and focuses on robust, fundamental indicators
    Designed to reduce backtest/live trading discrepancy
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
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
        """
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'tick_volume':
                    data[col] = 100
                else:
                    raise ValueError(f"Required column '{col}' missing from data")
        
        # Core indicators - simplified and robust
        data['ema_fast'] = data['close'].ewm(span=8).mean()
        data['ema_slow'] = data['close'].ewm(span=21).mean()
        data['rsi'] = self._calculate_rsi(data['close'], window=14)
        data['macd'], data['macd_signal'], data['macd_histogram'] = self._calculate_macd(data['close'])
        
        # Price momentum (simplified)
        data['price_momentum'] = data['close'].pct_change(3)
        data['volume_avg'] = data['tick_volume'].rolling(window=20).mean()
        
        # Generate signals using SIMPLIFIED GOLDEN FORMULA
        signal = []
        
        for i in range(max(26, len(data.columns)), len(data)):
            current_time = data.index[i] if hasattr(data, 'index') and hasattr(data.index[i], 'hour') else None
            
            # Check trading time validity
            if current_time and not self._is_valid_trading_time(current_time):
                signal.append('hold')
                continue
            
            # Current values
            close_curr = data['close'].iloc[i]
            volume_curr = data['tick_volume'].iloc[i]
            rsi_curr = data['rsi'].iloc[i]
            macd_hist = data['macd_histogram'].iloc[i]
            macd_hist_prev = data['macd_histogram'].iloc[i-1]
            price_momentum = data['price_momentum'].iloc[i]
            volume_avg = data['volume_avg'].iloc[i]
            
            # EMA conditions
            ema_fast = data['ema_fast'].iloc[i]
            ema_slow = data['ema_slow'].iloc[i]
            ema_fast_prev = data['ema_fast'].iloc[i-1]
            ema_slow_prev = data['ema_slow'].iloc[i-1]
            
            # Skip if insufficient data or volume
            if (volume_curr < self.min_volume or 
                pd.isna(rsi_curr) or pd.isna(macd_hist)):
                signal.append('hold')
                continue
            
            # SIMPLIFIED BUY CONDITIONS (4 robust conditions, require ALL 4)
            buy_conditions = [
                # 1. Trend alignment - EMA crossover or strong alignment
                (ema_fast > ema_slow) and (ema_fast > ema_fast_prev),
                
                # 2. RSI not overbought and showing momentum
                25 < rsi_curr < 70,
                
                # 3. MACD showing positive momentum
                macd_hist > macd_hist_prev and macd_hist > -0.5,
                
                # 4. Price momentum positive and volume adequate
                price_momentum > 0.0001 and volume_curr > volume_avg * 0.8
            ]
            
            # SIMPLIFIED SELL CONDITIONS (4 robust conditions, require ALL 4)
            sell_conditions = [
                # 1. Trend alignment - EMA crossover or strong alignment
                (ema_fast < ema_slow) and (ema_fast < ema_fast_prev),
                
                # 2. RSI not oversold and showing momentum
                30 < rsi_curr < 75,
                
                # 3. MACD showing negative momentum
                macd_hist < macd_hist_prev and macd_hist < 0.5,
                
                # 4. Price momentum negative and volume adequate
                price_momentum < -0.0001 and volume_curr > volume_avg * 0.8
            ]
            
            # SIMPLIFIED FORMULA: Require ALL conditions (no partial scoring)
            # This reduces false signals and overfitting
            if all(buy_conditions):
                signal.append('buy')
            elif all(sell_conditions):
                signal.append('sell')
            else:
                signal.append('hold')
        
        # Fill initial signals
        for _ in range(len(data) - len(signal)):
            signal.insert(0, 'hold')
        
        data['signal'] = signal
        
        # Return essential columns only
        return_cols = ['open', 'high', 'low', 'close', 'signal', 'rsi', 'macd_histogram', 
                      'price_momentum', 'ema_fast', 'ema_slow']
        
        return data[return_cols]
