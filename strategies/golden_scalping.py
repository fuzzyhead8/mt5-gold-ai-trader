import numpy as np
import pandas as pd

class GoldenScalpingStrategy:
    """
    GOLDEN FORMULA - Advanced Professional Scalping Strategy
    Designed to maximize profit while maintaining strict risk control
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.min_volume = 40  # More lenient volume filter
        self.volatility_multiplier = 1.5  # Dynamic volatility scaling
        self.trend_strength_threshold = 0.0001  # 0.01% trend strength (more sensitive)
        self.momentum_threshold = 0.0001  # 0.01% momentum threshold (more sensitive)
        
    def _calculate_advanced_rsi(self, prices, window=7):
        """Enhanced RSI with smoothing"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        # Smooth RSI to reduce noise
        return rsi.rolling(window=3).mean()
    
    def _calculate_stochastic(self, high, low, close, k_window=8, d_window=3):
        """Stochastic oscillator for momentum"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def _calculate_vwap(self, high, low, close, volume):
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    def _calculate_atr_bands(self, high, low, close, window=10, multiplier=2.0):
        """ATR-based dynamic bands"""
        atr = (high - low).rolling(window=window).mean()
        middle = close.rolling(window=window).mean()
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        return upper, lower, middle, atr
    
    def _calculate_macd_histogram(self, prices, fast=8, slow=17, signal=9):
        """MACD histogram for momentum"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_market_microstructure(self, high, low, close, volume):
        """Advanced market microstructure analysis"""
        # Price impact
        price_range = high - low
        volume_impact = volume / volume.rolling(window=20).mean()
        price_efficiency = price_range / (close.rolling(window=5).std() + 1e-8)
        
        # Order flow imbalance
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        mfi_ratio = (positive_flow.rolling(window=10).sum() / 
                     (negative_flow.rolling(window=10).sum() + 1e-8))
        
        return volume_impact, price_efficiency, mfi_ratio
    
    def _detect_support_resistance(self, prices, window=10):
        """Dynamic support/resistance detection"""
        rolling_max = prices.rolling(window=window, center=True).max()
        rolling_min = prices.rolling(window=window, center=True).min()
        
        resistance = prices == rolling_max
        support = prices == rolling_min
        
        return support, resistance
    
    def _calculate_volatility_regime(self, prices, window=20):
        """Volatility regime identification"""
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std()
        vol_percentile = volatility.rolling(window=50).rank(pct=True)
        
        # Low: 0-33%, Medium: 33-66%, High: 66-100%
        regime = pd.cut(vol_percentile, bins=[0, 0.33, 0.66, 1.0], 
                       labels=['low', 'medium', 'high'])
        return regime, volatility
    
    def _is_premium_trading_session(self, timestamp):
        """Identify premium trading sessions"""
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            # London-NY overlap (13:00-17:00 UTC) and Asian close (7:00-9:00 UTC)
            premium_sessions = (13 <= hour <= 17) or (7 <= hour <= 9)
            # Avoid low liquidity periods
            avoid_periods = (22 <= hour <= 23) or (0 <= hour <= 2)
            return premium_sessions and not avoid_periods
        return True
    
    def generate_signals(self, data):
        """
        GOLDEN FORMULA: Advanced multi-factor scalping signal generation
        """
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'tick_volume':
                    data[col] = 100
                else:
                    raise ValueError(f"Required column '{col}' missing from data")
        
        # Core moving averages with Fibonacci numbers
        data['ema_3'] = data['close'].ewm(span=3).mean()
        data['ema_8'] = data['close'].ewm(span=8).mean()
        data['ema_13'] = data['close'].ewm(span=13).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        data['ema_34'] = data['close'].ewm(span=34).mean()
        
        # Advanced technical indicators
        data['rsi'] = self._calculate_advanced_rsi(data['close'], window=7)
        data['stoch_k'], data['stoch_d'] = self._calculate_stochastic(
            data['high'], data['low'], data['close'])
        
        # VWAP analysis
        data['vwap'] = self._calculate_vwap(
            data['high'], data['low'], data['close'], data['tick_volume'])
        data['vwap_distance'] = (data['close'] - data['vwap']) / data['vwap']
        
        # Dynamic ATR bands
        data['atr_upper'], data['atr_lower'], data['atr_middle'], data['atr'] = \
            self._calculate_atr_bands(data['high'], data['low'], data['close'])
        
        # MACD histogram
        data['macd'], data['macd_signal'], data['macd_histogram'] = \
            self._calculate_macd_histogram(data['close'])
        
        # Market microstructure
        data['volume_impact'], data['price_efficiency'], data['mfi_ratio'] = \
            self._calculate_market_microstructure(
                data['high'], data['low'], data['close'], data['tick_volume'])
        
        # Support/Resistance
        data['support'], data['resistance'] = self._detect_support_resistance(data['close'])
        
        # Volatility regime
        data['vol_regime'], data['volatility'] = self._calculate_volatility_regime(data['close'])
        
        # Trend analysis
        data['trend_short'] = (data['ema_8'] - data['ema_13']) / data['ema_13']
        data['trend_medium'] = (data['ema_13'] - data['ema_21']) / data['ema_21']
        data['trend_long'] = (data['ema_21'] - data['ema_34']) / data['ema_34']
        
        # Momentum indicators
        data['price_momentum'] = data['close'].pct_change(3)
        data['volume_momentum'] = data['tick_volume'].pct_change(5)
        data['momentum_divergence'] = (data['price_momentum'] - data['volume_momentum']).rolling(3).mean()
        
        # Market structure
        data['higher_high'] = (data['high'] > data['high'].shift(1)) & (data['high'].shift(1) > data['high'].shift(2))
        data['lower_low'] = (data['low'] < data['low'].shift(1)) & (data['low'].shift(1) < data['low'].shift(2))
        
        # Price action patterns
        data['doji'] = abs(data['open'] - data['close']) < (data['high'] - data['low']) * 0.1
        data['hammer'] = ((data['close'] - data['low']) > 2 * abs(data['open'] - data['close'])) & \
                        ((data['high'] - data['close']) < abs(data['open'] - data['close']))
        
        # Generate signals using GOLDEN FORMULA
        signal = []
        
        for i in range(max(34, len(data.columns)), len(data)):
            current_time = data.index[i] if hasattr(data, 'index') and hasattr(data.index[i], 'hour') else None
            
            # Check premium trading session
            if current_time and not self._is_premium_trading_session(current_time):
                signal.append('hold')
                continue
            
            # Current values
            close_curr = data['close'].iloc[i]
            volume_curr = data['tick_volume'].iloc[i]
            rsi_curr = data['rsi'].iloc[i]
            stoch_k = data['stoch_k'].iloc[i]
            stoch_d = data['stoch_d'].iloc[i]
            macd_hist = data['macd_histogram'].iloc[i]
            macd_hist_prev = data['macd_histogram'].iloc[i-1]
            vwap_dist = data['vwap_distance'].iloc[i]
            volume_impact = data['volume_impact'].iloc[i]
            mfi_ratio = data['mfi_ratio'].iloc[i]
            trend_short = data['trend_short'].iloc[i]
            trend_medium = data['trend_medium'].iloc[i]
            price_momentum = data['price_momentum'].iloc[i]
            vol_regime = data['vol_regime'].iloc[i]
            
            # EMA conditions
            ema_3 = data['ema_3'].iloc[i]
            ema_8 = data['ema_8'].iloc[i]
            ema_13 = data['ema_13'].iloc[i]
            ema_21 = data['ema_21'].iloc[i]
            
            # ATR conditions
            atr_upper = data['atr_upper'].iloc[i]
            atr_lower = data['atr_lower'].iloc[i]
            atr_middle = data['atr_middle'].iloc[i]
            
            # Skip if insufficient volume or extreme conditions
            if (volume_curr < self.min_volume or 
                pd.isna(rsi_curr) or pd.isna(volume_impact) or
                vol_regime == 'high'):
                signal.append('hold')
                continue
            
            # GOLDEN BUY CONDITIONS - Optimized multi-factor confirmation
            golden_buy_conditions = [
                # 1. Primary trend alignment (key EMAs)
                ema_3 > ema_8 and ema_8 > ema_13,
                
                # 2. RSI in favorable range (more lenient)
                20 < rsi_curr < 65,
                
                # 3. Stochastic momentum (more aggressive)
                stoch_k > stoch_d and stoch_k < 80,
                
                # 4. MACD histogram improvement
                macd_hist > macd_hist_prev,
                
                # 5. Price relationship to VWAP (more lenient)
                -0.0015 < vwap_dist < 0.003,
                
                # 6. Volume confirmation (less strict)
                volume_impact > 1.0,
                
                # 7. Money flow positive (less strict)
                mfi_ratio > 1.05,
                
                # 8. Short-term trend strength (more sensitive)
                trend_short > self.trend_strength_threshold,
                
                # 9. Positive price momentum (more sensitive)
                price_momentum > self.momentum_threshold,
                
                # 10. Price position (more flexible)
                close_curr > atr_lower,
                
                # 11. Overall trend not opposing (more lenient)
                trend_medium > -0.0005,
                
                # 12. Additional momentum confirmation
                stoch_k > 20
            ]
            
            # GOLDEN SELL CONDITIONS - Optimized multi-factor confirmation
            golden_sell_conditions = [
                # 1. Primary trend alignment (key EMAs)
                ema_3 < ema_8 and ema_8 < ema_13,
                
                # 2. RSI in favorable range (more lenient)
                35 < rsi_curr < 80,
                
                # 3. Stochastic momentum (more aggressive)
                stoch_k < stoch_d and stoch_k > 20,
                
                # 4. MACD histogram deteriorating
                macd_hist < macd_hist_prev,
                
                # 5. Price relationship to VWAP (more lenient)
                -0.003 < vwap_dist < 0.0015,
                
                # 6. Volume confirmation (less strict)
                volume_impact > 1.0,
                
                # 7. Money flow negative (less strict)
                mfi_ratio < 0.95,
                
                # 8. Short-term trend weakness (more sensitive)
                trend_short < -self.trend_strength_threshold,
                
                # 9. Negative price momentum (more sensitive)
                price_momentum < -self.momentum_threshold,
                
                # 10. Price position (more flexible)
                close_curr < atr_upper,
                
                # 11. Overall trend not opposing (more lenient)
                trend_medium < 0.0005,
                
                # 12. Additional momentum confirmation
                stoch_k < 80
            ]
            
            # GOLDEN FORMULA: Require 7+ conditions for signal (balanced approach)
            buy_score = sum(golden_buy_conditions)
            sell_score = sum(golden_sell_conditions)
            
            if buy_score >= 7:
                signal.append('buy')
            elif sell_score >= 7:
                signal.append('sell')
            else:
                signal.append('hold')
        
        # Fill initial signals
        for _ in range(len(data) - len(signal)):
            signal.insert(0, 'hold')
        
        data['signal'] = signal
        
        # Return comprehensive dataset
        return_cols = ['open', 'high', 'low', 'close', 'signal', 'rsi', 'stoch_k', 'stoch_d',
                      'macd_histogram', 'vwap_distance', 'volume_impact', 'mfi_ratio',
                      'trend_short', 'trend_medium', 'price_momentum', 'atr_upper', 'atr_lower']
        
        return data[return_cols]
