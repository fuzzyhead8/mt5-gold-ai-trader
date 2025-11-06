"""
VWAP Trading Strategy
Integrates VWAP-based signals into the MT5 Gold AI Trader system
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

class VWAPStrategy:
    def __init__(self, symbol="XAUUSD"):
        """
        Initialize VWAP Strategy
        
        Args:
            symbol: Trading symbol (default: XAUUSD)
        """
        self.symbol = symbol
        self.logger = logging.getLogger(f"VWAPStrategy_{symbol}")
        
        # Strategy Parameters
        self.atr_period = 14
        self.rsi_period = 14
        self.ema_fast = 50
        self.ema_slow = 200
        self.rsi_oversold = 40
        self.breakout_candles = 3
        self.signal_validity_candles = 2
        
        # Risk Parameters
        self.stop_loss_atr_mult = 1.0
        self.tp1_atr_mult = 0.8
        self.tp2_atr_mult = 1.6
        self.trailing_atr_mult = 0.6
        self.min_stop_usd = 6.0
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rolling VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()
        return vwap
    
    def detect_engulfing(self, df: pd.DataFrame, index: int) -> bool:
        """Detect bullish engulfing pattern"""
        if index < 1:
            return False
        
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        
        # Bullish engulfing: current green candle engulfs previous red
        bullish = (
            previous['close'] < previous['open'] and  # Previous red
            current['close'] > current['open'] and    # Current green
            current['open'] <= previous['close'] and  # Opens at/below prev close
            current['close'] >= previous['open']      # Closes at/above prev open
        )
        return bullish
    
    def detect_bearish_engulfing(self, df: pd.DataFrame, index: int) -> bool:
        """Detect bearish engulfing pattern"""
        if index < 1:
            return False
        
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        
        # Bearish engulfing: current red candle engulfs previous green
        bearish = (
            previous['close'] > previous['open'] and  # Previous green
            current['close'] < current['open'] and    # Current red
            current['open'] >= previous['close'] and  # Opens at/above prev close
            current['close'] <= previous['open']      # Closes at/below prev open
        )
        return bearish
    
    def detect_rsi_divergence(self, df: pd.DataFrame, index: int, lookback: int = 3, divergence_type: str = 'bullish') -> bool:
        """Detect RSI divergence"""
        if index < lookback:
            return False
        
        price_slice = df['close'].iloc[index - lookback:index + 1]
        rsi_slice = df['rsi'].iloc[index - lookback:index + 1]
        
        if divergence_type == 'bullish':
            # Find lows
            price_min_idx = price_slice.idxmin()
            rsi_min_idx = rsi_slice.idxmin()
            
            # Check if price made lower low but RSI made higher low
            if price_min_idx == price_slice.index[-1]:  # Recent price low
                prev_price_low = price_slice.iloc[:-1].min()
                prev_rsi_low = rsi_slice.iloc[:-1].min()
                
                current_price = price_slice.iloc[-1]
                current_rsi = rsi_slice.iloc[-1]
                
                if current_price < prev_price_low and current_rsi > prev_rsi_low:
                    return True
        
        elif divergence_type == 'bearish':
            # Find highs
            price_max_idx = price_slice.idxmax()
            rsi_max_idx = rsi_slice.idxmax()
            
            # Check if price made higher high but RSI made lower high
            if price_max_idx == price_slice.index[-1]:  # Recent price high
                prev_price_high = price_slice.iloc[:-1].max()
                prev_rsi_high = rsi_slice.iloc[:-1].max()
                
                current_price = price_slice.iloc[-1]
                current_rsi = rsi_slice.iloc[-1]
                
                if current_price > prev_price_high and current_rsi < prev_rsi_high:
                    return True
        
        return False
    
    def get_trend(self, df: pd.DataFrame, index: int) -> str:
        """Determine trend: bullish, bearish, or neutral"""
        if index >= len(df):
            index = len(df) - 1
            
        ema_fast = df['ema_fast'].iloc[index]
        ema_slow = df['ema_slow'].iloc[index]
        
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return "neutral"
        
        if ema_fast > ema_slow:
            return "bullish"
        elif ema_fast < ema_slow:
            return "bearish"
        else:
            return "neutral"
    
    def check_entry_signals(self, df: pd.DataFrame, index: int) -> str:
        """Check all entry rules and return signal"""
        if index < self.ema_slow:  # Not enough data
            return 'hold'
        
        trend = self.get_trend(df, index)
        current = df.iloc[index]
        atr = current['atr']
        
        if pd.isna(atr) or atr == 0:
            return 'hold'
        
        # Entry Rule A: Pullback to VWAP + bullish engulfing (long)
        if (
            trend == "bullish" and
            abs(current['close'] - current['vwap']) / current['vwap'] < 0.002 and
            self.detect_engulfing(df, index)
        ):
            return 'buy'
        
        # Entry Rule A: Pullback to VWAP + bearish engulfing (short)
        if (
            trend == "bearish" and
            abs(current['close'] - current['vwap']) / current['vwap'] < 0.002 and
            self.detect_bearish_engulfing(df, index)
        ):
            return 'sell'
        
        # Entry Rule B: RSI oversold + bullish divergence (long)
        if (
            trend == "bullish" and
            current['rsi'] < self.rsi_oversold and
            self.detect_rsi_divergence(df, index, divergence_type='bullish')
        ):
            return 'buy'
        
        # Entry Rule B: RSI overbought + bearish divergence (short)
        if (
            trend == "bearish" and
            current['rsi'] > (100 - self.rsi_oversold) and
            self.detect_rsi_divergence(df, index, divergence_type='bearish')
        ):
            return 'sell'
        
        # Entry Rule C: Breakout above/below range
        if index >= self.breakout_candles:
            lookback = df.iloc[index - self.breakout_candles:index]
            range_high = lookback['high'].max()
            range_low = lookback['low'].min()
            
            # Bullish breakout
            if trend == "bullish" and current['close'] > range_high:
                avg_volume = lookback['tick_volume'].mean()
                if current['tick_volume'] > avg_volume * 1.2:
                    return 'buy'
            
            # Bearish breakdown
            if trend == "bearish" and current['close'] < range_low:
                avg_volume = lookback['tick_volume'].mean()
                if current['tick_volume'] > avg_volume * 1.2:
                    return 'sell'
        
        return 'hold'
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the given dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and indicators
        """
        result = df.copy()
        
        # Calculate indicators
        result['atr'] = self.calculate_atr(result, self.atr_period)
        result['rsi'] = self.calculate_rsi(result, self.rsi_period)
        result['ema_fast'] = self.calculate_ema(result, self.ema_fast)
        result['ema_slow'] = self.calculate_ema(result, self.ema_slow)
        result['vwap'] = self.calculate_vwap(result)
        
        # Initialize signal column
        result['signal'] = 'hold'
        
        # Generate signals for each bar
        for i in range(len(result)):
            signal = self.check_entry_signals(result, i)
            result.loc[result.index[i], 'signal'] = signal
        
        # Add risk management levels
        result['stop_loss'] = np.nan
        result['take_profit_1'] = np.nan
        result['take_profit_2'] = np.nan
        
        for i in range(len(result)):
            if result['signal'].iloc[i] in ['buy', 'sell']:
                atr = result['atr'].iloc[i]
                close = result['close'].iloc[i]
                idx = result.index[i]
                
                if result['signal'].iloc[i] == 'buy':
                    result.loc[idx, 'stop_loss'] = close - (atr * self.stop_loss_atr_mult)
                    result.loc[idx, 'take_profit_1'] = close + (atr * self.tp1_atr_mult)
                    result.loc[idx, 'take_profit_2'] = close + (atr * self.tp2_atr_mult)
                else:  # sell
                    result.loc[idx, 'stop_loss'] = close + (atr * self.stop_loss_atr_mult)
                    result.loc[idx, 'take_profit_1'] = close - (atr * self.tp1_atr_mult)
                    result.loc[idx, 'take_profit_2'] = close - (atr * self.tp2_atr_mult)
        
        self.logger.info(f"Generated {len(result)} signals for {self.symbol}")
        return result
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "VWAP"
    
    def get_recommended_timeframe(self) -> str:
        """Return recommended timeframe for this strategy"""
        return "M15"  # 15-minute timeframe as per original VWAP strategy
