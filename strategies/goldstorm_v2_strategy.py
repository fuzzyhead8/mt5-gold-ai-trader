import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .goldstorm import BotConfig, VolatilityAnalyzer, MomentumAnalyzer
from .base_strategy import BaseStrategy

class GoldStormV2Strategy(BaseStrategy):
    """
    GoldStorm V2 - Enhanced version with improved filters to reduce gross loss
    
    Key Improvements:
    1. Trend filters using EMA to avoid counter-trend trades
    2. ADX filter to avoid choppy/ranging markets
    3. MACD confirmation for momentum
    4. Volume filter to confirm genuine moves
    5. Stricter RSI conditions to avoid false signals
    6. Multi-timeframe confirmation (using longer-period indicators)
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        super().__init__(symbol)
        
        self.config = BotConfig(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_M15,
            min_deposit=300.0,
            risk_percentage=2.0,
            fixed_lot_size=None,
            max_positions=3,
            volatility_period=20,
            momentum_period=14,
            trailing_stop_points=200,
            min_volatility_threshold=0.85,  # Balanced - not too strict
            pyramid_distance_points=100,
            magic_number=101002  # Different magic number for V2
        )
        
        # Strategy-specific parameters
        self.timeframe = mt5.TIMEFRAME_M15
        self.sl_pips = 150  # Stop loss in pips
        self.tp_pips = 300  # Take profit in pips
        self.min_distance_pips = 50.0  # Minimum distance for stops
        self.strategy_name = "goldstormv2"
        
        # Enhanced filter parameters
        self.ema_fast = 20
        self.ema_slow = 50
        self.adx_period = 14
        self.adx_threshold = 20  # Relaxed from 25 to allow more trades
        self.volume_ma_period = 20
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI from DataFrame"""
        df = df.copy()
        df['price_change'] = df['close'].diff()
        df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
        df['loss'] = -df['price_change'].where(df['price_change'] < 0, 0)
        
        avg_gain = df['gain'].rolling(window=period).mean()
        avg_loss = df['loss'].rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range from DataFrame"""
        df = df.copy()
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
        
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        atr = df['true_range'].rolling(window=period).mean()
        
        return atr
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)
        ADX > 25 indicates trending market (good for trading)
        ADX < 20 indicates ranging/choppy market (avoid trading)
        """
        df = df.copy()
        
        # Calculate +DM and -DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()
        
        df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
        df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)
        
        # Calculate ATR
        atr = self.calculate_atr(df, period)
        
        # Calculate smoothed +DI and -DI
        plus_di = 100 * (df['plus_dm'].rolling(window=period).mean() / atr)
        minus_di = 100 * (df['minus_dm'].rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    def calculate_macd(self, df: pd.DataFrame) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns: (macd_line, signal_line, histogram)
        """
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_volume_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume ratio compared to moving average"""
        if 'tick_volume' in df.columns:
            volume = df['tick_volume']
        elif 'volume' in df.columns:
            volume = df['volume']
        else:
            return pd.Series(1.0, index=df.index)  # Default to 1 if no volume data
        
        volume_ma = volume.rolling(window=self.volume_ma_period).mean()
        volume_ratio = volume / volume_ma
        
        return volume_ratio.fillna(1.0)
    
    def calculate_volatility_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility ratio for signal generation"""
        current_atr = self.calculate_atr(df, self.config.volatility_period)
        historical_atr = self.calculate_atr(df, 50)
        
        volatility_ratio = current_atr / historical_atr
        return volatility_ratio.fillna(0)
    
    def check_trend_alignment(self, df: pd.DataFrame, idx: int, direction: str) -> bool:
        """
        Check if price is aligned with trend direction
        
        Args:
            df: DataFrame with price data and EMAs
            idx: Current index
            direction: 'buy' or 'sell'
            
        Returns:
            True if trend is aligned, False otherwise
        """
        if idx < max(self.ema_fast, self.ema_slow):
            return False
        
        price = df['close'].iloc[idx]
        ema_fast = df['ema_fast'].iloc[idx]
        ema_slow = df['ema_slow'].iloc[idx]
        
        if direction == 'buy':
            # For buy: price above both EMAs and fast EMA above slow EMA
            return price > ema_fast and ema_fast > ema_slow
        else:  # sell
            # For sell: price below both EMAs and fast EMA below slow EMA
            return price < ema_fast and ema_fast < ema_slow
    
    def check_macd_confirmation(self, df: pd.DataFrame, idx: int, direction: str) -> bool:
        """
        Check MACD confirmation for trade direction
        
        Args:
            df: DataFrame with MACD data
            idx: Current index
            direction: 'buy' or 'sell'
            
        Returns:
            True if MACD confirms, False otherwise
        """
        if idx < max(self.macd_slow, self.macd_signal):
            return False
        
        macd = df['macd'].iloc[idx]
        macd_signal = df['macd_signal'].iloc[idx]
        macd_hist = df['macd_hist'].iloc[idx]
        
        if direction == 'buy':
            # For buy: MACD above signal line and histogram positive
            return macd > macd_signal and macd_hist > 0
        else:  # sell
            # For sell: MACD below signal line and histogram negative
            return macd < macd_signal and macd_hist < 0
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with enhanced filters to reduce losing trades
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional columns: signal, momentum, volatility_ratio, atr, etc.
        """
        df = df.copy()
        
        # Calculate all technical indicators
        df['rsi'] = self.calculate_rsi(df, self.config.momentum_period)
        df['atr'] = self.calculate_atr(df, self.config.volatility_period)
        df['volatility_ratio'] = self.calculate_volatility_ratio(df)
        
        # Trend indicators
        df['ema_fast'] = self.calculate_ema(df, self.ema_fast)
        df['ema_slow'] = self.calculate_ema(df, self.ema_slow)
        
        # Momentum indicators
        df['adx'] = self.calculate_adx(df, self.adx_period)
        macd, macd_signal, macd_hist = self.calculate_macd(df)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Volume indicator
        df['volume_ratio'] = self.calculate_volume_ratio(df)
        
        # Initialize signal columns
        df['signal'] = 'hold'
        df['momentum'] = 'NEUTRAL'
        df['filter_reasons'] = ''  # Track why signals were filtered out
        
        # Generate signals for each row
        for i in range(len(df)):
            # Skip insufficient data
            min_period = max(self.ema_slow, self.config.volatility_period, 
                           self.config.momentum_period, 50, self.adx_period)
            if i < min_period:
                continue
            
            # Get current values
            rsi = df['rsi'].iloc[i]
            volatility_ratio = df['volatility_ratio'].iloc[i]
            adx = df['adx'].iloc[i]
            volume_ratio = df['volume_ratio'].iloc[i]
            price = df['close'].iloc[i]
            
            # Filter 1: Volatility check (increased threshold)
            if volatility_ratio <= self.config.min_volatility_threshold:
                df.loc[df.index[i], 'filter_reasons'] = 'low_volatility'
                continue
            
            # Filter 2: ADX check - only trade in trending markets
            if adx < self.adx_threshold:
                df.loc[df.index[i], 'filter_reasons'] = 'low_adx_choppy_market'
                continue
            
            # Filter 3: Volume confirmation - avoid low volume moves
            if volume_ratio < 0.8:  # Volume should be at least 80% of average
                df.loc[df.index[i], 'filter_reasons'] = 'low_volume'
                continue
            
            # Determine momentum direction with balanced RSI conditions
            momentum_signal = "NEUTRAL"
            price_change = df['close'].iloc[i] - df['close'].iloc[i-5] if i >= 5 else 0
            
            # Balanced RSI ranges - not too strict
            if rsi > 50 and rsi < 80 and price_change > 0:
                momentum_signal = "BULLISH"
            elif rsi < 50 and rsi > 20 and price_change < 0:
                momentum_signal = "BEARISH"
            
            df.loc[df.index[i], 'momentum'] = momentum_signal
            
            # Skip if momentum is not clear
            if momentum_signal not in ["BULLISH", "BEARISH"]:
                df.loc[df.index[i], 'filter_reasons'] = f'weak_momentum_{momentum_signal}'
                continue
            
            # Generate BUY signals with multiple confirmations
            if momentum_signal == "BULLISH":
                # Filter 4: Trend alignment - must be above EMAs
                if not self.check_trend_alignment(df, i, 'buy'):
                    df.loc[df.index[i], 'filter_reasons'] = 'trend_not_aligned_buy'
                    continue
                
                # Filter 5: MACD confirmation
                if not self.check_macd_confirmation(df, i, 'buy'):
                    df.loc[df.index[i], 'filter_reasons'] = 'macd_not_confirmed_buy'
                    continue
                
                # All filters passed - generate BUY signal
                df.loc[df.index[i], 'signal'] = 'buy'
                
            # Generate SELL signals with multiple confirmations
            elif momentum_signal == "BEARISH":
                # Filter 4: Trend alignment - must be below EMAs
                if not self.check_trend_alignment(df, i, 'sell'):
                    df.loc[df.index[i], 'filter_reasons'] = 'trend_not_aligned_sell'
                    continue
                
                # Filter 5: MACD confirmation
                if not self.check_macd_confirmation(df, i, 'sell'):
                    df.loc[df.index[i], 'filter_reasons'] = 'macd_not_confirmed_sell'
                    continue
                
                # All filters passed - generate SELL signal
                df.loc[df.index[i], 'signal'] = 'sell'
        
        # Add position sizing information
        df['position_size'] = 1.0  # Standard position size
        
        # Add stop loss and take profit levels using ATR
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        for i in range(len(df)):
            if df['signal'].iloc[i] in ['buy', 'sell']:
                atr_value = df['atr'].iloc[i]
                current_price = df['close'].iloc[i]
                
                # Use ATR-based stops with better risk/reward ratio
                if df['signal'].iloc[i] == 'buy':
                    df.loc[df.index[i], 'stop_loss'] = current_price - (atr_value * 1.5)
                    df.loc[df.index[i], 'take_profit'] = current_price + (atr_value * 3.0)
                else:  # sell
                    df.loc[df.index[i], 'stop_loss'] = current_price + (atr_value * 1.5)
                    df.loc[df.index[i], 'take_profit'] = current_price - (atr_value * 3.0)
        
        return df
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, 
                        account_balance: float) -> bool:
        """
        Execute the GoldStorm V2 strategy with complete trade management
        
        Args:
            df: OHLCV data DataFrame
            sentiment: Market sentiment ('bullish', 'bearish', 'neutral')
            account_balance: Account balance for position sizing
            
        Returns:
            bool: True if trade was executed, False otherwise
        """
        # Generate signals
        signals_df = self.generate_signals(df)
        latest_signal = signals_df['signal'].iloc[-1]
        
        if latest_signal not in ['buy', 'sell']:
            return False
        
        self.logger.info(f"GoldStorm V2 signal: {latest_signal}")
        
        # Get current market price
        entry_price = self.get_market_price(latest_signal)
        if not entry_price:
            return False
        
        # Calculate stop loss and take profit
        try:
            stop_loss, take_profit = self.calculate_stop_take_levels(
                latest_signal, entry_price, self.sl_pips, self.tp_pips
            )
        except ValueError as e:
            self.logger.error(f"Failed to calculate stop levels: {e}")
            return False
        
        # Validate stop distances
        if not self.validate_stop_distances(entry_price, stop_loss, take_profit):
            return False
        
        # Calculate position size
        lot_size = self.calculate_position_size(
            account_balance, entry_price, stop_loss, self.config.risk_percentage
        )
        
        self.logger.info(f"Trade levels - Price: {entry_price}, SL: {stop_loss} "
                        f"({abs(stop_loss-entry_price)*10000:.1f} pips), "
                        f"TP: {take_profit} ({abs(take_profit-entry_price)*10000:.1f} pips)")
        
        # Execute trade with validation
        result = self.execute_trade(
            signal=latest_signal,
            sentiment=sentiment, 
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            strategy_name=self.strategy_name
        )
        
        return result is not None

    def get_strategy_config(self) -> dict:
        """Get strategy configuration"""
        return {
            'name': 'goldstormv2',
            'timeframe': 'M15',
            'sleep_time': 900,  # 15 minutes
            'risk_per_trade': self.config.risk_percentage,
            'max_positions': self.config.max_positions,
            'volatility_threshold': self.config.min_volatility_threshold,
            'trailing_stop_points': self.config.trailing_stop_points,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'min_distance_pips': self.min_distance_pips,
            'adx_threshold': self.adx_threshold,
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'filters': 'EMA+ADX+MACD+Volume+Price Action'
        }
