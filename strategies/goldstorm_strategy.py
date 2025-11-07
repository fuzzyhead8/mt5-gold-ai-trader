import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .goldstorm import BotConfig, VolatilityAnalyzer, MomentumAnalyzer
from .base_strategy import BaseStrategy

class GoldStormStrategy(BaseStrategy):
    """
    GoldStorm trading strategy with sentiment validation and proper trade execution
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
            min_volatility_threshold=0.8,
            pyramid_distance_points=100,
            magic_number=101001
        )
        
        # Strategy-specific parameters
        self.timeframe = mt5.TIMEFRAME_M15
        self.sl_pips = 150  # Stop loss in pips
        self.tp_pips = 300  # Take profit in pips
        self.min_distance_pips = 50.0  # Minimum distance for stops
        self.strategy_name = "goldstorm"
    
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
    
    def calculate_volatility_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility ratio for signal generation"""
        current_atr = self.calculate_atr(df, self.config.volatility_period)
        historical_atr = self.calculate_atr(df, 50)
        
        volatility_ratio = current_atr / historical_atr
        return volatility_ratio.fillna(0)
    
    def get_momentum_signal(self, df: pd.DataFrame, idx: int) -> str:
        """Get momentum direction signal for specific index"""
        if idx < self.config.momentum_period:
            return "NEUTRAL"
            
        rsi_series = self.calculate_rsi(df, self.config.momentum_period)
        rsi = rsi_series.iloc[idx] if not pd.isna(rsi_series.iloc[idx]) else 50.0
        
        # Get price movement over last 5 periods
        if idx >= 5:
            price_change = df['close'].iloc[idx] - df['close'].iloc[idx-5]
        else:
            price_change = 0
        
        # Determine momentum direction
        if rsi > 70 and price_change > 0:
            return "STRONG_BULLISH"
        elif rsi < 30 and price_change < 0:
            return "STRONG_BEARISH"
        elif rsi > 50 and price_change > 0:
            return "BULLISH"
        elif rsi < 50 and price_change < 0:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on GoldStorm strategy logic
        Vectorized version for performance optimization
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional columns: signal, momentum, volatility_ratio, atr
        """
        df = df.copy()
        
        # Calculate technical indicators (vectorized)
        df['rsi'] = self.calculate_rsi(df, self.config.momentum_period)
        df['atr'] = self.calculate_atr(df, self.config.volatility_period)
        df['volatility_ratio'] = self.calculate_volatility_ratio(df)
        
        # Initialize columns
        df['signal'] = 'hold'
        df['momentum'] = 'NEUTRAL'
        df['position_size'] = 1.0  # Standard position size
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Minimum data requirement
        min_data_idx = max(self.config.volatility_period, self.config.momentum_period, 50)
        valid_mask = pd.Series([i >= min_data_idx for i in range(len(df))], index=df.index)
        
        # Volatility condition (vectorized)
        high_volatility = df['volatility_ratio'] > self.config.min_volatility_threshold
        
        # Vectorized momentum logic (inline get_momentum_signal)
        # Price change over last 5 periods
        df['price_change_5'] = df['close'] - df['close'].shift(5)
        
        # Momentum conditions
        strong_bullish = (df['rsi'] > 70) & (df['price_change_5'] > 0)
        strong_bearish = (df['rsi'] < 30) & (df['price_change_5'] < 0)
        bullish = (df['rsi'] > 50) & (df['price_change_5'] > 0)
        bearish = (df['rsi'] < 50) & (df['price_change_5'] < 0)
        
        # Assign momentum (priority: strong > normal)
        df.loc[strong_bullish, 'momentum'] = 'STRONG_BULLISH'
        df.loc[strong_bearish, 'momentum'] = 'STRONG_BEARISH'
        df.loc[(~strong_bullish) & bullish, 'momentum'] = 'BULLISH'
        df.loc[(~strong_bearish) & bearish, 'momentum'] = 'BEARISH'
        
        # Generate signals (vectorized)
        # Bullish signals
        bullish_signals = df['momentum'].isin(['STRONG_BULLISH', 'BULLISH']) & (df['rsi'] < 80)
        df.loc[valid_mask & high_volatility & bullish_signals, 'signal'] = 'buy'
        
        # Bearish signals
        bearish_signals = df['momentum'].isin(['STRONG_BEARISH', 'BEARISH']) & (df['rsi'] > 20)
        df.loc[valid_mask & high_volatility & bearish_signals, 'signal'] = 'sell'
        
        # Vectorized stop loss and take profit calculation
        atr_multiplier_sl = 1.5
        atr_multiplier_tp = 3.0
        
        buy_mask = (df['signal'] == 'buy') & valid_mask
        sell_mask = (df['signal'] == 'sell') & valid_mask
        
        df.loc[buy_mask, 'stop_loss'] = df.loc[buy_mask, 'close'] - (df.loc[buy_mask, 'atr'] * atr_multiplier_sl)
        df.loc[buy_mask, 'take_profit'] = df.loc[buy_mask, 'close'] + (df.loc[buy_mask, 'atr'] * atr_multiplier_tp)
        
        df.loc[sell_mask, 'stop_loss'] = df.loc[sell_mask, 'close'] + (df.loc[sell_mask, 'atr'] * atr_multiplier_sl)
        df.loc[sell_mask, 'take_profit'] = df.loc[sell_mask, 'close'] - (df.loc[sell_mask, 'atr'] * atr_multiplier_tp)
        
        # Drop temporary column
        if 'price_change_5' in df.columns:
            df.drop('price_change_5', axis=1, inplace=True)
        
        return df
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, 
                        account_balance: float) -> bool:
        """
        Execute the GoldStorm strategy with complete trade management
        
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
        
        self.logger.info(f"GoldStorm signal: {latest_signal}")
        
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
            'name': 'goldstorm',
            'timeframe': 'M15',
            'sleep_time': 900,  # 15 minutes
            'risk_per_trade': self.config.risk_percentage,
            'max_positions': self.config.max_positions,
            'volatility_threshold': self.config.min_volatility_threshold,
            'trailing_stop_points': self.config.trailing_stop_points,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'min_distance_pips': self.min_distance_pips
        }
