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
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional columns: signal, momentum, volatility_ratio, atr
        """
        df = df.copy()
        
        # Calculate technical indicators
        df['rsi'] = self.calculate_rsi(df, self.config.momentum_period)
        df['atr'] = self.calculate_atr(df, self.config.volatility_period)
        df['volatility_ratio'] = self.calculate_volatility_ratio(df)
        
        # Initialize signal column
        df['signal'] = 'hold'
        df['momentum'] = 'NEUTRAL'
        
        # Generate signals for each row
        for i in range(len(df)):
            # Skip insufficient data
            if i < max(self.config.volatility_period, self.config.momentum_period, 50):
                continue
            
            # Check volatility condition
            volatility_ratio = df['volatility_ratio'].iloc[i]
            is_high_volatility = volatility_ratio > self.config.min_volatility_threshold
            
            if not is_high_volatility:
                continue
            
            # Get momentum signal
            momentum_signal = self.get_momentum_signal(df, i)
            df.loc[df.index[i], 'momentum'] = momentum_signal
            
            if momentum_signal == "NEUTRAL":
                continue
            
            # Generate trading signals based on momentum
            if momentum_signal in ["STRONG_BULLISH", "BULLISH"]:
                # Additional confirmation: check if we're not overbought
                if df['rsi'].iloc[i] < 80:  # Not too overbought
                    df.loc[df.index[i], 'signal'] = 'buy'
                    
            elif momentum_signal in ["STRONG_BEARISH", "BEARISH"]:
                # Additional confirmation: check if we're not oversold
                if df['rsi'].iloc[i] > 20:  # Not too oversold
                    df.loc[df.index[i], 'signal'] = 'sell'
        
        # Add position sizing information (for backtesting)
        df['position_size'] = 1.0  # Standard position size
        
        # Add stop loss and take profit levels
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        for i in range(len(df)):
            if df['signal'].iloc[i] in ['buy', 'sell']:
                atr_value = df['atr'].iloc[i]
                current_price = df['close'].iloc[i]
                
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
