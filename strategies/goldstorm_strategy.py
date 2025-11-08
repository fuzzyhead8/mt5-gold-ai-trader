import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .goldstorm import BotConfig, VolatilityAnalyzer, MomentumAnalyzer
from .base_strategy import BaseStrategy
from utils.tp_magic import TPMagic, TPMagicMode

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

        # TPMagic parameters for GoldStorm (M15, larger targets)
        self.tp1_pips = 100
        self.tp2_pips = 150
        self.tp3_pips = 200
        self.tp_magic = None
    
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
    
    def generate_signals(self, df: pd.DataFrame, sentiment: str = None, backtest_mode: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on GoldStorm strategy logic
        Vectorized version for performance optimization
        If backtest_mode, use TPMagic for multi-TP simulation.
        
        Args:
            df: DataFrame with OHLCV data
            sentiment: Market sentiment (for backtest compatibility)
            backtest_mode: If True, return position and strategy_returns
            
        Returns:
            DataFrame with additional columns: signal, momentum, volatility_ratio, atr, [position, strategy_returns if backtest]
        """
        df = df.copy()
        
        if backtest_mode:
            # Ensure required columns for backtest
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = df['close']
            df['returns'] = df['close'].pct_change().fillna(0)

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
        
        if backtest_mode:
            df['position'] = 0.0
            df['strategy_returns'] = 0.0

        # Minimum data requirement
        min_data_idx = max(self.config.volatility_period, self.config.momentum_period, 50)
        valid_mask = pd.Series([i >= min_data_idx for i in range(len(df))], index=df.index)
        
        # Volatility condition (vectorized)
        high_volatility = df['volatility_ratio'] > self.config.min_volatility_threshold
        
        # Vectorized momentum logic
        df['price_change_5'] = df['close'] - df['close'].shift(5)
        
        # Momentum conditions
        strong_bullish = (df['rsi'] > 70) & (df['price_change_5'] > 0)
        strong_bearish = (df['rsi'] < 30) & (df['price_change_5'] < 0)
        bullish = (df['rsi'] > 50) & (df['price_change_5'] > 0)
        bearish = (df['rsi'] < 50) & (df['price_change_5'] < 0)
        
        # Assign momentum
        df.loc[strong_bullish, 'momentum'] = 'STRONG_BULLISH'
        df.loc[strong_bearish, 'momentum'] = 'STRONG_BEARISH'
        df.loc[(~strong_bullish) & bullish, 'momentum'] = 'BULLISH'
        df.loc[(~strong_bearish) & bearish, 'momentum'] = 'BEARISH'
        
        # Generate signals
        bullish_signals = df['momentum'].isin(['STRONG_BULLISH', 'BULLISH']) & (df['rsi'] < 80)
        df.loc[valid_mask & high_volatility & bullish_signals, 'signal'] = 'buy'
        
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
        
        if backtest_mode:
            # Backtest simulation with TPMagic
            pip_size = 0.1
            tp_magic = TPMagic(
                tp1_pips=self.tp1_pips, tp2_pips=self.tp2_pips, tp3_pips=self.tp3_pips,
                sl_pips=self.sl_pips, symbol=self.symbol, pip_size=pip_size,
                mode=TPMagicMode.BACKTEST, initial_lot=0.01,
                logger=self.logger
            )

            for i in range(len(df)):
                if i == 0:
                    continue

                high_i = df['high'].iloc[i]
                low_i = df['low'].iloc[i]
                close_i = df['close'].iloc[i]
                signal_i = df['signal'].iloc[i]
                asset_return = df['returns'].iloc[i]
                prev_close = df['close'].iloc[i-1]

                if not tp_magic.is_open:
                    if signal_i == 'buy':
                        tp_magic.open_position(1, prev_close)
                    elif signal_i == 'sell':
                        tp_magic.open_position(-1, prev_close)

                if tp_magic.is_open:
                    update_result = tp_magic.update(high_i, low_i, close_i, asset_return)
                    df.loc[df.index[i], 'strategy_returns'] = update_result['strategy_returns']
                    df.loc[df.index[i], 'position'] = update_result['position']
                else:
                    df.loc[df.index[i], 'strategy_returns'] = 0.0
                    df.loc[df.index[i], 'position'] = 0.0

        return df
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the GoldStorm strategy using TPMagic for consistent multi-TP management.
        """
        # Generate signals (normal mode)
        signals_df = self.generate_signals(df)
        if len(signals_df) == 0:
            self.logger.info("No data available for signal generation")
            return {"status": "no_data", "signal": "hold"}

        latest_signal = signals_df['signal'].iloc[-1]
        if latest_signal not in ['buy', 'sell']:
            self.logger.info(f"No actionable signal generated: {latest_signal}")
            return self._manage_existing_position(df, sentiment)

        # Validate signal with sentiment
        if not self.validate_signal_with_sentiment(latest_signal, sentiment):
            self.logger.info(f"Signal '{latest_signal}' rejected due to sentiment '{sentiment}'")
            return self._manage_existing_position(df, sentiment)

        # Get current market price
        current_price = self.get_market_price(latest_signal)
        if current_price is None:
            self.logger.error("Failed to get current market price")
            return {"status": "no_price", "signal": latest_signal}

        # Check if we have an existing TPMagic position
        if self.tp_magic is None or not self.tp_magic.is_open:
            # No position, open new one
            direction = 1 if latest_signal == 'buy' else -1
            # Calculate position size using base SL
            base_sl = current_price - self.sl_pips * 0.1 * direction  # pip_size=0.1
            lot_size = self.calculate_position_size(balance, current_price, base_sl, risk_percent=2.0)

            self.tp_magic = TPMagic(
                tp1_pips=self.tp1_pips, tp2_pips=self.tp2_pips, tp3_pips=self.tp3_pips,
                sl_pips=self.sl_pips, symbol=self.symbol, pip_size=0.1,
                mode=TPMagicMode.LIVE, initial_lot=lot_size,
                logger=self.logger
            )

            if self.tp_magic.open_position(direction, current_price, lot_size):
                self.logger.info(f"New TPMagic position opened: {latest_signal} {lot_size} lots at {current_price}")
                return {
                    "status": "executed",
                    "signal": latest_signal,
                    "price": current_price,
                    "lot_size": lot_size,
                    "tp_magic_state": self.tp_magic.get_state()
                }
            else:
                self.tp_magic = None
                return {"status": "execution_failed", "signal": latest_signal}
        else:
            # Position exists, manage it
            return self._manage_existing_position(df, sentiment)

    def _manage_existing_position(self, df: pd.DataFrame, sentiment: str) -> dict:
        """Manage existing TPMagic position"""
        if self.tp_magic is None or not self.tp_magic.is_open:
            return {"status": "no_position", "signal": "hold"}

        # Get latest bar data
        latest = df.iloc[-1]
        high = latest['high']
        low = latest['low']
        close = latest['close']

        # Update TPMagic
        result = self.tp_magic.update(high, low, close)
        actions = result['actions']
        current_sl = result['current_sl']

        executed_actions = []
        for action in actions:
            if action['type'] in ['partial_close', 'close_sl', 'close_retrace', 'close_all']:
                self.logger.info(f"Executed action: {action}")
                executed_actions.append(action)

        state = self.tp_magic.get_state()
        self.logger.info(f"Position managed. Stage: {state['stage']}, Remaining lot: {state['remaining_lot']}, SL: {current_sl}")

        return {
            "status": "managed",
            "signal": "hold",
            "actions": executed_actions,
            "tp_magic_state": state
        }

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
