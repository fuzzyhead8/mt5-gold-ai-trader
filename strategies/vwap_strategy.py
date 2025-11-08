"""
VWAP Trading Strategy
Integrates VWAP-based signals into the MT5 Gold AI Trader system
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import logging
from strategies.base_strategy import BaseStrategy
from utils.tp_magic import TPMagic, TPMagicMode

class VWAPStrategy(BaseStrategy):
    def __init__(self, symbol="XAUUSD"):
        """
        Initialize VWAP Strategy
        
        Args:
            symbol: Trading symbol (default: XAUUSD)
        """
        super().__init__(symbol)
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

        # TPMagic parameters for VWAP (M15 timeframe)
        self.tp1_pips = 25
        self.tp2_pips = 37.5
        self.tp3_pips = 50
        self.sl_pips = 30
        self.tp_magic = None

        
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
    
    def generate_signals(self, df: pd.DataFrame, sentiment: str = None, backtest_mode: bool = False) -> pd.DataFrame:
        """
        Generate trading signals for the given dataframe
        Vectorized version for performance optimization
        If backtest_mode, use TPMagic for multi-TP simulation.
        
        Args:
            df: DataFrame with OHLCV data
            sentiment: Market sentiment (for compatibility)
            backtest_mode: If True, return position and strategy_returns
            
        Returns:
            DataFrame with signals and indicators
        """
        result = df.copy()
        
        if backtest_mode:
            # Ensure required columns for backtest
            required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            for col in required_cols:
                if col not in result.columns:
                    if col == 'tick_volume':
                        result[col] = 100
                    else:
                        result[col] = result['close']
            result['returns'] = result['close'].pct_change().fillna(0)

        # Calculate indicators (vectorized)
        result['atr'] = self.calculate_atr(result, self.atr_period)
        result['rsi'] = self.calculate_rsi(result, self.rsi_period)
        result['ema_fast'] = self.calculate_ema(result, self.ema_fast)
        result['ema_slow'] = self.calculate_ema(result, self.ema_slow)
        result['vwap'] = self.calculate_vwap(result)
        
        # Initialize signal column
        result['signal'] = 'hold'
        
        # Vectorized trend determination
        result['trend'] = np.where(result['ema_fast'] > result['ema_slow'], 'bullish', 
                                   np.where(result['ema_fast'] < result['ema_slow'], 'bearish', 'neutral'))
        
        # Vectorized VWAP proximity (within 0.2%)
        result['near_vwap'] = abs(result['close'] - result['vwap']) / result['vwap'] < 0.002
        
        # Vectorized engulfing patterns
        prev_open = result['open'].shift(1)
        prev_close = result['close'].shift(1)
        prev_high = result['high'].shift(1)
        prev_low = result['low'].shift(1)
        
        # Bullish engulfing
        bullish_engulfing = (
            (prev_close < prev_open) &  # Previous red
            (result['close'] > result['open']) &  # Current green
            (result['open'] <= prev_close) &  # Opens at/below prev close
            (result['close'] >= prev_open)    # Closes at/above prev open
        )
        
        # Bearish engulfing
        bearish_engulfing = (
            (prev_close > prev_open) &  # Previous green
            (result['close'] < result['open']) &  # Current red
            (result['open'] >= prev_close) &  # Opens at/above prev close
            (result['close'] <= prev_open)    # Closes at/below prev open
        )
        
        # Vectorized RSI divergence (simplified - check recent lows/highs)
        # For bullish divergence: price lower low, RSI higher low
        price_low_3 = result['close'].rolling(3).min()
        rsi_low_3 = result['rsi'].rolling(3).min()
        prev_price_low = price_low_3.shift(3)
        prev_rsi_low = rsi_low_3.shift(3)
        
        bullish_divergence = (
            (result['close'] < prev_price_low) &  # Lower price low
            (result['rsi'] > prev_rsi_low) &      # Higher RSI low
            (result['rsi'] < self.rsi_oversold)   # In oversold territory
        )
        
        # Bearish divergence
        price_high_3 = result['close'].rolling(3).max()
        rsi_high_3 = result['rsi'].rolling(3).max()
        prev_price_high = price_high_3.shift(3)
        prev_rsi_high = rsi_high_3.shift(3)
        
        bearish_divergence = (
            (result['close'] > prev_price_high) &  # Higher price high
            (result['rsi'] < prev_rsi_high) &      # Lower RSI high
            (result['rsi'] > (100 - self.rsi_oversold))  # In overbought territory
        )
        
        # Vectorized breakout (last 3 candles)
        range_high_3 = result['high'].rolling(3).max().shift(1)
        range_low_3 = result['low'].rolling(3).min().shift(1)
        avg_volume_3 = result['tick_volume'].rolling(3).mean().shift(1)
        
        bullish_breakout = (
            (result['close'] > range_high_3) &
            (result['trend'] == 'bullish') &
            (result['tick_volume'] > avg_volume_3 * 1.2)
        )
        
        bearish_breakdown = (
            (result['close'] < range_low_3) &
            (result['trend'] == 'bearish') &
            (result['tick_volume'] > avg_volume_3 * 1.2)
        )
        
        # Generate buy signals (OR conditions)
        buy_signals = (
            ((result['trend'] == 'bullish') & result['near_vwap'] & bullish_engulfing) |
            ((result['trend'] == 'bullish') & (result['rsi'] < self.rsi_oversold) & bullish_divergence) |
            bullish_breakout
        )
        result.loc[buy_signals, 'signal'] = 'buy'
        
        # Generate sell signals (OR conditions)
        sell_signals = (
            ((result['trend'] == 'bearish') & result['near_vwap'] & bearish_engulfing) |
            ((result['trend'] == 'bearish') & (result['rsi'] > (100 - self.rsi_oversold)) & bearish_divergence) |
            bearish_breakdown
        )
        result.loc[sell_signals, 'signal'] = 'sell'
        
        # Vectorized risk management levels
        buy_mask = result['signal'] == 'buy'
        sell_mask = result['signal'] == 'sell'
        
        result.loc[buy_mask, 'stop_loss'] = result.loc[buy_mask, 'close'] - (result.loc[buy_mask, 'atr'] * self.stop_loss_atr_mult)
        result.loc[buy_mask, 'take_profit_1'] = result.loc[buy_mask, 'close'] + (result.loc[buy_mask, 'atr'] * self.tp1_atr_mult)
        result.loc[buy_mask, 'take_profit_2'] = result.loc[buy_mask, 'close'] + (result.loc[buy_mask, 'atr'] * self.tp2_atr_mult)
        
        result.loc[sell_mask, 'stop_loss'] = result.loc[sell_mask, 'close'] + (result.loc[sell_mask, 'atr'] * self.stop_loss_atr_mult)
        result.loc[sell_mask, 'take_profit_1'] = result.loc[sell_mask, 'close'] - (result.loc[sell_mask, 'atr'] * self.tp1_atr_mult)
        result.loc[sell_mask, 'take_profit_2'] = result.loc[sell_mask, 'close'] - (result.loc[sell_mask, 'atr'] * self.tp2_atr_mult)
        
        # Drop temporary columns
        temp_cols = ['trend', 'near_vwap', 'price_low_3', 'rsi_low_3', 'prev_price_low', 'prev_rsi_low',
                     'price_high_3', 'rsi_high_3', 'prev_price_high', 'prev_rsi_high', 'range_high_3',
                     'range_low_3', 'avg_volume_3']
        result.drop(columns=[col for col in temp_cols if col in result.columns], inplace=True, errors='ignore')
        
        if backtest_mode:
            # Backtest simulation with TPMagic
            result['position'] = 0.0
            result['strategy_returns'] = 0.0
            pip_size = 0.1
            tp_magic = TPMagic(
                tp1_pips=self.tp1_pips, tp2_pips=self.tp2_pips, tp3_pips=self.tp3_pips,
                sl_pips=self.sl_pips, symbol=self.symbol, pip_size=pip_size,
                mode=TPMagicMode.BACKTEST, initial_lot=0.01,
                logger=self.logger
            )

            for i in range(len(result)):
                if i == 0:
                    continue

                high_i = result['high'].iloc[i]
                low_i = result['low'].iloc[i]
                close_i = result['close'].iloc[i]
                signal_i = result['signal'].iloc[i]
                asset_return = (result['close'].iloc[i] - result['close'].iloc[i-1]) / result['close'].iloc[i-1] if i > 0 else 0.0
                prev_close = result['close'].iloc[i-1]

                if not tp_magic.is_open:
                    if signal_i == 'buy':
                        tp_magic.open_position(1, prev_close)
                    elif signal_i == 'sell':
                        tp_magic.open_position(-1, prev_close)

                if tp_magic.is_open:
                    update_result = tp_magic.update(high_i, low_i, close_i, asset_return)
                    result.loc[result.index[i], 'strategy_returns'] = update_result['strategy_returns']
                    result.loc[result.index[i], 'position'] = update_result['position']
                else:
                    result.loc[result.index[i], 'strategy_returns'] = 0.0
                    result.loc[result.index[i], 'position'] = 0.0

        self.logger.info(f"Generated {len(result)} signals for {self.symbol}")
        return result
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "VWAP"
    
    def get_recommended_timeframe(self) -> str:
        """Return recommended timeframe for this strategy"""
        return "M15"  # 15-minute timeframe as per original VWAP strategy
    
    def get_strategy_config(self) -> dict:
        """Return strategy configuration parameters"""
        return {
            "strategy_name": "VWAP",
            "symbol": self.symbol,
            "timeframe": "M15",
            "parameters": {
                "atr_period": self.atr_period,
                "rsi_period": self.rsi_period,
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
                "rsi_oversold": self.rsi_oversold,
                "breakout_candles": self.breakout_candles,
                "signal_validity_candles": self.signal_validity_candles
            },
            "risk_management": {
                "stop_loss_atr_mult": self.stop_loss_atr_mult,
                "tp1_atr_mult": self.tp1_atr_mult,
                "tp2_atr_mult": self.tp2_atr_mult,
                "trailing_atr_mult": self.trailing_atr_mult,
                "min_stop_usd": self.min_stop_usd
            }
        }

    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the VWAP strategy using TPMagic for consistent multi-TP management.
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
