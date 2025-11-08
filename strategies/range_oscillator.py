import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from utils.tp_magic import TPMagic, TPMagicMode

class RangeOscillatorStrategy(BaseStrategy):
    def __init__(self, symbol: str = "XAUUSD"):
        super().__init__(symbol)
        # Adapted parameters for XAUUSD
        self.length = 50
        self.mult = 2.0
        self.levels_inp = 2
        self.heat_thresh = 1
        self.heat_lookback = 300
        self.min_bars_entry = 200  # Minimum bars for entry logic (simplified)
        # TP/SL in pips, adapted for gold (1 pip = 0.1 for XAUUSD typically)
        self.tp1_pips = 16
        self.tp2_pips = 24
        self.tp3_pips = 32
        self.sl_pips = 20   # Base SL
        self.tp_pips = 32   # Base TP for single position
        self.timeframe = mt5.TIMEFRAME_M15  # Adapted to M15 like other strategies
        self.lot_size = 0.01
        self.enable_zone_lock = False  # Simplified, disabled for base integration
        self.zone_lock_pips = 100

        # TPMagic for consistent multi-TP handling
        self.tp_magic = None

    def calculate_atr_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate simple ATR proxy: rolling mean of (high - low)"""
        atr = df['high'] - df['low']
        atr_raw = atr.rolling(window=200, min_periods=50).mean().fillna(atr.mean()) * self.mult
        return atr_raw

    def calculate_weighted_ma(self, df: pd.Series, length: int) -> pd.Series:
        """Calculate custom weighted moving average for each bar"""
        ma = pd.Series(index=df.index, dtype=float)
        for i in range(length - 1, len(df)):
            if i < length - 1:
                ma.iloc[i] = df.iloc[i]
                continue
            weighted_sum = 0.0
            total_weight = 0.0
            for j in range(1, length + 1):
                idx = i - j + 1
                prev_idx = idx - 1
                if prev_idx < 0:
                    delta = 0
                else:
                    delta = abs(df.iloc[idx] - df.iloc[prev_idx])
                w = delta / df.iloc[prev_idx] if prev_idx >= 0 and df.iloc[prev_idx] != 0 else 0
                weighted_sum += df.iloc[idx] * w
                total_weight += w
            ma.iloc[i] = weighted_sum / total_weight if total_weight != 0 else df.iloc[i]
        return ma

    def get_heat_zone(self, osc_series: pd.Series, i: int, trend_dir: int) -> int:
        """Calculate heat zone for bar i using historical osc"""
        if i < self.heat_lookback:
            return 0
        source = osc_series.iloc[max(0, i - self.heat_lookback + 1):i + 1]
        if len(source) < self.heat_lookback:
            return 0
        val = osc_series.iloc[i]
        hi = source.max()
        lo = source.min()
        rng = hi - lo
        if rng == 0:
            return 0
        step = rng / self.levels_inp

        level_vals = [lo + step * k for k in range(self.levels_inp)]
        level_counts = [0] * self.levels_inp

        for k in range(self.levels_inp):
            lvl = level_vals[k]
            cnt = sum(1 for s in source if abs(s - lvl) < step / 2)
            level_counts[k] = cnt

        # Find closest level
        min_d = float('inf')
        best_count = 0
        for k in range(self.levels_inp):
            d = abs(val - level_vals[k])
            if d < min_d:
                min_d = d
                best_count = level_counts[k]

        # Zone logic
        zone = 0
        thresh_high = self.heat_thresh + 15
        thresh_mid = self.heat_thresh + 3
        if trend_dir == 1:
            if best_count >= thresh_high:
                zone = 2  # Strong bullish
            elif best_count >= thresh_mid:
                zone = 1  # Weak bullish
        elif trend_dir == -1:
            if best_count >= thresh_high:
                zone = -2  # Strong bearish
            elif best_count >= thresh_mid:
                zone = -1  # Weak bearish
        return zone

    def generate_signals(self, df: pd.DataFrame, sentiment: str = None, backtest_mode: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on Range Oscillator.
        If backtest_mode, simulate detailed returns with multi-TP logic using TPMagic.
        """
        if len(df) < max(self.length, 200, self.heat_lookback):
            df['signal'] = 'hold'
            if backtest_mode:
                df['position'] = 0.0
                df['strategy_returns'] = 0.0
            return df[['close', 'signal', 'osc', 'heat_zone', 'wma']] if not backtest_mode else df[['close', 'signal', 'position', 'strategy_returns', 'osc', 'heat_zone', 'wma']]

        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                df[col] = df['close']  # fallback if missing

        # Compute indicators
        close_series = df['close']
        atr_raw = self.calculate_atr_proxy(df)

        # Compute weighted MA for all bars
        wma = self.calculate_weighted_ma(close_series, self.length)

        # Compute oscillator and trend_dir
        osc = pd.Series(index=df.index, dtype=float)
        trend_dir = pd.Series(index=df.index, dtype=int)
        for i in range(len(df)):
            if i < self.length:
                osc.iloc[i] = 0
                trend_dir.iloc[i] = 0
                continue
            ma_val = wma.iloc[i]
            atr_val = atr_raw.iloc[i]
            close_val = df['close'].iloc[i]
            osc.iloc[i] = 100 * (close_val - ma_val) / atr_val if atr_val != 0 else 0
            if close_val > ma_val:
                trend_dir.iloc[i] = 1
            elif close_val < ma_val:
                trend_dir.iloc[i] = -1
            else:
                trend_dir.iloc[i] = 0

        # Compute heat_zone for each bar
        heat_zone = pd.Series(index=df.index, dtype=int)
        for i in range(len(df)):
            if i < self.min_bars_entry:
                heat_zone.iloc[i] = 0
            else:
                heat_zone.iloc[i] = self.get_heat_zone(osc, i, trend_dir.iloc[i])

        df['heat_zone'] = heat_zone
        df['osc'] = osc
        df['wma'] = wma
        df['atr_raw'] = atr_raw
        df['returns'] = df['close'].pct_change().fillna(0)
        df['signal'] = 'hold'

        if not backtest_mode:
            conditions_buy = (heat_zone > 0) & (df.index >= df.index[self.min_bars_entry])
            conditions_sell = (heat_zone < 0) & (df.index >= df.index[self.min_bars_entry])

            df.loc[conditions_buy, 'signal'] = 'buy'
            df.loc[conditions_sell, 'signal'] = 'sell'

            return_cols = ['close', 'signal', 'osc', 'heat_zone', 'wma']
            return df[return_cols].copy()
        else:
            # Backtest mode: use TPMagic for consistent multi-TP simulation
            df['position'] = 0.0
            df['strategy_returns'] = 0.0
            pip_size = 0.1  # For XAUUSD
            tp_magic = TPMagic(
                tp1_pips=self.tp1_pips, tp2_pips=self.tp2_pips, tp3_pips=self.tp3_pips,
                sl_pips=self.sl_pips, symbol=self.symbol, pip_size=pip_size,
                mode=TPMagicMode.BACKTEST, initial_lot=self.lot_size,
                logger=self.logger
            )

            for i in range(len(df)):
                if i == 0:
                    df.loc[df.index[i], 'position'] = 0.0
                    df.loc[df.index[i], 'strategy_returns'] = 0.0
                    continue

                high_i = df['high'].iloc[i]
                low_i = df['low'].iloc[i]
                close_i = df['close'].iloc[i]
                heat_zone_i = heat_zone.iloc[i]
                asset_return = df['returns'].iloc[i]
                signal = 'hold'
                prev_close = df['close'].iloc[i-1]

                # Open new position if no exposure and signal
                if not tp_magic.is_open:
                    if heat_zone_i > 0:
                        signal = 'buy'
                        direction = 1
                        if tp_magic.open_position(direction, prev_close):
                            pass  # Opened
                        else:
                            signal = 'hold'
                    elif heat_zone_i < 0:
                        signal = 'sell'
                        direction = -1
                        if tp_magic.open_position(direction, prev_close):
                            pass  # Opened
                        else:
                            signal = 'hold'

                # Update position if open
                if tp_magic.is_open:
                    update_result = tp_magic.update(high_i, low_i, close_i, asset_return)
                    df.loc[df.index[i], 'strategy_returns'] = update_result['strategy_returns']
                    df.loc[df.index[i], 'position'] = update_result['position']
                else:
                    df.loc[df.index[i], 'strategy_returns'] = 0.0
                    df.loc[df.index[i], 'position'] = 0.0

                df.loc[df.index[i], 'signal'] = signal

            return df[['close', 'signal', 'position', 'strategy_returns', 'osc', 'heat_zone', 'wma']]

    def get_strategy_config(self) -> dict:
        """Return strategy configuration parameters"""
        return {
            "strategy_name": "Range Oscillator Strategy",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "parameters": {
                "length": self.length,
                "mult": self.mult,
                "levels_inp": self.levels_inp,
                "heat_thresh": self.heat_thresh,
                "heat_lookback": self.heat_lookback,
                "min_bars_entry": self.min_bars_entry,
                "tp1_pips": self.tp1_pips,
                "tp2_pips": self.tp2_pips,
                "tp3_pips": self.tp3_pips,
                "sl_pips": self.sl_pips,
                "tp_pips": self.tp_pips,
                "lot_size": self.lot_size,
                "enable_zone_lock": self.enable_zone_lock,
                "description": "Adapted Range Oscillator strategy with weighted MA, oscillator, and heat zones for signal generation. Simplified for single position trading with base SL/TP."
            }
        }

    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Range Oscillator strategy using TPMagic for consistent multi-TP management.
        """
        # Generate signals (normal mode, no backtest)
        signals_df = self.generate_signals(df)
        if len(signals_df) == 0:
            self.logger.info("No data available for signal generation")
            return {"status": "no_data", "signal": "hold"}

        latest_signal = signals_df['signal'].iloc[-1]
        if latest_signal not in ['buy', 'sell']:
            self.logger.info(f"No actionable signal generated: {latest_signal}")
            # Still check for position management
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
            # Calculate position size (use base SL for risk calc)
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
            # Position exists, just manage it
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
                # Execute the close (TPMagic already handles execution in LIVE mode)
                # But log it
                self.logger.info(f"Executed action: {action}")
                executed_actions.append(action)

        # Update SL on position if needed (TPMagic manages, but verify)
        state = self.tp_magic.get_state()
        self.logger.info(f"Position managed. Stage: {state['stage']}, Remaining lot: {state['remaining_lot']}, SL: {current_sl}")

        return {
            "status": "managed",
            "signal": "hold",
            "actions": executed_actions,
            "tp_magic_state": state
        }
