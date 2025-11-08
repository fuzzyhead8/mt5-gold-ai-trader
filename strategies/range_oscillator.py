import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from typing import Dict

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
        If backtest_mode, simulate detailed returns with multi-TP logic.
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
            # Backtest mode: simulate multi-TP logic
            df['position'] = 0.0
            df['strategy_returns'] = 0.0
            exposure = 0.0
            direction = 0
            entry_price = 0.0
            sl = 0.0
            stage = 0
            pip_size = 0.1  # For XAUUSD

            for i in range(len(df)):
                if i == 0:
                    df.loc[df.index[i], 'position'] = 0.0
                    df.loc[df.index[i], 'strategy_returns'] = 0.0
                    continue

                asset_return = df['returns'].iloc[i]
                high_i = df['high'].iloc[i]
                low_i = df['low'].iloc[i]
                heat_zone_i = heat_zone.iloc[i]
                strategy_return = 0.0
                signal = 'hold'
                prev_close = df['close'].iloc[i-1]

                if exposure == 0.0:
                    if heat_zone_i > 0:
                        signal = 'buy'
                        direction = 1
                        entry_price = prev_close
                        sl = entry_price - self.sl_pips * pip_size * direction
                        exposure = 1.0
                        stage = 0
                    elif heat_zone_i < 0:
                        signal = 'sell'
                        direction = -1
                        entry_price = prev_close
                        sl = entry_price - self.sl_pips * pip_size * direction
                        exposure = 1.0
                        stage = 0

                if exposure > 0.0:
                    # Define TP prices
                    tp1_price = entry_price + self.tp1_pips * pip_size * direction
                    tp2_price = entry_price + self.tp2_pips * pip_size * direction
                    tp3_price = entry_price + self.tp3_pips * pip_size * direction

                    # Check hits
                    hit_tp3 = (high_i >= tp3_price if direction == 1 else low_i <= tp3_price)
                    hit_tp2 = (high_i >= tp2_price if direction == 1 else low_i <= tp2_price)
                    hit_tp1 = (high_i >= tp1_price if direction == 1 else low_i <= tp1_price)
                    hit_sl = (low_i <= sl if direction == 1 else high_i >= sl)

                    realized_pnl_pct = 0.0
                    closed_this_bar = False

                    if hit_tp3:
                        # Close all remaining at TP3
                        pnl_pct = direction * (tp3_price - entry_price) / entry_price * exposure
                        realized_pnl_pct += pnl_pct
                        exposure = 0.0
                        direction = 0
                        stage = 0
                        closed_this_bar = True
                    elif hit_tp2:
                        if stage == 1:
                            # Normal partial close at TP2
                            close_size = 1/3
                            pnl_pct = direction * (tp2_price - entry_price) / entry_price * close_size
                            realized_pnl_pct += pnl_pct
                            exposure -= close_size
                            dist2 = self.tp2_pips * pip_size
                            sl = tp2_price - direction * (0.2 * dist2)
                            stage = 2
                            closed_this_bar = True
                        elif stage == 0:
                            # Hit TP2 directly, close two portions
                            close_size1 = 1/3
                            tp1_price_local = entry_price + self.tp1_pips * pip_size * direction
                            pnl1 = direction * (tp1_price_local - entry_price) / entry_price * close_size1
                            close_size2 = 1/3
                            pnl2 = direction * (tp2_price - entry_price) / entry_price * close_size2
                            realized_pnl_pct += pnl1 + pnl2
                            exposure -= (close_size1 + close_size2)
                            dist2 = self.tp2_pips * pip_size
                            sl = tp2_price - direction * (0.2 * dist2)
                            stage = 2
                            closed_this_bar = True
                    elif hit_tp1 and stage == 0:
                        # Partial close at TP1
                        close_size = 1/3
                        pnl_pct = direction * (tp1_price - entry_price) / entry_price * close_size
                        realized_pnl_pct += pnl_pct
                        exposure -= close_size
                        dist1 = self.tp1_pips * pip_size
                        sl = tp1_price - direction * (0.2 * dist1)
                        stage = 1
                        closed_this_bar = True

                    if hit_sl and exposure > 0.0 and not closed_this_bar:
                        pnl_pct = direction * (sl - entry_price) / entry_price * exposure
                        realized_pnl_pct += pnl_pct
                        exposure = 0.0
                        direction = 0
                        stage = 0
                        closed_this_bar = True

                    # Unrealized return for remaining exposure
                    if exposure > 0.0:
                        unrealized = direction * asset_return * exposure
                    else:
                        unrealized = 0.0

                    strategy_return = unrealized + realized_pnl_pct

                df.loc[df.index[i], 'strategy_returns'] = strategy_return
                df.loc[df.index[i], 'position'] = direction * exposure if exposure > 0.0 else 0.0
                df.loc[df.index[i], 'signal'] = signal

            return df[['close', 'signal', 'position', 'strategy_returns', 'osc', 'heat_zone', 'wma']]

    def get_strategy_config(self) -> Dict:
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

    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> Dict:
        """
        Execute the Range Oscillator strategy: generate signals and place trades if conditions met.
        """
        # Generate signals
        signals_df = self.generate_signals(df)
        if len(signals_df) == 0:
            self.logger.info("No data available for signal generation")
            return {"status": "no_data", "signal": "hold"}

        latest_signal = signals_df['signal'].iloc[-1]
        if latest_signal not in ['buy', 'sell']:
            self.logger.info(f"No actionable signal generated: {latest_signal}")
            return {"status": "no_signal", "signal": latest_signal}

        # Validate signal with sentiment
        if not self.validate_signal_with_sentiment(latest_signal, sentiment):
            self.logger.info(f"Signal '{latest_signal}' rejected due to sentiment '{sentiment}'")
            return {"status": "sentiment_rejected", "signal": latest_signal}

        # Get current market price
        current_price = self.get_market_price(latest_signal)
        if current_price is None:
            self.logger.error("Failed to get current market price")
            return {"status": "no_price", "signal": latest_signal}

        # Calculate SL and TP using base method
        stop_loss, take_profit = self.calculate_stop_take_levels(
            latest_signal, current_price, self.sl_pips, self.tp_pips
        )

        # Calculate position size
        lot_size = self.calculate_position_size(balance, current_price, stop_loss, risk_percent=2.0)

        # Validate stop distances
        if not self.validate_stop_distances(current_price, stop_loss, take_profit):
            self.logger.warning("Stop loss/take profit distances invalid")
            return {"status": "invalid_stops", "signal": latest_signal}

        # Execute the trade
        result = self.execute_trade(
            latest_signal, sentiment, current_price,
            stop_loss, take_profit, lot_size, "Range Oscillator"
        )

        if result:
            self.logger.info(f"Trade executed successfully: {latest_signal} at {current_price}, lot: {lot_size}")
            return {
                "status": "executed",
                "signal": latest_signal,
                "ticket": result.order,
                "price": current_price,
                "sl": stop_loss,
                "tp": take_profit,
                "lot_size": lot_size
            }
        else:
            self.logger.error(f"Failed to execute {latest_signal} trade")
            return {"status": "execution_failed", "signal": latest_signal}
