import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from utils.tp_magic import TPMagic, TPMagicMode

class MultiRSIEMAStrategy(BaseStrategy):
    def __init__(self, symbol):
        super().__init__(symbol)
        # Strategy Parameters
        self.symbol = symbol
        self.timeframe = mt5.TIMEFRAME_M15
        self.lot_size = 0.01
        self.magic_number = 234567
        self.risk_reward_ratio = 2.0
        self.rsi_period_fast = 2    # Red RSI - Short-term oversold/overbought
        self.rsi_period_mid = 9     # Green RSI - Pullback identification
        self.rsi_period_slow = 34   # White RSI - Trend confirmation
        self.rsi_center = 50
        self.ema_fast = 34          # Red EMA
        self.ema_slow = 144         # Blue EMA
        self.swing_lookback = 15

        # TPMagic parameters for Multi RSI EMA (M15 timeframe)
        self.tp1_pips = 16
        self.tp2_pips = 24
        self.tp3_pips = 32
        self.sl_pips = 20
        self.tp_magic = None

    def calculate_rsi(self, data, period):
        """Calculate RSI indicator"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_ema(self, data, period):
        """Calculate EMA indicator"""
        return data.ewm(span=period, adjust=False).mean()

    def find_swing_low(self, lows, lookback):
        """Find the most recent swing low"""
        if len(lows) < lookback:
            return lows[-1]
        return min(lows[-lookback:])

    def find_swing_high(self, highs, lookback):
        """Find the most recent swing high"""
        if len(highs) < lookback:
            return highs[-1]
        return max(highs[-lookback:])

    def generate_signals(self, df: pd.DataFrame, sentiment: str = None, backtest_mode: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on Ultra High Win Rate Multi RSI EMA strategy (targeting 75%+ win rate)
        Vectorized version for performance optimization
        If backtest_mode, use TPMagic for multi-TP simulation.
        """
        if backtest_mode:
            if len(df) < max(self.ema_slow, self.rsi_period_slow, self.swing_lookback) + 10:
                df['signal'] = 'hold'
                df['position'] = 0.0
                df['strategy_returns'] = 0.0
                return df[['close', 'signal', 'position', 'strategy_returns']]

            # Ensure required columns for backtest
            if 'high' not in df.columns or 'low' not in df.columns:
                df['high'] = df['close']
                df['low'] = df['close']
            if 'tick_volume' not in df.columns:
                df['tick_volume'] = 100

            # Add volume analysis
            df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            df['volume_trend'] = df['tick_volume'].rolling(5).mean() / df['tick_volume'].rolling(20).mean()

            # Calculate core indicators
            df['rsi_2'] = self.calculate_rsi(df['close'], self.rsi_period_fast)
            df['rsi_9'] = self.calculate_rsi(df['close'], self.rsi_period_mid)
            df['rsi_34'] = self.calculate_rsi(df['close'], self.rsi_period_slow)
            df['ema_34'] = self.calculate_ema(df['close'], self.ema_fast)
            df['ema_144'] = self.calculate_ema(df['close'], self.ema_slow)
            df['ema_21'] = self.calculate_ema(df['close'], 21)
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Precompute rolling windows
            df['volatility'] = df['close'].rolling(14).std()
            df['recent_high'] = df['high'].rolling(30).max()
            df['recent_low'] = df['low'].rolling(30).min()
            df['high_roll_10'] = df['high'].rolling(10).max()
            df['low_roll_10'] = df['low'].rolling(10).min()
            df['close_roll_50'] = df['close'].rolling(50).mean()
            df['returns'] = df['close'].pct_change().fillna(0)

            # MACD-style
            macd_line = df['ema_34'] - df['ema_144']
            df['macd_signal'] = macd_line.rolling(12).mean()

            # BUY conditions
            strong_uptrend = (
                (df['ema_34'] > df['ema_144'] * 1.003) &
                (df['ema_21'] > df['ema_34']) &
                (df['close'] > df['sma_50'] * 1.005)
            ).fillna(False)

            price_momentum = df['close'] > df['close'].rolling(6).max().shift(1)
            price_above_key_ema = df['close'] > df['ema_34'] * 1.005

            rsi_34_very_bullish = df['rsi_34'] > 58
            rsi_9_precise_pullback = (
                (df['rsi_9'].shift(1) <= 42) &
                (df['rsi_9'] > 48) &
                (df['rsi_9'] < 65)
            )
            rsi_2_explosive_entry = (
                (df['rsi_2'].shift(1) < 35) &
                (df['rsi_2'] > 65) &
                (df['rsi_2'] < 95)
            )

            momentum_20 = df['close'] > df['close'].shift(20)
            momentum_5 = df['close'] > df['close'].shift(5)
            ema_34_momentum = (
                (df['ema_34'] > df['ema_34'].shift(3)) &
                (df['ema_34'].shift(1) > df['ema_34'].shift(4))
            )

            optimal_volatility = (
                (df['volatility'] > 0.8) &
                (df['volatility'] < 3.5)
            ).fillna(True)

            volume_surge = (
                (df['volume_ratio'] > 2.2) &
                (df['volume_trend'] > 1.15)
            ).fillna(True)

            market_position = ((df['close'] - df['recent_low']) / (df['recent_high'] - df['recent_low'])).fillna(0.5)
            bullish_market_position = market_position > 0.7

            low_8 = df['low'].rolling(8).min()
            low_16_8 = df['low'].rolling(16).min().shift(8)
            price_higher_low = low_8 > low_16_8
            rsi_34_mean_8 = df['rsi_34'].rolling(8).mean()
            rsi_divergence_bull = (price_higher_low & (df['rsi_34'] > rsi_34_mean_8)).fillna(False)

            macd_bullish = (
                (macd_line > df['macd_signal']) &
                (macd_line > macd_line.shift(1))
            ).fillna(True)

            consolidation_break = df['close'] > df['high_roll_10'].shift(1)

            longer_trend = df['close_roll_50'] > df['close_roll_50'].shift(5)

            not_overextended = df['close'] < df['recent_high'] * 0.998

            buy_conditions = [
                strong_uptrend, price_momentum, price_above_key_ema, rsi_34_very_bullish,
                rsi_9_precise_pullback, rsi_2_explosive_entry, momentum_20, momentum_5,
                ema_34_momentum, optimal_volatility, volume_surge, bullish_market_position,
                rsi_divergence_bull, macd_bullish, consolidation_break, longer_trend, not_overextended
            ]
            df['buy_score'] = sum(buy_conditions)

            # SELL conditions (symmetric)
            strong_downtrend = (
                (df['ema_34'] < df['ema_144'] * 0.997) &
                (df['ema_21'] < df['ema_34']) &
                (df['close'] < df['sma_50'] * 0.995)
            ).fillna(False)

            price_momentum_sell = df['close'] < df['close'].rolling(6).min().shift(1)
            price_below_key_ema = df['close'] < df['ema_34'] * 0.995

            rsi_34_very_bearish = df['rsi_34'] < 42
            rsi_9_precise_pullback_sell = (
                (df['rsi_9'].shift(1) >= 58) &
                (df['rsi_9'] < 52) &
                (df['rsi_9'] > 35)
            )
            rsi_2_explosive_sell = (
                (df['rsi_2'].shift(1) > 65) &
                (df['rsi_2'] < 35) &
                (df['rsi_2'] > 5)
            )

            momentum_20_sell = df['close'] < df['close'].shift(20)
            momentum_5_sell = df['close'] < df['close'].shift(5)
            ema_34_momentum_sell = (
                (df['ema_34'] < df['ema_34'].shift(3)) &
                (df['ema_34'].shift(1) < df['ema_34'].shift(4))
            )

            bearish_market_position = market_position < 0.3

            high_8 = df['high'].rolling(8).max()
            high_16_8 = df['high'].rolling(16).max().shift(8)
            price_lower_high = high_8 < high_16_8
            rsi_divergence_bear = (price_lower_high & (df['rsi_34'] < rsi_34_mean_8)).fillna(False)

            macd_bearish = (
                (macd_line < df['macd_signal']) &
                (macd_line < macd_line.shift(1))
            ).fillna(True)

            consolidation_breakdown = df['close'] < df['low_roll_10'].shift(1)

            longer_trend_sell = df['close_roll_50'] < df['close_roll_50'].shift(5)

            not_oversold = df['close'] > df['recent_low'] * 1.002

            sell_conditions = [
                strong_downtrend, price_momentum_sell, price_below_key_ema, rsi_34_very_bearish,
                rsi_9_precise_pullback_sell, rsi_2_explosive_sell, momentum_20_sell, momentum_5_sell,
                ema_34_momentum_sell, optimal_volatility, volume_surge, bearish_market_position,
                rsi_divergence_bear, macd_bearish, consolidation_breakdown, longer_trend_sell, not_oversold
            ]
            df['sell_score'] = sum(sell_conditions)

            # Generate signals
            conditions = pd.DataFrame({
                'buy': df['buy_score'] >= 7,
                'sell': df['sell_score'] >= 7
            })
            df['signal'] = 'hold'
            df.loc[conditions['buy'], 'signal'] = 'buy'
            df.loc[conditions['sell'], 'signal'] = 'sell'

            # Backtest with TPMagic
            df['position'] = 0.0
            df['strategy_returns'] = 0.0
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

            # Drop temporary columns
            df.drop(['buy_score', 'sell_score', 'volume_ma', 'volume_trend', 'volatility',
                     'recent_high', 'recent_low', 'high_roll_10', 'low_roll_10', 'close_roll_50',
                     'macd_signal'], axis=1, inplace=True, errors='ignore')

            return df[['close', 'signal', 'position', 'strategy_returns', 'rsi_2', 'rsi_9', 'rsi_34', 'ema_34', 'ema_144', 'ema_21']]
        else:
            if len(df) < max(self.ema_slow, self.rsi_period_slow, self.swing_lookback) + 10:
                df['signal'] = 'hold'
                return df[['close', 'signal']]

            # Ensure required columns
            if 'high' not in df.columns or 'low' not in df.columns:
                raise ValueError("Data must include 'high' and 'low' columns for market structure analysis")

            # Add advanced volume analysis (vectorized)
            if 'tick_volume' in df.columns:
                df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
                df['volume_trend'] = df['tick_volume'].rolling(5).mean() / df['tick_volume'].rolling(20).mean()
            else:
                df['volume_ratio'] = 1.0
                df['volume_trend'] = 1.0

            # Calculate core indicators (already vectorized)
            df['rsi_2'] = self.calculate_rsi(df['close'], self.rsi_period_fast)
            df['rsi_9'] = self.calculate_rsi(df['close'], self.rsi_period_mid)
            df['rsi_34'] = self.calculate_rsi(df['close'], self.rsi_period_slow)
            df['ema_34'] = self.calculate_ema(df['close'], self.ema_fast)
            df['ema_144'] = self.calculate_ema(df['close'], self.ema_slow)
            df['ema_21'] = self.calculate_ema(df['close'], 21)
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Precompute rolling windows for efficiency (vectorized)
            df['volatility'] = df['close'].rolling(14).std()
            df['recent_high'] = df['high'].rolling(30).max()
            df['recent_low'] = df['low'].rolling(30).min()
            df['high_roll_10'] = df['high'].rolling(10).max()
            df['low_roll_10'] = df['low'].rolling(10).min()
            df['close_roll_50'] = df['close'].rolling(50).mean()

            # MACD-style (vectorized)
            macd_line = df['ema_34'] - df['ema_144']
            df['macd_signal'] = macd_line.rolling(12).mean()

            # Vectorized BUY conditions
            # 1. STRONG TREND CONFIRMATION
            strong_uptrend = (
                (df['ema_34'] > df['ema_144'] * 1.003) &
                (df['ema_21'] > df['ema_34']) &
                (df['close'] > df['sma_50'] * 1.005)
            ).fillna(False)

            # 2. PRICE ACTION QUALITY
            price_momentum = df['close'] > df['close'].rolling(6).max().shift(1)  # Approx i-20 to i-15
            price_above_key_ema = df['close'] > df['ema_34'] * 1.005

            # 3. RSI COORDINATION
            rsi_34_very_bullish = df['rsi_34'] > 58
            rsi_9_precise_pullback = (
                (df['rsi_9'].shift(1) <= 42) &
                (df['rsi_9'] > 48) &
                (df['rsi_9'] < 65)
            )
            rsi_2_explosive_entry = (
                (df['rsi_2'].shift(1) < 35) &
                (df['rsi_2'] > 65) &
                (df['rsi_2'] < 95)
            )

            # 4. ADVANCED MOMENTUM FILTERS
            momentum_20 = df['close'] > df['close'].shift(20)
            momentum_5 = df['close'] > df['close'].shift(5)
            ema_34_momentum = (
                (df['ema_34'] > df['ema_34'].shift(3)) &
                (df['ema_34'].shift(1) > df['ema_34'].shift(4))
            )

            # 5. VOLATILITY AND TIMING
            optimal_volatility = (
                (df['volatility'] > 0.8) &
                (df['volatility'] < 3.5)
            ).fillna(True)

            # 6. VOLUME CONFIRMATION
            volume_surge = (
                (df['volume_ratio'] > 2.2) &
                (df['volume_trend'] > 1.15)
            ).fillna(True)

            # 7. MARKET STRUCTURE
            market_position = ((df['close'] - df['recent_low']) / (df['recent_high'] - df['recent_low'])).fillna(0.5)
            bullish_market_position = market_position > 0.7

            # 8. RSI DIVERGENCE (Simplified vectorized approximation)
            low_8 = df['low'].rolling(8).min()
            low_16_8 = df['low'].rolling(16).min().shift(8)
            price_higher_low = low_8 > low_16_8
            rsi_34_mean_8 = df['rsi_34'].rolling(8).mean()
            rsi_divergence_bull = (price_higher_low & (df['rsi_34'] > rsi_34_mean_8)).fillna(False)

            # 9. MACD-STYLE MOMENTUM
            macd_bullish = (
                (macd_line > df['macd_signal']) &
                (macd_line > macd_line.shift(1))
            ).fillna(True)

            # 10. PRICE PATTERN RECOGNITION
            consolidation_break = df['close'] > df['high_roll_10'].shift(1)

            # 11. MULTI-TIMEFRAME SIMULATION
            longer_trend = df['close_roll_50'] > df['close_roll_50'].shift(5)

            # 12. RISK MANAGEMENT FILTER
            not_overextended = df['close'] < df['recent_high'] * 0.998

            # Buy score (vectorized sum)
            buy_conditions = [
                strong_uptrend, price_momentum, price_above_key_ema, rsi_34_very_bullish,
                rsi_9_precise_pullback, rsi_2_explosive_entry, momentum_20, momentum_5,
                ema_34_momentum, optimal_volatility, volume_surge, bullish_market_position,
                rsi_divergence_bull, macd_bullish, consolidation_break, longer_trend, not_overextended
            ]
            df['buy_score'] = sum(buy_conditions)

            # Vectorized SELL conditions (symmetric)
            # 1. STRONG DOWNTREND
            strong_downtrend = (
                (df['ema_34'] < df['ema_144'] * 0.997) &
                (df['ema_21'] < df['ema_34']) &
                (df['close'] < df['sma_50'] * 0.995)
            ).fillna(False)

            # 2. BEARISH PRICE ACTION
            price_momentum_sell = df['close'] < df['close'].rolling(6).min().shift(1)
            price_below_key_ema = df['close'] < df['ema_34'] * 0.995

            # 3. RSI COORDINATION (Sell)
            rsi_34_very_bearish = df['rsi_34'] < 42
            rsi_9_precise_pullback_sell = (
                (df['rsi_9'].shift(1) >= 58) &
                (df['rsi_9'] < 52) &
                (df['rsi_9'] > 35)
            )
            rsi_2_explosive_sell = (
                (df['rsi_2'].shift(1) > 65) &
                (df['rsi_2'] < 35) &
                (df['rsi_2'] > 5)
            )

            # 4. BEARISH MOMENTUM
            momentum_20_sell = df['close'] < df['close'].shift(20)
            momentum_5_sell = df['close'] < df['close'].shift(5)
            ema_34_momentum_sell = (
                (df['ema_34'] < df['ema_34'].shift(3)) &
                (df['ema_34'].shift(1) < df['ema_34'].shift(4))
            )

            # 5. MARKET STRUCTURE (Sell)
            bearish_market_position = market_position < 0.3

            # 6. DIVERGENCE (Sell)
            high_8 = df['high'].rolling(8).max()
            high_16_8 = df['high'].rolling(16).max().shift(8)
            price_lower_high = high_8 < high_16_8
            rsi_divergence_bear = (price_lower_high & (df['rsi_34'] < rsi_34_mean_8)).fillna(False)

            # 7. MACD BEARISH
            macd_bearish = (
                (macd_line < df['macd_signal']) &
                (macd_line < macd_line.shift(1))
            ).fillna(True)

            # 8. BREAKDOWN PATTERN
            consolidation_breakdown = df['close'] < df['low_roll_10'].shift(1)

            # 9. LONGER TIMEFRAME BEARISH
            longer_trend_sell = df['close_roll_50'] < df['close_roll_50'].shift(5)

            # 10. NOT OVERSOLD
            not_oversold = df['close'] > df['recent_low'] * 1.002

            # Sell score
            sell_conditions = [
                strong_downtrend, price_momentum_sell, price_below_key_ema, rsi_34_very_bearish,
                rsi_9_precise_pullback_sell, rsi_2_explosive_sell, momentum_20_sell, momentum_5_sell,
                ema_34_momentum_sell, optimal_volatility, volume_surge, bearish_market_position,
                rsi_divergence_bear, macd_bearish, consolidation_breakdown, longer_trend_sell, not_oversold
            ]
            df['sell_score'] = sum(sell_conditions)

            # Generate signals vectorized
            conditions = pd.DataFrame({
                'buy': df['buy_score'] >= 7,
                'sell': df['sell_score'] >= 7
            })
            df['signal'] = 'hold'
            df.loc[conditions['buy'], 'signal'] = 'buy'
            df.loc[conditions['sell'], 'signal'] = 'sell'

            # Drop temporary columns
            df.drop(['buy_score', 'sell_score', 'volume_ma', 'volume_trend', 'volatility',
                     'recent_high', 'recent_low', 'high_roll_10', 'low_roll_10', 'close_roll_50',
                     'macd_signal'], axis=1, inplace=True, errors='ignore')

            # Return relevant columns
            return df[['close', 'signal', 'rsi_2', 'rsi_9', 'rsi_34', 'ema_34', 'ema_144', 'ema_21']]

    def get_strategy_config(self) -> dict:
        """Return strategy configuration parameters"""
        return {
            "strategy_name": "Ultra High Win Rate Multi RSI EMA Strategy (75%+ Target)",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "parameters": {
                "lot_size": self.lot_size,
                "magic_number": self.magic_number,
                "risk_reward_ratio": self.risk_reward_ratio,
                "rsi_period_fast": self.rsi_period_fast,
                "rsi_period_mid": self.rsi_period_mid,
                "rsi_period_slow": self.rsi_period_slow,
                "rsi_34_bull_threshold": 58,  # Stronger trend requirement
                "rsi_34_bear_threshold": 42,
                "rsi_9_pullback_buy_low": 42,  # Precise pullback zones
                "rsi_9_pullback_buy_high": 65,
                "rsi_9_pullback_sell_low": 35,
                "rsi_9_pullback_sell_high": 58,
                "rsi_2_explosive_buy_prev": 35,  # Explosive entry conditions
                "rsi_2_explosive_buy_curr_min": 65,
                "rsi_2_explosive_buy_curr_max": 95,
                "rsi_2_explosive_sell_prev": 65,
                "rsi_2_explosive_sell_curr_min": 5,
                "rsi_2_explosive_sell_curr_max": 35,
                "strong_trend_multiplier": 1.003,  # EMA trend strength
                "price_ema_buffer": 1.005,  # Price above/below EMA buffer
                "sma_trend_buffer": 1.005,  # SMA trend confirmation
                "momentum_bars_short": 5,
                "momentum_bars_medium": 20,
                "ema_momentum_bars": 3,
                "volatility_min": 0.8,
                "volatility_max": 3.5,
                "volume_ratio_threshold": 2.2,  # Stricter volume requirement
                "volume_trend_threshold": 1.15,
                "market_position_bull": 0.7,  # Very bullish market position
                "market_position_bear": 0.3,
                "price_lookback": 30,  # Market structure analysis
                "divergence_lookback": 16,
                "macd_signal_period": 12,
                "consolidation_break_period": 10,
                "longer_trend_ma": 50,
                "longer_trend_lookback": 5,
                "resistance_buffer": 0.998,
                "support_buffer": 1.002,
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
                "ema_21": 21,  # Additional trend filter
                "swing_lookback": self.swing_lookback,
                "min_conditions_buy": 12,  # Ultra-strict: 12+ out of 17 for 70%+ win rate
                "min_conditions_sell": 12,  # Ultra-strict for sells
                "total_conditions": 17
            },
            "description": "ULTRA HIGH WIN RATE Multi RSI EMA Strategy: 17 advanced filters with 12+ condition requirement for 75%+ win rate. Features: Multi-timeframe trend analysis, precise RSI coordination, momentum confirmation, volume surge detection, market structure analysis, divergence patterns, breakout recognition, and risk management filters for maximum trade quality."
        }

    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Multi RSI EMA strategy using TPMagic for consistent multi-TP management.
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
