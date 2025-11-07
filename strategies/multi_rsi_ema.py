import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Ultra High Win Rate Multi RSI EMA strategy (targeting 75%+ win rate)
        """
        if len(df) < max(self.ema_slow, self.rsi_period_slow, self.swing_lookback) + 10:
            df['signal'] = 'hold'
            return df[['close', 'signal']]

        # Add advanced volume analysis
        if 'tick_volume' in df.columns:
            df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            df['volume_trend'] = df['tick_volume'].rolling(5).mean() / df['tick_volume'].rolling(20).mean()

        # Calculate core indicators
        df['rsi_2'] = self.calculate_rsi(df['close'], self.rsi_period_fast)    # Ultra short-term
        df['rsi_9'] = self.calculate_rsi(df['close'], self.rsi_period_mid)     # Pullback detector
        df['rsi_34'] = self.calculate_rsi(df['close'], self.rsi_period_slow)   # Trend strength
        df['ema_34'] = self.calculate_ema(df['close'], self.ema_fast)          # Dynamic support/resistance
        df['ema_144'] = self.calculate_ema(df['close'], self.ema_slow)         # Main trend filter

        # Advanced trend analysis
        df['ema_21'] = self.calculate_ema(df['close'], 21)  # Additional trend filter
        df['sma_50'] = df['close'].rolling(window=50).mean()  # Long-term trend

        # Precompute position series
        pos = pd.Series(np.arange(len(df)), index=df.index)

        # Precompute rolling windows for efficiency
        df['recent_high_30'] = df['high'].rolling(30).max()
        df['recent_low_30'] = df['low'].rolling(30).min()
        df['volatility_14'] = df['close'].rolling(14).std()
        df['high_roll10_max'] = df['high'].rolling(10).max().shift(1)
        df['low_roll10_min'] = df['low'].rolling(10).min().shift(1)
        df['sma_50_shift5'] = df['sma_50'].shift(5)
        df['close_prev_max'] = df['close'].shift(17).rolling(5).max()
        df['close_prev_min'] = df['close'].shift(17).rolling(5).min()
        df['close_roll20'] = df['close'].shift(20)
        df['close_roll5'] = df['close'].shift(5)
        df['ema_34_shift3'] = df['ema_34'].shift(3)
        df['ema_34_shift1'] = df['ema_34'].shift(1)
        df['ema_34_shift4'] = df['ema_34'].shift(4)
        df['low_prev8_min'] = df['low'].shift(1).rolling(8).min()
        df['low_prev16_min'] = df['low'].shift(9).rolling(8).min()
        df['rsi_34_prev8_mean'] = df['rsi_34'].shift(1).rolling(8).mean()
        df['high_prev8_max'] = df['high'].shift(1).rolling(8).max()
        df['high_prev16_max'] = df['high'].shift(9).rolling(8).max()
        df['macd_line'] = df['ema_34'] - df['ema_144']
        df['macd_signal'] = df['macd_line'].rolling(12).mean()
        df['macd_line_shift1'] = df['macd_line'].shift(1)

        # Create previous values
        df['rsi_9_prev'] = df['rsi_9'].shift(1)
        df['rsi_34_prev'] = df['rsi_34'].shift(1)
        df['rsi_2_prev'] = df['rsi_2'].shift(1)

        # Volume surge
        if 'tick_volume' in df.columns:
            volume_surge = (df['volume_ratio'] > 2.5) & (df['volume_trend'] > 1.2)
        else:
            volume_surge = pd.Series(True, index=df.index)

        # Market position
        market_position = np.where(
            (df['recent_high_30'] > df['recent_low_30']),
            (df['close'] - df['recent_low_30']) / (df['recent_high_30'] - df['recent_low_30']),
            0.5
        )
        bullish_market_position = pd.Series(market_position > 0.7, index=df.index)
        bearish_market_position = pd.Series(market_position < 0.3, index=df.index)

        # Vectorized conditions for BUY
        n = len(df)
        strong_uptrend = (
            (df['ema_34'] > df['ema_144'] * 1.003) &
            (df['ema_21'] > df['ema_34']) &
            ((pos >= 50) & (df['close'] > df['sma_50'] * 1.005) | (pos < 50))
        )

        price_momentum = (pos >= 20) & (df['close'] > df['close_prev_max'])
        price_above_key_ema = df['close'] > df['ema_34'] * 1.005

        rsi_34_very_bullish = df['rsi_34'] > 60
        rsi_9_precise_pullback = (
            (df['rsi_9_prev'] <= 40) &
            (df['rsi_9'] > 50) &
            (df['rsi_9'] < 60)
        )
        rsi_9_cross_above_34 = (
            (df['rsi_9_prev'] <= df['rsi_34_prev']) &
            (df['rsi_9'] > df['rsi_34'])
        )
        rsi_2_explosive_entry = (
            (df['rsi_2_prev'] < 30) &
            (df['rsi_2'] > 70) &
            (df['rsi_2'] < 90)
        )

        momentum_20 = (pos >= 20) & (df['close'] > df['close_roll20'])
        momentum_5 = (pos >= 5) & (df['close'] > df['close_roll5'])
        ema_34_momentum = (
            (pos >= 4) &
            (df['ema_34'] > df['ema_34_shift3']) &
            (df['ema_34_shift1'] > df['ema_34_shift4'])
        ) | (pos < 4)

        optimal_volatility = ((pos >= 14) & (df['volatility_14'] > 0.8) & (df['volatility_14'] < 3.5)) | (pos < 14)

        price_higher_low = (pos >= 16) & (df['low_prev8_min'] > df['low_prev16_min'])
        rsi_divergence_bull = price_higher_low & (df['rsi_34'] > df['rsi_34_prev8_mean'])

        macd_bullish = (
            (pos >= 12) &
            (df['macd_line'] > df['macd_signal']) &
            (df['macd_line'] > df['macd_line_shift1'])
        ) | (pos < 12)

        consolidation_break = (pos >= 10) & (df['close'] > df['high_roll10_max'])

        longer_trend = (
            (pos >= 55) &
            (df['sma_50'] > df['sma_50_shift5'])
        ) | (pos < 55)

        not_overextended = df['close'] < df['recent_high_30'] * 0.998

        # Buy score
        buy_conditions = pd.concat([
            strong_uptrend.astype(int),
            price_momentum.astype(int),
            price_above_key_ema.astype(int),
            rsi_34_very_bullish.astype(int),
            rsi_9_precise_pullback.astype(int),
            rsi_9_cross_above_34.astype(int),
            rsi_2_explosive_entry.astype(int),
            momentum_20.astype(int),
            momentum_5.astype(int),
            ema_34_momentum.astype(int),
            optimal_volatility.astype(int),
            volume_surge.astype(int),
            bullish_market_position.astype(int),
            rsi_divergence_bull.astype(int),
            macd_bullish.astype(int),
            consolidation_break.astype(int),
            longer_trend.astype(int),
            not_overextended.astype(int)
        ], axis=1).sum(axis=1)

        # Vectorized conditions for SELL
        strong_downtrend = (
            (df['ema_34'] < df['ema_144'] * 0.997) &
            (df['ema_21'] < df['ema_34']) &
            ((pos >= 50) & (df['close'] < df['sma_50'] * 0.995) | (pos < 50))
        )

        price_momentum_sell = (pos >= 20) & (df['close'] < df['close_prev_min'])
        price_below_key_ema = df['close'] < df['ema_34'] * 0.995

        rsi_34_very_bearish = df['rsi_34'] < 40
        rsi_9_precise_pullback_sell = (
            (df['rsi_9_prev'] >= 60) &
            (df['rsi_9'] < 50) &
            (df['rsi_9'] > 40)
        )
        rsi_9_cross_below_34 = (
            (df['rsi_9_prev'] >= df['rsi_34_prev']) &
            (df['rsi_9'] < df['rsi_34'])
        )
        rsi_2_explosive_sell = (
            (df['rsi_2_prev'] > 70) &
            (df['rsi_2'] < 30) &
            (df['rsi_2'] > 10)
        )

        momentum_20_sell = (pos >= 20) & (df['close'] < df['close_roll20'])
        momentum_5_sell = (pos >= 5) & (df['close'] < df['close_roll5'])
        ema_34_momentum_sell = (
            (pos >= 4) &
            (df['ema_34'] < df['ema_34_shift3']) &
            (df['ema_34_shift1'] < df['ema_34_shift4'])
        ) | (pos < 4)

        price_lower_high = (pos >= 16) & (df['high_prev8_max'] < df['high_prev16_max'])
        rsi_divergence_bear = price_lower_high & (df['rsi_34'] < df['rsi_34_prev8_mean'])

        macd_bearish = (
            (pos >= 12) &
            (df['macd_line'] < df['macd_signal']) &
            (df['macd_line'] < df['macd_line_shift1'])
        ) | (pos < 12)

        consolidation_breakdown = (pos >= 10) & (df['close'] < df['low_roll10_min'])

        longer_trend_sell = (
            (pos >= 55) &
            (df['sma_50'] < df['sma_50_shift5'])
        ) | (pos < 55)

        not_oversold = df['close'] > df['recent_low_30'] * 1.002

        # Sell score
        sell_conditions = pd.concat([
            strong_downtrend.astype(int),
            price_momentum_sell.astype(int),
            price_below_key_ema.astype(int),
            rsi_34_very_bearish.astype(int),
            rsi_9_precise_pullback_sell.astype(int),
            rsi_9_cross_below_34.astype(int),
            rsi_2_explosive_sell.astype(int),
            momentum_20_sell.astype(int),
            momentum_5_sell.astype(int),
            ema_34_momentum_sell.astype(int),
            optimal_volatility.astype(int),
            volume_surge.astype(int),
            bearish_market_position.astype(int),
            rsi_divergence_bear.astype(int),
            macd_bearish.astype(int),
            consolidation_breakdown.astype(int),
            longer_trend_sell.astype(int),
            not_oversold.astype(int)
        ], axis=1).sum(axis=1)

        # Generate signals vectorized
        signals = pd.Series(['hold'] * n, index=df.index)
        signals[buy_conditions >= 12] = 'buy'
        signals[sell_conditions >= 12] = 'sell'

        df['signal'] = signals

        # Return relevant columns with new indicators
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
                "min_conditions_buy": 10,  # Adjusted: 10+ out of 18 for better quality
                "min_conditions_sell": 10,  # Adjusted for sells
                "total_conditions": 18
            },
            "description": "ULTRA HIGH WIN RATE Multi RSI EMA Strategy: 17 advanced filters with 12+ condition requirement for 75%+ win rate. Features: Multi-timeframe trend analysis, precise RSI coordination, momentum confirmation, volume surge detection, market structure analysis, divergence patterns, breakout recognition, and risk management filters for maximum trade quality."
        }

    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Multi RSI EMA strategy: generate signals and place trades if conditions met.
        
        Args:
            df: Market data DataFrame
            sentiment: Current market sentiment ('bullish', 'bearish', 'neutral')
            balance: Current account balance
            
        Returns:
            dict: Execution result with status and details
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
        
        # Get symbol info for point and digits
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            self.logger.error(f"Failed to get symbol info for {self.symbol}")
            return {"status": "no_symbol_info", "signal": latest_signal}
        
        point = symbol_info.point
        digits = symbol_info.digits
        
        # Define SL and TP distances (50 pips SL, 100 pips TP for 1:2 RR)
        # For XAUUSD, 1 pip = 0.1, point = 0.01, so 50 pips = 5.0 price units
        sl_distance = 5.0  # 50 pips
        tp_distance = 10.0  # 100 pips
        
        # Calculate SL and TP levels
        if latest_signal == 'buy':
            stop_loss = round(current_price - sl_distance, digits)
            take_profit = round(current_price + tp_distance, digits)
        else:  # sell
            stop_loss = round(current_price + sl_distance, digits)
            take_profit = round(current_price - tp_distance, digits)
        
        # Calculate position size based on risk (2% of balance)
        lot_size = self.calculate_position_size(balance, current_price, stop_loss, risk_percent=2.0)
        
        # Validate stop distances
        if not self.validate_stop_distances(current_price, stop_loss, take_profit):
            self.logger.warning("Stop loss/take profit distances invalid")
            return {"status": "invalid_stops", "signal": latest_signal}
        
        # Execute the trade
        result = self.execute_trade(
            latest_signal, sentiment, current_price, 
            stop_loss, take_profit, lot_size, "MultiRSIEMA"
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
