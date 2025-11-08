import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from strategies.base_strategy import BaseStrategy
from utils.tp_magic import TPMagic, TPMagicMode

class DayTradingStrategy(BaseStrategy):
    def __init__(self, symbol):
        super().__init__(symbol)
        # Parameters optimized for M15 timeframe (daily trading) - Made less conservative
        self.rsi_period = 14  # Shorter period for more responsive signals
        self.rsi_overbought = 65  # Less extreme levels for more signals
        self.rsi_oversold = 35
        self.ema_fast = 12  # For trend filtering
        self.ema_slow = 26
        self.volume_threshold = 50  # Lower minimum volume threshold

        # TPMagic parameters for day trading (M30 timeframe)
        self.tp1_pips = 20
        self.tp2_pips = 30
        self.tp3_pips = 40
        self.sl_pips = 25
        self.tp_magic = None


    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI with proper handling of edge cases"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value

    def _calculate_ema(self, prices, span):
        """Calculate exponential moving average"""
        return prices.ewm(span=span, adjust=False).mean()

    def _is_valid_signal_time(self, timestamp):
        """Check if time is suitable for day trading (avoid low liquidity periods)"""
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            # Active trading hours: 07:00-17:00 (European + US overlap)
            return 7 <= hour <= 17
        return True  # Default to valid if can't determine time

    def generate_signals(self, data, sentiment: str = None, backtest_mode: bool = False):
        """
        Enhanced day trading strategy with RSI, trend filtering, and volume confirmation
        Vectorized version for performance optimization
        If backtest_mode, use TPMagic for multi-TP simulation.
        """
        if backtest_mode:
            # Ensure required columns for backtest
            if 'tick_volume' not in data.columns:
                data['tick_volume'] = 100
            required_cols = ['close', 'high', 'low']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = data['close']

            # Calculate indicators (same as normal)
            data['RSI'] = self._calculate_rsi(data['close'], self.rsi_period)
            data['EMA_fast'] = self._calculate_ema(data['close'], self.ema_fast)
            data['EMA_slow'] = self._calculate_ema(data['close'], self.ema_slow)
            data['volume_ma'] = data['tick_volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['tick_volume'] / data['volume_ma']
            data['price_change'] = data['close'].pct_change()
            data['volatility'] = data['close'].rolling(window=20).std()
            data['atr'] = (data['high'] - data['low']).rolling(window=14).mean()
            data['trend'] = np.where(data['EMA_fast'] > data['EMA_slow'], 1, -1)
            data['trend_up'] = data['trend'] == 1
            data['trend_down'] = data['trend'] == -1
            data['ema_fast_rising'] = data['EMA_fast'] > data['EMA_fast'].shift(1)
            data['ema_fast_falling'] = data['EMA_fast'] < data['EMA_fast'].shift(1)
            data['returns'] = data['close'].pct_change().fillna(0)

            # Valid data mask
            valid_data = (
                (~data['RSI'].isna()) &
                (~data['EMA_fast'].isna()) &
                (~data['EMA_slow'].isna()) &
                (data['tick_volume'] >= self.volume_threshold) &
                (data['volume_ratio'] > 0)
            )

            # Signals
            data['signal'] = 'hold'

            buy_condition1 = (
                (data['RSI'] < self.rsi_oversold) &
                data['trend_up'] &
                (data['volume_ratio'] > 1.0) &
                data['ema_fast_rising']
            )

            buy_condition2 = (
                (data['RSI'] < 30) &
                data['trend_up']
            )

            sell_condition1 = (
                (data['RSI'] > self.rsi_overbought) &
                data['trend_down'] &
                (data['volume_ratio'] > 1.0) &
                data['ema_fast_falling']
            )

            sell_condition2 = (
                (data['RSI'] > 70) &
                data['trend_down']
            )

            data.loc[valid_data & buy_condition1, 'signal'] = 'buy'
            data.loc[valid_data & buy_condition2 & ~buy_condition1, 'signal'] = 'buy'
            data.loc[valid_data & sell_condition1, 'signal'] = 'sell'
            data.loc[valid_data & sell_condition2 & ~sell_condition1, 'signal'] = 'sell'

            data.iloc[0, data.columns.get_loc('signal')] = 'hold'

            # Backtest with TPMagic
            data['position'] = 0.0
            data['strategy_returns'] = 0.0
            pip_size = 0.1
            tp_magic = TPMagic(
                tp1_pips=self.tp1_pips, tp2_pips=self.tp2_pips, tp3_pips=self.tp3_pips,
                sl_pips=self.sl_pips, symbol=self.symbol, pip_size=pip_size,
                mode=TPMagicMode.BACKTEST, initial_lot=0.01,
                logger=self.logger
            )

            for i in range(len(data)):
                if i == 0:
                    continue

                high_i = data['high'].iloc[i]
                low_i = data['low'].iloc[i]
                close_i = data['close'].iloc[i]
                signal_i = data['signal'].iloc[i]
                asset_return = data['returns'].iloc[i]
                prev_close = data['close'].iloc[i-1]

                if not tp_magic.is_open:
                    if signal_i == 'buy':
                        tp_magic.open_position(1, prev_close)
                    elif signal_i == 'sell':
                        tp_magic.open_position(-1, prev_close)

                if tp_magic.is_open:
                    update_result = tp_magic.update(high_i, low_i, close_i, asset_return)
                    data.loc[data.index[i], 'strategy_returns'] = update_result['strategy_returns']
                    data.loc[data.index[i], 'position'] = update_result['position']
                else:
                    data.loc[data.index[i], 'strategy_returns'] = 0.0
                    data.loc[data.index[i], 'position'] = 0.0

            # Drop temp
            temp_cols = ['trend_up', 'trend_down', 'ema_fast_rising', 'ema_fast_falling']
            data.drop(columns=[col for col in temp_cols if col in data.columns], inplace=True, errors='ignore')

            return data[['close', 'RSI', 'EMA_fast', 'EMA_slow', 'signal', 'position', 'strategy_returns', 'volume_ratio', 'volatility', 'trend']]
        else:
            # Normal mode
            # Ensure we have required columns
            if 'tick_volume' not in data.columns:
                data['tick_volume'] = 100  # Default volume if missing
            
            # Calculate technical indicators (vectorized)
            data['RSI'] = self._calculate_rsi(data['close'], self.rsi_period)
            data['EMA_fast'] = self._calculate_ema(data['close'], self.ema_fast)
            data['EMA_slow'] = self._calculate_ema(data['close'], self.ema_slow)
            
            # Volume analysis (vectorized)
            data['volume_ma'] = data['tick_volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['tick_volume'] / data['volume_ma']
            
            # Price momentum and volatility (vectorized)
            data['price_change'] = data['close'].pct_change()
            data['volatility'] = data['close'].rolling(window=20).std()
            data['atr'] = (data['high'] - data['low']).rolling(window=14).mean()
            
            # Trend direction (vectorized)
            data['trend'] = np.where(data['EMA_fast'] > data['EMA_slow'], 1, -1)
            data['trend_up'] = data['trend'] == 1
            data['trend_down'] = data['trend'] == -1
            
            # EMA momentum (vectorized)
            data['ema_fast_rising'] = data['EMA_fast'] > data['EMA_fast'].shift(1)
            data['ema_fast_falling'] = data['EMA_fast'] < data['EMA_fast'].shift(1)
            
            # Valid data mask (vectorized)
            valid_data = (
                (~data['RSI'].isna()) &
                (~data['EMA_fast'].isna()) &
                (~data['EMA_slow'].isna()) &
                (data['tick_volume'] >= self.volume_threshold) &
                (data['volume_ratio'] > 0)  # Avoid division issues
            )
            
            # Initialize signals
            data['signal'] = 'hold'
            
            # BUY conditions (vectorized)
            buy_condition1 = (
                (data['RSI'] < self.rsi_oversold) &
                data['trend_up'] &
                (data['volume_ratio'] > 1.0) &
                data['ema_fast_rising']
            )
            
            buy_condition2 = (
                (data['RSI'] < 30) &
                data['trend_up']
            )
            
            # SELL conditions (vectorized)
            sell_condition1 = (
                (data['RSI'] > self.rsi_overbought) &
                data['trend_down'] &
                (data['volume_ratio'] > 1.0) &
                data['ema_fast_falling']
            )
            
            sell_condition2 = (
                (data['RSI'] > 70) &
                data['trend_down']
            )
            
            # Apply signals where data is valid
            data.loc[valid_data & buy_condition1, 'signal'] = 'buy'
            data.loc[valid_data & buy_condition2 & ~buy_condition1, 'signal'] = 'buy'  # Additional only if not already buy
            data.loc[valid_data & sell_condition1, 'signal'] = 'sell'
            data.loc[valid_data & sell_condition2 & ~sell_condition1, 'signal'] = 'sell'  # Additional only if not already sell
            
            # Set first row to hold
            data.iloc[0, data.columns.get_loc('signal')] = 'hold'
            
            # Drop temporary columns (keep 'trend' for return)
            temp_cols = ['trend_up', 'trend_down', 'ema_fast_rising', 'ema_fast_falling']
            data.drop(columns=[col for col in temp_cols if col in data.columns], inplace=True, errors='ignore')
            
            # Return enhanced dataset with indicators
            return data[['close', 'RSI', 'EMA_fast', 'EMA_slow', 'signal', 'volume_ratio', 'volatility', 'trend']]
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Day Trading strategy using TPMagic for consistent multi-TP management.
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
        """Return strategy configuration parameters"""
        return {
            "strategy_name": "Day Trading",
            "symbol": self.symbol,
            "timeframe": "M15",
            "parameters": {
                "rsi_period": self.rsi_period,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold,
                "ema_fast_period": self.ema_fast,
                "ema_slow_period": self.ema_slow,
                "volume_threshold": self.volume_threshold,
                "volume_ma_window": 20,
                "volatility_window": 20,
                "atr_window": 14
            },
            "signal_conditions": {
                "rsi_extreme_levels": [30, 70],
                "volume_ratio_threshold": 1.0,
                "trend_confirmation_required": True,
                "ema_momentum_required": True
            },
            "description": "Day trading strategy optimized for 15-minute timeframe with RSI and trend filtering"
        }
