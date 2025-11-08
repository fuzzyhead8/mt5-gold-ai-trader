import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from strategies.base_strategy import BaseStrategy
from utils.tp_magic import TPMagic, TPMagicMode

class SwingTradingStrategy(BaseStrategy):
    def __init__(self, symbol):
        super().__init__(symbol)

        # TPMagic parameters for swing trading (H1 timeframe)
        self.tp1_pips = 40
        self.tp2_pips = 60
        self.tp3_pips = 80
        self.sl_pips = 50
        self.tp_magic = None

    def generate_signals(self, data, sentiment: str = None, backtest_mode: bool = False):
        """
        MACD crossover strategy for swing trading.
        If backtest_mode, use TPMagic for multi-TP simulation.
        """
        if backtest_mode:
            # Ensure required columns
            required_cols = ['close', 'high', 'low']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = data['close']

            # Calculate MACD
            data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['returns'] = data['close'].pct_change().fillna(0)

            # Crossover detection
            buy_crossover = (
                (data['MACD'] > data['Signal_Line']) &
                (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
            ).fillna(False)

            sell_crossover = (
                (data['MACD'] < data['Signal_Line']) &
                (data['MACD'].shift(1) >= data['Signal_Line'].shift(1))
            ).fillna(False)

            data['signal'] = 'hold'
            data.loc[buy_crossover, 'signal'] = 'buy'
            data.loc[sell_crossover, 'signal'] = 'sell'

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

            return data[['close', 'MACD', 'Signal_Line', 'signal', 'position', 'strategy_returns']]
        else:
            # Normal mode
            data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Vectorized crossover detection
            buy_crossover = (
                (data['MACD'] > data['Signal_Line']) &
                (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
            ).fillna(False)

            sell_crossover = (
                (data['MACD'] < data['Signal_Line']) &
                (data['MACD'].shift(1) >= data['Signal_Line'].shift(1))
            ).fillna(False)

            # Generate signals vectorized
            data['signal'] = 'hold'
            data.loc[buy_crossover, 'signal'] = 'buy'
            data.loc[sell_crossover, 'signal'] = 'sell'

            # Ensure first signal is hold
            data.iloc[0, data.columns.get_loc('signal')] = 'hold'

            return data[['close', 'MACD', 'Signal_Line', 'signal']]

    def get_strategy_config(self) -> dict:
        """Return strategy configuration parameters"""
        return {
            "strategy_name": "Swing Trading",
            "symbol": self.symbol,
            "timeframe": "H1",
            "parameters": {
                "ema_fast_period": 12,
                "ema_slow_period": 26,
                "macd_signal_period": 9
            },
            "signal_conditions": {
                "macd_crossover": "MACD crosses above Signal Line for buy, below for sell",
                "confirmation_required": False
            },
            "description": "Simple MACD crossover strategy for swing trading on hourly timeframe"
        }

    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Swing Trading strategy using TPMagic for consistent multi-TP management.
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
