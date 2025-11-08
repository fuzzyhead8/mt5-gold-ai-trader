import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from strategies.base_strategy import BaseStrategy
from utils.tp_magic import TPMagic, TPMagicMode

class ScalpingStrategy(BaseStrategy):
    def __init__(self, symbol):
        super().__init__(symbol)
        self.min_volume = 50  # Moderate minimum tick volume 
        self.volatility_threshold = 2.5  # Dynamic volatility threshold

        # TPMagic parameters for scalping (adjusted for M1 timeframe)
        self.tp1_pips = 8   # Smaller TPs for scalping
        self.tp2_pips = 12
        self.tp3_pips = 16
        self.sl_pips = 10   # Tighter SL for scalping
        self.tp_magic = None

        
    def _calculate_rsi(self, prices, window=5):
        """Calculate RSI optimized for scalping (faster response)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices, window=12, num_std=1.8):
        """Calculate Bollinger Bands optimized for gold scalping"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band, rolling_mean
    
    def _calculate_momentum(self, prices, window=3):
        """Calculate short-term momentum"""
        return prices.pct_change(window)
    
    def _is_market_active(self, timestamp):
        """Check if market is in active trading hours (more lenient for scalping)"""
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            # Avoid very low activity hours (weekends handled elsewhere)
            # Allow most of the day except for the quietest hours
            return not (4 <= hour <= 6)  # Only avoid 4-6 AM (quietest period)
        return True  # Default to active if can't determine
    
    def generate_signals(self, data, sentiment: str = None, backtest_mode: bool = False):
        """
        Enhanced scalping strategy with multiple indicators and risk management
        Vectorized version for performance optimization
        If backtest_mode, use TPMagic for multi-TP simulation.
        """
        if backtest_mode:
            # Ensure required columns for backtest
            required_cols = ['close', 'high', 'low', 'tick_volume']
            for col in required_cols:
                if col not in data.columns:
                    if col == 'tick_volume':
                        data[col] = 100  # Default volume if missing
                    else:
                        data[col] = data['close']  # Fallback

            # Compute indicators (same as normal)
            data['ema_fast'] = data['close'].ewm(span=5).mean()
            data['ema_slow'] = data['close'].ewm(span=13).mean()
            data['rsi'] = self._calculate_rsi(data['close'], window=5)
            data['ema_trend'] = data['close'].ewm(span=21).mean()
            data['trend_strength'] = (data['close'] - data['ema_trend']) / data['ema_trend']
            data['bb_upper'], data['bb_lower'], data['bb_middle'] = self._calculate_bollinger_bands(data['close'])
            data['volume_ma'] = data['tick_volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['tick_volume'] / data['volume_ma']
            data['atr'] = (data['high'] - data['low']).rolling(window=7).mean()
            data['volatility'] = data['close'].rolling(window=10).std()
            data['volatility_ma20'] = data['volatility'].rolling(20).mean()
            data['momentum_3'] = self._calculate_momentum(data['close'], 3)
            data['momentum_5'] = self._calculate_momentum(data['close'], 5)
            data['price_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            data['price_vs_ema'] = (data['close'] - data['ema_slow']) / data['ema_slow']
            data['returns'] = data['close'].pct_change().fillna(0)

            # Market active mask
            if hasattr(data.index, 'hour'):
                data['hour'] = data.index.hour
                active_market = ~((data['hour'] >= 4) & (data['hour'] <= 6))
                data.drop('hour', axis=1, inplace=True)
            else:
                active_market = pd.Series(True, index=data.index)

            # Skip conditions
            insufficient_volume = data['tick_volume'] < self.min_volume
            invalid_volume = data['volume_ratio'].isna()
            extreme_volatility = data['volatility'] > data['close'] * 0.001
            skip_mask = insufficient_volume | invalid_volume | extreme_volatility

            # EMA crossovers
            buy_crossover = (data['ema_fast'] > data['ema_slow']) & (data['ema_fast'].shift(1) <= data['ema_slow'].shift(1))
            sell_crossover = (data['ema_fast'] < data['ema_slow']) & (data['ema_fast'].shift(1) >= data['ema_slow'].shift(1))

            # BUY CONDITIONS
            buy_rsi = (data['rsi'] > 30) & (data['rsi'] < 65)
            buy_bb = data['price_position'] < 0.4
            buy_momentum = data['momentum_3'] > 0
            buy_volume = data['volume_ratio'] > 1.0
            buy_volatility = data['volatility'] < data['volatility_ma20'] * 1.3

            buy_conditions = pd.DataFrame({
                'crossover': buy_crossover,
                'rsi': buy_rsi,
                'bb': buy_bb,
                'momentum': buy_momentum,
                'volume': buy_volume,
                'volatility': buy_volatility
            }, index=data.index)

            # SELL CONDITIONS
            sell_rsi = (data['rsi'] > 35) & (data['rsi'] < 70)
            sell_bb = data['price_position'] > 0.6
            sell_momentum = data['momentum_3'] < 0
            sell_volume = data['volume_ratio'] > 1.0
            sell_volatility = data['volatility'] < data['volatility_ma20'] * 1.3

            sell_conditions = pd.DataFrame({
                'crossover': sell_crossover,
                'rsi': sell_rsi,
                'bb': sell_bb,
                'momentum': sell_momentum,
                'volume': sell_volume,
                'volatility': sell_volatility
            }, index=data.index)

            # Scores
            buy_score = buy_conditions.sum(axis=1)
            sell_score = sell_conditions.sum(axis=1)

            # Generate signals
            data['signal'] = 'hold'
            valid_mask = active_market & ~skip_mask

            data.loc[valid_mask & (buy_score >= 4), 'signal'] = 'buy'
            data.loc[valid_mask & (sell_score >= 4), 'signal'] = 'sell'

            if data['signal'].iloc[0] != 'hold':
                data.iloc[0, data.columns.get_loc('signal')] = 'hold'

            # Backtest mode: use TPMagic
            data['position'] = 0.0
            data['strategy_returns'] = 0.0
            pip_size = 0.1
            tp_magic = TPMagic(
                tp1_pips=self.tp1_pips, tp2_pips=self.tp2_pips, tp3_pips=self.tp3_pips,
                sl_pips=self.sl_pips, symbol=self.symbol, pip_size=pip_size,
                mode=TPMagicMode.BACKTEST, initial_lot=0.01,  # Fixed lot for backtest
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

                # Open if no position and signal
                if not tp_magic.is_open:
                    if signal_i == 'buy':
                        direction = 1
                        tp_magic.open_position(direction, prev_close)
                    elif signal_i == 'sell':
                        direction = -1
                        tp_magic.open_position(direction, prev_close)

                # Update
                if tp_magic.is_open:
                    update_result = tp_magic.update(high_i, low_i, close_i, asset_return)
                    data.loc[data.index[i], 'strategy_returns'] = update_result['strategy_returns']
                    data.loc[data.index[i], 'position'] = update_result['position']
                else:
                    data.loc[data.index[i], 'strategy_returns'] = 0.0
                    data.loc[data.index[i], 'position'] = 0.0

            # Drop temp columns
            temp_cols = ['trend_strength', 'price_vs_ema', 'volatility_ma20', 'momentum_5']
            data.drop(columns=[col for col in temp_cols if col in data.columns], inplace=True, errors='ignore')

            return data[['close', 'signal', 'position', 'strategy_returns', 'rsi', 'volume_ratio', 'volatility', 'ema_fast', 'ema_slow', 'bb_upper', 'bb_lower']]
        else:
            # Normal mode (same as original)
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'tick_volume']
            for col in required_cols:
                if col not in data.columns:
                    if col == 'tick_volume':
                        data[col] = 100  # Default volume if missing
                    else:
                        raise ValueError(f"Required column '{col}' missing from data")
            
            # Calculate technical indicators optimized for scalping (vectorized)
            data['ema_fast'] = data['close'].ewm(span=5).mean()  # Fast EMA (optimized)
            data['ema_slow'] = data['close'].ewm(span=13).mean()   # Slow EMA (Fibonacci number)
            data['rsi'] = self._calculate_rsi(data['close'], window=5)
            
            # Additional trend confirmation
            data['ema_trend'] = data['close'].ewm(span=21).mean()  # Longer trend filter
            data['trend_strength'] = (data['close'] - data['ema_trend']) / data['ema_trend']
            
            # Bollinger Bands for overbought/oversold conditions
            data['bb_upper'], data['bb_lower'], data['bb_middle'] = self._calculate_bollinger_bands(data['close'])
            
            # Volume indicators
            data['volume_ma'] = data['tick_volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['tick_volume'] / data['volume_ma']
            
            # Volatility measures
            data['atr'] = (data['high'] - data['low']).rolling(window=7).mean()
            data['volatility'] = data['close'].rolling(window=10).std()
            data['volatility_ma20'] = data['volatility'].rolling(20).mean()
            
            # Momentum indicators
            data['momentum_3'] = self._calculate_momentum(data['close'], 3)
            data['momentum_5'] = self._calculate_momentum(data['close'], 5)
            data['price_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Price action indicators
            data['price_vs_ema'] = (data['close'] - data['ema_slow']) / data['ema_slow']
            
            # Market active mask (vectorized - assuming datetime index)
            if hasattr(data.index, 'hour'):
                data['hour'] = data.index.hour
                active_market = ~((data['hour'] >= 4) & (data['hour'] <= 6))
                data.drop('hour', axis=1, inplace=True)
            else:
                active_market = pd.Series(True, index=data.index)
            
            # Skip conditions (vectorized)
            insufficient_volume = data['tick_volume'] < self.min_volume
            invalid_volume = data['volume_ratio'].isna()
            extreme_volatility = data['volatility'] > data['close'] * 0.001
            skip_mask = insufficient_volume | invalid_volume | extreme_volatility
            
            # EMA crossovers (vectorized)
            buy_crossover = (data['ema_fast'] > data['ema_slow']) & (data['ema_fast'].shift(1) <= data['ema_slow'].shift(1))
            sell_crossover = (data['ema_fast'] < data['ema_slow']) & (data['ema_fast'].shift(1) >= data['ema_slow'].shift(1))
            
            # BUY CONDITIONS (vectorized boolean series)
            buy_rsi = (data['rsi'] > 30) & (data['rsi'] < 65)
            buy_bb = data['price_position'] < 0.4
            buy_momentum = data['momentum_3'] > 0
            buy_volume = data['volume_ratio'] > 1.0
            buy_volatility = data['volatility'] < data['volatility_ma20'] * 1.3
            
            buy_conditions = pd.DataFrame({
                'crossover': buy_crossover,
                'rsi': buy_rsi,
                'bb': buy_bb,
                'momentum': buy_momentum,
                'volume': buy_volume,
                'volatility': buy_volatility
            }, index=data.index)
            
            # SELL CONDITIONS
            sell_rsi = (data['rsi'] > 35) & (data['rsi'] < 70)
            sell_bb = data['price_position'] > 0.6
            sell_momentum = data['momentum_3'] < 0
            sell_volume = data['volume_ratio'] > 1.0
            sell_volatility = data['volatility'] < data['volatility_ma20'] * 1.3
            
            sell_conditions = pd.DataFrame({
                'crossover': sell_crossover,
                'rsi': sell_rsi,
                'bb': sell_bb,
                'momentum': sell_momentum,
                'volume': sell_volume,
                'volatility': sell_volatility
            }, index=data.index)
            
            # Calculate scores (sum of True values)
            buy_score = buy_conditions.sum(axis=1)
            sell_score = sell_conditions.sum(axis=1)
            
            # Generate signals (vectorized)
            data['signal'] = 'hold'
            valid_mask = active_market & ~skip_mask
            
            data.loc[valid_mask & (buy_score >= 4), 'signal'] = 'buy'
            data.loc[valid_mask & (sell_score >= 4), 'signal'] = 'sell'
            
            # Insert hold for first row if needed
            if data['signal'].iloc[0] != 'hold':
                data.iloc[0, data.columns.get_loc('signal')] = 'hold'
            
            # Drop temporary columns
            temp_cols = ['trend_strength', 'price_vs_ema', 'volatility_ma20', 'momentum_5']
            data.drop(columns=[col for col in temp_cols if col in data.columns], inplace=True, errors='ignore')
            
            # Return enhanced data with all indicators
            return data[['close', 'signal', 'rsi', 'volume_ratio', 'volatility', 'ema_fast', 'ema_slow', 'bb_upper', 'bb_lower']]
    
    def execute_strategy(self, df: pd.DataFrame, sentiment: str, balance: float) -> dict:
        """
        Execute the Scalping strategy using TPMagic for consistent multi-TP management.
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
            "strategy_name": "Scalping",
            "symbol": self.symbol,
            "timeframe": "M1",
            "parameters": {
                "min_volume": self.min_volume,
                "volatility_threshold": self.volatility_threshold,
                "ema_fast_period": 5,
                "ema_slow_period": 13,
                "ema_trend_period": 21,
                "rsi_period": 5,
                "bb_window": 12,
                "bb_std_dev": 1.8,
                "momentum_window": 3,
                "atr_window": 7,
                "volume_ma_window": 10,
                "volatility_window": 10
            },
            "signal_conditions": {
                "rsi_buy_range": [30, 65],
                "rsi_sell_range": [35, 70],
                "bb_position_buy_threshold": 0.4,
                "bb_position_sell_threshold": 0.6,
                "volume_ratio_threshold": 1.0,
                "conditions_required": 4
            },
            "description": "Fast scalping strategy optimized for 1-minute timeframe"
        }
