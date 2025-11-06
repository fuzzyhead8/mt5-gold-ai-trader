import numpy as np
import pandas as pd

class RobustTrendPyramidingStrategy:
    """
    Robust Trend-Following & Pyramiding Strategy for Gold (XAUUSD)
    
    This revised strategy corrects the core flaws of the original by implementing a
    clear, coherent trend-following system.
    
    Core Philosophy:
    1. Identify the macro trend using a long-period EMA.
    2. Only enter in the direction of the macro trend.
    3. Wait for a low-risk pullback (identified by the Stochastic Oscillator) to enter.
    4. Use ATR (Average True Range) for dynamic, volatility-based stops and pyramid thresholds.
    5. Add to winning positions during strong trend continuations.
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        
        # --- REASON: Using a long EMA for a robust trend filter ---
        self.ema_trend_period = 100
        
        # --- REASON: Standard periods for a reliable pullback indicator ---
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        self.stoch_slowing = 3
        
        # --- REASON: Using ATR for dynamic volatility-based stops & targets ---
        self.atr_period = 14
        self.atr_stop_multiplier = 2.0  # Wider stop to avoid whipsaws
        self.atr_tp_multiplier = 1.5    # Tighter take profit for higher win rate
        self.atr_pyramid_multiplier = 1.5 # Add position after 1.5x ATR profit
        
        # Pyramiding parameters
        self.max_pyramid_levels = 3  # Re-enable pyramiding
        self.position_scale = 0.5  # Each addition is 50% of the previous one
        
    # --- NEW: Calculate Stochastic Oscillator ---
    def _calculate_stochastics(self, high, low, close, k=14, d=3, slowing=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k).min()
        highest_high = high.rolling(window=k).max()
        
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_line = k_line.rolling(window=slowing).mean() # Apply slowing
        d_line = k_line.rolling(window=d).mean()
        return k_line, d_line

    # --- NEW: Calculate Average True Range (ATR) ---
    def _calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        h_l = high - low
        h_pc = np.abs(high - close.shift(1))
        l_pc = np.abs(low - close.shift(1))
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def generate_signals(self, data):
        """Generate trend-following signals with pyramiding"""
        
        # --- Data Validation ---
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' missing")
        
        # --- Calculate Indicators ---
        data['ema_trend'] = data['close'].ewm(span=self.ema_trend_period, adjust=False).mean()
        data['stoch_k'], data['stoch_d'] = self._calculate_stochastics(
            data['high'], data['low'], data['close'], 
            self.stoch_k_period, self.stoch_d_period, self.stoch_slowing)
        data['atr'] = self._calculate_atr(data['high'], data['low'], data['close'], self.atr_period)

        # Initialize tracking columns
        data['signal'] = 'hold'
        data['pyramid_level'] = 0
        data['position_size'] = 0.0
        data['stop_loss'] = 0.0
        
        # State variables
        current_position = None
        position_entries = []
        current_stop = None
        take_profit = None # NEW: Add take_profit
        
        for i in range(1, len(data)):
            
            # --- Unpack current and previous data points for clarity ---
            prev = data.iloc[i-1]
            curr = data.iloc[i]
            
            # --- POSITION MANAGEMENT ---
            if current_position == 'buy':
                # Check stop loss or take profit
                if curr['close'] <= current_stop or curr['close'] >= take_profit:
                    data.loc[data.index[i], 'signal'] = 'exit_buy'
                    current_position, position_entries, current_stop, take_profit = None, [], None, None
                    continue
                
                # Pyramid check: Add ONLY to winners after a significant move
                profit_since_last_add = curr['close'] - position_entries[-1]
                atr_threshold = prev['atr'] * self.atr_pyramid_multiplier
                
                if (len(position_entries) < self.max_pyramid_levels and 
                    profit_since_last_add >= atr_threshold):
                    
                    # Move stop to the previous entry price
                    current_stop = position_entries[-1]
                    
                    position_entries.append(curr['close'])
                    data.loc[data.index[i], 'signal'] = 'add_buy'
                    data.loc[data.index[i], 'pyramid_level'] = len(position_entries) - 1
                    data.loc[data.index[i], 'position_size'] = self.position_scale ** (len(position_entries) - 1)
                    data.loc[data.index[i], 'stop_loss'] = current_stop
                    continue
            
            elif current_position == 'sell':
                # Check stop loss or take profit
                if curr['close'] >= current_stop or curr['close'] <= take_profit:
                    data.loc[data.index[i], 'signal'] = 'exit_sell'
                    current_position, position_entries, current_stop, take_profit = None, [], None, None
                    continue

                # Pyramid check
                profit_since_last_add = position_entries[-1] - curr['close']
                atr_threshold = prev['atr'] * self.atr_pyramid_multiplier

                if (len(position_entries) < self.max_pyramid_levels and 
                    profit_since_last_add >= atr_threshold):
                    
                    current_stop = position_entries[-1] # Move stop
                    position_entries.append(curr['close'])
                    data.loc[data.index[i], 'signal'] = 'add_sell'
                    data.loc[data.index[i], 'pyramid_level'] = len(position_entries) - 1
                    data.loc[data.index[i], 'position_size'] = self.position_scale ** (len(position_entries) - 1)
                    data.loc[data.index[i], 'stop_loss'] = current_stop
                    continue

            # If in a position, just hold and update SL/Level info
            if current_position:
                data.loc[data.index[i], 'stop_loss'] = current_stop
                data.loc[data.index[i], 'pyramid_level'] = len(position_entries) - 1
                data.loc[data.index[i], 'position_size'] = self.position_scale ** (len(position_entries) - 1)
                continue

            # --- ENTRY SIGNALS (only if not in a position) ---
            
            # BUY ENTRY: Trend is UP, Stochastics confirm pullback is ending
            is_trending_up = curr['close'] > curr['ema_trend']
            is_pullback_ending_buy = prev['stoch_k'] < 30 and prev['stoch_d'] < 30 and \
                                     curr['stoch_k'] > curr['stoch_d'] and prev['stoch_k'] < prev['stoch_d']

            if is_trending_up and is_pullback_ending_buy:
                current_position = 'buy'
                position_entries = [curr['close']]
                atr_val = curr['atr']
                current_stop = curr['close'] - (atr_val * self.atr_stop_multiplier)
                take_profit = curr['close'] + (atr_val * self.atr_tp_multiplier)
                
                data.loc[data.index[i], 'signal'] = 'buy'
                data.loc[data.index[i], 'pyramid_level'] = 0
                data.loc[data.index[i], 'position_size'] = 1.0
                data.loc[data.index[i], 'stop_loss'] = current_stop
                continue

            # SELL ENTRY: Trend is DOWN, Stochastics confirm rally is ending
            is_trending_down = curr['close'] < curr['ema_trend']
            is_rally_ending_sell = prev['stoch_k'] > 70 and prev['stoch_d'] > 70 and \
                                     curr['stoch_k'] < curr['stoch_d'] and prev['stoch_k'] > prev['stoch_d']

            if is_trending_down and is_rally_ending_sell:
                current_position = 'sell'
                position_entries = [curr['close']]
                atr_val = curr['atr']
                current_stop = curr['close'] + (atr_val * self.atr_stop_multiplier)
                take_profit = curr['close'] - (atr_val * self.atr_tp_multiplier)
                
                data.loc[data.index[i], 'signal'] = 'sell'
                data.loc[data.index[i], 'pyramid_level'] = 0
                data.loc[data.index[i], 'position_size'] = 1.0
                data.loc[data.index[i], 'stop_loss'] = current_stop
                continue
        
        return data[['close', 'signal', 'pyramid_level', 'position_size', 'stop_loss', 
                     'ema_trend', 'stoch_k', 'atr']]
