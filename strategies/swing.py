import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from strategies.base_strategy import BaseStrategy

class SwingTradingStrategy(BaseStrategy):
    def __init__(self, symbol):
        super().__init__(symbol)

    def generate_signals(self, data):
        """
        Example: MACD cross strategy
        Vectorized version for performance optimization
        """
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
        Execute the Swing Trading strategy: generate signals and place trades if conditions met.
        
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
        
        # Define SL and TP distances (100 pips SL, 200 pips TP for 1:2 RR, suitable for swing)
        # For XAUUSD, 1 pip = 0.1, point = 0.01, so 100 pips = 10.0 price units
        sl_distance = 10.0  # 100 pips
        tp_distance = 20.0  # 200 pips
        
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
            stop_loss, take_profit, lot_size, "SwingTrading"
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
