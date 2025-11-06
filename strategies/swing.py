import numpy as np

class SwingTradingStrategy:
    def __init__(self, symbol):
        self.symbol = symbol

    def generate_signals(self, data):
        """
        Example: MACD cross strategy
        """
        data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        signals = []
        for i in range(1, len(data)):
            if data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1]:
                signals.append('buy')
            elif data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1]:
                signals.append('sell')
            else:
                signals.append('hold')

        signals.insert(0, 'hold')
        data['signal'] = signals
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
