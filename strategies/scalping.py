import numpy as np

class ScalpingStrategy:
    def __init__(self, symbol):
        self.symbol = symbol

    def generate_signals(self, data):
        """
        Example: Buy when short MA crosses above long MA and price volatility is low
        """
        data['ma_fast'] = data['close'].rolling(window=5).mean()
        data['ma_slow'] = data['close'].rolling(window=20).mean()
        data['volatility'] = data['close'].rolling(window=10).std()

        signal = []
        for i in range(1, len(data)):
            if (
                data['ma_fast'].iloc[i] > data['ma_slow'].iloc[i] and
                data['ma_fast'].iloc[i-1] <= data['ma_slow'].iloc[i-1] and
                data['volatility'].iloc[i] < data['close'].std()/2
            ):
                signal.append('buy')
            elif (
                data['ma_fast'].iloc[i] < data['ma_slow'].iloc[i] and
                data['ma_fast'].iloc[i-1] >= data['ma_slow'].iloc[i-1]
            ):
                signal.append('sell')
            else:
                signal.append('hold')

        signal.insert(0, 'hold')  # no signal for first row
        data['signal'] = signal
        return data[['close', 'signal']]
