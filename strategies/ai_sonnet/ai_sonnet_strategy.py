#!/usr/bin/env python3
"""
AI Sonnet Strategy

This strategy implements the indicator-based signal generation logic from the Gold AI Sonnet 4.5 system.
It uses a combination of EMAs, RSI, MACD, and momentum indicators to generate trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_strategy import BaseStrategy


class AISonnetStrategy(BaseStrategy):
    """
    AI Sonnet Strategy - Indicator-based trading strategy

    This strategy replicates the indicator logic from the Gold AI Sonnet system,
    using EMAs, RSI, MACD, and momentum for signal generation.
    """

    def __init__(self, symbol: str = "XAUUSD"):
        super().__init__(symbol)
        self.fast_ema = 8
        self.slow_ema = 21
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.momentum_period = 3

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using the AI Sonnet indicator logic

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signals column added
        """
        if len(df) < 30:
            df = df.copy()
            df['signal'] = 'hold'
            return df

        df = df.copy()

        # Calculate indicators
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema).mean()
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)

        # MACD calculation
        exp1 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Price momentum
        df['momentum'] = df['close'].pct_change(self.momentum_period)

        # Volume check (use tick_volume if available, otherwise assume sufficient volume)
        volume_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        if volume_col not in df.columns:
            df['volume_check'] = True  # Assume sufficient volume if no volume data
        else:
            volume_avg = df[volume_col].rolling(20).mean()
            df['volume_check'] = df[volume_col] > volume_avg * 0.8

        # Generate signals
        signals = []

        for i in range(len(df)):
            if i < max(self.slow_ema, self.rsi_period, self.macd_slow):
                signals.append('hold')
                continue

            # Current values
            ema_fast = df['ema_fast'].iloc[i]
            ema_slow = df['ema_slow'].iloc[i]
            rsi = df['rsi'].iloc[i]
            macd_hist = df['macd_histogram'].iloc[i]
            macd_hist_prev = df['macd_histogram'].iloc[i-1] if i > 0 else 0
            momentum = df['momentum'].iloc[i]
            volume_ok = df['volume_check'].iloc[i]

            # BUY conditions (simplified from original logic)
            buy_trend = ema_fast > ema_slow and ema_fast > df['ema_fast'].iloc[i-1] if i > 0 else False
            buy_rsi = 25 < rsi < 70
            buy_macd = macd_hist > macd_hist_prev and macd_hist > -0.5
            buy_momentum = momentum > 0.0001
            buy_volume = volume_ok

            # SELL conditions (simplified from original logic)
            sell_trend = ema_fast < ema_slow and ema_fast < df['ema_fast'].iloc[i-1] if i > 0 else False
            sell_rsi = 30 < rsi < 75
            sell_macd = macd_hist < macd_hist_prev and macd_hist < 0.5
            sell_momentum = momentum < -0.0001
            sell_volume = volume_ok

            # Determine signal
            if buy_trend and buy_rsi and buy_macd and buy_momentum and buy_volume:
                signals.append('buy')
            elif sell_trend and sell_rsi and sell_macd and sell_momentum and sell_volume:
                signals.append('sell')
            else:
                signals.append('hold')

        df['signal'] = signals
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_strategy_config(self) -> Dict:
        """Get strategy configuration"""
        return {
            'name': 'AI Sonnet Strategy',
            'description': 'Indicator-based strategy using EMAs, RSI, MACD, and momentum',
            'symbol': self.symbol,
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'rsi_period': self.rsi_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'momentum_period': self.momentum_period,
            'risk_per_trade': 0.02,  # 2%
            'max_daily_loss': 0.05   # 5%
        }
