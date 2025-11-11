"""
AI-Powered Backtesting for Gold AI Sonnet 4.5

This script performs backtesting of the AI trading strategy using historical data
to analyze profitability and performance metrics.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from main import NebulaAssistant, TradeSignal, ShieldProtocol
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Represents a trade in the backtest"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    profit: float
    commission: float
    ai_confidence: float
    ai_reasoning: str

@dataclass
class BacktestResult:
    """Results of the AI backtest"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    equity_curve: List[float]
    trades: List[BacktestTrade]

class AIBacktester:
    """AI-powered backtesting engine"""

    def __init__(self, symbol: str = 'XAUUSD', timeframe: str = 'H1'):
        self.symbol = symbol
        self.timeframe = timeframe

        # Initialize components
        self.nebula_assistant = NebulaAssistant()
        self.shield_protocol = ShieldProtocol()
        self.config = get_config()

        # Backtest parameters
        self.initial_balance = 10000.0
        self.leverage = 100
        self.commission_per_lot = 2.0  # $2 per lot round trip
        self.spread_pips = 2.0
        self.min_confidence = 0.6  # Minimum AI confidence to trade

        # Trading state
        self.current_balance = self.initial_balance
        self.open_position = None
        self.equity_curve = [self.initial_balance]

        logger.info(f"AI Backtester initialized for {symbol} {timeframe}")

    async def load_historical_data(self, csv_path: str) -> pd.DataFrame:
        """Load historical data from CSV file"""
        try:
            # Read CSV with proper column names
            df = pd.read_csv(csv_path)

            # Clean column names (remove leading comma/space issues)
            df.columns = df.columns.str.strip().str.lstrip(',')

            # Ensure we have the required columns
            required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            # Sort by time
            df.sort_index(inplace=True)

            # Add volume column if missing
            if 'volume' not in df.columns:
                df['volume'] = df['tick_volume']

            logger.info(f"Loaded {len(df)} bars of historical data")
            return df

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise

    async def run_backtest(self, historical_data: pd.DataFrame,
                          start_date: str = None, end_date: str = None) -> BacktestResult:
        """Run the AI backtest"""
        logger.info("Starting AI backtest...")

        # Filter data by date range if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            historical_data = historical_data[historical_data.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            historical_data = historical_data[historical_data.index <= end_dt]

        if len(historical_data) < 100:
            raise ValueError("Insufficient historical data for backtesting")

        trades = []
        consecutive_bars = 0

        # Process each bar
        for i in range(100, len(historical_data)):  # Start from bar 100 to have enough history
            current_bar = historical_data.iloc[i]
            market_data = historical_data.iloc[:i+1]

            # Check if we need to close any open positions
            closed_trade = await self._check_position_close(current_bar, current_bar.name)
            if closed_trade:
                trades.append(closed_trade)

            # Only generate new signals if no position is open
            if not self.open_position:
                # Step 1: Calculate indicator-based signals
                indicator_signal = self._calculate_indicator_signals(market_data)

                # Step 2: Only call AI for confirmation if indicators suggest a trade
                if indicator_signal in ['buy', 'sell']:
                    ai_confirmation = self._mock_ai_confirmation(indicator_signal, market_data, current_bar)

                    # Step 3: Generate signal only if AI confirms
                    if ai_confirmation['confirmed']:
                        signal = await self._generate_confirmed_signal(
                            indicator_signal, ai_confirmation, market_data, current_bar
                        )

                        if signal:
                            # Execute the confirmed signal
                            await self._execute_signal(signal, current_bar.name)

            consecutive_bars += 1
            if consecutive_bars % 100 == 0:
                logger.info(f"Processed {consecutive_bars} bars, {len(trades)} trades executed")

        # Calculate final metrics
        result = self._calculate_metrics(trades)

        logger.info(f"Backtest completed. Net profit: ${result.net_profit:.2f}, "
                   f"Win rate: {result.win_rate:.1%}, Total trades: {result.total_trades}")

        return result

    async def _generate_confirmed_signal(self, indicator_signal: str, ai_confirmation: Dict,
                                       market_data: pd.DataFrame, current_bar: pd.Series) -> Optional[TradeSignal]:
        """Generate trading signal based on AI-confirmed indicator signals"""
        try:
            confidence = ai_confirmation.get('confidence', 0.0)

            if confidence < self.min_confidence:
                return None

            # Get current price
            current_price = current_bar['close']

            # Calculate position size
            stop_loss_pips = self._calculate_atr_stop_loss(market_data)
            account_balance = self.current_balance

            volume = self.shield_protocol.calculate_position_size(
                account_balance, stop_loss_pips, self.symbol
            )

            # Calculate entry, stop loss, and take profit
            if indicator_signal == 'BUY':
                entry_price = current_price
                stop_loss = entry_price - stop_loss_pips
                take_profit = entry_price + (stop_loss_pips * 3)  # 3:1 reward ratio
            else:  # SELL
                entry_price = current_price
                stop_loss = entry_price + stop_loss_pips
                take_profit = entry_price - (stop_loss_pips * 3)

            # Create signal
            signal = TradeSignal(
                direction=indicator_signal.upper(),
                symbol=self.symbol,
                volume=volume,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=3.0,
                confidence=confidence,
                timestamp=current_bar.name,
                reasoning=f"Indicators: {indicator_signal.upper()} | AI Confirmation: {ai_confirmation.get('reasoning', 'N/A')}"
            )

            return signal

        except Exception as e:
            logger.error(f"Confirmed signal generation error: {e}")
            return None

    async def _generate_signal(self, market_data: pd.DataFrame,
                             analysis: Dict, current_bar: pd.Series) -> Optional[TradeSignal]:
        """Generate trading signal based on AI analysis (legacy method)"""
        try:
            # Check if AI recommends a trade
            recommendation = analysis.get('recommendation', 'WAIT')
            confidence = analysis.get('confidence', 0.0)

            if recommendation == 'WAIT' or confidence < self.min_confidence:
                return None

            # Get current price
            current_price = current_bar['close']

            # Calculate position size
            stop_loss_pips = self._calculate_atr_stop_loss(market_data)
            account_balance = self.current_balance

            volume = self.shield_protocol.calculate_position_size(
                account_balance, stop_loss_pips, self.symbol
            )

            # Calculate entry, stop loss, and take profit
            if recommendation == 'BUY':
                entry_price = current_price
                stop_loss = entry_price - stop_loss_pips
                take_profit = entry_price + (stop_loss_pips * 3)  # 3:1 reward ratio
            else:  # SELL
                entry_price = current_price
                stop_loss = entry_price + stop_loss_pips
                take_profit = entry_price - (stop_loss_pips * 3)

            # Create signal
            signal = TradeSignal(
                direction=recommendation,
                symbol=self.symbol,
                volume=volume,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=3.0,
                confidence=confidence,
                timestamp=current_bar.name,
                reasoning=f"AI Analysis: {analysis.get('reasoning', 'N/A')}"
            )

            return signal

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None

    def _calculate_atr_stop_loss(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR-based stop loss"""
        if len(market_data) < period + 1:
            return 50.0  # Default 50 pips

        # Calculate True Range
        high = market_data['high']
        low = market_data['low']
        close = market_data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        # Use 1.5 ATR as stop loss
        stop_loss_pips = atr * 1.5

        # Ensure minimum stop loss
        return max(stop_loss_pips, 10.0)  # Minimum 10 pips

    def _calculate_indicator_signals(self, market_data: pd.DataFrame) -> str:
        """Calculate indicator-based signals (simplified golden scalping logic)"""
        try:
            if len(market_data) < 30:
                return 'hold'

            # Calculate indicators
            ema_fast = market_data['close'].ewm(span=8).mean().iloc[-1]
            ema_slow = market_data['close'].ewm(span=21).mean().iloc[-1]
            rsi = self.nebula_assistant._calculate_rsi(market_data, 14)

            # MACD calculation
            exp1 = market_data['close'].ewm(span=12, adjust=False).mean()
            exp2 = market_data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal_line

            macd_hist = histogram.iloc[-1]
            macd_hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0

            # Price momentum
            momentum = market_data['close'].pct_change(3).iloc[-1]

            # Volume check (simplified)
            volume_avg = market_data['volume'].rolling(20).mean().iloc[-1] if 'volume' in market_data.columns else 100
            current_volume = market_data['volume'].iloc[-1] if 'volume' in market_data.columns else 100

            # BUY conditions (simplified)
            buy_trend = ema_fast > ema_slow and ema_fast > market_data['close'].ewm(span=8).mean().iloc[-2]
            buy_rsi = 25 < rsi < 70
            buy_macd = macd_hist > macd_hist_prev and macd_hist > -0.5
            buy_momentum = momentum > 0.0001 and current_volume > volume_avg * 0.8

            if buy_trend and buy_rsi and buy_macd and buy_momentum:
                return 'buy'

            # SELL conditions (simplified)
            sell_trend = ema_fast < ema_slow and ema_fast < market_data['close'].ewm(span=8).mean().iloc[-2]
            sell_rsi = 30 < rsi < 75
            sell_macd = macd_hist < macd_hist_prev and macd_hist < 0.5
            sell_momentum = momentum < -0.0001 and current_volume > volume_avg * 0.8

            if sell_trend and sell_rsi and sell_macd and sell_momentum:
                return 'sell'

            return 'hold'

        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return 'hold'

    def _mock_ai_confirmation(self, indicator_signal: str, market_data: pd.DataFrame, current_bar: pd.Series) -> Dict:
        """Mock AI confirmation for indicator signals (only called when indicators suggest a trade)"""
        try:
            if indicator_signal == 'hold':
                return {
                    'confirmed': False,
                    'confidence': 0.0,
                    'reasoning': 'No indicator signal to confirm'
                }

            # Calculate additional context for AI analysis
            rsi = self.nebula_assistant._calculate_rsi(market_data, 14)
            current_price = current_bar['close']

            # Trend strength
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            trend_strength = abs(current_price - sma_20) / sma_20

            # Volatility assessment
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.1

            # AI confirmation logic
            confirmed = False
            confidence = 0.0
            reasoning = ""

            if indicator_signal == 'buy':
                # Confirm buy signal
                if rsi < 65 and trend_strength > 0.005 and volatility < 0.25:
                    confirmed = True
                    confidence = 0.8
                    reasoning = f"AI confirms BUY: RSI {rsi:.1f} not overbought, strong trend ({trend_strength:.1%}), moderate volatility"
                elif rsi < 75 and trend_strength > 0.002:
                    confirmed = True
                    confidence = 0.6
                    reasoning = f"AI confirms BUY: Acceptable RSI {rsi:.1f}, moderate trend strength"
                else:
                    reasoning = f"AI rejects BUY: RSI {rsi:.1f} too high or weak trend ({trend_strength:.1%})"

            elif indicator_signal == 'sell':
                # Confirm sell signal
                if rsi > 35 and trend_strength > 0.005 and volatility < 0.25:
                    confirmed = True
                    confidence = 0.8
                    reasoning = f"AI confirms SELL: RSI {rsi:.1f} not oversold, strong trend ({trend_strength:.1f}), moderate volatility"
                elif rsi > 25 and trend_strength > 0.002:
                    confirmed = True
                    confidence = 0.6
                    reasoning = f"AI confirms SELL: Acceptable RSI {rsi:.1f}, moderate trend strength"
                else:
                    reasoning = f"AI rejects SELL: RSI {rsi:.1f} too low or weak trend ({trend_strength:.1f})"

            # Add some realistic rejection rate
            import random
            random.seed(int(current_bar.name.timestamp()))

            if confirmed and random.random() < 0.15:  # 15% chance of AI rejecting good signals
                confirmed = False
                confidence = 0.3
                reasoning += " - AI detects additional uncertainty"

            return {
                'confirmed': confirmed,
                'confidence': confidence,
                'reasoning': reasoning,
                'ai_called': True
            }

        except Exception as e:
            logger.error(f"AI confirmation error: {e}")
            return {
                'confirmed': False,
                'confidence': 0.0,
                'reasoning': 'AI confirmation failed',
                'ai_called': True
            }

    async def _execute_signal(self, signal: TradeSignal, timestamp: datetime) -> Optional[BacktestTrade]:
        """Execute a trading signal in the backtest"""
        try:
            # Check if we already have an open position
            if self.open_position:
                return None

            # Calculate commission
            commission = (signal.volume * self.commission_per_lot) / 2  # Half commission for entry

            # Open position
            self.open_position = {
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'volume': signal.volume,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'entry_time': timestamp,
                'entry_commission': commission,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            }

            # Update balance for commission
            self.current_balance -= commission
            self.equity_curve.append(self.current_balance)

            logger.debug(f"Opened {signal.direction} position at {signal.entry_price}")

            return None  # Position opened, no trade closed yet

        except Exception as e:
            logger.error(f"Signal execution error: {e}")
            return None

    async def _check_position_close(self, current_bar: pd.Series, timestamp: datetime) -> Optional[BacktestTrade]:
        """Check if open position should be closed (with AI confirmation for take profit)"""
        if not self.open_position:
            return None

        try:
            direction = self.open_position['direction']
            entry_price = self.open_position['entry_price']
            stop_loss = self.open_position['stop_loss']
            take_profit = self.open_position['take_profit']
            volume = self.open_position['volume']
            entry_time = self.open_position['entry_time']
            entry_commission = self.open_position['entry_commission']
            confidence = self.open_position['confidence']
            reasoning = self.open_position['reasoning']

            current_price = current_bar['close']
            close_reason = None
            exit_price = current_price

            # Always check stop loss (no AI needed for emergency close)
            if direction == 'BUY':
                if current_price <= stop_loss:
                    close_reason = 'stop_loss'
                    exit_price = stop_loss
            else:  # SELL
                if current_price >= stop_loss:
                    close_reason = 'stop_loss'
                    exit_price = stop_loss

            # For take profit, check if AI confirms closing
            if not close_reason:
                if direction == 'BUY' and current_price >= take_profit:
                    # Check if AI confirms taking profit
                    ai_confirm_close = self._mock_ai_confirm_close(direction, current_bar)
                    if ai_confirm_close['confirmed']:
                        close_reason = 'take_profit'
                        exit_price = take_profit
                elif direction == 'SELL' and current_price <= take_profit:
                    # Check if AI confirms taking profit
                    ai_confirm_close = self._mock_ai_confirm_close(direction, current_bar)
                    if ai_confirm_close['confirmed']:
                        close_reason = 'take_profit'
                        exit_price = take_profit

            if not close_reason:
                return None

            # Calculate P&L
            if direction == 'BUY':
                pip_movement = exit_price - entry_price
            else:
                pip_movement = entry_price - exit_price

            # Convert pips to profit (simplified)
            # For gold (XAUUSD), 1 pip = 0.01 USD per standard lot (100,000 units)
            # Since we're using lots, pip_value = volume * 1.0 (for 1 pip = $1 per lot)
            pip_value_per_lot = 1.0  # $1 per pip per lot for gold
            gross_profit = pip_movement * volume * pip_value_per_lot

            # Add exit commission
            exit_commission = (volume * self.commission_per_lot) / 2
            total_commission = entry_commission + exit_commission
            net_profit = gross_profit - total_commission

            # Create trade record
            trade = BacktestTrade(
                entry_time=entry_time,
                exit_time=timestamp,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                volume=volume,
                profit=net_profit,
                commission=total_commission,
                ai_confidence=confidence,
                ai_reasoning=reasoning
            )

            # Update balance
            self.current_balance += net_profit
            self.equity_curve.append(self.current_balance)

            # Close position
            self.open_position = None

            logger.debug(f"Closed {direction} position: {close_reason}, P&L: ${net_profit:.2f}")

            return trade

        except Exception as e:
            logger.error(f"Position close check error: {e}")
            return None

    def _mock_ai_confirm_close(self, direction: str, current_bar: pd.Series) -> Dict:
        """Mock AI confirmation for closing positions at take profit"""
        try:
            # Simple logic: AI confirms closing if profit target is reached
            # In reality, AI might analyze if market conditions suggest locking in profits

            # Basic confirmation logic - AI confirms most take profit scenarios
            import random
            random.seed(int(current_bar.name.timestamp()))

            # AI confirms 85% of take profit opportunities
            confirmed = random.random() < 0.85

            if confirmed:
                confidence = 0.75
                reasoning = f"AI confirms closing {direction} position - profit target reached, market conditions favorable"
            else:
                confidence = 0.4
                reasoning = f"AI suggests holding {direction} position - potential for further gains"

            return {
                'confirmed': confirmed,
                'confidence': confidence,
                'reasoning': reasoning,
                'ai_called': True
            }

        except Exception as e:
            logger.error(f"AI close confirmation error: {e}")
            return {
                'confirmed': False,
                'confidence': 0.0,
                'reasoning': 'AI confirmation failed',
                'ai_called': True
            }

    def _calculate_metrics(self, trades: List[BacktestTrade]) -> BacktestResult:
        """Calculate backtest performance metrics"""
        if not trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_profit=0.0,
                total_loss=0.0,
                net_profit=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                equity_curve=self.equity_curve,
                trades=trades
            )

        # Basic trade counts
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.profit > 0])
        losing_trades = len([t for t in trades if t.profit <= 0])

        # Profit metrics
        profits = [t.profit for t in trades]
        winning_profits = [t.profit for t in trades if t.profit > 0]
        losing_profits = [t.profit for t in trades if t.profit <= 0]

        total_profit = sum(winning_profits) if winning_profits else 0
        total_loss = abs(sum(losing_profits)) if losing_profits else 0
        net_profit = sum(profits)

        # Ratios and averages
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        largest_win = max(profits) if profits else 0
        largest_loss = min(profits) if profits else 0

        # Drawdown calculation
        equity_curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        max_drawdown = np.max(drawdown)

        # Sharpe ratio (simplified)
        if len(profits) > 1:
            returns = np.array(profits) / self.initial_balance
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            equity_curve=self.equity_curve,
            trades=trades
        )

    def plot_results(self, result: BacktestResult, save_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curve
        axes[0, 0].plot(result.equity_curve, linewidth=2)
        axes[0, 0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Trades')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=self.initial_balance, color='red', linestyle='--', alpha=0.7)

        # Trade P&L distribution
        if result.trades:
            profits = [trade.profit for trade in result.trades]
            axes[0, 1].hist(profits, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Profit/Loss ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)

        # Win/Loss pie chart
        if result.total_trades > 0:
            labels = ['Wins', 'Losses']
            sizes = [result.winning_trades, result.losing_trades]
            colors = ['green', 'red']
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Win/Loss Ratio', fontsize=14, fontweight='bold')

        # Performance metrics text
        metrics_text = ".2f"".2f"".2f"".2f"".2f"".2f"f"""
        GOLD AI SONNET 4.5 - AI BACKTEST REPORT
        =====================================

        TRADING PERFORMANCE
        ------------------
        Total Trades: {result.total_trades}
        Winning Trades: {result.winning_trades}
        Losing Trades: {result.losing_trades}
        Win Rate: {result.win_rate:.1%}

        FINANCIAL METRICS
        ----------------
        Net Profit: ${result.net_profit:.2f}
        Total Profit: ${result.total_profit:.2f}
        Total Loss: ${result.total_loss:.2f}
        Max Drawdown: ${result.max_drawdown:.2f}

        RISK METRICS
        -----------
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Profit Factor: {result.profit_factor:.2f}
        Avg Win: ${result.avg_win:.2f}
        Avg Loss: ${result.avg_loss:.2f}
        """
        axes[1, 1].text(0.05, 0.95, metrics_text, fontsize=10, verticalalignment='top',
                       fontfamily='monospace', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Backtest plot saved to {save_path}")
        else:
            plt.show()

    def generate_report(self, result: BacktestResult) -> str:
        """Generate detailed backtest report"""
        report = ".2f"".2f"".2f"".2f"".2f"".2f"".2f"".2f"".2f"".2f"".2f"".2f"f"""
        GOLD AI SONNET 4.5 - AI BACKTEST REPORT
        =======================================

        BACKTEST CONFIGURATION
        ---------------------
        Symbol: {self.symbol}
        Timeframe: {self.timeframe}
        Initial Balance: ${self.initial_balance:.2f}
        Leverage: {self.leverage}x
        Min AI Confidence: {self.min_confidence}
        Commission per Lot: ${self.commission_per_lot:.2f}

        TRADING PERFORMANCE
        ------------------
        Total Trades: {result.total_trades}
        Winning Trades: {result.winning_trades}
        Losing Trades: {result.losing_trades}
        Win Rate: {result.win_rate:.1%}

        FINANCIAL METRICS
        ----------------
        Net Profit: ${result.net_profit:.2f}
        Total Profit: ${result.total_profit:.2f}
        Total Loss: ${result.total_loss:.2f}
        Profit/Loss Ratio: {result.total_profit/result.total_loss:.2f} if result.total_loss > 0 else 'N/A'

        RISK METRICS
        -----------
        Max Drawdown: ${result.max_drawdown:.2f}
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Profit Factor: {result.profit_factor:.2f}
        Recovery Factor: {result.net_profit/result.max_drawdown:.2f} if result.max_drawdown > 0 else 'N/A'

        TRADE ANALYSIS
        -------------
        Average Win: ${result.avg_win:.2f}
        Average Loss: ${result.avg_loss:.2f}
        Largest Win: ${result.largest_win:.2f}
        Largest Loss: ${result.largest_loss:.2f}
        Avg Risk-Reward: {result.avg_win/abs(result.avg_loss):.2f} if result.avg_loss != 0 else 'N/A'

        CONCLUSION
        ---------
        """

        # Add conclusion based on results
        if result.net_profit > 0 and result.win_rate > 0.5:
            conclusion = "POSITIVE: The AI strategy shows profitable results with good win rate."
        elif result.net_profit > 0:
            conclusion = "MODERATE: The strategy is profitable but has room for improvement in win rate."
        else:
            conclusion = "NEGATIVE: The strategy is currently unprofitable and needs optimization."

        report += conclusion

        return report

async def main():
    """Main function to run AI backtest"""
    # Initialize backtester
    backtester = AIBacktester(symbol='XAUUSD', timeframe='H1')

    # Load historical data
    data_path = '../backtests/XAUUSD_H1_20251105_211938.csv'
    historical_data = await backtester.load_historical_data(data_path)

    # Run backtest
    result = await backtester.run_backtest(
        historical_data,
        start_date='2025-09-15',  # Start from a reasonable date
        end_date='2025-11-05'    # End date
    )

    # Generate and print report
    report = backtester.generate_report(result)
    print("\n" + "="*60)
    print(report)
    print("="*60)

    # Plot results
    backtester.plot_results(result, save_path='ai_backtest_results.png')

    # Save detailed trade log
    if result.trades:
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'volume': t.volume,
            'profit': t.profit,
            'commission': t.commission,
            'ai_confidence': t.ai_confidence,
            'ai_reasoning': t.ai_reasoning[:100] + '...' if len(t.ai_reasoning) > 100 else t.ai_reasoning
        } for t in result.trades])

        trades_df.to_csv('ai_backtest_trades.csv', index=False)
        logger.info("Detailed trade log saved to ai_backtest_trades.csv")

if __name__ == "__main__":
    asyncio.run(main())
