"""
Backtesting module for Gold AI Sonnet 4.5
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from config import BACKTEST_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Results of a backtest run"""
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
    max_consecutive_wins: int
    max_consecutive_losses: int
    equity_curve: List[float]
    trade_history: List[Dict]

class Backtester:
    """Backtesting engine for trading strategies"""

    def __init__(self, config: Dict = None):
        self.config = config or BACKTEST_CONFIG
        self.initial_balance = self.config.get('initial_balance', 10000)
        self.leverage = self.config.get('leverage', 100)
        self.commission = self.config.get('commission', 0.0002)
        self.spread = self.config.get('spread', 2.0)

    async def run_backtest(self, strategy_class, symbol: str, start_date: str,
                          end_date: str, timeframe: str = 'H1') -> BacktestResult:
        """Run backtest for a given strategy"""

        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")

        # Load historical data
        historical_data = await self._load_historical_data(symbol, start_date, end_date, timeframe)

        if historical_data is None or len(historical_data) < 100:
            raise ValueError("Insufficient historical data for backtesting")

        # Initialize strategy
        strategy = strategy_class(symbol=symbol, timeframe=timeframe)
        await strategy.initialize()

        # Run simulation
        results = await self._simulate_trading(strategy, historical_data)

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        logger.info(f"Backtest completed. Net profit: {metrics['net_profit']:.2f}")

        return BacktestResult(**metrics)

    async def _load_historical_data(self, symbol: str, start_date: str,
                                   end_date: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load historical data for backtesting"""
        try:
            # In a real implementation, this would load from a data source
            # For now, we'll create synthetic data
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')

            np.random.seed(42)  # For reproducible results

            # Generate synthetic OHLC data
            n_bars = len(date_range)
            base_price = 2000 if 'XAU' in symbol else 1.0

            # Random walk with trend
            price_changes = np.random.normal(0, 0.005, n_bars)
            prices = base_price * (1 + np.cumsum(price_changes))

            # Create OHLC data
            high_mult = 1 + np.random.uniform(0, 0.01, n_bars)
            low_mult = 1 - np.random.uniform(0, 0.01, n_bars)
            open_prices = prices * (1 + np.random.normal(0, 0.002, n_bars))
            close_prices = prices

            data = pd.DataFrame({
                'timestamp': date_range,
                'open': open_prices,
                'high': prices * high_mult,
                'low': prices * low_mult,
                'close': close_prices,
                'volume': np.random.randint(1000, 10000, n_bars)
            })

            data.set_index('timestamp', inplace=True)

            return data

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return None

    async def _simulate_trading(self, strategy, historical_data: pd.DataFrame) -> List[Dict]:
        """Simulate trading with historical data"""
        trades = []
        current_balance = self.initial_balance
        open_positions = []

        # Process each bar
        for i, (timestamp, bar) in enumerate(historical_data.iterrows()):
            current_data = historical_data.iloc[:i+1]

            # Update strategy with current data
            await strategy._update_market_data(current_data)

            # Generate signal
            signal = await strategy._generate_signal(current_data, {})

            if signal:
                # Execute trade
                trade_result = self._execute_trade(signal, bar, current_balance)

                if trade_result:
                    trades.append(trade_result)
                    current_balance += trade_result['profit']

                    # Update open positions
                    if signal.direction == 'BUY':
                        open_positions.append({
                            'entry_price': signal.entry_price,
                            'volume': signal.volume,
                            'direction': 'BUY'
                        })
                    else:
                        # Close opposite positions (simplified)
                        pass

            # Update open positions
            for position in open_positions:
                if position['direction'] == 'BUY':
                    # Calculate unrealized P&L
                    pass

        return trades

    def _execute_trade(self, signal, current_bar: pd.Series, current_balance: float) -> Optional[Dict]:
        """Execute a trade in backtest"""
        try:
            entry_price = signal.entry_price
            exit_price = current_bar['close']  # Simplified exit at bar close

            # Calculate pip movement
            if signal.direction == 'BUY':
                pip_movement = exit_price - entry_price
            else:
                pip_movement = entry_price - exit_price

            # Calculate profit/loss
            pip_value = current_balance * 0.0001  # Simplified pip value
            profit = pip_movement * signal.volume * (pip_value / 0.0001)

            # Apply commission
            commission_cost = abs(profit) * self.commission
            profit -= commission_cost

            return {
                'timestamp': current_bar.name,
                'direction': signal.direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'volume': signal.volume,
                'profit': profit,
                'commission': commission_cost,
                'symbol': signal.symbol
            }

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None

    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate backtest performance metrics"""
        if not trades:
            return self._empty_metrics()

        df_trades = pd.DataFrame(trades)

        # Basic counts
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['profit'] > 0])
        losing_trades = len(df_trades[df_trades['profit'] <= 0])

        # Profit metrics
        total_profit = df_trades[df_trades['profit'] > 0]['profit'].sum()
        total_loss = abs(df_trades[df_trades['profit'] <= 0]['profit'].sum())
        net_profit = df_trades['profit'].sum()

        # Ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Averages
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0

        # Extremes
        largest_win = df_trades['profit'].max() if not df_trades.empty else 0
        largest_loss = df_trades['profit'].min() if not df_trades.empty else 0

        # Drawdown
        cumulative = df_trades['profit'].cumsum() + self.initial_balance
        peak = cumulative.expanding().max()
        drawdown = peak - cumulative
        max_drawdown = drawdown.max()

        # Sharpe ratio (simplified)
        returns = df_trades['profit'] / self.initial_balance
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Consecutive wins/losses
        signs = np.sign(df_trades['profit'])
        consecutive_wins = self._max_consecutive_count(signs, 1)
        consecutive_losses = self._max_consecutive_count(signs, -1)

        # Equity curve
        equity_curve = [self.initial_balance] + cumulative.tolist()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'equity_curve': equity_curve,
            'trade_history': trades
        }

    def _empty_metrics(self) -> Dict:
        """Return empty metrics for no trades"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'total_loss': 0,
            'net_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'equity_curve': [self.initial_balance],
            'trade_history': []
        }

    def _max_consecutive_count(self, signs: np.ndarray, target: int) -> int:
        """Calculate maximum consecutive count of target value"""
        max_count = 0
        current_count = 0

        for sign in signs:
            if sign == target:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def plot_results(self, result: BacktestResult, save_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curve
        axes[0, 0].plot(result.equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Trades')
        axes[0, 0].set_ylabel('Balance')
        axes[0, 0].grid(True)

        # Trade P&L distribution
        if result.trade_history:
            profits = [trade['profit'] for trade in result.trade_history]
            axes[0, 1].hist(profits, bins=50, alpha=0.7)
            axes[0, 1].set_title('Trade P&L Distribution')
            axes[0, 1].set_xlabel('Profit/Loss')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(x=0, color='red', linestyle='--')

        # Win/Loss pie chart
        if result.total_trades > 0:
            labels = ['Wins', 'Losses']
            sizes = [result.winning_trades, result.losing_trades]
            colors = ['green', 'red']
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[1, 0].set_title('Win/Loss Ratio')

        # Performance metrics text
        metrics_text = ".2f"".2f"".2f"".2f"".2f"".2f"f"""
        Total Trades: {result.total_trades}
        Win Rate: {result.win_rate:.1%}
        Net Profit: ${result.net_profit:.2f}
        Max Drawdown: ${result.max_drawdown:.2f}
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Profit Factor: {result.profit_factor:.2f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Performance Metrics')
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
        GOLD AI SONNET 4.5 - BACKTEST REPORT
        ===================================

        TRADING PERFORMANCE
        ------------------
        Total Trades: {result.total_trades}
        Winning Trades: {result.winning_trades}
        Losing Trades: {result.losing_trades}
        Win Rate: {result.win_rate:.1%}

        FINANCIAL METRICS
        ----------------
        Total Profit: ${result.total_profit:.2f}
        Total Loss: ${result.total_loss:.2f}
        Net Profit: ${result.net_profit:.2f}
        Max Drawdown: ${result.max_drawdown:.2f}

        RISK METRICS
        -----------
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Profit Factor: {result.profit_factor:.2f}
        Average Win: ${result.avg_win:.2f}
        Average Loss: ${result.avg_loss:.2f}
        Largest Win: ${result.largest_win:.2f}
        Largest Loss: ${result.largest_loss:.2f}

        STREAK ANALYSIS
        --------------
        Max Consecutive Wins: {result.max_consecutive_wins}
        Max Consecutive Losses: {result.max_consecutive_losses}

        BACKTEST CONFIGURATION
        --------------------
        Initial Balance: ${self.initial_balance:.2f}
        Leverage: {self.leverage}x
        Commission: {self.commission:.4f}
        Spread: {self.spread} pips
        """

        return report
