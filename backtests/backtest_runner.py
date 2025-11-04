import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy
import matplotlib.pyplot as plt
from datetime import datetime

class BacktestRunner:
    def __init__(self, strategy_param: str = 'all', symbol: str = 'XAUUSD'):
        """
        Initialize BacktestRunner with strategy parameter
        
        Args:
            strategy_param: 'all', 'daily', 'swing', 'scalping'
            symbol: Trading symbol (default: 'XAUUSD')
        """
        self.symbol = symbol
        self.strategy_param = strategy_param.lower()
        self.data = None
        
        # Load the M15 data
        self._load_data()
        
        # Validate strategy parameter
        valid_strategies = ['all', 'daily', 'swing', 'scalping']
        if self.strategy_param not in valid_strategies:
            raise ValueError(f"Invalid strategy parameter. Must be one of: {valid_strategies}")

    def _load_data(self):
        """Load the XAUUSD M15 data"""
        try:
            csv_path = os.path.join(os.path.dirname(__file__), "XAUUSD_M15_20251104_214837.csv")
            self.data = pd.read_csv(csv_path)
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data.set_index('time', inplace=True)
            print(f"Loaded {len(self.data)} M15 candles from {self.data.index[0]} to {self.data.index[-1]}")
        except FileNotFoundError:
            raise FileNotFoundError("XAUUSD_M15_20251104_214837.csv not found in backtests directory")

    def _get_strategy(self, strategy_type: str):
        """Get strategy instance based on type"""
        if strategy_type == 'scalping':
            return ScalpingStrategy(self.symbol)
        elif strategy_type == 'daily':
            return DayTradingStrategy(self.symbol)
        elif strategy_type == 'swing':
            return SwingTradingStrategy(self.symbol)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _calculate_performance_metrics(self, results):
        """Calculate comprehensive performance metrics"""
        # Basic returns
        total_return = results['cumulative'].iloc[-1] - 1
        
        # Trade statistics
        signals = results['signal']
        trades = signals[signals != 'hold']
        total_trades = len(trades)
        
        # Win rate calculation
        buy_trades = (signals == 'buy').sum()
        sell_trades = (signals == 'sell').sum()
        
        # Daily returns for Sharpe ratio
        daily_returns = results['strategy_returns'].resample('D').sum()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252**0.5) if daily_returns.std() > 0 else 0
        
        # Maximum drawdown
        running_max = results['cumulative'].cummax()
        drawdown = (results['cumulative'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return (%)': round(total_return * 100, 2),
            'Total Trades': total_trades,
            'Buy Signals': buy_trades,
            'Sell Signals': sell_trades,
            'Sharpe Ratio': round(sharpe_ratio, 3),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
            'Final Equity': round(results['cumulative'].iloc[-1], 4)
        }

    def _run_single_strategy(self, strategy_type: str):
        """Run a single strategy and return results"""
        print(f"\n{'='*50}")
        print(f"Running {strategy_type.upper()} Strategy")
        print(f"{'='*50}")
        
        strategy = self._get_strategy(strategy_type)
        result = strategy.generate_signals(self.data.copy())
        
        # Calculate performance
        result['returns'] = result['close'].pct_change()
        result['strategy_returns'] = result['returns'] * result['signal'].map({'buy': 1, 'sell': -1, 'hold': 0}).fillna(0)
        result['cumulative'] = (1 + result['strategy_returns']).cumprod()
        
        # Calculate and display metrics
        metrics = self._calculate_performance_metrics(result)
        print(f"\nPerformance Metrics for {strategy_type.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
        return result, metrics

    def _plot_results(self, results_dict):
        """Plot results for single or multiple strategies"""
        plt.figure(figsize=(15, 10))
        
        if len(results_dict) == 1:
            # Single strategy plot
            strategy_name, (result, metrics) = list(results_dict.items())[0]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Price and signals
            ax1.plot(result.index, result['close'], label='Price', alpha=0.7)
            buy_signals = result[result['signal'] == 'buy']
            sell_signals = result[result['signal'] == 'sell']
            
            ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=50, label='Buy', alpha=0.8)
            ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=50, label='Sell', alpha=0.8)
            ax1.set_title(f'{strategy_name.upper()} Strategy - Price and Signals')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Equity curve
            ax2.plot(result.index, result['cumulative'], label=f'{strategy_name.upper()} Equity', linewidth=2)
            ax2.set_title(f'{strategy_name.upper()} Strategy - Equity Curve')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Return')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        else:
            # Multiple strategies comparison
            plt.subplot(2, 1, 1)
            colors = ['blue', 'green', 'red', 'orange']
            
            for i, (strategy_name, (result, metrics)) in enumerate(results_dict.items()):
                plt.plot(result.index, result['cumulative'], 
                        label=f"{strategy_name.upper()} ({metrics['Total Return (%)']}%)", 
                        linewidth=2, color=colors[i % len(colors)])
            
            plt.title(f'Strategy Comparison - Equity Curves ({self.symbol})')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Performance comparison bar chart
            plt.subplot(2, 1, 2)
            strategies = list(results_dict.keys())
            returns = [metrics['Total Return (%)'] for _, (_, metrics) in results_dict.items()]
            sharpe_ratios = [metrics['Sharpe Ratio'] for _, (_, metrics) in results_dict.items()]
            
            x = range(len(strategies))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], returns, width, label='Total Return (%)', alpha=0.8)
            plt.bar([i + width/2 for i in x], [s*10 for s in sharpe_ratios], width, label='Sharpe Ratio (x10)', alpha=0.8)
            
            plt.xlabel('Strategy')
            plt.ylabel('Performance')
            plt.title('Strategy Performance Comparison')
            plt.xticks(x, [s.upper() for s in strategies])
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def run(self):
        """Run backtest based on strategy parameter"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        results_dict = {}
        
        if self.strategy_param == 'all':
            print(f"Running ALL strategies on {self.symbol} M15 data")
            print(f"Data period: {self.data.index[0]} to {self.data.index[-1]}")
            
            # Run all strategies
            for strategy in ['scalping', 'daily', 'swing']:
                result, metrics = self._run_single_strategy(strategy)
                results_dict[strategy] = (result, metrics)
            
            # Print comparison summary
            print(f"\n{'='*60}")
            print("STRATEGY COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"{'Strategy':<12} {'Return %':<10} {'Trades':<8} {'Sharpe':<8} {'Max DD %':<10}")
            print("-" * 60)
            
            for strategy, (_, metrics) in results_dict.items():
                print(f"{strategy.upper():<12} {metrics['Total Return (%)']:<10} "
                      f"{metrics['Total Trades']:<8} {metrics['Sharpe Ratio']:<8} "
                      f"{metrics['Max Drawdown (%)']:<10}")
                
        else:
            # Run single strategy
            result, metrics = self._run_single_strategy(self.strategy_param)
            results_dict[self.strategy_param] = (result, metrics)
        
        # Plot results
        self._plot_results(results_dict)
        
        return results_dict

def run_backtest(strategy_param='all', symbol='XAUUSD'):
    """
    Convenience function to run backtest
    
    Args:
        strategy_param: 'all', 'daily', 'swing', 'scalping'
        symbol: Trading symbol (default: 'XAUUSD')
    
    Returns:
        Dictionary with results for each strategy
    """
    runner = BacktestRunner(strategy_param=strategy_param, symbol=symbol)
    return runner.run()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MT5 Gold AI Trader Backtest')
    parser.add_argument('--strategy', '-s', choices=['all', 'daily', 'swing', 'scalping'], 
                       default='all', help='Strategy to run (default: all)')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol (default: XAUUSD)')
    
    args = parser.parse_args()
    
    print(f"MT5 Gold AI Trader - Backtest Runner")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Symbol: {args.symbol}")
    
    try:
        results = run_backtest(strategy_param=args.strategy, symbol=args.symbol)
        print(f"\nBacktest completed successfully!")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
