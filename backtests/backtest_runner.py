import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy
from strategies.golden_scalping import GoldenScalpingStrategy
from strategies.golden_risk_manager import GoldenRiskManager
import matplotlib.pyplot as plt
import glob
from datetime import datetime

class BacktestRunner:
    def __init__(self, strategy_param: str = 'all', symbol: str = 'XAUUSD'):
        """
        Initialize BacktestRunner with strategy parameter
        
        Args:
            strategy_param: 'all', 'daily', 'swing', 'scalping', 'golden'
            symbol: Trading symbol (default: 'XAUUSD')
        """
        self.symbol = symbol
        self.strategy_param = strategy_param.lower()
        self.data = None
        self.timeframe = None
        
        # Validate strategy parameter
        valid_strategies = ['all', 'daily', 'swing', 'scalping', 'golden']
        if self.strategy_param not in valid_strategies:
            raise ValueError(f"Invalid strategy parameter. Must be one of: {valid_strategies}")

    def _load_data(self, timeframe: str):
        """Load the XAUUSD data for given timeframe"""
        try:
            dir_path = os.path.dirname(__file__)
            pattern = os.path.join(dir_path, f"XAUUSD_{timeframe}_*.csv")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(f"No XAUUSD_{timeframe}_*.csv found in {dir_path}")
            # Get the most recent file
            latest_file = max(files, key=os.path.getctime)
            self.data = pd.read_csv(latest_file)
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data.set_index('time', inplace=True)
            self.timeframe = timeframe
            print(f"Loaded {len(self.data)} {timeframe} candles from {self.data.index[0]} to {self.data.index[-1]} from {latest_file}")
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))

    def _get_strategy(self, strategy_type: str):
        """Get strategy instance based on type"""
        if strategy_type == 'scalping':
            return ScalpingStrategy(self.symbol)
        elif strategy_type == 'daily':
            return DayTradingStrategy(self.symbol)
        elif strategy_type == 'swing':
            return SwingTradingStrategy(self.symbol)
        elif strategy_type == 'golden':
            return GoldenScalpingStrategy(self.symbol)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _get_timeframe(self, strategy: str) -> str:
        """Get timeframe for strategy"""
        mapping = {
            'daily': 'M30',
            'swing': 'H4',
            'scalping': 'M1',
            'golden': 'M15'
        }
        return mapping.get(strategy.lower(), 'M15')

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
        
        # Save results to CSV
        output_path = os.path.join(os.path.dirname(__file__), f"{strategy_type}_backtest.csv")
        result.to_csv(output_path)
        print(f"Backtest results saved to {output_path}")
        
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
        results_dict = {}
        
        if self.strategy_param == 'all':
            print(f"Running ALL strategies on {self.symbol} with respective timeframes")
            
            # Run all strategies
            for strategy in ['scalping', 'daily', 'swing', 'golden']:
                timeframe = self._get_timeframe(strategy)
                self._load_data(timeframe)
                print(f"Data period for {strategy}: {self.data.index[0]} to {self.data.index[-1]}")
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
            timeframe = self._get_timeframe(self.strategy_param)
            self._load_data(timeframe)
            print(f"Data period: {self.data.index[0]} to {self.data.index[-1]}")
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
    parser.add_argument('--strategy', '-s', choices=['all', 'daily', 'swing', 'scalping', 'golden'], 
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
