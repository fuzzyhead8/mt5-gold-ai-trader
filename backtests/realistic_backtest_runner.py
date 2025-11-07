import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy
from strategies.golden_scalping import GoldenScalpingStrategy
from strategies.golden_scalping_simplified import GoldenScalpingStrategySimplified
from lot_optimizer.optimizer import LotSizeOptimizer
import matplotlib.pyplot as plt
from datetime import datetime

class RealisticBacktestRunner:
    """
    Realistic backtest runner that includes transaction costs, slippage, and execution delays
    to better align with live trading performance
    """
    
    def __init__(self, strategy_param: str = 'all', symbol: str = 'XAUUSD'):
        self.symbol = symbol
        self.strategy_param = strategy_param.lower()
        self.data = None
        self.lot_optimizer = LotSizeOptimizer(max_risk_percent=2)
        
        # Realistic trading costs for XAUUSD
        self.spread_pips = 4.0  # XAUUSD typical spread: 3-5 pips
        self.slippage_pips = 2.0  # Average slippage: 1-3 pips
        self.commission_per_lot = 5.0  # Per lot round-turn commission
        self.execution_delay_bars = 1  # Signal acts on next bar (realistic)
        
        # Account settings
        self.initial_balance = 1000
        self.max_risk_per_trade = 2.0  # 2% risk per trade
        
        # Load data
        self._load_data()
        
        # Validate strategy parameter
        valid_strategies = ['all', 'daily', 'swing', 'scalping', 'golden', 'golden_simplified']
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
        elif strategy_type == 'golden':
            return GoldenScalpingStrategy(self.symbol)
        elif strategy_type == 'golden_simplified':
            return GoldenScalpingStrategySimplified(self.symbol)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _calculate_transaction_costs(self, lot_size, entry_price):
        """Calculate realistic transaction costs"""
        # Convert pips to price for XAUUSD (1 pip = 0.1)
        spread_cost = (self.spread_pips * 0.1) * lot_size * 100  # $1 per pip per lot
        slippage_cost = (self.slippage_pips * 0.1) * lot_size * 100
        commission_cost = lot_size * self.commission_per_lot
        
        total_cost = spread_cost + slippage_cost + commission_cost
        return total_cost

    def _get_realistic_execution_price(self, signal, current_price, next_bar_data):
        """
        Get realistic execution price accounting for:
        1. Execution delay (use next bar open)
        2. Spread (buy at ask, sell at bid)
        3. Slippage (random component)
        """
        if next_bar_data is None:
            return None
            
        base_price = next_bar_data['open']  # Execute at next bar open
        
        # Add spread
        if signal == 'buy':
            # Buy at ask (add half spread)
            execution_price = base_price + (self.spread_pips * 0.1 / 2)
        else:  # sell
            # Sell at bid (subtract half spread)
            execution_price = base_price - (self.spread_pips * 0.1 / 2)
        
        # Add random slippage (normal distribution)
        slippage = np.random.normal(0, self.slippage_pips * 0.1 / 2)
        if signal == 'buy':
            execution_price += abs(slippage)  # Always adverse for buy
        else:
            execution_price -= abs(slippage)  # Always adverse for sell
            
        return execution_price

    def _run_realistic_simulation(self, signals_data, strategy_name):
        """Run realistic trading simulation with transaction costs and execution delays"""
        
        balance = self.initial_balance
        equity_curve = []
        trades_log = []
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        
        i = 0
        while i < len(signals_data) - 1:  # -1 because we need next bar for execution
            current_signal = signals_data['signal'].iloc[i]
            
            if current_signal in ['buy', 'sell']:
                # Get execution data (next bar)
                next_bar = signals_data.iloc[i + 1]
                current_price = signals_data['close'].iloc[i]
                
                # Calculate realistic execution price
                execution_price = self._get_realistic_execution_price(
                    current_signal, current_price, next_bar)
                
                if execution_price is None:
                    equity_curve.append(balance)
                    i += 1
                    continue
                
                # Calculate lot size based on risk management
                # Set stop loss and take profit (simplified)
                if current_signal == 'buy':
                    stop_loss = execution_price - 60 * 0.1  # 60 pips SL
                    take_profit = execution_price + 120 * 0.1  # 120 pips TP
                else:
                    stop_loss = execution_price + 60 * 0.1
                    take_profit = execution_price - 120 * 0.1
                
                lot_size = self.lot_optimizer.optimize(balance, execution_price, stop_loss)
                lot_size = min(lot_size, 0.1)  # Max 0.1 lots as in live trading
                
                # Calculate transaction costs
                transaction_cost = self._calculate_transaction_costs(lot_size, execution_price)
                
                # Simulate trade holding and exit
                trade_pnl = 0
                exit_price = None
                exit_reason = "timeout"
                bars_held = 0
                max_hold_periods = 96  # Max 24 hours for M15 bars
                
                # Look ahead to find exit
                for j in range(i + 1, min(i + max_hold_periods, len(signals_data))):
                    bar = signals_data.iloc[j]
                    bars_held += 1
                    
                    # Check stop loss
                    if current_signal == 'buy':
                        if bar['low'] <= stop_loss:
                            exit_price = stop_loss
                            exit_reason = "stop_loss"
                            break
                        elif bar['high'] >= take_profit:
                            exit_price = take_profit
                            exit_reason = "take_profit"
                            break
                    else:  # sell
                        if bar['high'] >= stop_loss:
                            exit_price = stop_loss
                            exit_reason = "stop_loss"
                            break
                        elif bar['low'] <= take_profit:
                            exit_price = take_profit
                            exit_reason = "take_profit"
                            break
                
                # If no exit found, exit at last available price
                if exit_price is None:
                    exit_price = signals_data['close'].iloc[min(i + max_hold_periods - 1, len(signals_data) - 1)]
                    exit_reason = "timeout"
                
                # Calculate P&L
                if current_signal == 'buy':
                    price_diff = exit_price - execution_price
                else:
                    price_diff = execution_price - exit_price
                
                # XAUUSD: $1 per pip per lot movement
                gross_pnl = price_diff * lot_size * 100
                net_pnl = gross_pnl - transaction_cost
                
                # Update balance
                balance += net_pnl
                total_pnl += net_pnl
                total_trades += 1
                
                if net_pnl > 0:
                    winning_trades += 1
                
                # Log trade
                trades_log.append({
                    'entry_time': signals_data.index[i],
                    'exit_time': signals_data.index[min(i + bars_held, len(signals_data) - 1)],
                    'signal': current_signal,
                    'entry_price': execution_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'lot_size': lot_size,
                    'gross_pnl': gross_pnl,
                    'transaction_cost': transaction_cost,
                    'net_pnl': net_pnl,
                    'balance': balance,
                    'bars_held': bars_held
                })
                
                # Skip ahead to avoid overlapping trades
                i += bars_held
            else:
                i += 1
            
            equity_curve.append(balance)
        
        # Create results DataFrame
        results = pd.DataFrame(index=signals_data.index[:len(equity_curve)])
        results['balance'] = equity_curve
        results['returns'] = pd.Series(equity_curve).pct_change().fillna(0)
        results['cumulative_return'] = results['balance'] / self.initial_balance
        
        # Calculate performance metrics
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        returns_series = pd.Series(equity_curve).pct_change().dropna()
        if len(returns_series) > 1:
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252 * 24 * 4) if returns_series.std() > 0 else 0  # Annualized for M15
            drawdowns = (pd.Series(equity_curve).cummax() - pd.Series(equity_curve)) / pd.Series(equity_curve).cummax()
            max_drawdown = drawdowns.max() * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        metrics = {
            'Strategy': strategy_name.upper(),
            'Total Return (%)': round(total_return, 2),
            'Total Trades': total_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Avg Trade ($)': round(avg_trade, 2),
            'Sharpe Ratio': round(sharpe_ratio, 3),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Final Balance ($)': round(balance, 2),
            'Total Transaction Costs ($)': round(sum([t['transaction_cost'] for t in trades_log]), 2)
        }
        
        return results, metrics, trades_log

    def _run_single_strategy(self, strategy_type: str):
        """Run a single strategy with realistic simulation"""
        print(f"\n{'='*60}")
        print(f"Running REALISTIC {strategy_type.upper()} Strategy Simulation")
        print(f"{'='*60}")
        
        strategy = self._get_strategy(strategy_type)
        signals_result = strategy.generate_signals(self.data.copy())
        
        # Run realistic simulation
        results, metrics, trades_log = self._run_realistic_simulation(signals_result, strategy_type)
        
        # Display metrics
        print(f"\nRealistic Performance Metrics for {strategy_type.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
        # Display sample trades
        if trades_log:
            print(f"\nSample Trades (first 5):")
            for i, trade in enumerate(trades_log[:5]):
                print(f"  Trade {i+1}: {trade['signal'].upper()} {trade['lot_size']} lots, "
                      f"Entry: {trade['entry_price']:.2f}, Exit: {trade['exit_price']:.2f}, "
                      f"P&L: ${trade['net_pnl']:.2f} ({trade['exit_reason']})")
        
        return results, metrics, trades_log

    def run(self):
        """Run realistic backtest based on strategy parameter"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        print(f"=== REALISTIC MT5 GOLD AI TRADER BACKTEST ===")
        print(f"Including: Spreads ({self.spread_pips} pips), Slippage ({self.slippage_pips} pips), "
              f"Commission (${self.commission_per_lot}/lot), Execution delays")
        print(f"Initial Balance: ${self.initial_balance:,}")
        print(f"Max Risk per Trade: {self.max_risk_per_trade}%")
        print(f"Data period: {self.data.index[0]} to {self.data.index[-1]}")
        
        results_dict = {}
        
        if self.strategy_param == 'all':
            # Run all strategies
            for strategy in ['scalping', 'daily', 'swing', 'golden']:
                results, metrics, trades_log = self._run_single_strategy(strategy)
                results_dict[strategy] = (results, metrics, trades_log)
                
            # Print comparison
            print(f"\n{'='*80}")
            print("REALISTIC STRATEGY COMPARISON")
            print(f"{'='*80}")
            print(f"{'Strategy':<12} {'Return %':<10} {'Trades':<8} {'Win %':<8} {'Avg Trade':<12} {'Max DD %':<10} {'Costs $':<10}")
            print("-" * 80)
            
            for strategy, (_, metrics, _) in results_dict.items():
                print(f"{strategy.upper():<12} {metrics['Total Return (%)']:<10} "
                      f"{metrics['Total Trades']:<8} {metrics['Win Rate (%)']:<8} "
                      f"${metrics['Avg Trade ($)']:<11} {metrics['Max Drawdown (%)']:<10} "
                      f"${metrics['Total Transaction Costs ($)']:<9}")
        else:
            # Run single strategy
            results, metrics, trades_log = self._run_single_strategy(self.strategy_param)
            results_dict[self.strategy_param] = (results, metrics, trades_log)
        
        return results_dict

def run_realistic_backtest(strategy_param='golden', symbol='XAUUSD'):
    """Run realistic backtest with transaction costs and execution delays"""
    runner = RealisticBacktestRunner(strategy_param=strategy_param, symbol=symbol)
    return runner.run()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Realistic MT5 Gold AI Trader Backtest')
    parser.add_argument('--strategy', '-s', choices=['all', 'daily', 'swing', 'scalping', 'golden', 'golden_simplified'], 
                       default='golden', help='Strategy to run (default: golden)')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol (default: XAUUSD)')
    
    args = parser.parse_args()
    
    print(f"MT5 Gold AI Trader - REALISTIC Backtest Runner")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Symbol: {args.symbol}")
    
    try:
        results = run_realistic_backtest(strategy_param=args.strategy, symbol=args.symbol)
        print(f"\nRealistic backtest completed successfully!")
        
    except Exception as e:
        print(f"Error running realistic backtest: {e}")
        import traceback
        traceback.print_exc()
