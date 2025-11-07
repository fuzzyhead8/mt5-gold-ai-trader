import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy
from strategies.golden_scalping import GoldenScalpingStrategy
from strategies.golden_risk_manager import GoldenRiskManager
from strategies.goldstorm_strategy import GoldStormStrategy
from strategies.goldstorm_v2_strategy import GoldStormV2Strategy
from strategies.vwap_strategy import VWAPStrategy
from strategies.multi_rsi_ema import MultiRSIEMAStrategy
import matplotlib.pyplot as plt
import glob
from datetime import datetime

class BacktestRunner:
    def __init__(self, strategy_param: str = 'all', symbol: str = 'XAUUSD', chart_style: str = 'light', use_ml: bool = False):
        """
        Initialize BacktestRunner with strategy parameter
        
        Args:
            strategy_param: 'all', 'daily', 'swing', 'scalping', 'golden'
            symbol: Trading symbol (default: 'XAUUSD')
            chart_style: 'light' or 'dark' for TradingView-style themes
            use_ml: Use trained ML model for signals (default: False)
        """
        self.symbol = symbol
        self.strategy_param = strategy_param.lower()
        self.chart_style = chart_style.lower()
        self.use_ml = use_ml
        self.data = None
        self.timeframe = None
        
        # Validate strategy parameter
        valid_strategies = ['all', 'daily', 'swing', 'scalping', 'golden', 'goldstorm', 'goldstormv2', 'vwap', 'multi_rsi_ema']
        if self.strategy_param not in valid_strategies:
            raise ValueError(f"Invalid strategy parameter. Must be one of: {valid_strategies}")
        
        # Validate chart style
        valid_styles = ['light', 'dark']
        if self.chart_style not in valid_styles:
            raise ValueError(f"Invalid chart style. Must be one of: {valid_styles}")
        
        # Set up TradingView-style color schemes
        self._setup_chart_styles()
    
    def _setup_chart_styles(self):
        """Setup TradingView-style color schemes"""
        if self.chart_style == 'dark':
            # TradingView Dark Theme Colors
            self.colors = {
                'background': '#131722',
                'grid': '#363C4E',
                'text': '#D1D4DC',
                'price_line': '#2962FF',
                'equity_line': '#089981',
                'buy_signal': '#089981',
                'sell_signal': '#F23645',
                'volume_up': '#26A69A',
                'volume_down': '#EF5350',
                'profit_green': '#00C851',
                'loss_red': '#FF4444',
                'neutral': '#9E9E9E'
            }
        else:
            # Light Theme Colors (Default)
            self.colors = {
                'background': '#FFFFFF',
                'grid': '#E0E0E0',
                'text': '#333333',
                'price_line': '#2E7D32',
                'equity_line': '#1976D2',
                'buy_signal': '#4CAF50',
                'sell_signal': '#F44336',
                'volume_up': '#4CAF50',
                'volume_down': '#F44336',
                'profit_green': '#4CAF50',
                'loss_red': '#F44336',
                'neutral': '#757575'
            }

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
        elif strategy_type == 'goldstorm':
            return GoldStormStrategy(self.symbol)
        elif strategy_type == 'goldstormv2':
            return GoldStormV2Strategy(self.symbol)
        elif strategy_type == 'vwap':
            return VWAPStrategy(self.symbol)
        elif strategy_type == 'multi_rsi_ema':
            return MultiRSIEMAStrategy(self.symbol)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _get_timeframe(self, strategy: str) -> str:
        """Get timeframe for strategy"""
        mapping = {
            'daily': 'M30',
            'swing': 'H4',
            'scalping': 'M1',
            'golden': 'M15',
            'goldstorm': 'M15',
            'goldstormv2': 'M15',
            'vwap': 'M15',
            'multi_rsi_ema': 'M15'
        }
        return mapping.get(strategy.lower(), 'M1')

    def _calculate_performance_metrics(self, results):
        """Calculate comprehensive TradingView-style performance metrics"""
        initial_capital = 10000  # Assume $10K starting capital
        
        # Calculate individual trade results
        trade_results = self._extract_individual_trades(results, initial_capital)
        
        # Basic equity metrics
        final_equity = results['cumulative'].iloc[-1] * initial_capital
        net_profit = final_equity - initial_capital
        total_return_pct = (final_equity / initial_capital - 1) * 100
        
        # Trade statistics
        total_trades = len(trade_results)
        winning_trades = len([t for t in trade_results if t['pnl'] > 0])
        losing_trades = len([t for t in trade_results if t['pnl'] < 0])
        
        # P&L calculations
        gross_profit = sum([t['pnl'] for t in trade_results if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in trade_results if t['pnl'] < 0]))
        
        # Profitability metrics
        percent_profitable = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Average trade metrics
        avg_trade = net_profit / total_trades if total_trades > 0 else 0
        avg_winning_trade = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_losing_trade = -gross_loss / losing_trades if losing_trades > 0 else 0
        
        # Win/Loss ratio
        win_loss_ratio = avg_winning_trade / abs(avg_losing_trade) if avg_losing_trade != 0 else 0
        
        # Largest trades
        winning_pnls = [t['pnl'] for t in trade_results if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in trade_results if t['pnl'] < 0]
        
        largest_winning_trade = max(winning_pnls) if winning_pnls else 0
        largest_losing_trade = min(losing_pnls) if losing_pnls else 0
        
        # Percentage of largest trades
        largest_winning_pct = (largest_winning_trade / initial_capital * 100) if largest_winning_trade > 0 else 0
        largest_losing_pct = (abs(largest_losing_trade) / initial_capital * 100) if largest_losing_trade < 0 else 0
        
        # Drawdown calculations
        running_max = (results['cumulative'] * initial_capital).cummax()
        equity_curve = results['cumulative'] * initial_capital
        drawdown_dollar = equity_curve - running_max
        drawdown_pct = drawdown_dollar / running_max * 100
        
        max_drawdown_dollar = drawdown_dollar.min()
        max_drawdown_pct = drawdown_pct.min()
        
        # Max equity run-up (maximum peak from starting capital)
        max_equity_runup = running_max.max() - initial_capital
        max_equity_runup_pct = (max_equity_runup / initial_capital * 100)
        
        # Buy & Hold return calculation
        buy_hold_return = (results['close'].iloc[-1] / results['close'].iloc[0] - 1) * 100
        
        # Commission calculation (estimate 0.1% per trade)
        commission_rate = 0.001
        total_volume = sum([abs(t['pnl']) for t in trade_results])
        commission_paid = total_volume * commission_rate
        
        # Sharpe ratio
        daily_returns = results['strategy_returns'].resample('D').sum()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252**0.5) if daily_returns.std() > 0 else 0
        
        # Average bars in trades
        avg_bars_in_trades = sum([t['duration'] for t in trade_results]) / total_trades if total_trades > 0 else 0
        
        # Separate long and short trade analysis
        long_trades = [t for t in trade_results if t['type'] == 'long']
        short_trades = [t for t in trade_results if t['type'] == 'short']
        
        long_winning = len([t for t in long_trades if t['pnl'] > 0])
        short_winning = len([t for t in short_trades if t['pnl'] > 0])
        
        long_net_profit = sum([t['pnl'] for t in long_trades])
        short_net_profit = sum([t['pnl'] for t in short_trades])
        
        long_gross_profit = sum([t['pnl'] for t in long_trades if t['pnl'] > 0])
        short_gross_profit = sum([t['pnl'] for t in short_trades if t['pnl'] > 0])
        
        long_gross_loss = abs(sum([t['pnl'] for t in long_trades if t['pnl'] < 0]))
        short_gross_loss = abs(sum([t['pnl'] for t in short_trades if t['pnl'] < 0]))
        
        return {
            # Overview metrics (like TradingView Overview tab)
            'Initial Capital': f"${initial_capital:.2f}",
            'Net Profit': f"${net_profit:.2f} ({total_return_pct:+.2f}%)",
            'Gross Profit': f"${gross_profit:.2f} ({gross_profit/initial_capital*100:.2f}%)",
            'Gross Loss': f"${gross_loss:.2f} ({gross_loss/initial_capital*100:.2f}%)",
            'Commission Paid': f"${commission_paid:.2f}",
            
            # Performance ratios
            'Profit Factor': f"{profit_factor:.3f}",
            'Max Equity Run-up': f"${max_equity_runup:.2f} ({max_equity_runup_pct:.2f}%)",
            'Max Equity Drawdown': f"${abs(max_drawdown_dollar):.2f} ({abs(max_drawdown_pct):.2f}%)",
            'Buy & Hold Return': f"{buy_hold_return:+.2f}%",
            
            # Trade analysis (like TradingView Trades Analysis tab)
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Percent Profitable': f"{percent_profitable:.2f}%",
            
            'Avg Trade P&L': f"${avg_trade:.2f} ({avg_trade/initial_capital*100:.2f}%)",
            'Avg Winning Trade': f"${avg_winning_trade:.2f} ({avg_winning_trade/initial_capital*100:.2f}%)",
            'Avg Losing Trade': f"${avg_losing_trade:.2f} ({avg_losing_trade/initial_capital*100:.2f}%)",
            'Ratio Avg Win/Avg Loss': f"{win_loss_ratio:.3f}",
            
            'Largest Winning Trade': f"${largest_winning_trade:.2f}",
            'Largest Winning Trade %': f"{largest_winning_pct:.2f}%",
            'Largest Losing Trade': f"${largest_losing_trade:.2f}",
            'Largest Losing Trade %': f"{largest_losing_pct:.2f}%",
            
            'Avg # Bars in Trades': f"{avg_bars_in_trades:.0f}",
            
            # Long vs Short breakdown
            'Long Trades': len(long_trades),
            'Long Winning': long_winning,
            'Long Net Profit': f"${long_net_profit:.2f} ({long_net_profit/initial_capital*100:+.2f}%)",
            'Long Gross Profit': f"${long_gross_profit:.2f} ({long_gross_profit/initial_capital*100:.2f}%)",
            'Long Gross Loss': f"${long_gross_loss:.2f} ({long_gross_loss/initial_capital*100:.2f}%)",
            
            'Short Trades': len(short_trades),
            'Short Winning': short_winning,
            'Short Net Profit': f"${short_net_profit:.2f} ({short_net_profit/initial_capital*100:+.2f}%)",
            'Short Gross Profit': f"${short_gross_profit:.2f} ({short_gross_profit/initial_capital*100:.2f}%)",
            'Short Gross Loss': f"${short_gross_loss:.2f} ({short_gross_loss/initial_capital*100:.2f}%)",
            
            # Additional metrics for compatibility
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Final Equity': f"${final_equity:.2f}",
        }

    def _extract_individual_trades(self, results, initial_capital):
        """Extract individual trade results for detailed analysis"""
        trades = []
        current_trade = None
        current_position = None
        entry_equity = initial_capital
        current_equity = initial_capital
        
        # Use the position column if available (from fixed strategy_returns calc)
        if 'position' in results.columns:
            positions = results['position']
        else:
            # Fallback to signal-based position (for compatibility)
            positions = []
            position = 0
            for signal in results['signal']:
                if pd.isna(signal):
                    positions.append(position)
                    continue
                signal_str = str(signal).lower()
                if signal_str == 'buy' and position != 1:
                    position = 1
                elif signal_str == 'sell' and position != -1:
                    position = -1
                positions.append(position)
            results['position'] = positions
        
        for i in range(len(results)):
            signal = results['signal'].iloc[i]
            returns = results['strategy_returns'].iloc[i]
            position = results['position'].iloc[i]
            
            # Update current equity
            current_equity *= (1 + returns)
            
            if current_trade is None:
                entry_equity = current_equity
            
            if pd.isna(signal):
                continue
            
            signal_str = str(signal).lower()
            close_position = False
            new_trade_type = None
            
            if signal_str == 'buy' and position == 1 and current_position != 1:
                if current_position == -1:
                    close_position = True
                current_position = 1
                new_trade_type = 'long'
            elif signal_str == 'sell' and position == -1 and current_position != -1:
                if current_position == 1:
                    close_position = True
                current_position = -1
                new_trade_type = 'short'
            else:
                close_position = False
            
            if close_position and current_trade is not None:
                # Close current trade using accumulated returns
                trade_returns_sum = results['strategy_returns'].iloc[current_trade['entry_index']:i].sum()
                pnl = trade_returns_sum * current_trade['entry_equity']
                duration = i - current_trade['entry_index']
                exit_price = results['close'].iloc[i]
                
                trades.append({
                    'type': current_trade['type'],
                    'entry_price': current_trade['entry_price'],
                    'exit_price': exit_price,
                    'entry_time': current_trade['entry_time'],
                    'exit_time': results.index[i],
                    'pnl': pnl,
                    'duration': duration,
                    'entry_equity': current_trade['entry_equity']
                })
                current_trade = None
            
            if (signal_str in ['buy', 'sell']) and current_trade is None and new_trade_type:
                # Start new trade
                current_trade = {
                    'type': new_trade_type,
                    'entry_price': results['close'].iloc[i],
                    'entry_time': results.index[i],
                    'entry_equity': entry_equity,
                    'entry_index': i
                }
            
            # Handle explicit exit signals (for pyramiding)
            if signal_str in ['exit_buy', 'exit_sell'] and current_trade is not None:
                trade_returns_sum = results['strategy_returns'].iloc[current_trade['entry_index']:i+1].sum()
                pnl = trade_returns_sum * current_trade['entry_equity']
                duration = i - current_trade['entry_index'] + 1
                exit_price = results['close'].iloc[i]
                
                trades.append({
                    'type': current_trade['type'],
                    'entry_price': current_trade['entry_price'],
                    'exit_price': exit_price,
                    'entry_time': current_trade['entry_time'],
                    'exit_time': results.index[i],
                    'pnl': pnl,
                    'duration': duration,
                    'entry_equity': current_trade['entry_equity']
                })
                current_trade = None
                current_position = 0
        
        # Handle unclosed trade at end
        if current_trade is not None:
            i_end = len(results)
            trade_returns_sum = results['strategy_returns'].iloc[current_trade['entry_index']:i_end].sum()
            pnl = trade_returns_sum * current_trade['entry_equity']
            duration = i_end - current_trade['entry_index']
            exit_price = results['close'].iloc[-1]
            
            trades.append({
                'type': current_trade['type'],
                'entry_price': current_trade['entry_price'],
                'exit_price': exit_price,
                'entry_time': current_trade['entry_time'],
                'exit_time': results.index[-1],
                'pnl': pnl,
                'duration': duration,
                'entry_equity': current_trade['entry_equity']
            })
        
        return trades

    def _run_single_strategy(self, strategy_type: str):
        """Run a single strategy and return results - FIXED for no look-ahead bias"""
        print(f"\n{'='*70}")
        print(f"🚀 {strategy_type.upper()} STRATEGY BACKTEST RESULTS (No Look-Ahead Bias)")
        print(f"{'='*70}")
        
        strategy = self._get_strategy(strategy_type)
        
        # FIXED: Generate signals progressively without look-ahead bias
        result = self.data.copy()
        signals = []
        min_lookback = max(50, 200)  # Conservative minimum for indicators (adjust based on strategy)
        
        for i in range(len(result)):
            if i < min_lookback:
                signals.append('hold')
            else:
                # Use only past and current data up to i
                current_data = result.iloc[:i+1].copy()
                temp_result = strategy.generate_signals(current_data)
                # Take the latest signal
                latest_signal = temp_result['signal'].iloc[-1] if 'signal' in temp_result.columns else 'hold'
                signals.append(latest_signal)
        
        result['signal'] = signals
        
        # Calculate performance
        result['returns'] = result['close'].pct_change().fillna(0)
        
        # Simulate persistent position for non-pyramiding strategies
        if strategy_type == 'scalping_pyr':
            result['strategy_returns'] = self._calculate_pyramiding_returns(result)
        else:
            position = 0
            positions = []
            for signal in result['signal']:
                if pd.isna(signal):
                    positions.append(position)
                    continue
                signal_str = str(signal).lower()
                if signal_str == 'buy' and position != 1:
                    position = 1
                elif signal_str == 'sell' and position != -1:
                    position = -1
                # else hold, keep position
                positions.append(position)
            result['position'] = positions
            result['strategy_returns'] = result['position'] * result['returns']
        
        result['cumulative'] = (1 + result['strategy_returns']).cumprod()
        
        # Calculate and display metrics
        metrics = self._calculate_performance_metrics(result)
        self._display_tradingview_style_metrics(metrics, strategy_type)
        
        # Save results to CSV
        output_path = os.path.join(os.path.dirname(__file__), f"{strategy_type}_backtest_no_bias.csv")
        result.to_csv(output_path)
        print(f"\n💾 Results saved to: {output_path} (No Look-Ahead Bias)")
        
        return result, metrics


    def _display_tradingview_style_metrics(self, metrics, strategy_type):
        """Display metrics in TradingView style with organized sections"""
        
        # Overview Section (Performance Tab)
        print(f"\n📊 PERFORMANCE OVERVIEW")
        print(f"{'─'*50}")
        print(f"{'Metric':<25} {'All':<20} {'Long':<15} {'Short':<15}")
        print(f"{'─'*50}")
        
        # Extract numeric values for display formatting
        net_profit_match = metrics['Net Profit'].split(' (')[0].replace('$', '').replace(',', '')
        net_profit_num = float(net_profit_match)
        
        print(f"{'Initial Capital':<25} {metrics['Initial Capital']:<20}")
        print(f"{'Net Profit':<25} {metrics['Net Profit']:<20} {metrics['Long Net Profit']:<15} {metrics['Short Net Profit']:<15}")
        print(f"{'Gross Profit':<25} {metrics['Gross Profit']:<20} {metrics['Long Gross Profit']:<15} {metrics['Short Gross Profit']:<15}")
        print(f"{'Gross Loss':<25} {metrics['Gross Loss']:<20} {metrics['Long Gross Loss']:<15} {metrics['Short Gross Loss']:<15}")
        print(f"{'Commission Paid':<25} {metrics['Commission Paid']:<20}")
        print(f"{'Buy & Hold Return':<25} {metrics['Buy & Hold Return']:<20}")
        
        # Risk/Performance Ratios
        print(f"\n⚖️  RISK/PERFORMANCE RATIOS")
        print(f"{'─'*50}")
        print(f"{'Profit Factor':<25} {metrics['Profit Factor']:<20}")
        print(f"{'Max Equity Run-up':<25} {metrics['Max Equity Run-up']:<20}")
        print(f"{'Max Equity Drawdown':<25} {metrics['Max Equity Drawdown']:<20}")
        
        # Trades Analysis Section
        print(f"\n📈 TRADES ANALYSIS")
        print(f"{'─'*50}")
        print(f"{'Metric':<25} {'All':<15} {'Long':<15} {'Short':<15}")
        print(f"{'─'*50}")
        print(f"{'Total Trades':<25} {metrics['Total Trades']:<15} {metrics['Long Trades']:<15} {metrics['Short Trades']:<15}")
        print(f"{'Winning Trades':<25} {metrics['Winning Trades']:<15} {metrics['Long Winning']:<15} {metrics['Short Winning']:<15}")
        print(f"{'Losing Trades':<25} {metrics['Losing Trades']:<15}")
        print(f"{'Percent Profitable':<25} {metrics['Percent Profitable']:<15}")
        
        print(f"\n{'Avg Trade P&L':<25} {metrics['Avg Trade P&L']:<15}")
        print(f"{'Avg Winning Trade':<25} {metrics['Avg Winning Trade']:<15}")
        print(f"{'Avg Losing Trade':<25} {metrics['Avg Losing Trade']:<15}")
        print(f"{'Ratio Avg Win/Loss':<25} {metrics['Ratio Avg Win/Avg Loss']:<15}")
        
        print(f"\n{'Largest Winning Trade':<25} {metrics['Largest Winning Trade']:<15}")
        print(f"{'Largest Winning %':<25} {metrics['Largest Winning Trade %']:<15}")
        print(f"{'Largest Losing Trade':<25} {metrics['Largest Losing Trade']:<15}")
        print(f"{'Largest Losing %':<25} {metrics['Largest Losing Trade %']:<15}")
        
        print(f"\n{'Avg # Bars in Trades':<25} {metrics['Avg # Bars in Trades']:<15}")
        
        # Summary with color coding (simulate with symbols)
        print(f"\n🎯 STRATEGY SUMMARY")
        print(f"{'─'*50}")
        status_symbol = "✅" if net_profit_num > 0 else "❌"
        profit_color = "PROFITABLE" if net_profit_num > 0 else "LOSS-MAKING"
        
        print(f"{status_symbol} Strategy Status: {profit_color}")
        print(f"📊 Final Equity: {metrics['Final Equity']}")
        print(f"📈 Sharpe Ratio: {metrics['Sharpe Ratio']}")
    
    def _calculate_pyramiding_returns(self, result):
        """
        Calculate returns for pyramiding strategy
        Takes into account position sizing and pyramid additions
        """
        strategy_returns = []
        current_position = None
        total_position_size = 0
        entry_prices = []  # Track entry price for each pyramid level
        
        for i in range(len(result)):
            signal = result['signal'].iloc[i]
            close = result['close'].iloc[i]
            returns = result['returns'].iloc[i]
            
            # Handle position entries and exits
            if signal == 'buy':
                current_position = 'buy'
                total_position_size = 1.0
                entry_prices = [close]
                strategy_returns.append(0)  # No return on entry
                
            elif signal == 'sell':
                current_position = 'sell'
                total_position_size = 1.0
                entry_prices = [close]
                strategy_returns.append(0)  # No return on entry
                
            elif signal == 'add_buy' and current_position == 'buy':
                # Add to position
                add_size = result['position_size'].iloc[i]
                total_position_size += add_size
                entry_prices.append(close)
                # Calculate return on existing position
                strategy_returns.append(returns * (total_position_size - add_size))
                
            elif signal == 'add_sell' and current_position == 'sell':
                # Add to position
                add_size = result['position_size'].iloc[i]
                total_position_size += add_size
                entry_prices.append(close)
                # Calculate return on existing position (inverted for sell)
                strategy_returns.append(-returns * (total_position_size - add_size))
                
            elif signal == 'exit_buy':
                # Close buy position
                if current_position == 'buy':
                    strategy_returns.append(returns * total_position_size)
                else:
                    strategy_returns.append(0)
                current_position = None
                total_position_size = 0
                entry_prices = []
                
            elif signal == 'exit_sell':
                # Close sell position
                if current_position == 'sell':
                    strategy_returns.append(-returns * total_position_size)
                else:
                    strategy_returns.append(0)
                current_position = None
                total_position_size = 0
                entry_prices = []
                
            elif signal == 'hold':
                # Calculate return based on current position
                if current_position == 'buy':
                    strategy_returns.append(returns * total_position_size)
                elif current_position == 'sell':
                    strategy_returns.append(-returns * total_position_size)
                else:
                    strategy_returns.append(0)
            else:
                strategy_returns.append(0)
        
        return pd.Series(strategy_returns, index=result.index)

    def _extract_return_percentage(self, net_profit_str):
        """Extract return percentage from Net Profit string format"""
        try:
            # Extract percentage from string like "$1245.67 (+12.46%)"
            return float(net_profit_str.split('(')[1].replace('%)', '').replace('+', ''))
        except (IndexError, ValueError):
            return 0.0

    def _extract_numeric_value(self, formatted_str):
        """Extract numeric value from formatted string"""
        try:
            # Extract number from strings like "1.987" or "$1245.67"
            import re
            number_match = re.search(r'[-+]?\d*\.?\d+', formatted_str.replace('$', '').replace(',', ''))
            return float(number_match.group()) if number_match else 0.0
        except (ValueError, AttributeError):
            return 0.0

    def _plot_results(self, results_dict):
        """Plot results for single or multiple strategies with TradingView-style themes"""
        # Set up matplotlib style based on chart theme
        plt.style.use('default')  # Reset to prevent style conflicts
        
        if len(results_dict) == 1:
            # Single strategy plot with TradingView styling
            strategy_name, (result, metrics) = list(results_dict.items())[0]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # Apply TradingView theme
            fig.patch.set_facecolor(self.colors['background'])
            
            # Price and signals chart
            ax1.set_facecolor(self.colors['background'])
            ax1.plot(result.index, result['close'], color=self.colors['price_line'], 
                    label='Price', alpha=0.8, linewidth=1.5)
            
            buy_signals = result[result['signal'] == 'buy']
            sell_signals = result[result['signal'] == 'sell']
            
            ax1.scatter(buy_signals.index, buy_signals['close'], 
                       color=self.colors['buy_signal'], marker='^', s=60, 
                       label='Buy', alpha=0.9, edgecolors='white', linewidth=0.5)
            ax1.scatter(sell_signals.index, sell_signals['close'], 
                       color=self.colors['sell_signal'], marker='v', s=60, 
                       label='Sell', alpha=0.9, edgecolors='white', linewidth=0.5)
            
            # TradingView-style title and labels
            ax1.set_title(f'{strategy_name.upper()} Strategy - Price Action & Signals', 
                         color=self.colors['text'], fontsize=14, fontweight='bold', pad=15)
            ax1.set_ylabel('Price (USD)', color=self.colors['text'], fontsize=11)
            ax1.tick_params(colors=self.colors['text'])
            ax1.grid(True, alpha=0.2, color=self.colors['grid'], linewidth=0.5)
            ax1.spines['bottom'].set_color(self.colors['grid'])
            ax1.spines['top'].set_color(self.colors['grid'])
            ax1.spines['right'].set_color(self.colors['grid'])
            ax1.spines['left'].set_color(self.colors['grid'])
            
            legend1 = ax1.legend(loc='upper left', framealpha=0.9)
            legend1.get_frame().set_facecolor(self.colors['background'])
            for text in legend1.get_texts():
                text.set_color(self.colors['text'])
            
            # Equity curve chart
            ax2.set_facecolor(self.colors['background'])
            return_pct = self._extract_return_percentage(metrics['Net Profit'])
            equity_color = self.colors['profit_green'] if return_pct > 0 else self.colors['loss_red']
            
            ax2.plot(result.index, result['cumulative'], 
                    color=equity_color, linewidth=3, alpha=0.9,
                    label=f'{strategy_name.upper()} Equity ({return_pct:+.2f}%)')
            
            # Fill area under equity curve
            ax2.fill_between(result.index, 1, result['cumulative'], 
                           color=equity_color, alpha=0.1)
            
            ax2.set_title(f'{strategy_name.upper()} Strategy - Equity Growth', 
                         color=self.colors['text'], fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('Time Period', color=self.colors['text'], fontsize=11)
            ax2.set_ylabel('Cumulative Return', color=self.colors['text'], fontsize=11)
            ax2.tick_params(colors=self.colors['text'])
            ax2.grid(True, alpha=0.2, color=self.colors['grid'], linewidth=0.5)
            ax2.spines['bottom'].set_color(self.colors['grid'])
            ax2.spines['top'].set_color(self.colors['grid'])
            ax2.spines['right'].set_color(self.colors['grid'])
            ax2.spines['left'].set_color(self.colors['grid'])
            
            legend2 = ax2.legend(loc='upper left', framealpha=0.9)
            legend2.get_frame().set_facecolor(self.colors['background'])
            for text in legend2.get_texts():
                text.set_color(self.colors['text'])
            
        else:
            # Multiple strategies comparison with TradingView styling
            fig = plt.figure(figsize=(18, 12))
            fig.patch.set_facecolor(self.colors['background'])
            
            # Equity curves comparison
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_facecolor(self.colors['background'])
            
            # TradingView-inspired color palette for multiple lines
            tv_colors = ['#2962FF', '#089981', '#F23645', '#FF9800', '#9C27B0', '#795548', '#E91E63']
            
            for i, (strategy_name, (result, metrics)) in enumerate(results_dict.items()):
                return_pct = self._extract_return_percentage(metrics['Net Profit'])
                color = tv_colors[i % len(tv_colors)]
                
                ax1.plot(result.index, result['cumulative'], 
                        label=f"{strategy_name.upper()} ({return_pct:+.2f}%)", 
                        linewidth=2.5, color=color, alpha=0.9)
            
            ax1.set_title(f'Multi-Strategy Performance Comparison - {self.symbol}', 
                         color=self.colors['text'], fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylabel('Cumulative Return Multiplier', color=self.colors['text'], fontsize=12)
            ax1.tick_params(colors=self.colors['text'])
            ax1.grid(True, alpha=0.2, color=self.colors['grid'], linewidth=0.5)
            ax1.spines['bottom'].set_color(self.colors['grid'])
            ax1.spines['top'].set_color(self.colors['grid'])
            ax1.spines['right'].set_color(self.colors['grid'])
            ax1.spines['left'].set_color(self.colors['grid'])
            
            legend1 = ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
            legend1.get_frame().set_facecolor(self.colors['background'])
            for text in legend1.get_texts():
                text.set_color(self.colors['text'])
            
            # Performance metrics bar chart
            ax2 = plt.subplot(2, 1, 2)
            ax2.set_facecolor(self.colors['background'])
            
            strategies = list(results_dict.keys())
            returns = [self._extract_return_percentage(metrics['Net Profit']) for _, (_, metrics) in results_dict.items()]
            sharpe_ratios = [self._extract_numeric_value(metrics['Sharpe Ratio']) for _, (_, metrics) in results_dict.items()]
            
            x = range(len(strategies))
            width = 0.35
            
            # Color bars based on performance
            return_colors = [self.colors['profit_green'] if r > 0 else self.colors['loss_red'] for r in returns]
            sharpe_colors = [self.colors['equity_line'] for _ in sharpe_ratios]
            
            bars1 = ax2.bar([i - width/2 for i in x], returns, width, 
                           label='Total Return (%)', alpha=0.8, color=return_colors, 
                           edgecolor=self.colors['text'], linewidth=0.5)
            bars2 = ax2.bar([i + width/2 for i in x], [s*10 for s in sharpe_ratios], width, 
                           label='Sharpe Ratio (x10)', alpha=0.7, color=sharpe_colors,
                           edgecolor=self.colors['text'], linewidth=0.5)
            
            ax2.set_xlabel('Trading Strategy', color=self.colors['text'], fontsize=12)
            ax2.set_ylabel('Performance Metrics', color=self.colors['text'], fontsize=12)
            ax2.set_title('Strategy Performance Metrics Comparison', 
                         color=self.colors['text'], fontsize=14, fontweight='bold', pad=15)
            ax2.set_xticks(x)
            ax2.set_xticklabels([s.upper() for s in strategies], rotation=45, ha='right')
            ax2.tick_params(colors=self.colors['text'])
            ax2.grid(True, alpha=0.2, color=self.colors['grid'], linewidth=0.5)
            ax2.spines['bottom'].set_color(self.colors['grid'])
            ax2.spines['top'].set_color(self.colors['grid'])
            ax2.spines['right'].set_color(self.colors['grid'])
            ax2.spines['left'].set_color(self.colors['grid'])
            
            legend2 = ax2.legend(framealpha=0.9, fontsize=10)
            legend2.get_frame().set_facecolor(self.colors['background'])
            for text in legend2.get_texts():
                text.set_color(self.colors['text'])
        
        plt.tight_layout(pad=3)
        plt.show()

    def run(self):
        """Run backtest based on strategy parameter"""
        results_dict = {}
        
        if self.strategy_param == 'all':
            print(f"Running ALL strategies on {self.symbol} with respective timeframes")
            
            # Run all strategies
            for strategy in ['scalping', 'daily', 'swing', 'golden', 'goldstorm', 'vwap', 'multi_rsi_ema']:
                timeframe = self._get_timeframe(strategy)
                self._load_data(timeframe)
                print(f"Data period for {strategy}: {self.data.index[0]} to {self.data.index[-1]}")
                result, metrics = self._run_single_strategy(strategy)
                results_dict[strategy] = (result, metrics)
            
            # Print comparison summary
            print(f"\n{'='*80}")
            print("🏆 STRATEGY COMPARISON SUMMARY")
            print(f"{'='*80}")
            print(f"{'Strategy':<12} {'Net Profit':<18} {'Trades':<8} {'Win %':<8} {'Profit Factor':<12} {'Sharpe':<8}")
            print("-" * 80)
            
            for strategy, (_, metrics) in results_dict.items():
                return_pct = self._extract_return_percentage(metrics['Net Profit'])
                profit_factor = self._extract_numeric_value(metrics['Profit Factor'])
                sharpe = self._extract_numeric_value(metrics['Sharpe Ratio'])
                
                print(f"{strategy.upper():<12} {return_pct:+6.2f}%{'':<10} "
                      f"{metrics['Total Trades']:<8} {metrics['Percent Profitable']:<8} "
                      f"{profit_factor:<12.3f} {sharpe:<8.3f}")
                
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

def run_backtest(strategy_param='all', symbol='XAUUSD', chart_style='light', use_ml=False):
    """
    Convenience function to run backtest
    
    Args:
        strategy_param: 'all', 'daily', 'swing', 'scalping'
        symbol: Trading symbol (default: 'XAUUSD')
        chart_style: 'light' or 'dark' for TradingView-style themes (default: 'light')
        use_ml: Use trained ML model for signals (default: False)
    
    Returns:
        Dictionary with results for each strategy
    """
    runner = BacktestRunner(strategy_param=strategy_param, symbol=symbol, chart_style=chart_style, use_ml=use_ml)
    return runner.run()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MT5 Gold AI Trader Backtest with TradingView-Style Charts')
    parser.add_argument('--strategy', '-s', choices=['all', 'daily', 'swing', 'scalping', 'golden', 'goldstorm', 'goldstormv2', 'vwap', 'multi_rsi_ema'], 
                       default='all', help='Strategy to run (default: all)')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol (default: XAUUSD)')
    parser.add_argument('--style', choices=['light', 'dark'], default='light', 
                       help='Chart style theme: light (default) or dark (TradingView-style)')
    parser.add_argument('--use_ml', action='store_true', help='Use trained ML model for signals (default: False)')
    
    args = parser.parse_args()
    
    style_emoji = "🌙" if args.style == 'dark' else "☀️"
    print(f"MT5 Gold AI Trader - Backtest Runner")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Symbol: {args.symbol}")
    print(f"Chart Style: {style_emoji} {args.style.upper()} theme")
    print(f"Use ML: {args.use_ml}")
    
    try:
        results = run_backtest(strategy_param=args.strategy, symbol=args.symbol, chart_style=args.style, use_ml=args.use_ml)
        print(f"\nBacktest completed successfully!")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
