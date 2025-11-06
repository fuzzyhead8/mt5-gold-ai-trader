import pandas as pd
import numpy as np
import argparse

def analyze_backtest(csv_path, show_trades=False):
    """
    Accurate analysis of backtest CSV that matches the fixed backtest_runner metrics.
    Counts actual trades based on position changes and calculates P&L per trade.
    """
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    initial_capital = 10000
    
    # Assume 'position' column exists (1 long, -1 short, 0 flat)
    # If not, we can derive it, but from fixed runner, it should be there
    if 'position' not in df.columns:
        print("Warning: 'position' column not found. Deriving from signals.")
        position = 0
        positions = []
        for signal in df['signal']:
            if pd.isna(signal):
                positions.append(position)
                continue
            signal_str = str(signal).lower()
            if signal_str == 'buy' and position != 1:
                position = 1
            elif signal_str == 'sell' and position != -1:
                position = -1
            positions.append(position)
        df['position'] = positions
    
    # Detect trades mimicking the backtest_runner's logic: change on opposite signal
    trades = []
    current_trade = None
    current_position = 0  # 0 flat, 1 long, -1 short
    entry_equity = initial_capital
    current_equity = initial_capital
    
    for i in range(len(df)):
        signal = df['signal'].iloc[i]
        returns = df['strategy_returns'].iloc[i]
        position = df['position'].iloc[i]
        
        # Update current equity
        current_equity *= (1 + returns)
        
        if current_trade is None:
            entry_equity = current_equity
        
        if pd.isna(signal):
            continue
        
        signal_str = str(signal).lower()
        close_position = False
        new_trade_type = None
        
        if signal_str == 'buy' and current_position != 1:
            if current_position == -1:
                close_position = True
            current_position = 1
            new_trade_type = 'long'
        elif signal_str == 'sell' and current_position != -1:
            if current_position == 1:
                close_position = True
            current_position = -1
            new_trade_type = 'short'
        else:
            close_position = False
        
        if close_position and current_trade is not None:
            # Close current trade
            trade_returns_sum = df['strategy_returns'].iloc[current_trade['entry_index']:i].sum()
            pnl = trade_returns_sum * current_trade['entry_equity']
            duration = i - current_trade['entry_index']
            exit_price = df['close'].iloc[i]
            
            trades.append({
                'type': current_trade['type'],
                'entry_price': current_trade['entry_price'],
                'exit_price': exit_price,
                'entry_time': current_trade['entry_time'],
                'exit_time': df.index[i],
                'pnl': pnl,
                'duration': duration,
                'entry_equity': current_trade['entry_equity']
            })
            current_trade = None
        
        if (signal_str in ['buy', 'sell']) and current_trade is None and new_trade_type:
            # Start new trade
            current_trade = {
                'type': new_trade_type,
                'entry_price': df['close'].iloc[i],
                'entry_time': df.index[i],
                'entry_equity': entry_equity,
                'entry_index': i
            }
        
        # Handle explicit exit signals (if any)
        if signal_str in ['exit_buy', 'exit_sell'] and current_trade is not None:
            trade_returns_sum = df['strategy_returns'].iloc[current_trade['entry_index']:i+1].sum()
            pnl = trade_returns_sum * current_trade['entry_equity']
            duration = i - current_trade['entry_index'] + 1
            exit_price = df['close'].iloc[i]
            
            trades.append({
                'type': current_trade['type'],
                'entry_price': current_trade['entry_price'],
                'exit_price': exit_price,
                'entry_time': current_trade['entry_time'],
                'exit_time': df.index[i],
                'pnl': pnl,
                'duration': duration,
                'entry_equity': current_trade['entry_equity']
            })
            current_trade = None
            current_position = 0
    
    # Handle unclosed trade at end
    if current_trade is not None:
        i_end = len(df)
        trade_returns_sum = df['strategy_returns'].iloc[current_trade['entry_index']:i_end].sum()
        pnl = trade_returns_sum * current_trade['entry_equity']
        duration = i_end - current_trade['entry_index']
        exit_price = df['close'].iloc[-1]
        
        trades.append({
            'type': current_trade['type'],
            'entry_price': current_trade['entry_price'],
            'exit_price': exit_price,
            'entry_time': current_trade['entry_time'],
            'exit_time': df.index[-1],
            'pnl': pnl,
            'duration': duration,
            'entry_equity': current_trade['entry_equity']
        })
    
    # Calculate metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    losing_trades = len([t for t in trades if t['pnl'] < 0])
    percent_profitable = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
    net_profit = sum([t['pnl'] for t in trades])
    final_equity = initial_capital * df['cumulative'].iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    
    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = -gross_loss / losing_trades if losing_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    long_trades = [t for t in trades if t['type'] == 'long']
    short_trades = [t for t in trades if t['type'] == 'short']
    
    long_wins = len([t for t in long_trades if t['pnl'] > 0])
    short_wins = len([t for t in short_trades if t['pnl'] > 0])
    long_net = sum([t['pnl'] for t in long_trades])
    short_net = sum([t['pnl'] for t in short_trades])
    
    avg_bars = sum([t['duration'] for t in trades]) / total_trades if total_trades > 0 else 0
    
    # Output
    print('=== ACCURATE BACKTEST ANALYSIS ===')
    print(f'Total Trades: {total_trades}')
    print(f'Winning Trades: {winning_trades}')
    print(f'Losing Trades: {losing_trades}')
    print(f'Win Rate: {percent_profitable:.2f}%')
    print(f'Net Profit: ${net_profit:.2f} ({total_return:.2f}%)')
    print(f'Gross Profit: ${gross_profit:.2f}')
    print(f'Gross Loss: ${gross_loss:.2f}')
    print(f'Profit Factor: {profit_factor:.3f}')
    print(f'Average Win: ${avg_win:.2f}')
    print(f'Average Loss: ${avg_loss:.2f}')
    print(f'Avg Bars per Trade: {avg_bars:.0f}')
    
    print(f'\nLong Trades: {len(long_trades)}, Wins: {long_wins}, Net: ${long_net:.2f}')
    print(f'Short Trades: {len(short_trades)}, Wins: {short_wins}, Net: ${short_net:.2f}')
    
    if show_trades:
        # Print individual trades for transparency
        print(f'\n=== INDIVIDUAL TRADES ({total_trades} trades) ===')
        for idx, trade in enumerate(trades, 1):
            pnl_sign = '+' if trade['pnl'] > 0 else ''
            print(f"Trade {idx}: {trade['type'].upper()} | Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} @ ${trade['entry_price']:.2f} | "
                  f"Exit: {trade['exit_time'].strftime('%Y-%m-%d %H:%M')} @ ${trade['exit_price']:.2f} | "
                  f"Duration: {trade['duration']} bars | P&L: {pnl_sign}${trade['pnl']:.2f} ({trade['pnl']/trade['entry_equity']*100:.2f}%)")
    else:
        print(f'\nRun with --with_trades to see individual trade details.')
    
    # Signal distribution for reference (excluding holds)
    buy_signals = len(df[df['signal'] == 'buy'])
    sell_signals = len(df[df['signal'] == 'sell'])
    print(f'\nSignal Distribution (Active Signals Only):')
    print(f'Buy Signals: {buy_signals}')
    print(f'Sell Signals: {sell_signals}')
    print(f'Total Active Signals: {buy_signals + sell_signals}')
    
    final_cumulative = df['cumulative'].iloc[-1] if 'cumulative' in df.columns else 1 + df['strategy_returns'].sum()
    print(f'\nFinal Cumulative Return: {(final_cumulative - 1) * 100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accurate Backtest Analysis')
    parser.add_argument('--csv', default='backtests/multi_rsi_ema_backtest.csv', help='Path to CSV file')
    parser.add_argument('--with_trades', action='store_true', help='Show individual trades')
    args = parser.parse_args()
    
    analyze_backtest(args.csv, show_trades=args.with_trades)
