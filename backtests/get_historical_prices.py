import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import argparse

# MT5 Credentials - REPLACE WITH YOUR ACTUAL DETAILS IF NEEDED
MT5_ACCOUNT = 98462975  # Your MT5 account number
MT5_PASSWORD = "@3VsItUl"  # Your MT5 password
MT5_SERVER = "MetaQuotes-Demo"  # Your MT5 server name (e.g., MetaQuotes-Demo for demo)

# Trading parameters (default values from the original script)
# SYMBOL = "XAUUSD"  # Gold symbol in MT5
# COUNT = 1000  # Number of historical bars to fetch
# TIMEFRAME = mt5.TIMEFRAME_D1  # D1 timeframe (1 day)

def initialize_mt5(symbol, account=None, password=None, server=None):
    """Initialize MT5 connection."""
    if account is None:
        account = MT5_ACCOUNT
    if password is None:
        password = MT5_PASSWORD
    if server is None:
        server = MT5_SERVER
    
    if not mt5.initialize():
        print("MT5 initialize() failed")
        return False
    
    if not mt5.login(account, password=password, server=server):
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"MT5 symbol_select({symbol}) failed")
        mt5.shutdown()
        return False
    
    print(f"MT5 connected to account {account} on {server}")
    return True

def get_historical_prices(symbol, count, timeframe):
    """Fetch historical prices from MT5 and return as DataFrame."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        print("Failed to fetch rates")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # Keep all columns: time, open, high, low, close, tick_volume, spread, real_volume
    return df

def save_to_csv(df, symbol, tf_str, filename=None):
    """Save DataFrame to CSV file."""
    if filename is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{tf_str}_{now}.csv"
    if df is not None and not df.empty:
        df.to_csv(filename)
        print(f"Historical prices saved to {filename}")
    else:
        print("No data to save")

def main():
    """Main function to fetch and save historical prices."""
    parser = argparse.ArgumentParser(description='Fetch historical prices from MT5')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol to fetch (default: XAUUSD)')
    parser.add_argument('--timeframe', type=str, default='M1', help='Timeframe: M1, M5, M15, D1, H4 (default: M1)')
    parser.add_argument('--count', type=int, default=1000, help='Number of bars to fetch (default: 1000)')
    args = parser.parse_args()

    symbol = args.symbol
    tf_str = args.timeframe.upper()
    count = args.count

    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    if tf_str not in timeframe_map:
        print(f"Invalid timeframe: {tf_str}. Supported: M1, M5, M15, D1")
        return
    timeframe = timeframe_map[tf_str]

    if not initialize_mt5(symbol):
        return
    
    try:
        df = get_historical_prices(symbol, count, timeframe)
        if df is not None:
            print(f"Fetched {len(df)} historical prices for {symbol} on {tf_str}")
            print(df.head())  # Preview the data
            save_to_csv(df, symbol, tf_str)
        else:
            print("Failed to fetch historical prices")
    finally:
        mt5.shutdown()
        print("MT5 connection closed")

if __name__ == "__main__":
    main()
