"""
XAU/USD M15 Trading Bot for MetaTrader 5
High-probability, low-risk strategy with strict risk controls

Requirements:
pip install MetaTrader5 pandas numpy ta-lib (or use custom indicators)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Optional, Tuple, List, Dict
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Connection
    MT5_LOGIN = 12345678  # Your MT5 account number
    MT5_PASSWORD = "YourPassword"
    MT5_SERVER = "YourBroker-Server"
    
    # Instrument
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M15
    MAGIC_NUMBER = 102002  # Unique identifier for VWAP bot trades
    
    # Strategy Parameters
    RISK_PER_TRADE = 0.0025  # 0.25% equity per trade
    MAX_TOTAL_RISK = 0.008  # 0.8% max total exposure
    MAX_OPEN_TRADES = 2
    
    # ATR-based Parameters (baseline values)
    STOP_LOSS_ATR_MULT = 1.0
    TP1_ATR_MULT = 0.8
    TP2_ATR_MULT = 1.6
    TRAILING_ATR_MULT = 0.6
    MIN_STOP_USD = 6.0
    
    # Indicator Periods
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    EMA_FAST = 50
    EMA_SLOW = 200
    
    # Entry Rules
    RSI_OVERSOLD = 40
    SIGNAL_VALIDITY_CANDLES = 2
    BREAKOUT_CANDLES = 3
    
    # Safety
    MAX_DRAWDOWN_PCT = 0.06  # Halt at 6% drawdown
    VOLATILITY_SPIKE_MULT = 3.0  # Circuit breaker
    NEWS_BUFFER_MIN = 10  # Minutes before/after news
    
    # Trading Hours (UTC) - London/NY overlap
    TRADE_START_HOUR = 12  # 12:00 UTC
    TRADE_END_HOUR = 20  # 20:00 UTC
    
    # Logging
    LOG_FILE = "xauusd_bot.log"
    TRADE_LOG_FILE = "trades.json"


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("XAU_Bot")


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class Indicators:
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate rolling 15-minute VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()
        return vwap
    
    @staticmethod
    def detect_engulfing(df: pd.DataFrame, index: int) -> bool:
        """Detect bullish engulfing pattern"""
        if index < 1:
            return False
        
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        
        # Bullish engulfing: current green candle engulfs previous red
        bullish = (
            previous['close'] < previous['open'] and  # Previous red
            current['close'] > current['open'] and    # Current green
            current['open'] <= previous['close'] and  # Opens at/below prev close
            current['close'] >= previous['open']      # Closes at/above prev open
        )
        return bullish
    
    @staticmethod
    def detect_rsi_divergence(df: pd.DataFrame, index: int, lookback: int = 3) -> bool:
        """Detect bullish RSI divergence (price lower low, RSI higher low)"""
        if index < lookback:
            return False
        
        price_slice = df['close'].iloc[index - lookback:index + 1]
        rsi_slice = df['rsi'].iloc[index - lookback:index + 1]
        
        # Find lows
        price_min_idx = price_slice.idxmin()
        rsi_min_idx = rsi_slice.idxmin()
        
        # Check if price made lower low but RSI made higher low
        if price_min_idx == price_slice.index[-1]:  # Recent price low
            prev_price_low = price_slice.iloc[:-1].min()
            prev_rsi_low = rsi_slice.iloc[:-1].min()
            
            current_price = price_slice.iloc[-1]
            current_rsi = rsi_slice.iloc[-1]
            
            if current_price < prev_price_low and current_rsi > prev_rsi_low:
                return True
        
        return False


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    def __init__(self):
        self.initial_equity = None
        self.peak_equity = None
        self.halted = False
        self.halt_until = None
        self.baseline_atr = None
        
    def initialize(self, equity: float):
        """Initialize equity tracking"""
        self.initial_equity = equity
        self.peak_equity = equity
        logger.info(f"Risk Manager initialized with equity: ${equity:.2f}")
    
    def update_equity(self, current_equity: float) -> bool:
        """Update equity and check drawdown. Returns True if trading allowed."""
        if self.peak_equity is None:
            self.peak_equity = current_equity
        
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
        
        # Check if halt expired
        if self.halted and self.halt_until:
            if datetime.now() >= self.halt_until:
                self.halted = False
                logger.info("Trading halt expired. Resuming.")
        
        # Check drawdown threshold
        if drawdown_pct >= Config.MAX_DRAWDOWN_PCT and not self.halted:
            self.halted = True
            self.halt_until = datetime.now() + timedelta(hours=24)
            logger.critical(
                f"DRAWDOWN LIMIT HIT: {drawdown_pct:.2%}. "
                f"Trading halted for 24 hours."
            )
            return False
        
        return not self.halted
    
    def calculate_position_size(
        self, 
        equity: float, 
        entry_price: float, 
        stop_price: float,
        symbol_info
    ) -> float:
        """Calculate position size based on risk per trade"""
        risk_amount = equity * Config.RISK_PER_TRADE
        stop_distance = abs(entry_price - stop_price)
        
        # Calculate lots
        point_value = symbol_info.trade_contract_size
        value_per_point = point_value * symbol_info.point
        
        position_size = risk_amount / (stop_distance * value_per_point)
        
        # Round to lot step
        lot_step = symbol_info.volume_step
        position_size = round(position_size / lot_step) * lot_step
        
        # Apply min/max limits
        position_size = max(symbol_info.volume_min, position_size)
        position_size = min(symbol_info.volume_max, position_size)
        
        return position_size
    
    def can_open_trade(self, current_positions: int, total_risk_pct: float) -> bool:
        """Check if new trade is allowed based on exposure limits"""
        if self.halted:
            return False
        
        if current_positions >= Config.MAX_OPEN_TRADES:
            logger.warning(f"Max open trades reached: {current_positions}")
            return False
        
        if total_risk_pct >= Config.MAX_TOTAL_RISK:
            logger.warning(f"Max total risk reached: {total_risk_pct:.2%}")
            return False
        
        return True
    
    def check_volatility_spike(self, current_atr: float) -> bool:
        """Circuit breaker for volatility spikes"""
        if self.baseline_atr is None:
            self.baseline_atr = current_atr
            return False
        
        if current_atr > self.baseline_atr * Config.VOLATILITY_SPIKE_MULT:
            logger.warning(
                f"Volatility spike detected! ATR: {current_atr:.2f} "
                f"vs baseline: {self.baseline_atr:.2f}"
            )
            return True
        
        # Update baseline slowly
        self.baseline_atr = self.baseline_atr * 0.95 + current_atr * 0.05
        return False


# ============================================================================
# TRADE LOGGER
# ============================================================================

class TradeLogger:
    def __init__(self, filename: str = Config.TRADE_LOG_FILE):
        self.filename = filename
    
    def log_trade(self, trade_data: Dict):
        """Append trade data to JSON log"""
        try:
            # Load existing
            try:
                with open(self.filename, 'r') as f:
                    trades = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                trades = []
            
            # Add new trade
            trade_data['timestamp'] = datetime.now().isoformat()
            trades.append(trade_data)
            
            # Save
            with open(self.filename, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")


# ============================================================================
# MAIN BOT CLASS
# ============================================================================

class XAUBot:
    def __init__(self):
        self.risk_manager = RiskManager()
        self.trade_logger = TradeLogger()
        self.symbol_info = None
        self.running = False
        self.last_signal_time = {}
        
    def connect(self) -> bool:
        """Connect to MetaTrader 5"""
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # Login
        authorized = mt5.login(
            Config.MT5_LOGIN,
            password=Config.MT5_PASSWORD,
            server=Config.MT5_SERVER
        )
        
        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        logger.info(f"Connected to MT5: {mt5.account_info()}")
        
        # Get symbol info
        self.symbol_info = mt5.symbol_info(Config.SYMBOL)
        if self.symbol_info is None:
            logger.error(f"Symbol {Config.SYMBOL} not found")
            return False
        
        # Enable symbol
        if not self.symbol_info.visible:
            if not mt5.symbol_select(Config.SYMBOL, True):
                logger.error(f"Failed to select {Config.SYMBOL}")
                return False
        
        # Initialize risk manager
        account = mt5.account_info()
        self.risk_manager.initialize(account.equity)
        
        logger.info(f"Bot initialized for {Config.SYMBOL}")
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def get_candles(self, count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch M15 candles and calculate indicators"""
        rates = mt5.copy_rates_from_pos(
            Config.SYMBOL,
            Config.TIMEFRAME,
            0,
            count
        )
        
        if rates is None or len(rates) == 0:
            logger.error("Failed to fetch candles")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate indicators
        df['atr'] = Indicators.calculate_atr(df, Config.ATR_PERIOD)
        df['rsi'] = Indicators.calculate_rsi(df, Config.RSI_PERIOD)
        df['ema_fast'] = Indicators.calculate_ema(df, Config.EMA_FAST)
        df['ema_slow'] = Indicators.calculate_ema(df, Config.EMA_SLOW)
        df['vwap'] = Indicators.calculate_vwap(df)
        
        return df
    
    def is_trading_hours(self) -> bool:
        """Check if within allowed trading hours"""
        now = datetime.utcnow()
        return Config.TRADE_START_HOUR <= now.hour < Config.TRADE_END_HOUR
    
    def get_trend(self, df: pd.DataFrame, index: int) -> str:
        """Determine trend: bullish, bearish, or neutral"""
        ema_fast = df['ema_fast'].iloc[index]
        ema_slow = df['ema_slow'].iloc[index]
        
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return "neutral"
        
        if ema_fast > ema_slow:
            return "bullish"
        elif ema_fast < ema_slow:
            return "bearish"
        else:
            return "neutral"
    
    def check_entry_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Check all entry rules and return signals"""
        signals = []
        index = len(df) - 1  # Current candle
        
        if index < Config.EMA_SLOW:  # Not enough data
            return signals
        
        trend = self.get_trend(df, index)
        if trend != "bullish":  # Only long trades per spec
            return signals
        
        current = df.iloc[index]
        atr = current['atr']
        
        if pd.isna(atr) or atr == 0:
            return signals
        
        # Entry Rule A: Pullback to VWAP + bullish engulfing
        if (
            abs(current['close'] - current['vwap']) / current['vwap'] < 0.002
            and Indicators.detect_engulfing(df, index)
        ):
            signals.append({
                'type': 'VWAP_PULLBACK',
                'entry_price': current['close'],
                'stop_loss': current['close'] - (atr * Config.STOP_LOSS_ATR_MULT),
                'tp1': current['close'] + (atr * Config.TP1_ATR_MULT),
                'tp2': current['close'] + (atr * Config.TP2_ATR_MULT),
                'reason': 'Pullback to VWAP with bullish engulfing',
                'atr': atr
            })
        
        # Entry Rule B: RSI oversold + divergence
        if (
            current['rsi'] < Config.RSI_OVERSOLD
            and Indicators.detect_rsi_divergence(df, index)
        ):
            signals.append({
                'type': 'RSI_DIVERGENCE',
                'entry_price': current['close'],
                'stop_loss': current['close'] - (atr * Config.STOP_LOSS_ATR_MULT),
                'tp1': current['close'] + (atr * Config.TP1_ATR_MULT),
                'tp2': current['close'] + (atr * Config.TP2_ATR_MULT),
                'reason': 'RSI oversold with bullish divergence',
                'atr': atr
            })
        
        # Entry Rule C: Breakout above 3-candle range
        if index >= Config.BREAKOUT_CANDLES:
            lookback = df.iloc[index - Config.BREAKOUT_CANDLES:index]
            range_high = lookback['high'].max()
            
            if current['close'] > range_high:
                # Check volume uptick (simplified: current > avg)
                avg_volume = lookback['tick_volume'].mean()
                if current['tick_volume'] > avg_volume * 1.2:
                    signals.append({
                        'type': 'BREAKOUT',
                        'entry_price': current['close'],
                        'stop_loss': current['close'] - (atr * 1.4),  # Wider stop
                        'tp1': current['close'] + (atr * Config.TP1_ATR_MULT),
                        'tp2': current['close'] + (atr * Config.TP2_ATR_MULT),
                        'reason': f'Breakout above {Config.BREAKOUT_CANDLES}-candle high',
                        'atr': atr
                    })
        
        return signals
    
    def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on signal"""
        account = mt5.account_info()
        
        # Risk checks
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        current_positions = len(positions) if positions else 0
        
        # Calculate total current risk
        total_risk = sum([
            abs(p.price_open - p.sl) * p.volume 
            for p in (positions or [])
        ])
        total_risk_pct = total_risk / account.equity if account.equity > 0 else 0
        
        if not self.risk_manager.can_open_trade(current_positions, total_risk_pct):
            return False
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            account.equity,
            signal['entry_price'],
            signal['stop_loss'],
            self.symbol_info
        )
        
        if position_size < self.symbol_info.volume_min:
            logger.warning("Position size too small")
            return False
        
        # Apply minimum stop
        stop_distance = signal['entry_price'] - signal['stop_loss']
        if stop_distance < Config.MIN_STOP_USD:
            signal['stop_loss'] = signal['entry_price'] - Config.MIN_STOP_USD
        
        # Prepare order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": Config.SYMBOL,
            "volume": position_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(Config.SYMBOL).ask,
            "sl": signal['stop_loss'],
            "tp": signal['tp1'],  # Set TP1 initially
            "deviation": 10,
            "magic": Config.MAGIC_NUMBER,
            "comment": f"XAU_Bot_{signal['type']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return False
        
        logger.info(
            f"✓ Trade executed: {signal['type']} | "
            f"Size: {position_size} | Entry: {signal['entry_price']:.2f} | "
            f"SL: {signal['stop_loss']:.2f} | TP1: {signal['tp1']:.2f}"
        )
        
        # Log trade
        self.trade_logger.log_trade({
            'signal_type': signal['type'],
            'reason': signal['reason'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'tp1': signal['tp1'],
            'tp2': signal['tp2'],
            'position_size': position_size,
            'atr': signal['atr'],
            'order_ticket': result.order,
            'deal_ticket': result.deal
        })
        
        return True
    
    def manage_positions(self, df: pd.DataFrame):
        """Manage open positions (trailing stop, partial close)"""
        positions = mt5.positions_get(
            symbol=Config.SYMBOL, 
            magic=Config.MAGIC_NUMBER
        )
        
        if not positions:
            return
        
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        for position in positions:
            # Check if TP1 hit (simplified: check if profit > TP1 distance)
            entry_price = position.price_open
            tp1_distance = current_atr * Config.TP1_ATR_MULT
            
            profit_distance = current_price - entry_price
            
            # If profit exceeds TP1, implement trailing stop
            if profit_distance >= tp1_distance:
                trailing_stop = current_price - (current_atr * Config.TRAILING_ATR_MULT)
                
                # Only update if new SL is higher
                if trailing_stop > position.sl:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": Config.SYMBOL,
                        "position": position.ticket,
                        "sl": trailing_stop,
                        "tp": position.tp,
                        "magic": Config.MAGIC_NUMBER
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(
                            f"Trailing stop updated for position {position.ticket}: "
                            f"New SL = {trailing_stop:.2f}"
                        )
    
    def run(self):
        """Main bot loop"""
        self.running = True
        logger.info("Bot started. Press Ctrl+C to stop.")
        
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                
                # Health check every 60 iterations (~15 minutes for 15s sleep)
                if iteration % 60 == 0:
                    if not mt5.terminal_info().connected:
                        logger.critical("MT5 connection lost!")
                        break
                
                # Update equity and check drawdown
                account = mt5.account_info()
                if not self.risk_manager.update_equity(account.equity):
                    logger.warning("Trading halted due to risk limits")
                    time.sleep(60)
                    continue
                
                # Check trading hours
                if not self.is_trading_hours():
                    time.sleep(60)
                    continue
                
                # Get data
                df = self.get_candles()
                if df is None:
                    time.sleep(15)
                    continue
                
                # Check volatility spike
                current_atr = df['atr'].iloc[-1]
                if self.risk_manager.check_volatility_spike(current_atr):
                    logger.warning("Volatility spike - skipping this cycle")
                    time.sleep(60)
                    continue
                
                # Manage existing positions
                self.manage_positions(df)
                
                # Check for new signals
                signals = self.check_entry_signals(df)
                
                for signal in signals:
                    # Check signal validity (not too old)
                    signal_key = f"{signal['type']}_{signal['entry_price']}"
                    if signal_key in self.last_signal_time:
                        age = time.time() - self.last_signal_time[signal_key]
                        if age > Config.SIGNAL_VALIDITY_CANDLES * 15 * 60:
                            continue
                    else:
                        self.last_signal_time[signal_key] = time.time()
                    
                    # Execute trade
                    self.execute_trade(signal)
                
                # Clean old signals
                current_time = time.time()
                self.last_signal_time = {
                    k: v for k, v in self.last_signal_time.items()
                    if current_time - v < 3600  # Keep for 1 hour
                }
                
                # Sleep (check every 15 seconds to catch new candles quickly)
                time.sleep(15)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
        finally:
            self.disconnect()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Start the bot"""
    bot = XAUBot()
    
    if not bot.connect():
        logger.error("Failed to connect. Exiting.")
        return
    
    try:
        bot.run()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
