import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
import threading

@dataclass
class BotConfig:
    """Configuration parameters for GoldStorm bot"""
    symbol: str = "XAUUSD"
    timeframe: int = mt5.TIMEFRAME_M15
    min_deposit: float = 300.0
    risk_percentage: float = 2.0  # Risk per trade as % of balance
    fixed_lot_size: Optional[float] = None  # If None, use dynamic sizing
    max_positions: int = 5  # Maximum pyramiding positions
    volatility_period: int = 20  # Period for volatility calculation
    momentum_period: int = 14  # Period for momentum calculation
    trailing_stop_points: int = 200  # Trailing stop in points
    min_volatility_threshold: float = 0.5  # Minimum volatility to trade
    pyramid_distance_points: int = 100  # Distance between pyramid levels
    magic_number: int = 101001  # Unique identifier for GoldStorm bot trades

class VolatilityAnalyzer:
    """Analyzes market volatility across multiple timeframes"""
    
    def __init__(self, symbol: str, config: BotConfig):
        self.symbol = symbol
        self.config = config
        
    def calculate_atr(self, timeframe: int, period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            return 0.0
            
        df = pd.DataFrame(rates)
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
        
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        atr = df['true_range'].rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def get_multi_timeframe_volatility(self) -> Dict[str, float]:
        """Get volatility across multiple timeframes"""
        timeframes = {
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4
        }
        
        volatility = {}
        for name, tf in timeframes.items():
            volatility[name] = self.calculate_atr(tf, self.config.volatility_period)
            
        return volatility
    
    def is_high_volatility_period(self) -> bool:
        """Determine if current market conditions show high volatility"""
        current_atr = self.calculate_atr(self.config.timeframe)
        historical_atr = self.calculate_atr(self.config.timeframe, 50)
        
        if historical_atr == 0:
            return False
            
        volatility_ratio = current_atr / historical_atr
        return volatility_ratio > self.config.min_volatility_threshold

class MomentumAnalyzer:
    """Analyzes price momentum for entry signals"""
    
    def __init__(self, symbol: str, config: BotConfig):
        self.symbol = symbol
        self.config = config
    
    def calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI for momentum analysis"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.config.timeframe, 0, period * 2)
        if rates is None or len(rates) < period * 2:
            return 50.0
            
        df = pd.DataFrame(rates)
        df['price_change'] = df['close'].diff()
        df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
        df['loss'] = -df['price_change'].where(df['price_change'] < 0, 0)
        
        avg_gain = df['gain'].rolling(window=period).mean()
        avg_loss = df['loss'].rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def get_momentum_signal(self) -> str:
        """Get momentum direction signal"""
        rsi = self.calculate_rsi(self.config.momentum_period)
        
        # Get price movement over last few candles
        rates = mt5.copy_rates_from_pos(self.symbol, self.config.timeframe, 0, 5)
        if rates is None or len(rates) < 5:
            return "NEUTRAL"
            
        df = pd.DataFrame(rates)
        price_change = df['close'].iloc[-1] - df['close'].iloc[-5]
        
        # Determine momentum direction
        if rsi > 70 and price_change > 0:
            return "STRONG_BULLISH"
        elif rsi < 30 and price_change < 0:
            return "STRONG_BEARISH"
        elif rsi > 50 and price_change > 0:
            return "BULLISH"
        elif rsi < 50 and price_change < 0:
            return "BEARISH"
        else:
            return "NEUTRAL"

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: BotConfig):
        self.config = config
    
    def calculate_position_size(self, stop_loss_points: int) -> float:
        """Calculate position size based on risk management"""
        if self.config.fixed_lot_size:
            return self.config.fixed_lot_size
            
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01
            
        balance = account_info.balance
        risk_amount = balance * (self.config.risk_percentage / 100)
        
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return 0.01
            
        point = symbol_info.point
        tick_value = symbol_info.trade_tick_value
        
        if tick_value == 0 or point == 0:
            return 0.01
            
        lot_size = risk_amount / (stop_loss_points * point * tick_value)
        
        # Ensure lot size is within limits
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_step = symbol_info.volume_step
        
        lot_size = max(min_lot, min(max_lot, lot_size))
        lot_size = round(lot_size / lot_step) * lot_step
        
        return lot_size
    
    def get_existing_positions_count(self) -> int:
        """Count existing positions opened by this bot"""
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None:
            return 0
            
        bot_positions = [pos for pos in positions if pos.magic == self.config.magic_number]
        return len(bot_positions)
    
    def should_add_pyramid_position(self, direction: str) -> bool:
        """Check if we should add a pyramid position"""
        current_positions = self.get_existing_positions_count()
        
        if current_positions >= self.config.max_positions:
            return False
            
        # Check if existing positions are in the same direction and profitable
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None:
            return True
            
        bot_positions = [pos for pos in positions if pos.magic == self.config.magic_number]
        
        if not bot_positions:
            return True
            
        # Check if all positions are in same direction and profitable
        for pos in bot_positions:
            if direction == "BUY" and pos.type != mt5.ORDER_TYPE_BUY:
                return False
            if direction == "SELL" and pos.type != mt5.ORDER_TYPE_SELL:
                return False
            if pos.profit <= 0:
                return False
                
        return True

class TrailingStopManager:
    """Manages trailing stops for open positions"""
    
    def __init__(self, config: BotConfig):
        self.config = config
    
    def update_trailing_stops(self):
        """Update trailing stops for all bot positions"""
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None:
            return
            
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return
            
        point = symbol_info.point
        
        for position in positions:
            if position.magic != self.config.magic_number:
                continue
                
            if position.type == mt5.ORDER_TYPE_BUY:
                # For buy positions, move stop loss up
                current_price = mt5.symbol_info_tick(self.config.symbol).bid
                new_sl = current_price - self.config.trailing_stop_points * point
                
                if position.sl == 0 or new_sl > position.sl:
                    self._modify_position(position.ticket, new_sl, position.tp)
                    
            elif position.type == mt5.ORDER_TYPE_SELL:
                # For sell positions, move stop loss down
                current_price = mt5.symbol_info_tick(self.config.symbol).ask
                new_sl = current_price + self.config.trailing_stop_points * point
                
                if position.sl == 0 or new_sl < position.sl:
                    self._modify_position(position.ticket, new_sl, position.tp)
    
    def _modify_position(self, ticket: int, sl: float, tp: float):
        """Modify position with new SL/TP"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        
        mt5.order_send(request)

class GoldStormBot:
    """Main GoldStorm trading bot class"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.volatility_analyzer = VolatilityAnalyzer(config.symbol, config)
        self.momentum_analyzer = MomentumAnalyzer(config.symbol, config)
        self.risk_manager = RiskManager(config)
        self.trailing_stop_manager = TrailingStopManager(config)
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('goldstorm_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize MT5 connection and bot"""
        if not mt5.initialize():
            self.logger.error("Failed to initialize MT5")
            return False
            
        # Check if symbol is available
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {self.config.symbol} not found")
            return False
            
        if not symbol_info.visible:
            if not mt5.symbol_select(self.config.symbol, True):
                self.logger.error(f"Failed to select symbol {self.config.symbol}")
                return False
        
        # Check account balance
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account information")
            return False
            
        if account_info.balance < self.config.min_deposit:
            self.logger.warning(f"Account balance {account_info.balance} below minimum {self.config.min_deposit}")
        
        self.logger.info(f"GoldStorm bot initialized for {self.config.symbol}")
        return True
    
    def analyze_market_conditions(self) -> Dict[str, any]:
        """Analyze current market conditions"""
        # Check volatility
        is_high_volatility = self.volatility_analyzer.is_high_volatility_period()
        volatility_data = self.volatility_analyzer.get_multi_timeframe_volatility()
        
        # Get momentum signal
        momentum_signal = self.momentum_analyzer.get_momentum_signal()
        
        # Get current price data
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return {"should_trade": False}
        
        return {
            "should_trade": is_high_volatility and momentum_signal != "NEUTRAL",
            "volatility_data": volatility_data,
            "momentum_signal": momentum_signal,
            "current_bid": tick.bid,
            "current_ask": tick.ask,
            "is_high_volatility": is_high_volatility
        }
    
    def calculate_entry_levels(self, direction: str, current_price: float) -> Dict[str, float]:
        """Calculate entry, stop loss, and take profit levels"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        point = symbol_info.point
        
        atr = self.volatility_analyzer.calculate_atr(self.config.timeframe)
        
        if direction == "BUY":
            entry_price = current_price
            stop_loss = entry_price - (atr * 1.5)  # 1.5 ATR stop loss
            take_profit = entry_price + (atr * 3.0)  # 3.0 ATR take profit (2:1 RR)
        else:  # SELL
            entry_price = current_price
            stop_loss = entry_price + (atr * 1.5)
            take_profit = entry_price - (atr * 3.0)
        
        return {
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr
        }
    
    def open_position(self, direction: str, levels: Dict[str, float]) -> bool:
        """Open a new position"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        point = symbol_info.point
        
        # Calculate position size
        sl_points = abs(levels["entry"] - levels["stop_loss"]) / point
        lot_size = self.risk_manager.calculate_position_size(int(sl_points))
        
        # Prepare order request
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = levels["entry"]
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": levels["stop_loss"],
            "tp": levels["take_profit"],
            "magic": self.config.magic_number,
            "comment": f"GoldStorm_{direction}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Failed to open {direction} position: {result.comment}")
            return False
        
        self.logger.info(f"Opened {direction} position: {lot_size} lots at {price}")
        return True
    
    def check_pyramid_opportunity(self, market_data: Dict[str, any]) -> Optional[str]:
        """Check if we should add a pyramid position"""
        momentum = market_data["momentum_signal"]
        
        if momentum in ["STRONG_BULLISH", "BULLISH"]:
            if self.risk_manager.should_add_pyramid_position("BUY"):
                return "BUY"
        elif momentum in ["STRONG_BEARISH", "BEARISH"]:
            if self.risk_manager.should_add_pyramid_position("SELL"):
                return "SELL"
                
        return None
    
    def run_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Analyze market conditions
            market_data = self.analyze_market_conditions()
            
            if not market_data["should_trade"]:
                return
            
            momentum = market_data["momentum_signal"]
            current_price = (market_data["current_bid"] + market_data["current_ask"]) / 2
            
            # Determine trading direction
            direction = None
            if momentum in ["STRONG_BULLISH", "BULLISH"]:
                direction = "BUY"
                entry_price = market_data["current_ask"]
            elif momentum in ["STRONG_BEARISH", "BEARISH"]:
                direction = "SELL"
                entry_price = market_data["current_bid"]
            
            if direction is None:
                return
            
            # Check if we should open a new position or add to existing
            current_positions = self.risk_manager.get_existing_positions_count()
            
            if current_positions == 0:
                # Open initial position
                levels = self.calculate_entry_levels(direction, entry_price)
                self.open_position(direction, levels)
                
            else:
                # Check for pyramid opportunity
                pyramid_direction = self.check_pyramid_opportunity(market_data)
                if pyramid_direction:
                    levels = self.calculate_entry_levels(pyramid_direction, entry_price)
                    self.open_position(pyramid_direction, levels)
            
            # Update trailing stops
            self.trailing_stop_manager.update_trailing_stops()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def start(self):
        """Start the trading bot"""
        if not self.initialize():
            return
        
        self.running = True
        self.logger.info("GoldStorm bot started")
        
        while self.running:
            try:
                self.run_trading_cycle()
                time.sleep(30)  # Wait 30 seconds between cycles
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait longer after error
        
        mt5.shutdown()
        self.logger.info("GoldStorm bot stopped")
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False

# Example usage and configuration
def main():
    # Configure the bot
    config = BotConfig(
        symbol="XAUUSD",
        timeframe=mt5.TIMEFRAME_M15,
        min_deposit=300.0,
        risk_percentage=2.0,  # 2% risk per trade
        fixed_lot_size=None,  # Use dynamic sizing
        max_positions=3,  # Maximum 3 pyramid positions
        volatility_period=20,
        momentum_period=14,
        trailing_stop_points=200,  # 20 pips trailing stop
        min_volatility_threshold=0.8,  # Higher threshold for more selective trading
        pyramid_distance_points=100,  # 10 pips between pyramid levels
        magic_number=101001
    )
    
    # Create and start the bot
    bot = GoldStormBot(config)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.stop()

if __name__ == "__main__":
    main()
