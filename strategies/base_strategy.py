#!/usr/bin/env python3
"""
Base Strategy Class

This class provides common functionality for all trading strategies including:
- Signal validation against sentiment
- Trade execution with proper logging
- Position monitoring
- Risk management
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals - must be implemented by each strategy"""
        pass
    
    @abstractmethod
    def get_strategy_config(self) -> Dict:
        """Get strategy configuration - must be implemented by each strategy"""
        pass
    
    def validate_signal_with_sentiment(self, signal: str, sentiment: str) -> bool:
        """
        Validate if a trading signal aligns with market sentiment
        
        Args:
            signal: 'buy' or 'sell'
            sentiment: 'bullish', 'bearish', or 'neutral'
            
        Returns:
            bool: True if signal is valid for the sentiment
        """
        # Trading logic:
        # - Buy only on bullish or neutral sentiment
        # - Sell only on bearish or neutral sentiment
        # - Never buy on bearish sentiment
        # - Never sell on bullish sentiment
        
        if signal == 'buy':
            return sentiment in ['bullish', 'neutral']
        elif signal == 'sell':
            return sentiment in ['bearish', 'neutral']
        else:
            return False
    
    def calculate_position_size(self, balance: float, entry_price: float, 
                              stop_loss: float, risk_percent: float = 2.0) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Account balance
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_percent: Risk percentage of balance (default 2%)
            
        Returns:
            float: Position size in lots
        """
        risk_amount = balance * (risk_percent / 100)
        price_difference = abs(entry_price - stop_loss)
        
        if price_difference == 0:
            return 0.01  # Minimum lot size
        
        # For XAUUSD: 1 lot = 100 oz, 1 pip = $1 for 0.01 lot
        lot_size = risk_amount / (price_difference * 100)
        
        # Clamp between 0.01 and 0.1
        return max(0.01, min(0.1, round(lot_size, 2)))
    
    def calculate_stop_take_levels(self, signal: str, entry_price: float, 
                                 sl_pips: int, tp_pips: int) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            signal: 'buy' or 'sell'
            entry_price: Entry price
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
            
        Returns:
            Tuple[float, float]: (stop_loss, take_profit)
        """
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            raise ValueError(f"Cannot get symbol info for {self.symbol}")
        
        point = symbol_info.point
        digits = symbol_info.digits
        
        if signal == 'buy':
            stop_loss = round(entry_price - (sl_pips * point), digits)
            take_profit = round(entry_price + (tp_pips * point), digits)
        else:  # sell
            stop_loss = round(entry_price + (sl_pips * point), digits)
            take_profit = round(entry_price - (tp_pips * point), digits)
        
        return stop_loss, take_profit
    
    def execute_trade(self, signal: str, sentiment: str, entry_price: float,
                     stop_loss: float, take_profit: float, lot_size: float,
                     strategy_name: str) -> Optional[object]:
        """
        Execute a trade with validation and logging
        
        Args:
            signal: 'buy' or 'sell'
            sentiment: Market sentiment
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            lot_size: Position size
            strategy_name: Name of the strategy
            
        Returns:
            MT5 order result or None if validation fails
        """
        # Validate signal against sentiment
        if not self.validate_signal_with_sentiment(signal, sentiment):
            self.logger.warning(f"Signal {signal} rejected due to conflicting sentiment ({sentiment}). "
                              f"Only buy on bullish/neutral, sell on bearish/neutral sentiment.")
            return None
        
        self.logger.info(f"Signal {signal} approved for {sentiment} sentiment")
        
        # Prepare order request
        action_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": action_type,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 100000,
            "comment": f"AI bot - {strategy_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        self.logger.info(f"Executing {signal.upper()} {lot_size} lots at {entry_price}, "
                        f"SL: {stop_loss}, TP: {take_profit}")
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Trade executed successfully: Ticket {result.order}")
            
            # Log trade with complete information
            self._log_trade(signal, sentiment, entry_price, lot_size, 
                          strategy_name, result.order)
            
            return result
        else:
            error_msg = f"Trade failed: {result.retcode if result else 'No result'}"
            if result:
                error_msg += f" - {result.comment}"
            self.logger.error(error_msg)
            return None
    
    def _log_trade(self, signal: str, sentiment: str, entry_price: float,
                  lot_size: float, strategy_name: str, ticket: int):
        """Log trade to JSON file"""
        try:
            trade_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "action": signal,
                "entry_price": entry_price,
                "exit_price": None,
                "volume": lot_size,
                "profit_usd": 0.0,
                "strategy": strategy_name,
                "sentiment": sentiment,
                "duration_minutes": 0,
                "ticket": ticket,
                "status": "open"
            }
            
            log_file = "logs/trade_logs.json"
            
            # Load existing trades
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    data = json.load(f)
                trades = data.get("trades", [])
            else:
                trades = []
            
            # Add new trade
            trades.append(trade_data)
            data = {"trades": trades}
            
            # Save to file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Trade logged with ticket {ticket}")
            
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")
    
    def get_market_price(self, signal: str) -> Optional[float]:
        """Get current market price for the signal"""
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            self.logger.error("Failed to get tick data")
            return None
        
        price = tick.ask if signal == 'buy' else tick.bid
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            return round(price, symbol_info.digits)
        return price
    
    def validate_stop_distances(self, entry_price: float, stop_loss: float, 
                               take_profit: float) -> bool:
        """Validate that stop distances meet broker requirements"""
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return False
        
        point = symbol_info.point
        min_distance = symbol_info.trade_stops_level * point if symbol_info.trade_stops_level > 0 else 0
        
        sl_distance = abs(stop_loss - entry_price)
        tp_distance = abs(take_profit - entry_price)
        
        if sl_distance < min_distance or tp_distance < min_distance:
            self.logger.error(f"Stop distances too small - SL: {sl_distance/point:.1f} pips, "
                            f"TP: {tp_distance/point:.1f} pips, Min required: {min_distance/point:.1f} pips")
            return False
        
        return True
