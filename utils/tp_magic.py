import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

class TPMagicMode(Enum):
    BACKTEST = "backtest"
    LIVE = "live"

class TPMagic:
    """
    TPMagic: Universal Multi-TP Management System
    
    Handles configurable 3-level TP with partial closes and trailing SL based on scenarios.
    Designed for use across all trading strategies.
    
    Default levels (configurable):
    - TP1: 16 pips (1/3 close, trail SL to 80% of TP1)
    - TP2: 24 pips (1/3 close, trail SL to 80% TP2; if from TP1, close TP2 at 80%, TP1 at 60%)
    - TP3: 32 pips (close all)
    
    Scenarios:
    1. Hit TP1 -> close 1/3, open TP2 portion
    2. After TP1, retrace -> close at 80% TP1
    3. Hit TP2 -> close next 1/3, trail; if retrace, close TP2@80% TP2, then TP1@60% TP2
    4. Hit TP3 -> close all; if retrace, close @80% TP3, then remaining @60% TP3
    
    Unified for backtest (simulates returns) and live (manages MT5 positions).
    """
    
    def __init__(self, tp1_pips: float, tp2_pips: float, tp3_pips: float, sl_pips: float,
                 symbol: str = "XAUUSD", pip_size: float = 0.1, mode: TPMagicMode = TPMagicMode.BACKTEST,
                 initial_lot: float = 0.01, logger=None):
        self.tp1_pips = tp1_pips
        self.tp2_pips = tp2_pips
        self.tp3_pips = tp3_pips
        self.sl_pips = sl_pips
        self.symbol = symbol
        self.pip_size = pip_size
        self.mode = mode
        self.initial_lot = initial_lot
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.is_open = False
        self.direction = 0  # 1 buy, -1 sell
        self.entry_price = 0.0
        self.total_lot = 0.0
        self.remaining_lot = 0.0
        self.stage = 0  # 0: initial, 1: after TP1, 2: after TP2
        self.current_sl = 0.0
        self.portions = []  # List of {'level': str, 'lot': float, 'sl': float}
        self.entry_time = None
        
        # For backtest simulation
        self.exposure = 0.0  # Remaining exposure fraction
        
        if self.mode == TPMagicMode.LIVE:
            if not mt5.initialize():
                raise ValueError("MT5 not initialized for live mode")
    
    def open_position(self, direction: int, entry_price: float, lot_size: Optional[float] = None) -> bool:
        """Open initial position"""
        if self.is_open:
            self.logger.warning("Position already open")
            return False
        
        lot_size = lot_size or self.initial_lot
        self.direction = direction
        self.entry_price = entry_price
        self.total_lot = lot_size
        self.remaining_lot = lot_size
        self.exposure = 1.0
        self.stage = 0
        self.current_sl = entry_price - self.sl_pips * self.pip_size * direction
        self.portions = [
            {'level': 'TP1', 'lot': lot_size / 3, 'sl': self.current_sl},
            {'level': 'TP2', 'lot': lot_size / 3, 'sl': self.current_sl},
            {'level': 'TP3', 'lot': lot_size / 3, 'sl': self.current_sl}
        ]
        self.entry_time = pd.Timestamp.now()
        self.is_open = True
        
        if self.mode == TPMagicMode.LIVE:
            # Send initial order without TP/SL (manage manually)
            action = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": action,
                "price": entry_price,
                "sl": 0,  # No initial SL, manage manually
                "tp": 0,
                "deviation": 20,
                "magic": 234567,  # Unique magic for TPMagic
                "comment": "TPMagic Initial",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Live position opened: {direction} {lot_size} lots at {entry_price}")
                return True
            else:
                self.logger.error(f"Failed to open live position: {result}")
                self._reset()
                return False
        else:
            self.logger.info(f"Backtest position opened: {direction} {lot_size} lots at {entry_price}")
            return True
    
    def _reset(self):
        """Reset state"""
        self.is_open = False
        self.direction = 0
        self.entry_price = 0.0
        self.total_lot = 0.0
        self.remaining_lot = 0.0
        self.stage = 0
        self.current_sl = 0.0
        self.portions = []
        self.entry_time = None
        self.exposure = 0.0
    
    def update(self, high: float, low: float, close: float, asset_return: float = 0.0) -> Dict:
        """
        Update position state and return actions/returns.
        
        For backtest: Returns {'strategy_returns': float, 'position': float, 'actions': List}
        For live: Returns {'actions': List[Dict], 'current_sl': float} where actions are close commands
        """
        if not self.is_open:
            return {'strategy_returns': 0.0, 'position': 0.0, 'actions': []} if self.mode == TPMagicMode.BACKTEST else {'actions': [], 'current_sl': 0.0}
        
        actions = []
        strategy_return = 0.0
        realized_pnl = 0.0
        closed_this_bar = False
        
        # Calculate TP levels
        tp1_price = self.entry_price + self.tp1_pips * self.pip_size * self.direction
        tp2_price = self.entry_price + self.tp2_pips * self.pip_size * self.direction
        tp3_price = self.entry_price + self.tp3_pips * self.pip_size * self.direction
        
        # Check hits (direction-aware)
        hit_tp3 = (high >= tp3_price if self.direction == 1 else low <= tp3_price)
        hit_tp2 = (high >= tp2_price if self.direction == 1 else low <= tp2_price) and not hit_tp3
        hit_tp1 = (high >= tp1_price if self.direction == 1 else low <= tp1_price) and not (hit_tp2 or hit_tp3)
        hit_sl = (low <= self.current_sl if self.direction == 1 else high >= self.current_sl)
        
        if hit_tp3:
            # Scenario 4: Close all at TP3
            pnl_pct = self.direction * (tp3_price - self.entry_price) / self.entry_price * self.exposure
            realized_pnl += pnl_pct
            if self.mode == TPMagicMode.BACKTEST:
                strategy_return = realized_pnl + (self.direction * asset_return * 0)  # No remaining
            actions.append({'type': 'close_all', 'price': tp3_price, 'lot': self.remaining_lot, 'reason': 'TP3'})
            if self.mode == TPMagicMode.LIVE:
                self._execute_close(self.remaining_lot, tp3_price)
            self._reset()
            closed_this_bar = True
        elif hit_tp2:
            if self.stage == 2:
                # Already at stage 2, close remaining at TP2? But stage 2 is after TP2
                pass  # Should not hit again
            elif self.stage == 1:
                # Close TP2 portion at TP2 (portions: [TP2, TP3])
                close_lot = self.portions[0]['lot']  # TP2 is now [0]
                pnl_pct = self.direction * (tp2_price - self.entry_price) / self.entry_price * (close_lot / self.total_lot)
                realized_pnl += pnl_pct
                self.remaining_lot -= close_lot
                self.exposure -= close_lot / self.total_lot
                self.portions.pop(0)  # Remove TP2
                # Trail SL for remaining (TP3) to 80% TP2
                dist2 = self.tp2_pips * self.pip_size
                new_sl = tp2_price - self.direction * (0.2 * dist2)
                self.current_sl = max(self.current_sl, new_sl) if self.direction == 1 else min(self.current_sl, new_sl)
                for p in self.portions:
                    p['sl'] = self.current_sl
                self.stage = 2
                actions.append({'type': 'partial_close', 'level': 'TP2', 'lot': close_lot, 'price': tp2_price, 'reason': 'TP2 hit'})
                if self.mode == TPMagicMode.LIVE:
                    self._execute_close(close_lot, tp2_price)
                closed_this_bar = True
            elif self.stage == 0:
                # Hit TP2 directly: close TP1 and TP2 portions
                close_lot1 = self.portions[0]['lot']
                tp1_price_local = self.entry_price + self.tp1_pips * self.pip_size * self.direction
                pnl1 = self.direction * (tp1_price_local - self.entry_price) / self.entry_price * (close_lot1 / self.total_lot)
                close_lot2 = self.portions[1]['lot']
                pnl2 = self.direction * (tp2_price - self.entry_price) / self.entry_price * (close_lot2 / self.total_lot)
                realized_pnl += pnl1 + pnl2
                self.remaining_lot -= (close_lot1 + close_lot2)
                self.exposure -= (close_lot1 + close_lot2) / self.total_lot
                # Remove TP1 and TP2 portions
                self.portions = self.portions[2:]
                # Trail SL for TP3 to 80% TP2
                dist2 = self.tp2_pips * self.pip_size
                new_sl = tp2_price - self.direction * (0.2 * dist2)
                self.current_sl = max(self.current_sl, new_sl) if self.direction == 1 else min(self.current_sl, new_sl)
                for p in self.portions:
                    p['sl'] = self.current_sl
                self.stage = 2
                actions.extend([
                    {'type': 'partial_close', 'level': 'TP1', 'lot': close_lot1, 'price': tp1_price_local, 'reason': 'TP2 direct TP1'},
                    {'type': 'partial_close', 'level': 'TP2', 'lot': close_lot2, 'price': tp2_price, 'reason': 'TP2 hit'}
                ])
                if self.mode == TPMagicMode.LIVE:
                    self._execute_close(close_lot1, tp1_price_local)
                    self._execute_close(close_lot2, tp2_price)
                closed_this_bar = True
        elif hit_tp1 and self.stage == 0:
            # Scenario 1 & 2: Close TP1 portion, trail SL to 80% TP1 for remaining
            close_lot = self.portions[0]['lot']
            pnl_pct = self.direction * (tp1_price - self.entry_price) / self.entry_price * (close_lot / self.total_lot)
            realized_pnl += pnl_pct
            self.remaining_lot -= close_lot
            self.exposure -= close_lot / self.total_lot
            # Remove TP1 portion
            self.portions.pop(0)
            # Trail SL for remaining to 80% TP1
            dist1 = self.tp1_pips * self.pip_size
            new_sl = tp1_price - self.direction * (0.2 * dist1)
            self.current_sl = max(self.current_sl, new_sl) if self.direction == 1 else min(self.current_sl, new_sl)
            for p in self.portions:
                p['sl'] = self.current_sl
            self.stage = 1
            actions.append({'type': 'partial_close', 'level': 'TP1', 'lot': close_lot, 'price': tp1_price, 'reason': 'TP1 hit'})
            if self.mode == TPMagicMode.LIVE:
                self._execute_close(close_lot, tp1_price)
            closed_this_bar = True
        
        # Handle SL hit if not closed by TP
        if hit_sl and self.remaining_lot > 0 and not closed_this_bar:
            # Close remaining at SL
            pnl_pct = self.direction * (self.current_sl - self.entry_price) / self.entry_price * self.exposure
            realized_pnl += pnl_pct
            actions.append({'type': 'close_sl', 'lot': self.remaining_lot, 'price': self.current_sl, 'reason': 'SL hit'})
            if self.mode == TPMagicMode.LIVE:
                self._execute_close(self.remaining_lot, self.current_sl)
            self._reset()
            closed_this_bar = True
        
        # Handle retracements for open positions (scenarios 2,3,4)
        if self.remaining_lot > 0 and not closed_this_bar:
            if self.stage == 1:  # After TP1, check retrace to 80% TP1
                retrace_level = self.entry_price + (0.8 * self.tp1_pips * self.pip_size * self.direction)
                hit_retrace = (low <= retrace_level if self.direction == 1 else high >= retrace_level)
                if hit_retrace:
                    # Scenario 2: Close remaining at 80% TP1
                    pnl_pct = self.direction * (retrace_level - self.entry_price) / self.entry_price * self.exposure
                    realized_pnl += pnl_pct
                    actions.append({'type': 'close_retrace', 'level': 'TP1_80', 'lot': self.remaining_lot, 'price': retrace_level, 'reason': 'Retrace after TP1'})
                    if self.mode == TPMagicMode.LIVE:
                        self._execute_close(self.remaining_lot, retrace_level)
                    self._reset()
            elif self.stage == 2:  # After TP2, remaining TP3, check retrace to 80% TP2 for TP3 close
                tp2_80 = self.entry_price + (0.8 * self.tp2_pips * self.pip_size * self.direction)
                tp2_60 = self.entry_price + (0.6 * self.tp2_pips * self.pip_size * self.direction)
                hit_tp2_80 = (low <= tp2_80 if self.direction == 1 else high >= tp2_80)
                if hit_tp2_80 and self.portions:  # Close remaining TP3 at 80% TP2
                    close_lot = self.remaining_lot
                    pnl_pct = self.direction * (tp2_80 - self.entry_price) / self.entry_price * self.exposure
                    realized_pnl += pnl_pct
                    self.remaining_lot = 0
                    self.exposure = 0
                    self.portions = []
                    actions.append({'type': 'close_retrace', 'level': 'TP3_at_TP2_80', 'lot': close_lot, 'price': tp2_80, 'reason': 'Retrace after TP2 80%'})
                    if self.mode == TPMagicMode.LIVE:
                        self._execute_close(close_lot, tp2_80)
                    self._reset()
                elif not hit_tp2_80:
                    # If not hit 80%, check 60% for full close (adjusted for scenario)
                    hit_tp2_60 = (low <= tp2_60 if self.direction == 1 else high >= tp2_60)
                    if hit_tp2_60 and self.remaining_lot > 0:
                        close_lot = self.remaining_lot
                        pnl_pct = self.direction * (tp2_60 - self.entry_price) / self.entry_price * self.exposure
                        realized_pnl += pnl_pct
                        actions.append({'type': 'close_retrace', 'level': 'TP2_60', 'lot': close_lot, 'price': tp2_60, 'reason': 'Retrace 60% TP2'})
                        if self.mode == TPMagicMode.LIVE:
                            self._execute_close(close_lot, tp2_60)
                        self._reset()
        
        # For backtest: unrealized + realized
        if self.mode == TPMagicMode.BACKTEST:
            if self.exposure > 0:
                unrealized = self.direction * asset_return * self.exposure
            else:
                unrealized = 0.0
            strategy_return = unrealized + realized_pnl
            position = self.direction * self.exposure if self.exposure > 0 else 0.0
            return {'strategy_returns': strategy_return, 'position': position, 'actions': actions}
        
        # For live: return actions and current SL (to update if needed)
        return {'actions': actions, 'current_sl': self.current_sl}
    
    def _execute_close(self, lot: float, price: float):
        """Execute partial close in live mode"""
        if lot <= 0:
            return
        
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            self.logger.warning("No position to close")
            return
        
        # Find our position (by magic)
        our_pos = next((p for p in positions if p.magic == 234567), None)
        if not our_pos:
            self.logger.warning("TPMagic position not found")
            return
        
        close_type = mt5.ORDER_TYPE_SELL if our_pos.type == 0 else mt5.ORDER_TYPE_BUY  # Opposite
        close_price = mt5.symbol_info_tick(self.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": close_type,
            "position": our_pos.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 234567,
            "comment": f"TPMagic Close {lot}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Live partial close: {lot} lots at ~{close_price}")
            self.remaining_lot -= lot
            if self.remaining_lot <= 0:
                self._reset()
        else:
            self.logger.error(f"Failed partial close: {result}")
    
    def get_state(self) -> Dict:
        """Get current state for logging/persistence"""
        return {
            'is_open': self.is_open,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'remaining_lot': self.remaining_lot,
            'stage': self.stage,
            'current_sl': self.current_sl,
            'portions': self.portions,
            'exposure': self.exposure
        }
    
    def close_all(self, reason: str = "Manual"):
        """Force close all"""
        if not self.is_open:
            return
        actions = [{'type': 'force_close', 'lot': self.remaining_lot, 'reason': reason}]
        if self.mode == TPMagicMode.LIVE:
            self._execute_close(self.remaining_lot, 0)  # Use market price
        self._reset()
        if self.mode == TPMagicMode.BACKTEST:
            return {'strategy_returns': 0.0, 'position': 0.0, 'actions': actions}
        return {'actions': actions, 'current_sl': 0.0}
