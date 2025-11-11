"""
MT5 Connector for Gold AI Sonnet 4.5
"""

import MetaTrader5 as mt5
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class MT5Connector:
    """MT5 integration for Gold AI Sonnet"""

    def __init__(self):
        self.connected = False
        self.login = int(os.getenv('MT5_LOGIN', '0'))
        self.password = os.getenv('MT5_PASSWORD', '')
        self.server = os.getenv('MT5_SERVER', '')
        self.path = os.getenv('MT5_PATH', '')

    async def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            # Initialize MT5
            if not mt5.initialize(self.path):
                logger.error(f"MT5 initialization failed. Path: {self.path}")
                return False

            # Login to account
            if not mt5.login(self.login, self.password, self.server):
                logger.error(f"MT5 login failed. Login: {self.login}, Server: {self.server}")
                return False

            self.connected = True
            logger.info(f"Successfully connected to MT5. Account: {self.login}")
            return True

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    async def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.connected:
            return None

        try:
            info = mt5.account_info()
            if info is None:
                logger.error("Failed to get account info")
                return None

            return {
                'login': info.login,
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'margin_free': info.margin_free,
                'margin_level': info.margin_level,
                'leverage': info.leverage,
                'currency': info.currency
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def get_historical_data(self, symbol: str, timeframe: str,
                                bars: int = 100) -> Optional[pd.DataFrame]:
        """Get historical market data"""
        if not self.connected:
            return None

        try:
            # Map timeframe string to MT5 constants
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }

            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get historical data for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Rename columns to match our format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None

    async def get_positions(self) -> Optional[List[Dict]]:
        """Get open positions"""
        if not self.connected:
            return None

        try:
            positions = mt5.positions_get()

            if positions is None:
                return []

            result = []
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'direction': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'time': pos.time
                })

            return result

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return None

    async def place_order(self, symbol: str, direction: str, volume: float,
                         price: float, stop_loss: float = 0.0,
                         take_profit: float = 0.0) -> Optional[Dict]:
        """Place a market order"""
        if not self.connected:
            return None

        try:
            # Determine order type
            order_type = mt5.ORDER_TYPE_BUY if direction.upper() == 'BUY' else mt5.ORDER_TYPE_SELL

            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 123456,  # Magic number for our EA
                "comment": "Gold AI Sonnet 4.5",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None

            logger.info(f"Order placed successfully: {direction} {symbol} {volume} lots")

            return {
                'ticket': result.order,
                'price': result.price,
                'volume': result.volume
            }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def modify_position(self, ticket: int, stop_loss: float = 0.0,
                            take_profit: float = 0.0) -> bool:
        """Modify an existing position"""
        if not self.connected:
            return False

        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": stop_loss,
                "tp": take_profit
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Position modification failed: {result.retcode}")
                return False

            logger.info(f"Position {ticket} modified successfully")
            return True

        except Exception as e:
            logger.error(f"Error modifying position {ticket}: {e}")
            return False

    async def close_position(self, ticket: int, volume: float = 0.0) -> bool:
        """Close a position"""
        if not self.connected:
            return False

        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error(f"Position {ticket} not found")
                return False

            pos = positions[0]

            # Determine close type (opposite of position type)
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            # Use full volume if not specified
            close_volume = volume if volume > 0 else pos.volume

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": close_type,
                "price": mt5.symbol_info_tick(pos.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 20,
                "magic": 123456,
                "comment": "Gold AI Sonnet Close"
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Position close failed: {result.retcode}")
                return False

            logger.info(f"Position {ticket} closed successfully")
            return True

        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False

    async def close_position_partial(self, ticket: int, volume: float) -> bool:
        """Close part of a position"""
        return await self.close_position(ticket, volume)

    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.connected:
            return None

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None

            return {
                'symbol': info.name,
                'bid': info.bid,
                'ask': info.ask,
                'spread': info.spread,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step,
                'point': info.point,
                'digits': info.digits
            }

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    async def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for symbol"""
        if not self.connected:
            return False

        try:
            info = mt5.symbol_info(symbol)
            return info is not None and info.visible

        except Exception as e:
            logger.error(f"Error checking market status for {symbol}: {e}")
            return False
