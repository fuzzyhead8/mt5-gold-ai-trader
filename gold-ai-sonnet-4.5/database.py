"""
Database models for Gold AI Sonnet
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite database manager for trade persistence"""

    def __init__(self, db_path: str = "gold_ai_sonnet.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database and create tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT,
                    close_price REAL,
                    final_profit REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create trades table for historical records
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    open_time TEXT NOT NULL,
                    close_time TEXT,
                    profit REAL DEFAULT 0,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            logger.info("Database initialized successfully")

    def save_position(self, ticket: int, symbol: str, direction: str, volume: float,
                     entry_price: float, stop_loss: float, take_profit: float,
                     open_time: datetime):
        """Save or update a position"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO positions
                (ticket, symbol, direction, volume, entry_price, stop_loss, take_profit,
                 open_time, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (ticket, symbol, direction, volume, entry_price, stop_loss, take_profit,
                  open_time.isoformat()))

            conn.commit()
            logger.debug(f"Position {ticket} saved to database")

    def update_position_profit(self, ticket: int, current_profit: float):
        """Update position current profit"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE positions
                SET final_profit = ?, updated_at = CURRENT_TIMESTAMP
                WHERE ticket = ?
            ''', (current_profit, ticket))

            conn.commit()

    def close_position(self, ticket: int, close_price: float, profit: float,
                      close_time: datetime, reason: str = ""):
        """Close a position and move to trades table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get position data
            cursor.execute('SELECT * FROM positions WHERE ticket = ?', (ticket,))
            position = cursor.fetchone()

            if position:
                # Insert into trades table
                cursor.execute('''
                    INSERT INTO trades
                    (ticket, symbol, direction, volume, entry_price, exit_price,
                     stop_loss, take_profit, open_time, close_time, profit, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (position[0], position[1], position[2], position[3], position[4],
                      close_price, position[5], position[6], position[7],
                      close_time.isoformat(), profit, reason))

                # Remove from positions table
                cursor.execute('DELETE FROM positions WHERE ticket = ?', (ticket,))

                conn.commit()
                logger.info(f"Position {ticket} closed and moved to trades history")

    def partial_close_position(self, ticket: int, closed_volume: float,
                              close_price: float, profit: float, close_time: datetime):
        """Handle partial position closure"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get current position
            cursor.execute('SELECT * FROM positions WHERE ticket = ?', (ticket,))
            position = cursor.fetchone()

            if position:
                current_volume = position[3]  # volume
                remaining_volume = current_volume - closed_volume

                if remaining_volume <= 0:
                    # Full close
                    self.close_position(ticket, close_price, profit, close_time, "partial_close_full")
                else:
                    # Partial close - update volume and add to trades
                    cursor.execute('''
                        INSERT INTO trades
                        (ticket, symbol, direction, volume, entry_price, exit_price,
                         stop_loss, take_profit, open_time, close_time, profit, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (position[0], position[1], position[2], closed_volume, position[4],
                          close_price, position[5], position[6], position[7],
                          close_time.isoformat(), profit, "partial_close"))

                    # Update remaining position volume
                    cursor.execute('''
                        UPDATE positions
                        SET volume = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE ticket = ?
                    ''', (remaining_volume, ticket))

                    conn.commit()
                    logger.info(f"Partial close for position {ticket}: {closed_volume} lots closed")

    def load_open_positions(self) -> List[Dict]:
        """Load all open positions from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM positions WHERE status = "open"')
            positions = cursor.fetchall()

            result = []
            for pos in positions:
                result.append({
                    'ticket': pos[0],
                    'symbol': pos[1],
                    'direction': pos[2],
                    'volume': pos[3],
                    'entry_price': pos[4],
                    'stop_loss': pos[5],
                    'take_profit': pos[6],
                    'open_time': datetime.fromisoformat(pos[7]),
                    'current_profit': pos[9] or 0.0
                })

            return result

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM trades
                ORDER BY close_time DESC
                LIMIT ?
            ''', (limit,))

            trades = cursor.fetchall()

            result = []
            for trade in trades:
                result.append({
                    'id': trade[0],
                    'ticket': trade[1],
                    'symbol': trade[2],
                    'direction': trade[3],
                    'volume': trade[4],
                    'entry_price': trade[5],
                    'exit_price': trade[6],
                    'stop_loss': trade[7],
                    'take_profit': trade[8],
                    'open_time': datetime.fromisoformat(trade[9]) if trade[9] else None,
                    'close_time': datetime.fromisoformat(trade[10]) if trade[10] else None,
                    'profit': trade[11] or 0.0,
                    'reason': trade[12]
                })

            return result

    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total trades
            cursor.execute('SELECT COUNT(*) FROM trades')
            total_trades = cursor.fetchone()[0]

            # Winning trades
            cursor.execute('SELECT COUNT(*) FROM trades WHERE profit > 0')
            winning_trades = cursor.fetchone()[0]

            # Total profit
            cursor.execute('SELECT SUM(profit) FROM trades')
            total_profit = cursor.fetchone()[0] or 0

            # Win rate
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': round(win_rate, 2),
                'total_profit': round(total_profit, 2)
            }
