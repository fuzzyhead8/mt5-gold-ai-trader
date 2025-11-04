import numpy as np
import pandas as pd

class GoldenRiskManager:
    """
    GOLDEN RISK MANAGER - Advanced Position Sizing and Risk Management
    Designed to maximize profit while protecting capital
    """
    
    def __init__(self, account_balance=10000, max_risk_per_trade=0.02, max_daily_risk=0.06):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade max
        self.max_daily_risk = max_daily_risk  # 6% daily max
        self.daily_risk_used = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_drawdown = 0
        self.peak_balance = account_balance
        
        # Dynamic risk parameters
        self.base_lot_size = 0.1
        self.max_lot_size = 1.0
        self.min_lot_size = 0.01
        
        # Performance tracking
        self.trades_today = 0
        self.max_trades_per_day = 15
        self.win_rate_window = 20  # Last 20 trades for win rate calculation
        self.recent_trades = []
        
    def update_balance(self, new_balance):
        """Update account balance and related metrics"""
        old_balance = self.account_balance
        self.account_balance = new_balance
        
        # Update peak and drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_balance - new_balance) / self.peak_balance
    
    def calculate_position_size(self, entry_price, stop_loss_price, signal_strength=1.0, volatility_factor=1.0):
        """
        Calculate optimal position size using advanced risk management
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            signal_strength: Signal confidence (0.5 - 1.5)
            volatility_factor: Market volatility adjustment (0.5 - 2.0)
        """
        # Base risk calculation
        risk_distance = abs(entry_price - stop_loss_price)
        if risk_distance == 0:
            return 0
        
        # Dynamic risk percentage based on performance
        current_risk_pct = self._get_dynamic_risk_percentage()
        
        # Calculate base position size
        risk_amount = self.account_balance * current_risk_pct
        base_position_size = risk_amount / risk_distance
        
        # Apply signal strength adjustment
        position_size = base_position_size * signal_strength
        
        # Apply volatility adjustment (reduce size in high volatility)
        position_size = position_size / volatility_factor
        
        # Apply consecutive loss protection
        if self.consecutive_losses >= 3:
            position_size *= 0.5  # Half size after 3 consecutive losses
        elif self.consecutive_losses >= 5:
            position_size *= 0.25  # Quarter size after 5 consecutive losses
            
        # Apply win streak bonus (carefully)
        if self.consecutive_wins >= 3:
            position_size *= min(1.25, 1 + (self.consecutive_wins * 0.05))  # Max 25% increase
        
        # Apply drawdown protection
        if self.current_drawdown > 0.1:  # 10% drawdown
            position_size *= (1 - self.current_drawdown)
        
        # Ensure within limits
        position_size = max(self.min_lot_size, min(self.max_lot_size, position_size))
        
        return round(position_size, 2)
    
    def _get_dynamic_risk_percentage(self):
        """Calculate dynamic risk percentage based on performance"""
        base_risk = self.max_risk_per_trade
        
        # Reduce risk if approaching daily limit
        if self.daily_risk_used > self.max_daily_risk * 0.7:
            base_risk *= 0.5
            
        # Adjust based on recent win rate
        if len(self.recent_trades) >= 10:
            recent_wins = sum(1 for trade in self.recent_trades[-10:] if trade['profit'] > 0)
            win_rate = recent_wins / 10
            
            if win_rate > 0.6:
                base_risk *= 1.2  # Increase risk when winning
            elif win_rate < 0.4:
                base_risk *= 0.8  # Decrease risk when losing
        
        # Maximum drawdown protection
        if self.current_drawdown > 0.15:  # 15% drawdown
            base_risk *= 0.3  # Severely reduce risk
        elif self.current_drawdown > 0.08:  # 8% drawdown
            base_risk *= 0.6  # Moderately reduce risk
            
        return min(base_risk, self.max_risk_per_trade)
    
    def should_trade(self, signal_strength=1.0):
        """
        Determine if we should take a trade based on risk management rules
        """
        # Check daily risk limit
        if self.daily_risk_used >= self.max_daily_risk:
            return False, "Daily risk limit reached"
        
        # Check daily trade limit
        if self.trades_today >= self.max_trades_per_day:
            return False, "Daily trade limit reached"
        
        # Check drawdown protection
        if self.current_drawdown > 0.20:  # 20% drawdown - stop trading
            return False, "Maximum drawdown protection activated"
        
        # Check consecutive losses (circuit breaker)
        if self.consecutive_losses >= 7:
            return False, "Consecutive loss circuit breaker activated"
        
        # Reduce trading frequency during bad performance
        if self.consecutive_losses >= 4 and signal_strength < 0.8:
            return False, "Low confidence signal during loss streak"
        
        # Check minimum signal strength
        if signal_strength < 0.5:
            return False, "Signal strength too low"
        
        return True, "Trade approved"
    
    def calculate_stop_loss_take_profit(self, entry_price, signal_type, atr_value, volatility_regime='medium'):
        """
        Calculate dynamic stop loss and take profit levels
        
        Args:
            entry_price: Entry price
            signal_type: 'buy' or 'sell'
            atr_value: Current ATR value
            volatility_regime: 'low', 'medium', or 'high'
        """
        # Base multipliers
        if volatility_regime == 'low':
            stop_multiplier = 1.5
            tp_multiplier = 2.5
        elif volatility_regime == 'high':
            stop_multiplier = 2.5
            tp_multiplier = 3.5
        else:  # medium
            stop_multiplier = 2.0
            tp_multiplier = 3.0
        
        # Adjust based on recent performance
        if self.consecutive_losses >= 3:
            stop_multiplier *= 0.8  # Tighter stops during losses
            tp_multiplier *= 1.2   # Wider targets
        elif self.consecutive_wins >= 3:
            stop_multiplier *= 1.2  # Wider stops during wins
            tp_multiplier *= 0.9   # Tighter targets (take profits quicker)
        
        # Calculate levels
        stop_distance = atr_value * stop_multiplier
        tp_distance = atr_value * tp_multiplier
        
        if signal_type == 'buy':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # sell
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def record_trade(self, entry_price, exit_price, signal_type, position_size):
        """Record trade for performance tracking"""
        if signal_type == 'buy':
            profit = (exit_price - entry_price) * position_size
        else:
            profit = (entry_price - exit_price) * position_size
        
        trade_record = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'signal_type': signal_type,
            'position_size': position_size,
            'timestamp': pd.Timestamp.now()
        }
        
        self.recent_trades.append(trade_record)
        
        # Keep only recent trades for analysis
        if len(self.recent_trades) > self.win_rate_window:
            self.recent_trades.pop(0)
        
        # Update consecutive wins/losses
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Update daily risk used
        risk_used = abs(profit) / self.account_balance
        self.daily_risk_used += risk_used
        
        self.trades_today += 1
        
        return trade_record
    
    def reset_daily_counters(self):
        """Reset daily counters (call this at start of each trading day)"""
        self.daily_risk_used = 0
        self.trades_today = 0
    
    def get_risk_metrics(self):
        """Get current risk metrics"""
        win_rate = 0
        if len(self.recent_trades) > 0:
            wins = sum(1 for trade in self.recent_trades if trade['profit'] > 0)
            win_rate = wins / len(self.recent_trades)
        
        return {
            'account_balance': self.account_balance,
            'current_drawdown': self.current_drawdown,
            'daily_risk_used': self.daily_risk_used,
            'trades_today': self.trades_today,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'recent_win_rate': win_rate,
            'risk_per_trade': self._get_dynamic_risk_percentage()
        }
