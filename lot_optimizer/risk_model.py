class RiskModel:
    def __init__(self, max_risk_percent=2):
        self.max_risk_percent = max_risk_percent

    def calculate_risk_amount(self, balance):
        """
        Returns the dollar amount to risk on a single trade based on account balance.
        """
        return (self.max_risk_percent / 100.0) * balance

    def calculate_stop_loss_distance(self, entry_price, stop_price):
        """
        Calculates stop loss distance in pips.
        """
        return abs(entry_price - stop_price)

    def calculate_lot_size(self, balance, entry_price, stop_price, symbol="XAUUSD"):
        """
        Calculate lot size based on risk amount and stop loss distance
        Fixed for XAUUSD: 1 pip = 0.1, 1 lot = $1 per pip movement
        """
        risk_amount = self.calculate_risk_amount(balance)
        stop_loss_distance = abs(entry_price - stop_price)
        
        if stop_loss_distance == 0:
            return 0.01
            
        # XAUUSD: 1 standard lot = $1 per 0.1 price movement
        # So for 0.01 lot = $0.01 per 0.1 price movement
        # Dollar risk per lot = stop_loss_distance * 100 (for 0.01 lot)
        dollar_risk_per_lot = stop_loss_distance * 100
        
        if dollar_risk_per_lot == 0:
            return 0.01
            
        lot_size = risk_amount / dollar_risk_per_lot
        
        # Ensure minimum 0.01 lots and maximum reasonable size
        lot_size = max(0.01, min(lot_size, 1.0))
        return round(lot_size, 2)
