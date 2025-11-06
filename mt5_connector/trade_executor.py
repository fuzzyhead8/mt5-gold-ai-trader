import MetaTrader5 as mt5
import logging

class TradeExecutor:
    def __init__(self, symbol):
        self.symbol = symbol

    def send_order(self, action, lot, price, sl, tp, deviation=20, magic=100000, comment="AI Bot Trade"):
        action_type = mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": action_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logging.info(f"Sending order: {action.upper()} {lot} lots at {price}, SL: {sl}, TP: {tp}")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Get error description
            error_descriptions = {
                10003: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume in the request",
                10004: "TRADE_RETCODE_INVALID_PRICE - Invalid price in the request",
                10006: "TRADE_RETCODE_REJECTED - Request rejected",
                10007: "TRADE_RETCODE_CANCEL - Request canceled by trader",
                10008: "TRADE_RETCODE_PLACED - Order placed",
                10009: "TRADE_RETCODE_DONE_PARTIAL - Request completed partially",
                10010: "TRADE_RETCODE_ERROR - Request processing error",
                10011: "TRADE_RETCODE_TIMEOUT - Request canceled by timeout",
                10012: "TRADE_RETCODE_INVALID - Invalid request",
                10013: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume in the request",
                10014: "TRADE_RETCODE_INVALID_PRICE - Invalid price in the request",
                10015: "TRADE_RETCODE_INVALID_STOPS - Invalid stops in the request",
                10016: "TRADE_RETCODE_TRADE_DISABLED - Trade is disabled",
                10017: "TRADE_RETCODE_MARKET_CLOSED - Market is closed",
                10018: "TRADE_RETCODE_NO_MONEY - There is not enough money to complete the request",
                10019: "TRADE_RETCODE_PRICE_CHANGED - Prices changed",
                10020: "TRADE_RETCODE_PRICE_OFF - There are no quotes to process the request",
                10021: "TRADE_RETCODE_INVALID_EXPIRATION - Invalid order expiration date in the request",
                10022: "TRADE_RETCODE_ORDER_CHANGED - Order state changed",
                10023: "TRADE_RETCODE_TOO_MANY_REQUESTS - Too frequent requests",
                10024: "TRADE_RETCODE_NO_CHANGES - Request does not contain changes",
                10025: "TRADE_RETCODE_SERVER_DISABLES_AT - Autotrading disabled by server",
                10026: "TRADE_RETCODE_CLIENT_DISABLES_AT - Autotrading disabled by client",
                10027: "TRADE_RETCODE_LOCKED - Request locked for processing",
                10028: "TRADE_RETCODE_FROZEN - Order or position frozen",
                10029: "TRADE_RETCODE_INVALID_FILL - Invalid order filling type",
                10030: "TRADE_RETCODE_CONNECTION - No connection with the trade server",
                10031: "TRADE_RETCODE_ONLY_REAL - Operation is allowed only for live accounts",
                10032: "TRADE_RETCODE_LIMIT_ORDERS - The number of pending orders has reached the limit",
                10033: "TRADE_RETCODE_LIMIT_VOLUME - The volume of orders and positions for the symbol has reached the limit",
                10034: "TRADE_RETCODE_INVALID_ORDER - Incorrect or prohibited order type",
                10035: "TRADE_RETCODE_POSITION_CLOSED - Position with the specified POSITION_IDENTIFIER has already been closed"
            }
            
            error_desc = error_descriptions.get(result.retcode, f"Unknown error code: {result.retcode}")
            logging.error(f"Order failed: {result.retcode} - {error_desc}")
            logging.error(f"Order comment: {result.comment}")
            
            # Log additional details for debugging
            if result.retcode == 10016:  # Invalid stops
                symbol_info = mt5.symbol_info(self.symbol)
                if symbol_info:
                    min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
                    actual_sl_distance = abs(sl - price)
                    actual_tp_distance = abs(tp - price)
                    logging.error(f"Stop validation - Min required: {min_stop_distance/symbol_info.point:.1f} pips, "
                                f"SL distance: {actual_sl_distance/symbol_info.point:.1f} pips, "
                                f"TP distance: {actual_tp_distance/symbol_info.point:.1f} pips")
        else:
            logging.info(f"Order successful: Ticket {result.order}, Volume {result.volume}")
            
        return result
