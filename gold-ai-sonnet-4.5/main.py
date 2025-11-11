"""
Gold AI Sonnet 4.5 - Reverse Engineered AuriON AI System EA

A cognitive trading system that integrates algorithmic execution, machine learning, and artificial intelligence.
Built on the Deep Neural Cognition framework with an embedded Multilayer GPT Integration Engine.

Trading Instrument: XAUUSD (Gold) on H1 timeframe
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
import os

# Local imports
from mt5_connector import MT5Connector
from config import get_config
from database import DatabaseManager

# AI imports
import openai
import anthropic
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gold_ai_sonnet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Represents a trading signal with all necessary parameters"""
    direction: str  # 'BUY' or 'SELL'
    symbol: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: float
    timestamp: datetime
    reasoning: str

@dataclass
class Position:
    """Represents an open position"""
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    current_profit: float = 0.0

class ShieldProtocol:
    """Shield Protocol 3.0 - Risk Management System"""

    def __init__(self, max_risk_per_trade: float = 0.02, max_daily_loss: float = 0.05):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_daily_loss = max_daily_loss  # 5% daily loss
        self.daily_loss = 0.0
        self.equity_lock_triggered = False

    def calculate_position_size(self, account_balance: float, stop_loss_pips: float,
                              symbol: str, risk_amount: float = None) -> float:
        """Calculate position size based on risk management rules"""
        if risk_amount is None:
            risk_amount = account_balance * self.max_risk_per_trade

        # Dynamic risk limiter - adjust based on volatility
        volatility_multiplier = self._get_volatility_multiplier(symbol)

        adjusted_risk = risk_amount * volatility_multiplier

        # Convert pips to price
        pip_value = self._get_pip_value(symbol, account_balance)

        if pip_value == 0 or stop_loss_pips == 0:
            return 0.01  # Minimum lot size

        position_size = adjusted_risk / (stop_loss_pips * pip_value)

        # Ensure minimum and maximum lot sizes
        position_size = max(0.01, min(position_size, 100.0))

        return round(position_size, 2)

    def _get_volatility_multiplier(self, symbol: str) -> float:
        """Get volatility multiplier for dynamic risk adjustment"""
        # Simplified volatility calculation
        # In real implementation, this would use ATR or similar
        base_volatility = 1.0

        if 'XAU' in symbol:  # Gold specific
            base_volatility = 1.2  # Gold is more volatile

        return base_volatility

    def _get_pip_value(self, symbol: str, account_balance: float) -> float:
        """Calculate pip value for the symbol"""
        # Simplified pip value calculation
        if 'XAU' in symbol:
            return account_balance * 0.00001  # Approximate for gold
        return account_balance * 0.0001  # Standard forex

    def check_equity_lock(self, current_equity: float, initial_balance: float) -> bool:
        """Check if equity lock should be triggered"""
        loss_percentage = (initial_balance - current_equity) / initial_balance

        if loss_percentage >= 0.1:  # 10% loss triggers lock
            self.equity_lock_triggered = True
            logger.warning(f"Equity lock triggered. Loss: {loss_percentage:.2%}")
            return True

        return False

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        return not self.equity_lock_triggered

class NebulaAssistant:
    """Nebula Assistant - AI Interaction Layer with GPT/Claude Integration"""

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        self.conversation_history = []
        self.market_context = {}

        # Initialize clients if API keys are available
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')

        if openai_key:
            self.openai_client = openai.OpenAI(api_key=openai_key)
        if anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

    async def analyze_market_conditions(self, symbol: str, timeframe: str,
                                      market_data: pd.DataFrame) -> Dict:
        """Analyze current market conditions using AI reasoning"""
        try:
            if len(market_data) < 20:
                return {
                    'trend': 'insufficient_data',
                    'volatility': 'unknown',
                    'support_resistance': {'resistance': 0, 'support': 0},
                    'momentum': 0.0,
                    'recommendation': 'WAIT',
                    'confidence': 0.0,
                    'reasoning': 'Insufficient market data for analysis'
                }

            # Prepare market data summary
            data_summary = self._prepare_market_data_summary(market_data, symbol, timeframe)

            # Get AI analysis from multiple sources
            gpt_analysis = await self._get_gpt_analysis(data_summary) if self.openai_client else None
            claude_analysis = await self._get_claude_analysis(data_summary) if self.anthropic_client else None

            # Combine analyses
            combined_analysis = self._combine_ai_analyses(gpt_analysis, claude_analysis, market_data)

            return combined_analysis

        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            # Fallback to technical analysis
            return self._fallback_technical_analysis(market_data)

    async def _get_gpt_analysis(self, data_summary: str) -> Dict:
        """Get market analysis from GPT"""
        try:
            prompt = f"""
            You are an expert forex trader analyzing {data_summary['symbol']} on {data_summary['timeframe']} timeframe.

            Market Data Summary:
            {data_summary['summary']}

            Recent Price Action:
            - Current Price: {data_summary['current_price']}
            - 24h Change: {data_summary['change_24h']}%
            - Volatility: {data_summary['volatility']}
            - Volume Trend: {data_summary['volume_trend']}

            Technical Indicators:
            - Trend: {data_summary['trend']}
            - RSI: {data_summary['rsi']}
            - MACD: {data_summary['macd']}
            - Support/Resistance: {data_summary['support_resistance']}

            Based on this data, provide a trading analysis in the following JSON format:
            {{
                "trend": "bullish|bearish|sideways",
                "volatility": "low|moderate|high",
                "momentum": "strong_bullish|bullish|neutral|bearish|strong_bearish",
                "recommendation": "BUY|SELL|WAIT",
                "confidence": 0.0-1.0,
                "key_levels": {{"support": price, "resistance": price}},
                "timeframe": "short|medium|long",
                "reasoning": "brief explanation of your analysis"
            }}

            Be conservative and only recommend trades with high confidence (>0.7).
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            # Extract JSON from response
            import json
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("GPT response not in expected JSON format")
                return None

        except Exception as e:
            logger.error(f"GPT analysis error: {e}")
            return None

    async def _get_claude_analysis(self, data_summary: str) -> Dict:
        """Get market analysis from Claude"""
        try:
            prompt = f"""
            You are a professional forex trader specializing in gold (XAUUSD) analysis.

            Analyze the following market data and provide a trading recommendation:

            Market Data Summary:
            {data_summary['summary']}

            Current Market Conditions:
            - Current Price: {data_summary['current_price']}
            - 24h Change: {data_summary['change_24h']}%
            - Volatility Level: {data_summary['volatility']}
            - Volume Trend: {data_summary['volume_trend']}

            Technical Analysis:
            - Trend Direction: {data_summary['trend']}
            - RSI(14): {data_summary['rsi']}
            - MACD Signal: {data_summary['macd']}
            - Key Levels: {data_summary['support_resistance']}

            Provide your analysis in this exact JSON format:
            {{
                "trend": "bullish|bearish|sideways",
                "volatility": "low|moderate|high",
                "momentum": "strong_bullish|bullish|neutral|bearish|strong_bearish",
                "recommendation": "BUY|SELL|WAIT",
                "confidence": 0.0-1.0,
                "key_levels": {{"support": price, "resistance": price}},
                "timeframe": "short|medium|long",
                "reasoning": "detailed analysis explanation"
            }}

            Focus on risk management and only recommend trades when conditions are clearly favorable.
            """

            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip()
            # Extract JSON from response
            import json
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("Claude response not in expected JSON format")
                return None

        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return None

    def _prepare_market_data_summary(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Prepare market data summary for AI analysis"""
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
        change_24h = ((current_price - prev_price) / prev_price) * 100

        # Calculate technical indicators
        rsi = self._calculate_rsi(data, 14)
        macd = self._calculate_macd(data)
        trend = self._determine_trend(data)
        volatility = self._assess_volatility(data)
        support_resistance = self._find_key_levels(data)

        # Volume trend
        volume_trend = "neutral"
        if len(data) > 10:
            recent_volume = data['volume'].tail(5).mean()
            older_volume = data['volume'].tail(10).head(5).mean()
            if recent_volume > older_volume * 1.2:
                volume_trend = "increasing"
            elif recent_volume < older_volume * 0.8:
                volume_trend = "decreasing"

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': f"{current_price:.5f}",
            'change_24h': f"{change_24h:.2f}",
            'volatility': volatility,
            'volume_trend': volume_trend,
            'trend': trend,
            'rsi': f"{rsi:.1f}",
            'macd': macd,
            'support_resistance': f"Support: {support_resistance['support']:.5f}, Resistance: {support_resistance['resistance']:.5f}",
            'summary': f"Latest {len(data)} bars of {symbol} {timeframe} data"
        }

    def _combine_ai_analyses(self, gpt_analysis: Dict, claude_analysis: Dict, market_data: pd.DataFrame) -> Dict:
        """Combine analyses from multiple AI sources"""
        analyses = []
        if gpt_analysis:
            analyses.append(gpt_analysis)
        if claude_analysis:
            analyses.append(claude_analysis)

        if not analyses:
            return self._fallback_technical_analysis(market_data)

        # If only one analysis, use it
        if len(analyses) == 1:
            analysis = analyses[0]
        else:
            # Combine multiple analyses
            analysis = self._merge_analyses(analyses)

        # Ensure all required fields are present
        result = {
            'trend': analysis.get('trend', 'sideways'),
            'volatility': analysis.get('volatility', 'moderate'),
            'support_resistance': analysis.get('key_levels', self._find_key_levels(market_data)),
            'momentum': self._convert_momentum_to_float(analysis.get('momentum', 'neutral')),
            'recommendation': analysis.get('recommendation', 'WAIT'),
            'confidence': analysis.get('confidence', 0.5),
            'reasoning': analysis.get('reasoning', 'AI analysis completed'),
            'timeframe': analysis.get('timeframe', 'medium'),
            'ai_sources': len(analyses)
        }

        return result

    def _merge_analyses(self, analyses: List[Dict]) -> Dict:
        """Merge multiple AI analyses into one"""
        # Simple majority voting for categorical fields
        trends = [a.get('trend') for a in analyses]
        trend = max(set(trends), key=trends.count) if trends else 'sideways'

        recommendations = [a.get('recommendation') for a in analyses]
        recommendation = max(set(recommendations), key=recommendations.count) if recommendations else 'WAIT'

        volatilities = [a.get('volatility') for a in analyses]
        volatility = max(set(volatilities), key=volatilities.count) if volatilities else 'moderate'

        # Average confidence
        confidences = [a.get('confidence', 0.5) for a in analyses]
        confidence = sum(confidences) / len(confidences)

        # Combine reasoning
        reasonings = [a.get('reasoning', '') for a in analyses]
        reasoning = ' | '.join(reasonings)

        # Use first key levels (could be improved)
        key_levels = analyses[0].get('key_levels', {'support': 0, 'resistance': 0})

        return {
            'trend': trend,
            'recommendation': recommendation,
            'volatility': volatility,
            'confidence': confidence,
            'reasoning': reasoning,
            'key_levels': key_levels,
            'momentum': analyses[0].get('momentum', 'neutral'),
            'timeframe': analyses[0].get('timeframe', 'medium')
        }

    def _convert_momentum_to_float(self, momentum: str) -> float:
        """Convert momentum string to float value"""
        momentum_map = {
            'strong_bullish': 0.8,
            'bullish': 0.3,
            'neutral': 0.0,
            'bearish': -0.3,
            'strong_bearish': -0.8
        }
        return momentum_map.get(momentum, 0.0)

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(data) < period + 1:
            return 50.0

        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

    def _calculate_macd(self, data: pd.DataFrame) -> str:
        """Calculate MACD signal"""
        if len(data) < 26:
            return "neutral"

        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        if hist.iloc[-1] > 0 and hist.iloc[-2] < 0:
            return "bullish_crossover"
        elif hist.iloc[-1] < 0 and hist.iloc[-2] > 0:
            return "bearish_crossover"
        elif hist.iloc[-1] > 0:
            return "bullish"
        else:
            return "bearish"

    def _fallback_technical_analysis(self, market_data: pd.DataFrame) -> Dict:
        """Fallback to technical analysis when AI fails"""
        return {
            'trend': self._determine_trend(market_data),
            'volatility': self._assess_volatility(market_data),
            'support_resistance': self._find_key_levels(market_data),
            'momentum': self._calculate_momentum(market_data),
            'recommendation': self._generate_recommendation(market_data),
            'confidence': 0.5,
            'reasoning': 'Fallback technical analysis - AI unavailable',
            'timeframe': 'medium',
            'ai_sources': 0
        }

    # Keep original technical methods as fallbacks
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine market trend"""
        if len(data) < 20:
            return 'insufficient_data'

        recent_close = data['close'].iloc[-1]
        ma_20 = data['close'].rolling(20).mean().iloc[-1]
        ma_50 = data['close'].rolling(50).mean().iloc[-1]

        if recent_close > ma_20 > ma_50:
            return 'bullish'
        elif recent_close < ma_20 < ma_50:
            return 'bearish'
        else:
            return 'sideways'

    def _assess_volatility(self, data: pd.DataFrame) -> str:
        """Assess market volatility"""
        if len(data) < 20:
            return 'unknown'

        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        if volatility > 0.3:
            return 'high'
        elif volatility > 0.15:
            return 'moderate'
        else:
            return 'low'

    def _find_key_levels(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        high_20 = data['high'].rolling(20).max().iloc[-1]
        low_20 = data['low'].rolling(20).min().iloc[-1]

        return {
            'resistance': high_20,
            'support': low_20
        }

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum indicator"""
        if len(data) < 14:
            return 0.0

        # Simple momentum calculation
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-14]) / data['close'].iloc[-14]
        return momentum

    def _generate_recommendation(self, data: pd.DataFrame) -> str:
        """Generate trading recommendation"""
        trend = self._determine_trend(data)
        momentum = self._calculate_momentum(data)

        if trend == 'bullish' and momentum > 0.01:
            return 'BUY'
        elif trend == 'bearish' and momentum < -0.01:
            return 'SELL'
        else:
            return 'WAIT'

class GoldAISonnet:
    """Main Gold AI Sonnet trading system"""

    def __init__(self, symbol: str = 'XAUUSD', timeframe: str = 'H1'):
        self.symbol = symbol
        self.timeframe = timeframe

        # Initialize components
        self.mt5_connector = MT5Connector()
        self.shield_protocol = ShieldProtocol()
        self.nebula_assistant = NebulaAssistant()
        self.database = DatabaseManager()
        self.config = get_config()

        # Trading state
        self.positions: Dict[int, Position] = {}
        self.last_analysis_time = None
        self.is_running = False

        logger.info(f"Gold AI Sonnet initialized for {symbol} on {timeframe}")

    async def initialize(self) -> bool:
        """Initialize the trading system"""
        try:
            # Connect to MT5
            if not await self.mt5_connector.connect():
                logger.error("Failed to connect to MT5")
                return False

            # Load existing positions from database
            await self._load_positions_from_database()

            # Models initialization would go here if they existed

            logger.info("Gold AI Sonnet initialization complete")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def start_trading(self):
        """Start the trading loop"""
        self.is_running = True
        logger.info("Starting Gold AI Sonnet trading system")

        while self.is_running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                await asyncio.sleep(60)

    async def stop_trading(self):
        """Stop the trading system"""
        self.is_running = False
        await self.mt5_connector.disconnect()
        logger.info("Gold AI Sonnet trading stopped")

    async def _trading_cycle(self):
        """Main trading cycle"""
        # Get market data
        market_data = await self.mt5_connector.get_historical_data(
            self.symbol, self.timeframe, 100
        )

        if market_data is None or len(market_data) < 50:
            logger.warning("Insufficient market data")
            return

        # Update positions
        await self._update_positions()

        # Check if trading is allowed
        if not self.shield_protocol.is_trading_allowed():
            logger.info("Trading not allowed by Shield Protocol")
            return

        # AI market analysis
        analysis = await self.nebula_assistant.analyze_market_conditions(
            self.symbol, self.timeframe, market_data
        )

        # Generate trading signal
        signal = await self._generate_signal(market_data, analysis)

        if signal:
            await self._execute_signal(signal)

        # Manage existing positions
        await self._manage_positions(market_data)

    async def _generate_signal(self, market_data: pd.DataFrame,
                             analysis: Dict) -> Optional[TradeSignal]:
        """Generate trading signal using AI analysis"""
        try:
            # Get current price
            current_price = market_data['close'].iloc[-1]

            # AI recommendation
            recommendation = analysis.get('recommendation', 'WAIT')

            if recommendation == 'WAIT':
                return None

            # Calculate stop loss and take profit using ATR
            atr = self._calculate_atr(market_data, 14)
            stop_loss_pips = atr * 1.5  # 1.5 ATR stop loss
            take_profit_pips = atr * 3.0  # 3:1 reward ratio

            if recommendation == 'BUY':
                entry_price = current_price
                stop_loss = entry_price - stop_loss_pips
                take_profit = entry_price + take_profit_pips
            else:  # SELL
                entry_price = current_price
                stop_loss = entry_price + stop_loss_pips
                take_profit = entry_price - take_profit_pips

            # Calculate position size
            account_info = await self.mt5_connector.get_account_info()
            if not account_info:
                return None

            volume = self.shield_protocol.calculate_position_size(
                account_info['balance'], stop_loss_pips, self.symbol
            )

            risk_reward_ratio = take_profit_pips / stop_loss_pips

            # Use AI confidence score
            confidence = analysis.get('confidence', 0.5)

            if confidence < 0.6:  # Minimum confidence threshold
                return None

            signal = TradeSignal(
                direction=recommendation,
                symbol=self.symbol,
                volume=volume,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                confidence=confidence,
                timestamp=datetime.now(),
                reasoning=f"AI Analysis: {analysis}"
            )

            return signal

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(data) < period + 1:
            return data['high'].iloc[-1] - data['low'].iloc[-1]  # Simple range

        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else (high.iloc[-1] - low.iloc[-1])

    def _calculate_confidence(self, analysis: Dict, market_data: pd.DataFrame) -> float:
        """Calculate AI confidence score"""
        confidence = 0.5  # Base confidence

        # Trend alignment
        if analysis.get('trend') in ['bullish', 'bearish']:
            confidence += 0.1

        # Momentum strength
        momentum = abs(analysis.get('momentum', 0))
        if momentum > 0.02:
            confidence += 0.1

        # Volatility consideration
        volatility = analysis.get('volatility', 'moderate')
        if volatility == 'moderate':
            confidence += 0.1
        elif volatility == 'high':
            confidence -= 0.1

        return min(confidence, 1.0)

    async def _execute_signal(self, signal: TradeSignal):
        """Execute trading signal"""
        try:
            # Place order
            order_result = await self.mt5_connector.place_order(
                symbol=signal.symbol,
                direction=signal.direction,
                volume=signal.volume,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

            if order_result and 'ticket' in order_result:
                # Create position object
                position = Position(
                    ticket=order_result['ticket'],
                    symbol=signal.symbol,
                    direction=signal.direction,
                    volume=signal.volume,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    open_time=signal.timestamp
                )

                self.positions[order_result['ticket']] = position

                # Save position to database
                await self._save_position_to_database(position)

                logger.info(f"Order executed: {signal.direction} {signal.symbol} "
                          f"Volume: {signal.volume} Entry: {signal.entry_price}")

        except Exception as e:
            logger.error(f"Signal execution error: {e}")

    async def _update_positions(self):
        """Update position information"""
        try:
            positions_data = await self.mt5_connector.get_positions()

            if positions_data:
                # Check for closed positions (not in MT5 response)
                current_mt5_tickets = {pos_data['ticket'] for pos_data in positions_data}
                for ticket in list(self.positions.keys()):
                    if ticket not in current_mt5_tickets:
                        # Position was closed externally, remove from memory and database
                        position = self.positions[ticket]
                        await self._close_position_in_database(
                            ticket, position.entry_price, position.current_profit, datetime.now()
                        )
                        del self.positions[ticket]
                        logger.info(f"Position {ticket} closed externally, removed from database")

                # Update remaining positions
                for pos_data in positions_data:
                    ticket = pos_data['ticket']
                    if ticket in self.positions:
                        old_profit = self.positions[ticket].current_profit
                        self.positions[ticket].current_profit = pos_data.get('profit', 0.0)

                        # Update profit in database if it changed significantly
                        if abs(self.positions[ticket].current_profit - old_profit) > 0.01:
                            self.database.update_position_profit(ticket, self.positions[ticket].current_profit)

        except Exception as e:
            logger.error(f"Position update error: {e}")

    async def _manage_positions(self, market_data: pd.DataFrame):
        """Manage existing positions"""
        try:
            for ticket, position in list(self.positions.items()):
                # Check for partial close conditions
                await self._check_partial_close(ticket, position, market_data)

                # Update trailing stops
                await self._update_trailing_stop(ticket, position, market_data)

        except Exception as e:
            logger.error(f"Position management error: {e}")

    async def _check_partial_close(self, ticket: int, position: Position,
                                 market_data: pd.DataFrame):
        """Check conditions for partial position closure"""
        try:
            # Simple partial close logic - close 50% at 50% profit target
            profit_target = abs(position.take_profit - position.entry_price) * 0.5

            if position.current_profit >= profit_target:
                # Close 50% of position
                partial_volume = position.volume * 0.5

                await self.mt5_connector.close_position_partial(
                    ticket, partial_volume
                )

                # Update position volume
                position.volume -= partial_volume

                logger.info(f"Partial close executed for ticket {ticket}")

        except Exception as e:
            logger.error(f"Partial close error for ticket {ticket}: {e}")

    async def _update_trailing_stop(self, ticket: int, position: Position,
                                  market_data: pd.DataFrame):
        """Update trailing stop loss"""
        try:
            current_price = market_data['close'].iloc[-1]
            atr = self._calculate_atr(market_data, 14)

            if position.direction == 'BUY':
                new_stop = current_price - (atr * 1.5)
                if new_stop > position.stop_loss:
                    await self.mt5_connector.modify_position(
                        ticket, stop_loss=new_stop
                    )
                    position.stop_loss = new_stop
            else:  # SELL
                new_stop = current_price + (atr * 1.5)
                if new_stop < position.stop_loss:
                    await self.mt5_connector.modify_position(
                        ticket, stop_loss=new_stop
                    )
                    position.stop_loss = new_stop

        except Exception as e:
            logger.error(f"Trailing stop update error for ticket {ticket}: {e}")

    async def _load_positions_from_database(self):
        """Load open positions from database on startup"""
        try:
            db_positions = self.database.load_open_positions()

            for pos_data in db_positions:
                position = Position(
                    ticket=pos_data['ticket'],
                    symbol=pos_data['symbol'],
                    direction=pos_data['direction'],
                    volume=pos_data['volume'],
                    entry_price=pos_data['entry_price'],
                    stop_loss=pos_data['stop_loss'],
                    take_profit=pos_data['take_profit'],
                    open_time=pos_data['open_time'],
                    current_profit=pos_data['current_profit']
                )

                self.positions[pos_data['ticket']] = position
                logger.info(f"Loaded position {pos_data['ticket']} from database")

        except Exception as e:
            logger.error(f"Error loading positions from database: {e}")

    async def _save_position_to_database(self, position: Position):
        """Save position to database"""
        try:
            self.database.save_position(
                ticket=position.ticket,
                symbol=position.symbol,
                direction=position.direction,
                volume=position.volume,
                entry_price=position.entry_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                open_time=position.open_time
            )
        except Exception as e:
            logger.error(f"Error saving position {position.ticket} to database: {e}")

    async def _close_position_in_database(self, ticket: int, close_price: float,
                                        profit: float, close_time: datetime):
        """Close position in database and move to trades history"""
        try:
            self.database.close_position(ticket, close_price, profit, close_time)
        except Exception as e:
            logger.error(f"Error closing position {ticket} in database: {e}")

    async def get_system_status(self) -> Dict:
        """Get current system status"""
        account_info = await self.mt5_connector.get_account_info()

        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'is_running': self.is_running,
            'positions_count': len(self.positions),
            'account_balance': account_info.get('balance', 0) if account_info else 0,
            'account_equity': account_info.get('equity', 0) if account_info else 0,
            'trading_allowed': self.shield_protocol.is_trading_allowed(),
            'equity_lock_triggered': self.shield_protocol.equity_lock_triggered
        }

async def main():
    """Main function"""
    # Create trading system
    trader = GoldAISonnet(symbol='XAUUSD', timeframe='H1')

    # Initialize
    if not await trader.initialize():
        logger.error("Failed to initialize trading system")
        return

    # Start trading
    try:
        await trader.start_trading()
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    finally:
        await trader.stop_trading()

if __name__ == "__main__":
    asyncio.run(main())
