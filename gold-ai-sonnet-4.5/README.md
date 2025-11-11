# Gold AI Sonnet 4.5

**Reverse Engineered AuriON AI System EA**

A cognitive trading system that integrates algorithmic execution, machine learning, and artificial intelligence. Built on the Deep Neural Cognition framework with an embedded Multilayer GPT Integration Engine.

## 🚀 Features

### Core Architecture
- **Deep Neural Cognition Framework**: Advanced AI-driven decision making
- **Multilayer GPT Integration Engine**: Supports GPT-4o, GPT-4.1, and GPT-5 models
- **Cognitive Trading Concept**: Statistical analysis with probabilistic modeling
- **Real-time Market Adaptation**: Dynamic analysis and interpretation

### Trading Instruments
- **Primary**: XAUUSD (Gold) on H1 timeframe
- **Extensible**: Support for major Forex pairs and CFDs

### Risk Management (Shield Protocol 3.0)
- **Zero-Martingale Policy**: No grid, averaging, or multiplier strategies
- **Dynamic Risk Limiter**: Position sizing adjusted to volatility
- **Smart Equity Lock**: Automatic protection during market anomalies
- **Maximum 2% risk per trade**

### AI Components
- **Nebula Assistant**: Human-AI interaction interface
- **Context-aware Analysis**: Market condition interpretation
- **Sentiment Analysis**: News and social media integration
- **Trend Prediction**: Advanced pattern recognition

### Position Management
- **Partial Closes**: 10%, 25%, 50% position closure options
- **Adaptive Trailing Stops**: ATR-based dynamic stops
- **Breakeven Management**: Automatic profit locking
- **OCO Orders**: One-Cancels-Other logic

### Technical Analysis
- **Sparkline Vision**: Multi-timeframe visualization (M1-D1)
- **Advanced Indicators**: ATR, RSI, MACD, Bollinger Bands
- **Support/Resistance Detection**: Dynamic key level identification
- **Volatility Analysis**: ADR-based market assessment

## 📋 Requirements

- Python 3.8+
- MetaTrader 5 terminal
- Valid MT5 trading account
- OpenAI API key (optional, for GPT integration)

## 🛠️ Installation

1. **Clone and Setup**:
   ```bash
   cd gold-ai-sonnet-4.5
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your MT5 credentials and API keys
   ```

3. **MT5 Setup**:
   - Install MetaTrader 5 terminal
   - Enable automated trading
   - Configure API access

## ⚙️ Configuration

### Key Settings (`config.py`)

```python
# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'XAUUSD',
    'timeframe': 'H1',
    'max_positions': 3,
    'min_confidence': 0.6
}

# Shield Protocol
SHIELD_CONFIG = {
    'max_risk_per_trade': 0.02,  # 2%
    'equity_lock_threshold': 0.10  # 10%
}

# AI Integration
AI_CONFIG = {
    'gpt_integration': {
        'enabled': True,
        'model': 'gpt-4o-mini'
    }
}
```

### Environment Variables

```bash
# MT5 Configuration
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=/path/to/mt5/terminal

# AI APIs (Optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## 🚀 Usage

### Basic Operation

```python
from gold_ai_sonnet import GoldAISonnet

# Initialize system
trader = GoldAISonnet(symbol='XAUUSD', timeframe='H1')

# Start trading
await trader.initialize()
await trader.start_trading()
```

### Command Line

```bash
# Start trading
python main.py

# With custom configuration
python main.py --symbol XAUUSD --timeframe H1 --max-risk 0.02
```

### Interactive Mode

```python
# Get system status
status = await trader.get_system_status()
print(f"Active positions: {status['positions_count']}")
print(f"Trading allowed: {status['trading_allowed']}")
```

## 🧠 AI Integration

### Nebula Assistant
The AI assistant provides:
- Real-time market analysis
- Trade signal explanations
- Risk assessment
- Performance insights

### GPT Integration
When enabled, the system uses:
- **GPT-4o**: Primary cognitive analysis
- **GPT-4o-mini**: Real-time operations
- **GPT-5**: Advanced scenario modeling

### Confidence Scoring
AI signals include confidence levels:
- **High (0.8-1.0)**: Strong recommendation
- **Medium (0.6-0.8)**: Moderate confidence
- **Low (<0.6)**: Signal rejected

## 📊 Risk Management

### Shield Protocol Rules
1. **Maximum 2% risk per trade**
2. **5% maximum daily loss**
3. **10% equity lock threshold**
4. **No martingale strategies**
5. **Volatility-adjusted position sizing**

### Position Management
- **Partial closes** at profit levels
- **Trailing stops** with ATR
- **Breakeven** activation
- **Maximum holding time** limits

## 📈 Performance Monitoring

### Real-time Metrics
- Win/Loss ratio
- Profit/Loss tracking
- Risk exposure
- AI confidence levels

### Reporting
- Daily performance reports
- Weekly summaries
- Monthly analysis
- Risk alerts

## 🔧 Advanced Features

### Backtesting
```python
from backtesting import Backtester

backtester = Backtester(config=BACKTEST_CONFIG)
results = backtester.run_strategy(GoldAISonnet, '2023-01-01', '2024-01-01')
```

### Custom Indicators
```python
# Add custom technical indicators
class CustomIndicator:
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # Your indicator logic
        return custom_signal
```

### Multi-Symbol Support
```python
# Trade multiple symbols
symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
for symbol in symbols:
    trader = GoldAISonnet(symbol=symbol)
    # Initialize and run
```

## 🛡️ Safety Features

### Circuit Breakers
- Automatic trading suspension on excessive losses
- Spread monitoring
- Connection stability checks
- Emergency stop functions

### Validation
- Pre-trade signal validation
- Position size verification
- Risk limit enforcement
- Configuration validation

## 📚 API Reference

### Main Classes

#### `GoldAISonnet`
Main trading system class.

**Methods:**
- `initialize()`: Setup system components
- `start_trading()`: Begin automated trading
- `stop_trading()`: Graceful shutdown
- `get_system_status()`: Current system state

#### `ShieldProtocol`
Risk management system.

**Methods:**
- `calculate_position_size()`: Risk-based sizing
- `check_equity_lock()`: Equity protection
- `is_trading_allowed()`: Trading permission

#### `NebulaAssistant`
AI analysis interface.

**Methods:**
- `analyze_market_conditions()`: Market analysis
- `generate_recommendation()`: Trading signals

## 🐛 Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   - Verify MT5 terminal is running
   - Check login credentials
   - Ensure API access is enabled

2. **Low AI Confidence**
   - Check market data quality
   - Verify indicator calculations
   - Review volatility conditions

3. **High Risk Alerts**
   - Reduce position sizes
   - Adjust risk parameters
   - Check Shield Protocol settings

### Logs
All system activity is logged to `logs/gold_ai_sonnet.log`

## 📄 License

This project is for educational and research purposes. Commercial use requires proper licensing.

## ⚠️ Disclaimer

This software is provided for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly before live trading.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the logs for error details

---

**Gold AI Sonnet 4.5** - Trading Reinvented by Intelligence
