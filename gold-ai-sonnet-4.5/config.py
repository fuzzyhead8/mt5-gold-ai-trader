"""
Configuration file for Gold AI Sonnet 4.5
"""

import os
import datetime
from typing import Dict, Any

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'XAUUSD',
    'timeframe': 'H1',
    'max_positions': 3,
    'min_confidence': 0.6,
    'trading_hours': {
        'start': '00:00',
        'end': '23:59',
        'timezone': 'UTC'
    }
}

# Shield Protocol Configuration
SHIELD_CONFIG = {
    'max_risk_per_trade': 0.02,  # 2% per trade
    'max_daily_loss': 0.05,      # 5% daily loss
    'max_total_risk': 0.10,      # 10% total risk
    'equity_lock_threshold': 0.10,  # 10% loss triggers lock
    'volatility_multiplier_gold': 1.2,
    'volatility_multiplier_forex': 1.0
}

# AI Configuration
AI_CONFIG = {
    'nebula_assistant': {
        'enabled': True,
        'analysis_interval': 60,  # seconds
        'confidence_threshold': 0.6
    },
    'gpt_integration': {
        'enabled': True,  # Enable GPT integration for AI analysis
        'model': 'gpt-4o-mini',
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'max_tokens': 1000,
        'temperature': 0.3
    },
    'sentiment_analysis': {
        'enabled': True,
        'sources': ['news', 'social_media'],
        'update_interval': 300  # 5 minutes
    }
}

# MT5 Configuration
MT5_CONFIG = {
    'login': int(os.getenv('MT5_LOGIN', '0')),
    'password': os.getenv('MT5_PASSWORD', ''),
    'server': os.getenv('MT5_SERVER', ''),
    'path': os.getenv('MT5_PATH', ''),
    'timeout': 60000
}

# Risk Management
RISK_CONFIG = {
    'position_sizing': {
        'method': 'fixed_percentage',  # 'fixed_percentage', 'kelly_criterion', 'martingale'
        'fixed_percentage': 0.02,
        'kelly_fraction': 0.5,
        'martingale_multiplier': 1.0
    },
    'stop_loss': {
        'method': 'atr_based',  # 'fixed_pips', 'atr_based', 'percentage'
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'fixed_pips': 50,
        'percentage': 0.02
    },
    'take_profit': {
        'method': 'risk_reward',  # 'fixed_pips', 'risk_reward', 'atr_based'
        'risk_reward_ratio': 3.0,
        'fixed_pips': 100,
        'atr_multiplier': 3.0
    },
    'trailing_stop': {
        'enabled': True,
        'method': 'atr_based',  # 'percentage', 'atr_based', 'parabolic_sar'
        'atr_multiplier': 1.5,
        'percentage': 0.05,
        'activation_pips': 20
    }
}

# Position Management
POSITION_CONFIG = {
    'partial_closes': {
        'enabled': True,
        'levels': [
            {'profit_percentage': 0.5, 'close_percentage': 0.5},
            {'profit_percentage': 1.0, 'close_percentage': 0.3},
            {'profit_percentage': 2.0, 'close_percentage': 0.2}
        ]
    },
    'breakeven': {
        'enabled': True,
        'trigger_pips': 30,
        'lock_pips': 10
    },
    'max_holding_time': 24 * 60 * 60,  # 24 hours in seconds
    'max_spread': 3.0  # Maximum spread in pips
}

# Technical Indicators
INDICATOR_CONFIG = {
    'moving_averages': {
        'sma_periods': [20, 50, 200],
        'ema_periods': [12, 26, 50]
    },
    'oscillators': {
        'rsi_period': 14,
        'stochastic_k': 14,
        'stochastic_d': 3,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    },
    'volatility': {
        'atr_period': 14,
        'bollinger_period': 20,
        'bollinger_std': 2
    },
    'trend': {
        'adx_period': 14,
        'cci_period': 20
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/gold_ai_sonnet.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'enabled': True,
    'start_date': '2025-01-01',
    'end_date': datetime.date.today().isoformat(),
    'initial_balance': 10000,
    'leverage': 100,
    'commission': 0.0002,  # Commission per lot
    'spread': 2.0  # Spread in pips
}

# Performance Monitoring
MONITORING_CONFIG = {
    'enabled': True,
    'metrics_interval': 300,  # 5 minutes
    'alerts': {
        'daily_loss_threshold': 0.03,
        'weekly_loss_threshold': 0.10,
        'win_rate_threshold': 0.50
    },
    'reports': {
        'daily': True,
        'weekly': True,
        'monthly': True
    }
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration"""
    return {
        'trading': TRADING_CONFIG,
        'shield': SHIELD_CONFIG,
        'ai': AI_CONFIG,
        'mt5': MT5_CONFIG,
        'risk': RISK_CONFIG,
        'position': POSITION_CONFIG,
        'indicators': INDICATOR_CONFIG,
        'logging': LOGGING_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'monitoring': MONITORING_CONFIG
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    try:
        # Validate trading config
        assert config['trading']['symbol'] in ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
        assert config['trading']['timeframe'] in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        assert 0 < config['trading']['max_positions'] <= 10
        assert 0 < config['trading']['min_confidence'] <= 1

        # Validate shield config
        assert 0 < config['shield']['max_risk_per_trade'] <= 0.1
        assert 0 < config['shield']['max_daily_loss'] <= 0.2
        assert 0 < config['shield']['equity_lock_threshold'] <= 0.2

        # Validate risk config
        assert config['risk']['position_sizing']['method'] in ['fixed_percentage', 'kelly_criterion', 'martingale']
        assert config['risk']['stop_loss']['method'] in ['fixed_pips', 'atr_based', 'percentage']
        assert config['risk']['take_profit']['method'] in ['fixed_pips', 'risk_reward', 'atr_based']

        return True

    except (KeyError, AssertionError) as e:
        print(f"Configuration validation failed: {e}")
        return False
