#!/usr/bin/env python3
"""
Comprehensive Bot Training Script
Trains all trading bots with their respective timeframe data:
- Scalping: M1 data
- Day Trading: M15 data  
- Golden: M15 data
- Multi RSI EMA: M15 data
- Swing: H4 data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import strategy classes
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.golden_scalping import GoldenScalpingStrategy
from strategies.swing import SwingTradingStrategy
from strategies.multi_rsi_ema import MultiRSIEMAStrategy
from strategies.range_oscillator import RangeOscillatorStrategy

class BotTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        
    def load_data(self, file_path):
        """Load and prepare CSV data"""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Ensure proper datetime format
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Rename tick_volume to volume if needed
        if 'tick_volume' in df.columns and 'volume' not in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def prepare_features(self, data_with_signals, strategy_name):
        """Extract features from data with signals"""
        features = []
        labels = []
        
        # Convert signals to numerical labels
        signal_map = {'buy': 1, 'sell': -1, 'hold': 0}
        
        for i in range(len(data_with_signals)):
            if pd.isna(data_with_signals.iloc[i]).any():
                continue
                
            # Basic price features
            row = data_with_signals.iloc[i]
            feature_vector = []
            
            if strategy_name == 'scalping':
                # Scalping-specific features
                feature_vector = [
                    row.get('close', 0),
                    row.get('rsi', 50),
                    row.get('volume_ratio', 1),
                    row.get('volatility', 0),
                    row.get('ema_fast', row.get('close', 0)),
                    row.get('ema_slow', row.get('close', 0)),
                    row.get('bb_upper', row.get('close', 0)),
                    row.get('bb_lower', row.get('close', 0)),
                ]
                
            elif strategy_name == 'day_trading':
                # Day trading-specific features
                feature_vector = [
                    row.get('close', 0),
                    row.get('RSI', 50),
                    row.get('EMA_fast', row.get('close', 0)),
                    row.get('EMA_slow', row.get('close', 0)),
                    row.get('volume_ratio', 1),
                    row.get('volatility', 0),
                    row.get('trend', 0),
                ]
                
            elif strategy_name == 'golden':
                # Golden strategy-specific features
                feature_vector = [
                    row.get('close', 0),
                    row.get('rsi', 50),
                    row.get('stoch_k', 50),
                    row.get('stoch_d', 50),
                    row.get('macd_histogram', 0),
                    row.get('vwap_distance', 0),
                    row.get('volume_impact', 1),
                    row.get('mfi_ratio', 1),
                    row.get('trend_short', 0),
                    row.get('trend_medium', 0),
                    row.get('price_momentum', 0),
                ]
                
            elif strategy_name == 'multi_rsi_ema':
                # Multi RSI EMA-specific features
                feature_vector = [
                    row.get('close', 0),
                    row.get('rsi_2', 50),
                    row.get('rsi_9', 50),
                    row.get('rsi_34', 50),
                    row.get('ema_34', row.get('close', 0)),
                    row.get('ema_144', row.get('close', 0)),
                    row.get('ema_21', row.get('close', 0)),
                ]
                
            elif strategy_name == 'range_oscillator':
                # Range Oscillator-specific features
                feature_vector = [
                    row.get('close', 0),
                    row.get('osc', 0),
                    row.get('heat_zone', 0),
                    row.get('wma', row.get('close', 0)),
                ]
                
            elif strategy_name == 'swing':
                # Swing trading-specific features
                feature_vector = [
                    row.get('close', 0),
                    row.get('MACD', 0),
                    row.get('Signal_Line', 0),
                ]
            
            # Add signal label
            signal_label = signal_map.get(row.get('signal', 'hold'), 0)
            
            if len(feature_vector) > 0 and not any(pd.isna(v) for v in feature_vector):
                features.append(feature_vector)
                labels.append(signal_label)
        
        return np.array(features), np.array(labels)
    
    def train_strategy_model(self, strategy_name, data_files, strategy_class):
        """Train model for specific strategy"""
        print(f"\n{'='*50}")
        print(f"Training {strategy_name.upper()} Bot")
        print(f"{'='*50}")
        
        all_features = []
        all_labels = []
        
        for file_path in data_files:
            # Load data
            data = self.load_data(file_path)
            
            # Initialize strategy
            strategy = strategy_class("XAUUSD")
            
            # Generate signals
            print(f"Generating {strategy_name} signals...")
            try:
                data_with_signals = strategy.generate_signals(data)
                print(f"Generated signals shape: {data_with_signals.shape}")
                
                # Debug: Check signal distribution
                if 'signal' in data_with_signals.columns:
                    signal_counts = data_with_signals['signal'].value_counts()
                    print(f"Signal distribution: {signal_counts.to_dict()}")
                
                # Extract features
                features, labels = self.prepare_features(data_with_signals, strategy_name)
                
                if len(features) > 0:
                    all_features.extend(features)
                    all_labels.extend(labels)
                    print(f"Extracted {len(features)} feature vectors")
                else:
                    print("No valid features extracted from this file")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        if len(all_features) == 0:
            print(f"No training data available for {strategy_name}")
            return
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\nTotal training samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1] if len(X) > 0 else 0}")
        print(f"Label distribution: {np.bincount(y + 1)}")  # +1 to handle -1,0,1 labels
        
        # Check if we have enough data and variety in labels
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            print(f"Insufficient label variety for {strategy_name}. Skipping training.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with GridSearch
        print("Training model...")
        
        # Use Gradient Boosting for better performance
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.15, 0.2]
        }
        
        model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Evaluate
        train_predictions = best_model.predict(X_train_scaled)
        test_predictions = best_model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"\nModel Performance:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, test_predictions, 
                                  target_names=['Sell', 'Hold', 'Buy']))
        
        # Save model and scaler
        model_path = f'models/{strategy_name}_signal.joblib'
        scaler_path = f'models/{strategy_name}_scaler.joblib'
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Store in class for later use
        self.models[strategy_name] = best_model
        self.scalers[strategy_name] = scaler
        self.feature_columns[strategy_name] = list(range(X.shape[1]))
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def train_all_bots(self):
        """Train all trading bots with their respective data"""
        print("Starting Comprehensive Bot Training...")
        print(f"Training started at: {datetime.now()}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Define strategy configurations
        strategy_configs = {
            'scalping': {
                'files': ['backtests/XAUUSD_M1_20251107_001549.csv'],
                'class': ScalpingStrategy
            },
            'day_trading': {
                'files': ['backtests/XAUUSD_M15_20251107_001740.csv'],
                'class': DayTradingStrategy
            },
            'golden': {
                'files': ['backtests/XAUUSD_M15_20251107_001740.csv'],
                'class': GoldenScalpingStrategy
            },
            'multi_rsi_ema': {
                'files': ['backtests/XAUUSD_M15_20251107_001740.csv'],
                'class': MultiRSIEMAStrategy
            },
            'range_oscillator': {
                'files': ['backtests/XAUUSD_M15_20251107_001740.csv'],
                'class': RangeOscillatorStrategy
            },
            'swing': {
                'files': [
                    'backtests/XAUUSD_H4_20251104_230518.csv'
                ],
                'class': SwingTradingStrategy
            }
        }
        
        # Train each strategy
        for strategy_name, config in strategy_configs.items():
            # Check if files exist
            available_files = [f for f in config['files'] if os.path.exists(f)]
            if not available_files:
                print(f"\nWarning: No data files found for {strategy_name}")
                continue
            
            try:
                self.train_strategy_model(
                    strategy_name, 
                    available_files, 
                    config['class']
                )
            except Exception as e:
                print(f"Error training {strategy_name}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully trained models: {list(self.models.keys())}")
        print(f"Training completed at: {datetime.now()}")
        
        # Save training summary
        summary = {
            'trained_models': list(self.models.keys()),
            'training_date': datetime.now().isoformat(),
            'model_files': {strategy: f'models/{strategy}_signal.joblib' 
                          for strategy in self.models.keys()}
        }
        
        import json
        with open('models/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Training summary saved to: models/training_summary.json")

def main():
    """Main training function"""
    trainer = BotTrainer()
    trainer.train_all_bots()

if __name__ == "__main__":
    main()
