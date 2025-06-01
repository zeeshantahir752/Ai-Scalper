#!/usr/bin/env python3
"""
XGBoost Model Trainer for XAUUSDm AI Trading System
Trains machine learning model using historical data and technical indicators
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest
import talib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

class XGBoostTrainer:
    def __init__(self, symbol="XAUUSDm", config_file="config.json"):
        """Initialize the XGBoost trainer"""
        self.symbol = symbol
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Load configuration
        self.load_config(config_file)
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize MT5
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")
        
        print("XGBoost Trainer initialized successfully")
    
    def load_config(self, config_file):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "model_parameters": {
                    "lookforward_periods": 5,
                    "profit_threshold": 0.0002,
                    "max_features": 50
                }
            }
    
    def get_historical_data(self, timeframe, count=10000):
        """Get historical data from MT5"""
        print(f"Fetching {count} bars of {timeframe} data for {self.symbol}...")
        
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None:
            print(f"Failed to get data for {self.symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"Retrieved {len(df)} bars from {df['time'].min()} to {df['time'].max()}")
        return df
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if len(df) < 100:
            print("Insufficient data for indicator calculation")
            return df
        
        print("Calculating technical indicators...")
        
        try:
            # Price-based indicators
            df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
            
            # Momentum indicators
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_6'] = talib.RSI(df['close'], timeperiod=6)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            df['momentum'] = talib.MOM(df['close'], timeperiod=10)
            
            # Volume indicators
            df['obv'] = talib.OBV(df['close'], df['tick_volume'])
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['tick_volume'])
            
            # Volatility indicators
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            df['natr'] = talib.NATR(df['high'], df['low'], df['close'])
            df['trange'] = talib.TRANGE(df['high'], df['low'], df['close'])
            
            # Pattern recognition (key patterns only)
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
            
            # Custom technical features
            df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
            df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
            df['macd_bullish'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
            df['above_sma20'] = np.where(df['close'] > df['sma_20'], 1, 0)
            df['above_ema12'] = np.where(df['close'] > df['ema_12'], 1, 0)
            
            # Price action features
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['candle_type'] = np.where(df['close'] > df['open'], 1, -1)
            
            # Trend features
            df['trend_sma'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
            df['trend_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
            
            # Volatility features
            df['volatility_ratio'] = df['atr'] / df['close']
            df['price_range'] = (df['high'] - df['low']) / df['close']
            
            # Support/Resistance levels
            df['resistance_20'] = df['high'].rolling(20).max()
            df['support_20'] = df['low'].rolling(20).min()
            df['distance_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
            df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']
            
            print(f"Calculated {len([col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']])} technical indicators")
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    def create_multi_timeframe_features(self):
        """Create features from multiple timeframes"""
        timeframes = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]
        timeframe_names = ["M1", "M5", "M15", "H1"]
        
        print("Creating multi-timeframe features...")
        
        base_df = None
        
        for i, (tf, tf_name) in enumerate(zip(timeframes, timeframe_names)):
            print(f"Processing {tf_name} timeframe...")
            df = self.get_historical_data(tf, 5000)
            
            if df.empty:
                continue
            
            df = self.calculate_technical_indicators(df)
            
            if base_df is None:
                # Use M1 as base timeframe
                base_df = df.copy()
                base_df = base_df.add_prefix(f'{tf_name}_')
                base_df = base_df.rename(columns={f'{tf_name}_time': 'time'})
            else:
                # Select key features for higher timeframes
                feature_cols = ['time', 'rsi', 'macd', 'macd_signal', 'stoch_k', 'williams_r', 'cci',
                               'price_position', 'atr', 'volatility_ratio', 'trend_sma', 'trend_ema',
                               'rsi_oversold', 'rsi_overbought', 'macd_bullish', 'above_sma20']
                
                tf_features = df[feature_cols].copy()
                tf_features = tf_features.add_prefix(f'{tf_name}_')
                tf_features = tf_features.rename(columns={f'{tf_name}_time': 'time'})
                
                # Merge with base timeframe using forward fill
                tf_features_resampled = tf_features.set_index('time').resample('1min').ffill().reset_index()
                base_df = pd.merge(base_df, tf_features_resampled, on='time', how='left')
        
        if base_df is not None:
            print(f"Multi-timeframe dataset shape: {base_df.shape}")
            return base_df
        else:
            print("Failed to create multi-timeframe features")
            return pd.DataFrame()
    
    def create_target_variable(self, df, lookforward=None, threshold=None):
        """Create target variable based on future price movement"""
        lookforward = lookforward or self.config.get("model_parameters", {}).get("lookforward_periods", 5)
        threshold = threshold or self.config.get("model_parameters", {}).get("profit_threshold", 0.0002)
        
        print(f"Creating target variable with lookforward={lookforward}, threshold={threshold}")
        
        df = df.copy()
        
        # Use M1 close price for target calculation
        close_col = 'M1_close' if 'M1_close' in df.columns else 'close'
        
        # Calculate future high and low
        df['future_high'] = df[close_col].shift(-lookforward).rolling(lookforward, min_periods=1).max()
        df['future_low'] = df[close_col].shift(-lookforward).rolling(lookforward, min_periods=1).min()
        
        # Calculate percentage moves
        df['future_high_pct'] = (df['future_high'] - df[close_col]) / df[close_col]
        df['future_low_pct'] = (df['future_low'] - df[close_col]) / df[close_col]
        
        # Create target based on significant moves
        conditions = [
            df['future_high_pct'] > threshold,  # Strong upward move - BUY signal
            df['future_low_pct'] < -threshold,  # Strong downward move - SELL signal
        ]
        choices = [2, 0]  # 2=BUY, 0=SELL, 1=HOLD (default)
        
        df['target'] = np.select(conditions, choices, default=1)
        
        # Print target distribution
        target_counts = df['target'].value_counts().sort_index()
        print(f"Target distribution:")
        print(f"  SELL (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  HOLD (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
        print(f"  BUY (2): {target_counts.get(2, 0)} ({target_counts.get(2, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_training_data(self):
        """Prepare training dataset"""
        print("="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        
        df = self.create_multi_timeframe_features()
        
        if df is None or df.empty:
            raise Exception("No data available for training")
        
        df = self.create_target_variable(df)
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        print(f"Removed {initial_rows - final_rows} rows with NaN values")
        
        if len(df) < 1000:
            raise Exception(f"Insufficient data for training: {len(df)} rows")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in 
                       ['time', 'future_high', 'future_low', 'future_high_pct', 'future_low_pct', 'target']]
        
        X = df[feature_cols]
        y = df['target']
        
        print(f"Training data prepared:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, hyperparameter_tuning=False):
        """Train the XGBoost model"""
        print("="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        X, y = self.prepare_training_data()
        
        # Check for class imbalance
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < 100:
            print(f"Warning: Low sample count for some classes: {class_counts.to_dict()}")
        
        # Split data
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train anomaly detector
        print("Training anomaly detector...")
        self.anomaly_detector.fit(X_train_scaled)
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            best_params = self.hyperparameter_tuning(X_train_scaled, y_train)
        else:
            # Use default parameters optimized for trading
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
            **best_params
        }
        
        print("Training XGBoost model with parameters:")
        for key, value in xgb_params.items():
            print(f"  {key}: {value}")
        
        self.model = xgb.XGBClassifier(**xgb_params)
        
        # Train model (handle different XGBoost versions)
        try:
            # Try new XGBoost API (v1.6+)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
                verbose=False
            )
        except TypeError:
            # Fallback for older XGBoost versions
            try:
                self.model.fit(
                    X_train_scaled, y_train,
                    early_stopping_rounds=50,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
            except TypeError:
                # Simplest fallback - no early stopping
                self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Training accuracy
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Test accuracy
        y_test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred, target_names=['SELL', 'HOLD', 'BUY']))
        
        # Confusion matrix
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        # Cross-validation
        print(f"\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return test_accuracy, feature_importance
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        print("Performing hyperparameter tuning (this may take a while)...")
        
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def save_model(self):
        """Save trained model and preprocessors"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        print("\nSaving model and preprocessors...")
        
        # Save model
        model_path = "models/xgboost_model.pkl"
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save scaler
        scaler_path = "models/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        # Save anomaly detector
        anomaly_path = "models/anomaly_detector.pkl"
        joblib.dump(self.anomaly_detector, anomaly_path)
        print(f"✓ Anomaly detector saved to {anomaly_path}")
        
        # Save feature names for reference
        if hasattr(self.model, 'feature_names_in_'):
            features_path = "models/feature_names.pkl"
            joblib.dump(self.model.feature_names_in_, features_path)
            print(f"✓ Feature names saved to {features_path}")
        
        print("All models saved successfully!")
    
    def plot_feature_importance(self, feature_importance, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - XGBoost Model')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to models/feature_importance.png")
        plt.show()
    
    def plot_target_distribution(self, y):
        """Plot target variable distribution"""
        plt.figure(figsize=(8, 6))
        target_counts = y.value_counts().sort_index()
        target_labels = ['SELL', 'HOLD', 'BUY']
        
        plt.bar(range(len(target_counts)), target_counts.values)
        plt.xlabel('Signal Type')
        plt.ylabel('Count')
        plt.title('Target Variable Distribution')
        plt.xticks(range(len(target_counts)), target_labels)
        
        # Add percentage labels
        total = len(y)
        for i, count in enumerate(target_counts.values):
            plt.text(i, count + total*0.01, f'{count}\n({count/total*100:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('models/target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Target distribution plot saved to models/target_distribution.png")
        plt.show()
    
    def test_model_prediction(self):
        """Test the saved model with a sample prediction"""
        try:
            # Load the saved model
            model = joblib.load("models/xgboost_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
            
            print("\nTesting saved model...")
            
            # Get current market data for testing
            df = self.create_multi_timeframe_features()
            if not df.empty:
                # Get latest features
                feature_cols = [col for col in df.columns if col not in 
                               ['time', 'future_high', 'future_low', 'future_high_pct', 'future_low_pct', 'target']]
                
                latest_features = df[feature_cols].iloc[-1:].fillna(0)
                
                # Scale and predict
                features_scaled = scaler.transform(latest_features)
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                
                signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                print(f"Test prediction: {signal_map[prediction]}")
                print(f"Probabilities: SELL={probabilities[0]:.3f}, HOLD={probabilities[1]:.3f}, BUY={probabilities[2]:.3f}")
                print("✓ Model test successful!")
            else:
                print("⚠ Could not get current data for testing")
                
        except Exception as e:
            print(f"✗ Model test failed: {e}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train XGBoost model for XAUUSDmm trading')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--test', action='store_true', help='Test the saved model')
    args = parser.parse_args()
    
    try:
        print("="*80)
        print("AI SCALPER XAUUSD - MODEL TRAINING")
        print("="*80)
        
        # Initialize trainer
        trainer = XGBoostTrainer()
        
        # Train model
        accuracy, feature_importance = trainer.train_model(hyperparameter_tuning=args.tune)
        
        # Generate plots if requested
        if args.plot:
            trainer.plot_feature_importance(feature_importance)
            # Plot target distribution would require getting the target data
        
        # Save model
        trainer.save_model()
        
        # Test model if requested
        if args.test:
            trainer.test_model_prediction()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Final test accuracy: {accuracy:.4f}")
        print("Model files saved in 'models/' directory")
        print("You can now run the AI signal generator!")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()