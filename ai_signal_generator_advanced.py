#!/usr/bin/env python3
"""
Advanced AI Signal Generator for XAUUSD Trading
Uses XGBoost machine learning with multi-timeframe technical analysis
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import talib
import datetime
import time
import json
import logging
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class AdvancedAISignalGenerator:
    def __init__(self, config_file: str = "config.json"):
        """Initialize the AI Signal Generator with advanced features"""
        self.setup_logging()
        self.load_config(config_file)
        self.model = None
        self.scaler = None
        self.anomaly_detector = None
        self.last_signal = None
        self.signal_history = []
        
        # Initialize MT5 connection
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            raise Exception("Failed to initialize MT5")
        
        self.logger.info("AI Signal Generator initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_signals.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "symbol": "XAUUSDm",
                "timeframes": [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15],
                "lookback_periods": 500,
                "signal_threshold": 0.7,
                "anomaly_threshold": -0.1,
                "max_daily_signals": 20,
                "risk_per_trade": 0.02,
                "model_path": "models/xgboost_model.pkl",
                "scaler_path": "models/scaler.pkl",
                "anomaly_detector_path": "models/anomaly_detector.pkl",
                "signal_file": "signals/xauusdm_signal.txt"
            }
            # Save default config
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
    
    def get_market_data(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """Fetch market data from MT5"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                self.logger.error(f"Failed to get market data for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if len(df) < 50:
            return df
        
        try:
            # Price-based indicators
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
            
            # Pattern recognition
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            
            # Custom indicators
            df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
            df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
            df['macd_bullish'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
            
            # Price action features
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['true_range'] = np.maximum(df['high'] - df['low'], 
                                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                                 abs(df['low'] - df['close'].shift(1))))
            
            # Trend indicators
            df['trend_sma'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
            df['trend_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def calculate_multi_timeframe_features(self, symbol: str) -> pd.DataFrame:
        """Calculate features from multiple timeframes"""
        features_dict = {}
        
        try:
            for i, tf in enumerate(self.config["timeframes"]):
                df = self.get_market_data(symbol, tf, self.config["lookback_periods"])
                if df.empty:
                    continue
                    
                df = self.calculate_technical_indicators(df)
                
                # Get the latest values for each timeframe
                if len(df) > 0:
                    latest = df.iloc[-1]
                    tf_name = f"tf_{i}"
                    
                    # Select key features
                    feature_cols = ['rsi', 'macd', 'macd_signal', 'stoch_k', 'williams_r', 
                                   'cci', 'price_position', 'atr', 'rsi_oversold', 'rsi_overbought', 
                                   'macd_bullish', 'trend_sma', 'trend_ema']
                    
                    for col in feature_cols:
                        if col in df.columns and not pd.isna(latest[col]):
                            features_dict[f"{tf_name}_{col}"] = latest[col]
                    
                    # Add momentum features
                    if len(df) >= 20:
                        features_dict[f"{tf_name}_momentum"] = latest['close'] / df['close'].iloc[-20] - 1
                        features_dict[f"{tf_name}_volatility"] = df['true_range'].rolling(20).mean().iloc[-1] / latest['close']
            
            return pd.DataFrame([features_dict]) if features_dict else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-timeframe features: {e}")
            return pd.DataFrame()
    
    def detect_market_anomalies(self, features: pd.DataFrame) -> bool:
        """Detect market anomalies using Isolation Forest"""
        if self.anomaly_detector is None:
            return False
        
        try:
            anomaly_score = self.anomaly_detector.decision_function(features)
            return anomaly_score[0] < self.config["anomaly_threshold"]
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return False
    
    def load_models(self):
        """Load trained models and scalers"""
        try:
            if os.path.exists(self.config["model_path"]):
                self.model = joblib.load(self.config["model_path"])
                self.logger.info("XGBoost model loaded successfully")
            else:
                self.logger.error(f"Model file not found: {self.config['model_path']}")
                return False
                
            if os.path.exists(self.config["scaler_path"]):
                self.scaler = joblib.load(self.config["scaler_path"])
                self.logger.info("Scaler loaded successfully")
            else:
                self.logger.error(f"Scaler file not found: {self.config['scaler_path']}")
                return False
                
            if os.path.exists(self.config["anomaly_detector_path"]):
                self.anomaly_detector = joblib.load(self.config["anomaly_detector_path"])
                self.logger.info("Anomaly detector loaded successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def generate_signal(self) -> Dict:
        """Generate trading signal using AI model"""
        try:
            # Load models if not already loaded
            if self.model is None or self.scaler is None:
                if not self.load_models():
                    return {"signal": "NONE", "confidence": 0, "reason": "Models not available"}
            
            # Get multi-timeframe features
            features = self.calculate_multi_timeframe_features(self.config["symbol"])
            
            if features.empty:
                return {"signal": "NONE", "confidence": 0, "reason": "No data available"}
            
            # Check for missing values and fill them
            features_clean = features.fillna(0)
            
            # Ensure we have the expected number of features
            if features_clean.shape[1] == 0:
                return {"signal": "NONE", "confidence": 0, "reason": "No valid features"}
            
            # Scale features
            features_scaled = self.scaler.transform(features_clean)
            
            # Detect anomalies
            is_anomaly = self.detect_market_anomalies(features_scaled)
            
            if is_anomaly:
                return {"signal": "NONE", "confidence": 0, "reason": "Market anomaly detected"}
            
            # Generate prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Determine signal (assuming classes: 0=SELL, 1=HOLD, 2=BUY)
            if len(prediction_proba) >= 3:
                sell_prob = prediction_proba[0]
                hold_prob = prediction_proba[1] 
                buy_prob = prediction_proba[2]
            else:
                # Fallback for binary classification
                sell_prob = prediction_proba[0] if len(prediction_proba) > 0 else 0
                buy_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
                hold_prob = 0
            
            max_prob = max(buy_prob, sell_prob, hold_prob)
            
            if max_prob < self.config["signal_threshold"]:
                signal = "NONE"
                confidence = max_prob
            elif buy_prob == max_prob:
                signal = "BUY"
                confidence = buy_prob
            elif sell_prob == max_prob:
                signal = "SELL"
                confidence = sell_prob
            else:
                signal = "NONE"
                confidence = hold_prob
            
            # Get current market info
            tick = mt5.symbol_info_tick(self.config["symbol"])
            current_price = tick.ask if tick else 0
            spread = tick.spread if tick else 0
            
            signal_data = {
                "signal": signal,
                "confidence": round(confidence, 4),
                "price": current_price,
                "spread": spread,
                "timestamp": datetime.datetime.now().isoformat(),
                "features_count": len(features_clean.columns),
                "reason": f"AI prediction with {confidence:.2%} confidence"
            }
            
            self.last_signal = signal_data
            self.signal_history.append(signal_data)
            
            # Keep only last 100 signals in memory
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {"signal": "NONE", "confidence": 0, "reason": f"Error: {str(e)}"}
    
    def save_signal_to_file(self, signal_data: Dict):
        """Save signal to file for MT5 EA to read"""
        try:
            # Create signals directory if it doesn't exist
            os.makedirs("signals", exist_ok=True)
            
            with open(self.config["signal_file"], 'w') as f:
                json.dump(signal_data, f, indent=2)
            self.logger.info(f"Signal saved: {signal_data['signal']} with confidence {signal_data['confidence']}")
        except Exception as e:
            self.logger.error(f"Error saving signal to file: {e}")
    
    def is_market_open(self) -> bool:
        """Check if the market is open for trading"""
        try:
            symbol_info = mt5.symbol_info(self.config["symbol"])
            return symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def run_continuous(self, interval_seconds: int = 60):
        """Run the signal generator continuously"""
        self.logger.info("Starting continuous signal generation...")
        
        try:
            while True:
                # Check if market is open
                if not self.is_market_open():
                    self.logger.info("Market is closed, waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Generate signal
                signal_data = self.generate_signal()
                
                # Save signal to file
                self.save_signal_to_file(signal_data)
                
                # Log signal
                self.logger.info(f"Generated signal: {signal_data}")
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Signal generation stopped by user")
        except Exception as e:
            self.logger.error(f"Error in continuous run: {e}")
        finally:
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
    
    def get_signal_statistics(self) -> Dict:
        """Get statistics about generated signals"""
        if not self.signal_history:
            return {"total_signals": 0}
        
        signals_df = pd.DataFrame(self.signal_history)
        
        stats = {
            "total_signals": len(signals_df),
            "buy_signals": len(signals_df[signals_df['signal'] == 'BUY']),
            "sell_signals": len(signals_df[signals_df['signal'] == 'SELL']),
            "none_signals": len(signals_df[signals_df['signal'] == 'NONE']),
            "avg_confidence": signals_df['confidence'].mean(),
            "max_confidence": signals_df['confidence'].max(),
            "min_confidence": signals_df['confidence'].min()
        }
        
        return stats

if __name__ == "__main__":
    # Initialize and run the AI signal generator
    try:
        generator = AdvancedAISignalGenerator()
        
        # Generate a single signal for testing
        signal = generator.generate_signal()
        print(f"Generated signal: {signal}")
        
        # Get statistics
        stats = generator.get_signal_statistics()
        print(f"Signal statistics: {stats}")
        
        # Uncomment to run continuously
        # generator.run_continuous(60)  # Generate signal every minute
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mt5.shutdown()