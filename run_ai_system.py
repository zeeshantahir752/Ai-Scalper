#!/usr/bin/env python3
"""
Main runner script for AI Scalper XAUUSDm System
Handles system initialization, monitoring, and continuous operation
"""

import os
import sys
import time
import logging
import signal
import threading
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from ai_signal_generator_advanced import AdvancedAISignalGenerator
    from train_model_xgboost import XGBoostTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required Python files are in the current directory")
    sys.exit(1)

class AIScalperSystem:
    def __init__(self, config_file="config.json"):
        """Initialize the AI Scalper System"""
        self.config_file = config_file
        self.generator = None
        self.running = False
        self.system_stats = {
            'start_time': None,
            'signals_generated': 0,
            'last_signal_time': None,
            'errors': 0,
            'uptime': 0
        }
        
        self.setup_logging()
        self.setup_signal_handlers()
        self.load_config()
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AIScalperSystem')
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        
    def load_config(self):
        """Load system configuration"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_file} not found, using defaults")
            self.config = {
                "system": {
                    "check_interval": 60,
                    "max_errors": 10,
                    "restart_on_error": True
                }
            }
            
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['signals', 'logs', 'models', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.info(f"✓ Directory ready: {directory}")
            
    def check_dependencies(self):
        """Check if all dependencies are available"""
        self.logger.info("Checking dependencies...")
        
        required_modules = [
            'MetaTrader5', 'pandas', 'numpy', 'xgboost', 
            'sklearn', 'talib', 'matplotlib', 'seaborn', 'joblib'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.logger.error(f"Missing dependencies: {missing_modules}")
            return False
        
        self.logger.info("✓ All dependencies available")
        return True
        
    def check_model_availability(self):
        """Check if trained model exists"""
        model_files = [
            "models/xgboost_model.pkl",
            "models/scaler.pkl"
        ]
        
        missing_files = []
        for file_path in model_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.error(f"Missing model files: {missing_files}")
            return False
            
        self.logger.info("✓ Trained models found")
        return True
        
    def check_mt5_connection(self):
        """Check MetaTrader 5 connection"""
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                self.logger.error("Failed to initialize MT5 connection")
                return False
                
            # Check if XAUUSD symbol is available
            symbol_info = mt5.symbol_info("XAUUSDm")
            if symbol_info is None:
                self.logger.error("XAUUSD symbol not found")
                return False
                
            self.logger.info("✓ MT5 connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
            
    def train_model_interactive(self):
        """Interactive model training"""
        self.logger.info("="*60)
        self.logger.info("MODEL TRAINING REQUIRED")
        self.logger.info("="*60)
        
        print("\nThe AI model needs to be trained before use.")
        print("This process will:")
        print("1. Download historical XAUUSD data from MT5")
        print("2. Calculate technical indicators")
        print("3. Train XGBoost machine learning model")
        print("4. Save the trained model for use")
        print("\nEstimated time: 5-15 minutes depending on your system")
        
        response = input("\nDo you want to train the model now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            try:
                self.logger.info("Starting model training...")
                trainer = XGBoostTrainer()
                
                # Train the model
                accuracy, feature_importance = trainer.train_model()
                
                # Save the model
                trainer.save_model()
                
                self.logger.info(f"✓ Model training completed! Accuracy: {accuracy:.4f}")
                
                # Test the model
                trainer.test_model_prediction()
                
                return True
                
            except Exception as e:
                self.logger.error(f"✗ Model training failed: {e}")
                return False
        else:
            print("\nTo train the model manually, run:")
            print("python train_model_xgboost.py")
            return False
            
    def start_monitoring(self):
        """Start system monitoring thread"""
        def monitor():
            while self.running:
                try:
                    # Update uptime
                    if self.system_stats['start_time']:
                        self.system_stats['uptime'] = (
                            datetime.now() - self.system_stats['start_time']
                        ).total_seconds()
                    
                    # Log system status every 10 minutes
                    if int(time.time()) % 600 == 0:
                        self.log_system_status()
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("✓ System monitoring started")
        
    def log_system_status(self):
        """Log current system status"""
        uptime_str = str(timedelta(seconds=int(self.system_stats['uptime'])))
        
        status_msg = (
            f"System Status - "
            f"Uptime: {uptime_str}, "
            f"Signals: {self.system_stats['signals_generated']}, "
            f"Errors: {self.system_stats['errors']}"
        )
        
        self.logger.info(status_msg)
        
    def display_startup_banner(self):
        """Display startup banner"""
        banner = """
        ╔══════════════════════════════════════════════════════════════╗
        ║                    AI SCALPER XAUUSD                         ║
        ║                  Advanced Trading System                     ║
        ║                                                              ║
        ║  🤖 AI-Powered Signal Generation                             ║
        ║  📊 Multi-Timeframe Technical Analysis                      ║
        ║  ⚡ Real-time MT5 Integration                               ║
        ║  🛡️ Advanced Risk Management                                ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def display_system_info(self):
        """Display system information"""
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        
        # Python version
        print(f"Python Version: {sys.version.split()[0]}")
        
        # System configuration
        if hasattr(self, 'config'):
            print(f"Symbol: {self.config.get('symbol', 'XAUUSDm')}")
            print(f"Signal Threshold: {self.config.get('signal_threshold', 0.7)}")
            print(f"Max Daily Signals: {self.config.get('max_daily_signals', 20)}")
        
        # File status
        model_status = "✓ Available" if self.check_model_availability() else "✗ Missing"
        print(f"AI Model: {model_status}")
        
        mt5_status = "✓ Connected" if self.check_mt5_connection() else "✗ Not Connected"
        print(f"MT5 Connection: {mt5_status}")
        
        print("="*60)
        
    def start_signal_generation(self):
        """Start the AI signal generation process"""
        try:
            # Initialize the generator
            self.generator = AdvancedAISignalGenerator(self.config_file)
            self.logger.info("✓ AI Signal Generator initialized")
            
            # Update system stats
            self.system_stats['start_time'] = datetime.now()
            
            # Start monitoring
            self.start_monitoring()
            
            # Main signal generation loop
            self.logger.info("🚀 Starting signal generation...")
            print("\n" + "-"*60)
            print("LIVE SIGNAL GENERATION")
            print("-"*60)
            print("Press Ctrl+C to stop")
            print()
            
            interval = self.config.get('system', {}).get('check_interval', 60)
            
            while self.running:
                try:
                    # Check if market is open
                    if not self.generator.is_market_open():
                        self.logger.info("⏳ Market closed, waiting...")
                        time.sleep(300)  # Wait 5 minutes
                        continue
                    
                    # Generate signal
                    signal_data = self.generator.generate_signal()
                    
                    # Save signal to file
                    self.generator.save_signal_to_file(signal_data)
                    
                    # Update stats
                    self.system_stats['signals_generated'] += 1
                    self.system_stats['last_signal_time'] = datetime.now()
                    
                    # Display signal info
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    confidence_pct = signal_data['confidence'] * 100
                    
                    if signal_data['signal'] in ['BUY', 'SELL']:
                        print(f"[{timestamp}] 🎯 {signal_data['signal']} - "
                              f"Confidence: {confidence_pct:.1f}% - "
                              f"Price: {signal_data['price']:.2f}")
                    else:
                        print(f"[{timestamp}] ⏸️  {signal_data['signal']} - "
                              f"Confidence: {confidence_pct:.1f}%")
                    
                    # Wait for next iteration
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.system_stats['errors'] += 1
                    self.logger.error(f"Signal generation error: {e}")
                    
                    # Check if we should restart
                    max_errors = self.config.get('system', {}).get('max_errors', 10)
                    if self.system_stats['errors'] >= max_errors:
                        self.logger.error(f"Too many errors ({max_errors}), stopping system")
                        break
                    
                    time.sleep(30)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Fatal error in signal generation: {e}")
            
    def display_final_stats(self):
        """Display final statistics"""
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        
        if self.system_stats['start_time']:
            session_duration = datetime.now() - self.system_stats['start_time']
            print(f"Session Duration: {str(session_duration).split('.')[0]}")
        
        print(f"Signals Generated: {self.system_stats['signals_generated']}")
        print(f"Errors Encountered: {self.system_stats['errors']}")
        
        if self.system_stats['last_signal_time']:
            print(f"Last Signal: {self.system_stats['last_signal_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("="*60)
        
    def start_system(self):
        """Start the complete AI scalper system"""
        try:
            # Display banner
            self.display_startup_banner()
            
            # System checks
            self.logger.info("Performing system checks...")
            self.setup_directories()
            
            if not self.check_dependencies():
                print("❌ Please install missing dependencies: pip install -r requirements.txt")
                return False
                
            if not self.check_mt5_connection():
                print("❌ Please ensure MetaTrader 5 is running and logged in")
                return False
                
            if not self.check_model_availability():
                if not self.train_model_interactive():
                    return False
                    
            # Display system info
            self.display_system_info()
            
            # Start signal generation
            self.running = True
            self.start_signal_generation()
            
        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.shutdown()
            
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down AI Scalper system...")
        
        self.running = False
        
        # Close MT5 connection
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
            self.logger.info("✓ MT5 connection closed")
        except:
            pass
            
        # Display final stats
        self.display_final_stats()
        
        print("\n🛑 AI Scalper system stopped")
        print("Thank you for using AI Scalper XAUUSDm!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Scalper XAUUSDm Trading System')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--train-only', action='store_true', help='Only train the model and exit')
    parser.add_argument('--test-signal', action='store_true', help='Generate a test signal and exit')
    args = parser.parse_args()
    
    if args.train_only:
        # Train model only
        try:
            trainer = XGBoostTrainer()
            accuracy, _ = trainer.train_model()
            trainer.save_model()
            print(f"Model training completed with accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Training failed: {e}")
        return
    
    if args.test_signal:
        # Generate test signal only
        try:
            generator = AdvancedAISignalGenerator(args.config)
            signal = generator.generate_signal()
            print(f"Test signal: {signal}")
        except Exception as e:
            print(f"Signal test failed: {e}")
        return
    
    # Start full system
    system = AIScalperSystem(args.config)
    system.start_system()

if __name__ == "__main__":
    main()