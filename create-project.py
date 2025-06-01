#!/usr/bin/env python3
"""
Quick setup script for AI Scalper with Python 3.11.6 and Exness MT5
"""

import sys
import subprocess
import json
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor != 11:
        print("⚠️  Warning: This script is optimized for Python 3.11.x")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print("✅ Python 3.11.x detected - perfect!")

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "MetaTrader5>=5.0.45",
        "pandas>=1.5.0",
        "numpy>=1.21.0,<2.0.0", 
        "xgboost>=1.7.0",
        "scikit-learn>=1.1.0",
        "TA-Lib>=0.4.24",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "python-json-logger>=2.0.0",
        "python-dateutil>=2.8.0",
        "pytz>=2021.3",
        "psutil>=5.8.0",
        "flask>=2.0.0"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            if "TA-Lib" in package:
                print("💡 TA-Lib installation failed. Try manual installation:")
                print("   Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
                print("   For Python 3.11: TA_Lib‑0.4.24‑cp311‑cp311‑win_amd64.whl")

def create_exness_config():
    """Create Exness-specific configuration"""
    print("\n⚙️  Creating Exness configuration...")
    
    config = {
        "symbol": "XAUUSDm",
        "timeframes": [1, 5, 15, 60],
        "lookback_periods": 500,
        "signal_threshold": 0.75,
        "anomaly_threshold": -0.1,
        "max_daily_signals": 15,
        "risk_per_trade": 0.015,
        
        "paths": {
            "model_path": "models/xgboost_model.pkl",
            "scaler_path": "models/scaler.pkl", 
            "anomaly_detector_path": "models/anomaly_detector.pkl",
            "signal_file": "signals/xauusdm_signal.txt"
        },
        
        "mt5_settings": {
            "login": 0,
            "password": "",
            "server": "Exness-MT5Trial4",
            "timeout": 10000,
            "retries": 3,
            "symbol_suffix": ""
        },
        
        "trading_schedule": {
            "start_hour": 1,
            "end_hour": 23,
            "weekdays_only": True,
            "pause_on_friday": True,
            "friday_close_hour": 21,
            "avoid_news_hours": [
                {"start": "08:30", "end": "09:30", "description": "Economic data releases"},
                {"start": "13:30", "end": "14:30", "description": "US session opening"},
                {"start": "15:30", "end": "16:30", "description": "Federal Reserve announcements"}
            ]
        },
        
        "risk_management": {
            "max_drawdown_percent": 8.0,
            "equity_stop_percent": 4.0,
            "max_trades_per_day": 15,
            "min_confidence": 0.75,
            "max_spread_points": 50,
            "position_sizing": {
                "method": "percentage",
                "risk_per_trade": 0.015,
                "max_lot_size": 1.0,
                "min_lot_size": 0.01
            },
            "stop_loss": {
                "default_points": 250,
                "max_points": 500,
                "min_points": 100
            },
            "take_profit": {
                "default_points": 500,
                "risk_reward_ratio": 2.0
            }
        },
        
        "market_filters": {
            "spread_filter": {
                "enabled": True,
                "max_spread_points": 50
            },
            "volatility_filter": {
                "enabled": True,
                "max_volatility_percent": 2.5
            }
        },
        
        "exness_specific": {
            "server_timezone": "EET",
            "spread_type": "variable",
            "execution_type": "market",
            "account_currency": "USD",
            "leverage": 1000,
            "commission_per_lot": 0.0
        }
    }
    
    # Create directories
    dirs = ["models", "signals", "logs", "data"]
    for directory in dirs:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Save configuration
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("✅ Exness configuration created: config.json")

def test_mt5_connection():
    """Test MT5 connection"""
    print("\n🔌 Testing MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            print("💡 Make sure MT5 is running and you're logged in to Exness")
            return False
        
        # Test basic functionality
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Cannot get account info")
            mt5.shutdown()
            return False
        
        print(f"✅ MT5 connected successfully")
        print(f"   Account: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Balance: ${account_info.balance:,.2f}")
        print(f"   Leverage: 1:{account_info.leverage}")
        
        # Test XAUUSD symbol
        symbol_info = mt5.symbol_info("XAUUSDm")
        if symbol_info is None:
            print("❌ XAUUSD symbol not found")
            print("💡 Make sure XAUUSD is available in your Exness account")
        else:
            print(f"✅ XAUUSD symbol available")
            print(f"   Current spread: {symbol_info.spread} points")
            print(f"   Min lot: {symbol_info.volume_min}")
            print(f"   Max lot: {symbol_info.volume_max}")
        
        mt5.shutdown()
        return True
        
    except ImportError:
        print("❌ MetaTrader5 package not installed")
        print("💡 Install with: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"❌ MT5 connection error: {e}")
        return False

def test_talib():
    """Test TA-Lib installation"""
    print("\n📊 Testing TA-Lib...")
    
    try:
        import talib
        import numpy as np
        
        # Test basic calculation
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        sma = talib.SMA(data, timeperiod=5)
        rsi = talib.RSI(data, timeperiod=5)
        
        print(f"✅ TA-Lib {talib.__version__} working correctly")
        print(f"   Test SMA: {sma[-1]:.2f}")
        print(f"   Test RSI: {rsi[-1]:.2f}")
        return True
        
    except ImportError:
        print("❌ TA-Lib not installed")
        print("💡 For Python 3.11 on Windows:")
        print("   Download: TA_Lib‑0.4.24‑cp311‑cp311‑win_amd64.whl")
        print("   From: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("   Install: pip install TA_Lib‑0.4.24‑cp311‑cp311‑win_amd64.whl")
        return False
    except Exception as e:
        print(f"❌ TA-Lib error: {e}")
        return False

def create_exness_ea_settings():
    """Create recommended EA settings for Exness"""
    print("\n🎯 Creating Exness EA settings guide...")
    
    ea_settings = """
=== AI SCALPER PRO - EXNESS MT5 SETTINGS ===

Attach the EA to XAUUSD chart with these parameters:

=== AI Signal Settings ===
SignalFilePath = signals\\xau_signal.txt
SignalTimeoutSeconds = 300
MinConfidence = 0.75
UseAISignalsOnly = true

=== Risk Management ===
RiskPercentage = 1.5
MaxDrawdownPercent = 8.0
MaxTradesPerDay = 15
MaxLotSize = 1.0
MinLotSize = 0.01

=== Position Management ===
StopLossPoints = 250
TakeProfitPoints = 500
UseTrailingStop = true
TrailingStopPoints = 150
TrailingStepPoints = 50

=== Market Filters ===
MaxSpreadPoints = 50
UseVolatilityFilter = true
MaxVolatilityPercent = 2.5

=== Exness-Specific ===
- Chart timeframe: M1 or M5
- Enable AutoTrading in MT5
- Ensure Expert Advisors are allowed
- Check your Exness account has sufficient margin
- XAUUSD symbol should show in Market Watch

=== Trading Hours (Server Time) ===
StartHour = 1 (Monday 01:00)
EndHour = 23 (Friday 23:00)
Avoid trading during:
- Sunday 22:00 - Monday 01:00 (weekend gap)
- Major news events (especially US/EU economic data)

=== Live Trading Checklist ===
□ Demo tested for at least 1 week
□ Positive backtest results
□ Risk settings appropriate for account size
□ Stop loss and take profit levels set
□ Maximum daily loss limit configured
□ Emergency stop procedures understood
    """
    
    with open("EXNESS_EA_SETTINGS.txt", "w") as f:
        f.write(ea_settings)
    
    print("✅ EA settings guide created: EXNESS_EA_SETTINGS.txt")

def main():
    """Main setup function"""
    print("🚀 AI Scalper XAUUSDm Setup for Python 3.11.6 + Exness")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Test TA-Lib
    talib_ok = test_talib()
    
    # Test MT5
    mt5_ok = test_mt5_connection()
    
    # Create configuration
    create_exness_config()
    
    # Create EA settings guide
    create_exness_ea_settings()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 SETUP SUMMARY")
    print("=" * 60)
    print(f"✅ Python 3.11.6: Ready")
    print(f"{'✅' if talib_ok else '❌'} TA-Lib: {'Ready' if talib_ok else 'Needs manual installation'}")
    print(f"{'✅' if mt5_ok else '❌'} MT5 Exness: {'Connected' if mt5_ok else 'Check connection'}")
    print(f"✅ Configuration: Created for Exness")
    print(f"✅ EA Settings: Guide created")
    
    print("\n📋 NEXT STEPS:")
    print("1. Copy AI_Scalper_Pro_XAUUSD.mq5 to MT5 Experts folder")
    print("2. Compile EA in MetaEditor")
    print("3. Train AI model: python train_model_xgboost.py")
    print("4. Start system: python run_ai_system.py")
    print("5. Attach EA to XAUUSD chart with recommended settings")
    
    if not talib_ok:
        print("\n⚠️  IMPORTANT: Install TA-Lib manually before proceeding!")
    if not mt5_ok:
        print("\n⚠️  IMPORTANT: Fix MT5 connection before proceeding!")

if __name__ == "__main__":
    main()