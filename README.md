# 🤖 AI Scalper XAUUSDm - Complete Trading System

An advanced AI-powered scalping system for XAU/USD (Gold) trading, featuring machine learning signals, automated MT5 execution, and comprehensive monitoring tools.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📁 Complete File Structure](#-complete-file-structure)
- [💾 Download Instructions](#-download-instructions)
- [⚙️ Installation](#️-installation)
- [🎯 Usage Guide](#-usage-guide)
- [📊 Features](#-features)
- [🔧 Configuration](#-configuration)
- [📈 Monitoring & Analysis](#-monitoring--analysis)
- [⚠️ Risk Disclaimer](#️-risk-disclaimer)
- [🛠️ Troubleshooting](#️-troubleshooting)

## 🚀 Quick Start

### Prerequisites
- **MetaTrader 5** (latest version)
- **Python 3.8+** with pip
- **Windows/Linux/macOS** system
- **Demo/Live trading account** with XAU/USD access
- **4GB+ RAM** and stable internet connection

### 30-Second Setup
```bash
# 1. Download and extract the project
# 2. Run installer
./install.sh          # Linux/macOS
# OR
install.bat           # Windows

# 3. Train the AI model
python train_model_xgboost.py

# 4. Start the system
python run_ai_system.py
```

## 📁 Complete File Structure

```
ai-scalper-xauusd/
│
├── 📄 README.md                          # This comprehensive guide
├── 📄 requirements.txt                   # Python dependencies
├── 📄 config.json                       # Main configuration file
├── 📄 .gitignore                        # Git ignore rules
├── 📄 LICENSE                           # MIT license
│
├── 🐍 Core Python Files
│   ├── ai_signal_generator_advanced.py   # Main AI signal generator
│   ├── train_model_xgboost.py           # ML model training
│   ├── run_ai_system.py                 # System orchestrator
│   ├── utils.py                         # Utility functions
│   ├── monitoring.py                    # GUI/Console monitoring
│   ├── backtest.py                      # Backtesting engine
│   └── web_dashboard.py                 # Web-based dashboard
│
├── 📂 mt5_ea/
│   └── AI_Scalper_Pro_XAUUSD.mq5        # MetaTrader 5 Expert Advisor
│
├── 📂 models/                           # AI models (auto-created)
│   ├── xgboost_model.pkl               # Trained XGBoost model
│   ├── scaler.pkl                      # Feature scaler
│   └── anomaly_detector.pkl            # Anomaly detection model
│
├── 📂 signals/                          # Signal communication
│   └── xau_signal.txt                  # Current AI signal (JSON)
│
├── 📂 logs/                             # System logs
│   ├── ai_signals.log                  # Signal generation logs
│   ├── system.log                      # System operations
│   └── backtest.log                    # Backtesting logs
│
├── 📂 data/                             # Data storage
│   ├── signal_history.json             # Historical signals
│   └── price_cache/                    # Cached price data
│
├── 📂 scripts/                          # Installation & utilities
│   ├── install.sh                      # Linux/macOS installer
│   ├── install.bat                     # Windows installer
│   ├── start_ai_scalper.sh            # Quick launcher (Unix)
│   ├── start_ai_scalper.bat           # Quick launcher (Windows)
│   ├── train_model.sh                 # Model training script
│   └── launcher.sh                    # Interactive menu
│
└── 📂 documentation/                    # Additional docs
    ├── SETUP.md                        # Detailed setup guide
    ├── CONFIGURATION.md                # Configuration reference
    ├── API.md                          # API documentation
    └── TROUBLESHOOTING.md              # Common issues & solutions
```

## 💾 Download Instructions

### Method 1: Direct Download
1. **Download each file** from the artifacts I provided
2. **Create the folder structure** as shown above
3. **Place each file** in its corresponding location
4. **Run the installer** script for your operating system

### Method 2: Manual Creation
Save each code file from the artifacts with these exact names:

**Core Python Files:**
- `ai_signal_generator_advanced.py` - Main AI engine
- `train_model_xgboost.py` - Model training
- `run_ai_system.py` - System runner
- `utils.py` - Utility functions
- `monitoring.py` - Monitoring dashboard
- `backtest.py` - Backtesting engine
- `web_dashboard.py` - Web interface

**Configuration Files:**
- `config.json` - Main configuration
- `requirements.txt` - Python dependencies

**MT5 Expert Advisor:**
- `mt5_ea/AI_Scalper_Pro_XAUUSD.mq5` - Expert Advisor

**Installation Scripts:**
- `scripts/install.sh` (Linux/macOS)
- `scripts/install.bat` (Windows)

## ⚙️ Installation

### Automated Installation

#### Windows
```cmd
# Download and extract project files
# Open Command Prompt in project directory
scripts\install.bat
```

#### Linux/macOS
```bash
# Download and extract project files
# Open Terminal in project directory
chmod +x scripts/install.sh
scripts/install.sh
```

### Manual Installation

#### 1. Install Python Dependencies
```bash
# Create virtual environment
python -m venv ai_scalper_env

# Activate environment
# Windows:
ai_scalper_env\Scripts\activate
# Linux/macOS:
source ai_scalper_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

#### 2. Install TA-Lib (Technical Analysis Library)

**Windows:**
```cmd
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
```

**Linux:**
```bash
# Install dependencies
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

#### 3. Setup MetaTrader 5
1. **Install MT5** and create demo/live account
2. **Copy EA**: Place `AI_Scalper_Pro_XAUUSD.mq5` in `MQL5/Experts/` folder
3. **Compile EA**: Open in MetaEditor and compile
4. **Enable settings**: Allow DLL imports and external experts in MT5

## 🎯 Usage Guide

### Step 1: Train the AI Model
```bash
# Train with default settings
python train_model_xgboost.py

# Train with hyperparameter tuning (takes longer)
python train_model_xgboost.py --tune

# Train with visualization
python train_model_xgboost.py --plot

# Test the trained model
python train_model_xgboost.py --test
```

**Training Output:**
- Model accuracy and performance metrics
- Feature importance analysis
- Cross-validation results
- Saved models in `models/` directory

### Step 2: Configure the System
Edit `config.json` to customize:

```json
{
    "symbol": "XAUUSDm",
    "signal_threshold": 0.70,
    "risk_per_trade": 0.02,
    "max_daily_signals": 20,
    "trading_schedule": {
        "start_hour": 0,
        "end_hour": 23,
        "weekdays_only": true
    }
}
```

### Step 3: Deploy MT5 Expert Advisor
1. **Attach EA** to XAUUSD chart
2. **Configure parameters**:
   - Risk per trade: 1-2%
   - Stop Loss: 200 points
   - Take Profit: 400 points
   - AI confidence threshold: 70%
3. **Enable AutoTrading** in MT5

### Step 4: Start the AI System

#### Option A: Full System
```bash
python run_ai_system.py
```

#### Option B: Training Only
```bash
python run_ai_system.py --train-only
```

#### Option C: Test Signal
```bash
python run_ai_system.py --test-signal
```

### Step 5: Monitor the System

#### GUI Monitor
```bash
python monitoring.py
```

#### Console Monitor
```bash
python monitoring.py --console
```

#### Web Dashboard
```bash
python web_dashboard.py
# Open browser: http://localhost:5000
```

## 📊 Features

### 🧠 AI Engine
- **XGBoost Machine Learning** - Advanced gradient boosting classifier
- **Multi-Timeframe Analysis** - M1, M5, M15, H1 data fusion
- **50+ Technical Indicators** - RSI, MACD, Bollinger Bands, Stochastic, etc.
- **Anomaly Detection** - Isolation Forest for market irregularities
- **Confidence Scoring** - Only trade high-confidence signals
- **Feature Importance** - Understand what drives predictions

### 🎯 Trading Features
- **Real-time Signal Generation** - Continuous AI analysis
- **Automated MT5 Execution** - Seamless integration
- **Risk-based Position Sizing** - Dynamic lot calculation
- **Multi-layer Protection** - Stop losses, trailing stops, equity protection
- **Market Filters** - Spread, volatility, and news avoidance
- **Emergency Controls** - Instant stop and position closure

### 📈 Risk Management
- **Dynamic Position Sizing** - Based on account balance and risk tolerance
- **Stop Loss & Take Profit** - Configurable risk-reward ratios
- **Trailing Stops** - Lock in profits as positions move favorably
- **Daily Limits** - Maximum trades and loss limits per day
- **Drawdown Protection** - Automatic shutdown on excessive losses
- **Equity Monitoring** - Real-time account protection

### 📊 Monitoring & Analysis
- **Real-time Dashboard** - GUI and web-based monitoring
- **Performance Analytics** - Win rate, profit factor, Sharpe ratio
- **Signal History** - Track AI decision accuracy
- **System Health** - CPU, memory, and connection monitoring
- **Backtesting Engine** - Historical performance validation
- **Comprehensive Reporting** - HTML reports with charts

## 🔧 Configuration

### Main Configuration (`config.json`)

#### Trading Parameters
```json
{
    "symbol": "XAUUSDm",
    "signal_threshold": 0.70,
    "max_daily_signals": 20,
    "risk_per_trade": 0.02
}
```

#### Risk Management
```json
{
    "risk_management": {
        "max_drawdown_percent": 10.0,
        "equity_stop_percent": 5.0,
        "stop_loss": {
            "default_points": 200,
            "use_atr_multiplier": false
        },
        "take_profit": {
            "default_points": 400,
            "risk_reward_ratio": 2.0
        }
    }
}
```

#### Trading Schedule
```json
{
    "trading_schedule": {
        "start_hour": 0,
        "end_hour": 23,
        "weekdays_only": true,
        "avoid_news_hours": [
            {"start": "08:30", "end": "09:30"},
            {"start": "13:30", "end": "14:30"}
        ]
    }
}
```

### MT5 EA Configuration
Configure these parameters when attaching the EA:

| Parameter | Default | Description |
|-----------|---------|-------------|
| RiskPercentage | 2.0 | Risk per trade (%) |
| MinConfidence | 0.70 | Minimum AI confidence |
| StopLossPoints | 200 | Stop loss in points |
| TakeProfitPoints | 400 | Take profit in points |
| MaxTradesPerDay | 20 | Daily trade limit |
| UseTrailingStop | true | Enable trailing stops |

## 📈 Monitoring & Analysis

### Real-time Monitoring Options

#### 1. GUI Dashboard (`monitoring.py`)
- **Live account information** - Balance, equity, margin
- **Current positions** - Real-time P&L tracking
- **AI signal display** - Current signal and confidence
- **Price charts** - Live XAUUSD price visualization
- **System health** - Resource usage and status

#### 2. Web Dashboard (`web_dashboard.py`)
- **Browser-based interface** - Access from any device
- **Mobile-friendly** - Responsive design
- **Real-time updates** - Auto-refresh every 5 seconds
- **Emergency controls** - Stop system remotely

#### 3. Console Monitor (`monitoring.py --console`)
- **Terminal-based** - Perfect for headless servers
- **Low resource usage** - Minimal system impact
- **Structured output** - Easy to parse and log

### Performance Analysis

#### Backtesting (`backtest.py`)
```bash
# Run backtest for specific period
python backtest.py --start 2024-01-01 --end 2024-12-31

# Generate HTML report
python backtest.py --report

# Create performance charts
python backtest.py --plot
```

**Backtest Features:**
- Historical signal generation
- Realistic trading simulation
- Commission and slippage modeling
- Comprehensive performance metrics
- Visual equity curve and drawdown analysis

#### Key Performance Metrics
- **Total Return** - Percentage gain/loss
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss
- **Sharpe Ratio** - Risk-adjusted returns
- **Maximum Drawdown** - Largest equity decline
- **Average Trade** - Mean profit per trade

## ⚠️ Risk Disclaimer

### Trading Risks
- **Substantial Risk of Loss** - Trading involves significant financial risk
- **Past Performance** - Does not guarantee future results
- **Market Volatility** - Gold prices can be highly volatile
- **AI Limitations** - Machine learning models can fail during unusual market conditions
- **Technical Failures** - System outages or connectivity issues can result in losses

### Best Practices
1. **Start Small** - Begin with minimum position sizes
2. **Demo Testing** - Thoroughly test on demo accounts first
3. **Risk Management** - Never risk more than you can afford to lose
4. **Regular Monitoring** - Actively supervise the system
5. **Model Updates** - Retrain models periodically with fresh data
6. **Market Awareness** - Stay informed about major economic events

### Recommended Settings for Beginners
- **Risk per trade:** 0.5-1.0%
- **Maximum daily trades:** 5-10
- **AI confidence threshold:** 75%+
- **Stop loss:** 150-250 points
- **Take profit:** 300-500 points

## 🛠️ Troubleshooting

### Common Issues

#### 1. MT5 Connection Problems
**Symptoms:** "MT5 initialization failed"
**Solutions:**
- Ensure MT5 is running and logged in
- Check if trading is enabled for your account
- Verify XAUUSD symbol is available
- Restart MT5 and try again

#### 2. Model Training Failures
**Symptoms:** "No data available for training"
**Solutions:**
- Check MT5 connection
- Ensure sufficient historical data
- Verify symbol name is correct
- Check date ranges in configuration

#### 3. TA-Lib Import Errors
**Symptoms:** "No module named 'talib'"
**Solutions:**
- Install TA-Lib C library first
- Use pre-compiled wheels for Windows
- Check Python version compatibility
- Try: `pip install --no-cache-dir TA-Lib`

#### 4. Signal File Not Found
**Symptoms:** "Cannot open signal file"
**Solutions:**
- Ensure Python system is running
- Check file permissions
- Verify signal file path in MT5 EA
- Create signals directory manually

#### 5. High Memory Usage
**Symptoms:** System becomes slow
**Solutions:**
- Reduce lookback periods in config
- Close unnecessary applications
- Increase virtual memory
- Monitor with task manager

### Debug Mode
Enable detailed logging:
```bash
python run_ai_system.py --verbose
```

### Emergency Procedures
1. **Stop All Trading:**
   ```bash
   # Create emergency stop file
   touch EMERGENCY_STOP
   ```

2. **Close All Positions:**
   - Use MT5 "Close All" function
   - Or use web dashboard emergency button

3. **System Recovery:**
   - Restart MT5
   - Restart Python system
   - Check logs for errors
   - Verify all connections

### Getting Help
1. **Check logs** in `logs/` directory
2. **Review configuration** settings
3. **Test components** individually
4. **Verify system requirements**
5. **Check MT5 journal** for EA messages

### Performance Optimization
- **Reduce timeframes** if CPU usage is high
- **Limit concurrent processes**
- **Use SSD storage** for better I/O performance
- **Ensure stable internet** connection
- **Regular system maintenance**

## 📞 Support & Community

### System Requirements
- **OS:** Windows 10+, Linux (Ubuntu 18+), macOS 10.14+
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space
- **Network:** Stable broadband connection
- **MT5:** Latest version with trading permissions

### Updates & Maintenance
- **Weekly:** Review trading performance
- **Monthly:** Retrain AI model with fresh data
- **Quarterly:** Update technical indicators and features
- **Annually:** Review and optimize system parameters

### Version History
- **v2.0.0** - Complete system with web dashboard and backtesting
- **v1.5.0** - Added monitoring and risk management features  
- **v1.0.0** - Initial release with basic AI signal generation

---

## 🎉 Conclusion

The AI Scalper XAUUSDm system represents a comprehensive approach to algorithmic trading, combining:

- **Advanced machine learning** for signal generation
- **Professional risk management** for capital protection
- **Real-time monitoring** for system oversight
- **Comprehensive backtesting** for strategy validation
- **User-friendly interfaces** for easy operation

### Success Tips
1. **Understand the system** thoroughly before live trading
2. **Start conservative** with low risk settings
3. **Monitor regularly** and adjust as needed
4. **Keep learning** about markets and technology
5. **Maintain discipline** in risk management

### Final Reminders
- **Never invest more than you can afford to lose**
- **This system is for educational purposes**
- **Always test extensively on demo accounts**
- **Consider consulting financial advisors**
- **Trading success requires patience and discipline**

**Happy Trading! 🚀📈**

---

*This system is provided as-is for educational and research purposes. Trading involves substantial risk and may not be suitable for all investors.*