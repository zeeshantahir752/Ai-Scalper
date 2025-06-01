#!/usr/bin/env python3
"""
Utility functions for AI Scalper XAUUSDm System
Provides common functions used across the trading system
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import MetaTrader5 as mt5
from pathlib import Path

def setup_logging(log_file: str = "logs/system.log", level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('ai_scalper')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def save_config(config: Dict[str, Any], config_file: str = "config.json") -> None:
    """Save configuration to JSON file"""
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to save configuration: {e}")

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration parameters"""
    errors = []
    
    # Required fields
    required_fields = [
        'symbol', 'timeframes', 'signal_threshold', 'paths'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate ranges
    if 'signal_threshold' in config:
        if not 0.1 <= config['signal_threshold'] <= 1.0:
            errors.append("signal_threshold must be between 0.1 and 1.0")
    
    if 'risk_management' in config:
        rm = config['risk_management']
        if 'risk_per_trade' in rm:
            if not 0.001 <= rm['risk_per_trade'] <= 0.1:
                errors.append("risk_per_trade must be between 0.1% and 10%")
    
    return errors

def ensure_directories(directories: List[str]) -> None:
    """Ensure required directories exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_mt5_connection() -> bool:
    """Check if MT5 is connected and working"""
    try:
        if not mt5.initialize():
            return False
        
        # Test basic functionality
        account_info = mt5.account_info()
        if account_info is None:
            return False
        
        return True
    except Exception:
        return False

def get_trading_hours(symbol: str = "XAUUSDm") -> Dict[str, Any]:
    """Get trading hours for a symbol"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {}
        
        return {
            'trading_mode': symbol_info.trade_mode,
            'start_time': symbol_info.start_time,
            'expiration_time': symbol_info.expiration_time,
            'sessions_count': symbol_info.sessions_count
        }
    except Exception:
        return {}

def is_market_open(symbol: str = "XAUUSDm") -> bool:
    """Check if market is open for trading"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False
        
        return symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
    except Exception:
        return False

def get_symbol_info(symbol: str = "XAUUSDm") -> Dict[str, Any]:
    """Get comprehensive symbol information"""
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return {}
        
        tick = mt5.symbol_info_tick(symbol)
        
        return {
            'symbol': symbol,
            'bid': tick.bid if tick else 0,
            'ask': tick.ask if tick else 0,
            'spread': info.spread,
            'point': info.point,
            'tick_value': info.trade_tick_value,
            'tick_size': info.trade_tick_size,
            'contract_size': info.trade_contract_size,
            'min_lot': info.volume_min,
            'max_lot': info.volume_max,
            'lot_step': info.volume_step,
            'margin_required': info.margin_initial
        }
    except Exception:
        return {}

def calculate_lot_size(
    balance: float,
    risk_percent: float,
    stop_loss_points: float,
    symbol: str = "XAUUSDm"
) -> float:
    """Calculate lot size based on risk management"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.01
        
        risk_amount = balance * (risk_percent / 100)
        point_value = symbol_info.trade_tick_value
        stop_loss_value = stop_loss_points * point_value
        
        if stop_loss_value <= 0:
            return 0.01
        
        lot_size = risk_amount / stop_loss_value
        
        # Apply symbol constraints
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        
        # Round to lot step
        lot_step = symbol_info.volume_step
        lot_size = round(lot_size / lot_step) * lot_step
        
        return lot_size
    except Exception:
        return 0.01

def format_price(price: float, symbol: str = "XAUUSDm") -> str:
    """Format price according to symbol specifications"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return f"{price:.2f}"
        
        digits = symbol_info.digits
        return f"{price:.{digits}f}"
    except Exception:
        return f"{price:.2f}"

def calculate_pnl(
    position_type: str,
    open_price: float,
    current_price: float,
    lot_size: float,
    symbol: str = "XAUUSDm"
) -> float:
    """Calculate unrealized P&L for a position"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.0
        
        point_value = symbol_info.trade_tick_value
        
        if position_type.upper() == "BUY":
            price_diff = current_price - open_price
        else:  # SELL
            price_diff = open_price - current_price
        
        pnl = price_diff * lot_size * point_value / symbol_info.trade_tick_size
        return pnl
    except Exception:
        return 0.0

def get_account_info() -> Dict[str, Any]:
    """Get account information"""
    try:
        account = mt5.account_info()
        if account is None:
            return {}
        
        return {
            'login': account.login,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'margin_level': account.margin_level,
            'currency': account.currency,
            'server': account.server,
            'leverage': account.leverage
        }
    except Exception:
        return {}

def save_signal_history(signals: List[Dict], filename: str = "data/signal_history.json") -> None:
    """Save signal history to file"""
    try:
        ensure_directories(["data"])
        with open(filename, 'w') as f:
            json.dump(signals, f, indent=2, default=str)
    except Exception as e:
        logging.error(f"Failed to save signal history: {e}")

def load_signal_history(filename: str = "data/signal_history.json") -> List[Dict]:
    """Load signal history from file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        logging.error(f"Failed to load signal history: {e}")
        return []

def create_backup(source_dir: str, backup_dir: str = "backups") -> str:
    """Create backup of important files"""
    try:
        ensure_directories([backup_dir])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)
        
        import shutil
        shutil.copytree(source_dir, backup_path, ignore=shutil.ignore_patterns(
            '*.log', '__pycache__', '*.pyc', 'ai_scalper_env'
        ))
        
        return backup_path
    except Exception as e:
        logging.error(f"Failed to create backup: {e}")
        return ""

def clean_old_files(directory: str, days_old: int = 30) -> None:
    """Clean old files from directory"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < cutoff_date.timestamp():
                    os.remove(file_path)
                    
    except Exception as e:
        logging.error(f"Failed to clean old files: {e}")

def validate_signal(signal: Dict[str, Any]) -> bool:
    """Validate signal format and content"""
    required_fields = ['signal', 'confidence', 'timestamp']
    
    # Check required fields
    for field in required_fields:
        if field not in signal:
            return False
    
    # Validate signal type
    if signal['signal'] not in ['BUY', 'SELL', 'NONE']:
        return False
    
    # Validate confidence
    if not 0 <= signal['confidence'] <= 1:
        return False
    
    # Validate timestamp
    try:
        if isinstance(signal['timestamp'], str):
            datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
        elif not isinstance(signal['timestamp'], (int, float)):
            return False
    except:
        return False
    
    return True

def calculate_drawdown(equity_curve: List[float]) -> Dict[str, float]:
    """Calculate drawdown statistics"""
    if not equity_curve:
        return {'max_drawdown': 0, 'current_drawdown': 0}
    
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    
    return {
        'max_drawdown': abs(drawdown.min()),
        'current_drawdown': abs(drawdown.iloc[-1]),
        'drawdown_duration': len(drawdown[drawdown < 0])
    }

def calculate_performance_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """Calculate trading performance metrics"""
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    if 'profit' not in df.columns:
        return {}
    
    total_trades = len(df)
    winning_trades = len(df[df['profit'] > 0])
    losing_trades = len(df[df['profit'] < 0])
    
    total_profit = df['profit'].sum()
    gross_profit = df[df['profit'] > 0]['profit'].sum()
    gross_loss = abs(df[df['profit'] < 0]['profit'].sum())
    
    metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'total_profit': total_profit,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else 0,
        'average_win': (gross_profit / winning_trades) if winning_trades > 0 else 0,
        'average_loss': (gross_loss / losing_trades) if losing_trades > 0 else 0,
    }
    
    if metrics['average_loss'] > 0:
        metrics['risk_reward_ratio'] = metrics['average_win'] / metrics['average_loss']
    else:
        metrics['risk_reward_ratio'] = 0
    
    return metrics

def send_notification(message: str, config: Dict = None) -> None:
    """Send notification via configured channels"""
    if not config or not config.get('notifications', {}).get('enabled'):
        return
    
    notifications = config['notifications']
    
    # Telegram notification
    if notifications.get('telegram', {}).get('enabled'):
        try:
            import requests
            telegram = notifications['telegram']
            url = f"https://api.telegram.org/bot{telegram['bot_token']}/sendMessage"
            data = {
                'chat_id': telegram['chat_id'],
                'text': f"🤖 AI Scalper: {message}"
            }
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            logging.error(f"Failed to send Telegram notification: {e}")
    
    # Email notification
    if notifications.get('email', {}).get('enabled'):
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            email = notifications['email']
            msg = MIMEText(message)
            msg['Subject'] = f"{email['subject_prefix']} Notification"
            msg['From'] = email['username']
            msg['To'] = email['to_email']
            
            server = smtplib.SMTP(email['smtp_server'], email['smtp_port'])
            server.starttls()
            server.login(email['username'], email['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")

def get_system_health() -> Dict[str, Any]:
    """Get system health status"""
    try:
        import psutil
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        # MT5 connection
        mt5_connected = check_mt5_connection()
        
        # Files status
        model_exists = os.path.exists("models/xgboost_model.pkl")
        config_exists = os.path.exists("config.json")
        
        health = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'available_memory_gb': memory.available / (1024**3)
            },
            'mt5': {
                'connected': mt5_connected,
                'account_info': get_account_info() if mt5_connected else {}
            },
            'files': {
                'model_exists': model_exists,
                'config_exists': config_exists,
                'logs_dir_exists': os.path.exists("logs"),
                'signals_dir_exists': os.path.exists("signals")
            },
            'overall_health': 'healthy' if all([
                cpu_percent < 80,
                memory.percent < 80,
                mt5_connected,
                model_exists,
                config_exists
            ]) else 'unhealthy'
        }
        
        return health
    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'overall_health': 'error'
        }

def optimize_memory_usage() -> None:
    """Optimize memory usage by cleaning up"""
    try:
        import gc
        gc.collect()
        
        # Clean matplotlib cache if it exists
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
            
    except Exception as e:
        logging.error(f"Memory optimization failed: {e}")

# Emergency functions
def emergency_stop(reason: str = "Emergency stop triggered") -> None:
    """Emergency stop - close all positions and stop trading"""
    try:
        logger = logging.getLogger('ai_scalper')
        logger.critical(f"EMERGENCY STOP: {reason}")
        
        # Close all positions
        if check_mt5_connection():
            positions = mt5.positions_get()
            if positions:
                for position in positions:
                    mt5.Close(position.ticket)
                logger.info(f"Closed {len(positions)} positions")
        
        # Create emergency flag file
        with open("EMERGENCY_STOP", 'w') as f:
            f.write(f"Emergency stop at {datetime.now()}\nReason: {reason}")
            
    except Exception as e:
        print(f"Emergency stop failed: {e}")

def check_emergency_stop() -> bool:
    """Check if emergency stop is active"""
    return os.path.exists("EMERGENCY_STOP")

def clear_emergency_stop() -> None:
    """Clear emergency stop flag"""
    try:
        if os.path.exists("EMERGENCY_STOP"):
            os.remove("EMERGENCY_STOP")
            logging.info("Emergency stop cleared")
    except Exception as e:
        logging.error(f"Failed to clear emergency stop: {e}")

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test logging
    logger = setup_logging()
    logger.info("Logging test successful")
    
    # Test MT5 connection
    mt5_status = check_mt5_connection()
    print(f"MT5 Connection: {'✓' if mt5_status else '✗'}")
    
    # Test directory creation
    ensure_directories(['test_dir'])
    print(f"Directory creation: {'✓' if os.path.exists('test_dir') else '✗'}")
    
    # Clean up test directory
    import shutil
    if os.path.exists('test_dir'):
        shutil.rmtree('test_dir')
    
    # Test system health
    health = get_system_health()
    print(f"System health: {health['overall_health']}")
    
    print("Utility functions test completed!")