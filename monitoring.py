#!/usr/bin/env python3
"""
Real-time Monitoring and Dashboard for AI Scalper XAUUSDm System
Provides live monitoring, statistics, and performance tracking
"""

import os
import sys
import time
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List, Any, Optional

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import get_account_info, get_symbol_info, load_signal_history, get_system_health
except ImportError:
    print("Warning: Could not import utils module")

class TradingMonitor:
    """Real-time trading monitor with GUI dashboard"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Scalper XAUUSDm - Live Monitor")
        self.root.geometry("1200x800")
        
        # Data storage
        self.price_data = []
        self.signal_data = []
        self.performance_data = []
        self.account_data = []
        
        # Control variables
        self.monitoring = False
        self.update_interval = 5  # seconds
        
        # Initialize MT5
        self.mt5_connected = False
        self.init_mt5()
        
        # Setup GUI
        self.setup_gui()
        
        # Start monitoring thread
        self.monitor_thread = None
        
    def init_mt5(self):
        """Initialize MT5 connection"""
        try:
            if mt5.initialize():
                self.mt5_connected = True
                print("✓ MT5 connected successfully")
            else:
                print("✗ MT5 connection failed")
        except Exception as e:
            print(f"MT5 initialization error: {e}")
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_overview_tab()
        self.create_signals_tab()
        self.create_performance_tab()
        self.create_system_tab()
        
        # Create control frame
        self.create_control_frame()
        
    def create_control_frame(self):
        """Create control buttons frame"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Start/Stop button
        self.start_button = tk.Button(
            control_frame, 
            text="Start Monitoring", 
            command=self.toggle_monitoring,
            bg="green", 
            fg="white",
            font=("Arial", 12, "bold")
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        tk.Button(
            control_frame, 
            text="Refresh", 
            command=self.refresh_data,
            font=("Arial", 10)
        ).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            control_frame, 
            text="Status: Disconnected", 
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Connection indicator
        self.connection_indicator = tk.Label(
            control_frame, 
            text="●", 
            font=("Arial", 16),
            fg="red" if not self.mt5_connected else "green"
        )
        self.connection_indicator.pack(side=tk.RIGHT)
        
    def create_overview_tab(self):
        """Create overview tab with key metrics"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")
        
        # Account info frame
        account_frame = tk.LabelFrame(overview_frame, text="Account Information")
        account_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Account info labels
        self.account_labels = {}
        account_fields = [
            ("Balance", "balance"),
            ("Equity", "equity"),
            ("Margin", "margin"),
            ("Free Margin", "free_margin"),
            ("Margin Level", "margin_level")
        ]
        
        for i, (label, key) in enumerate(account_fields):
            tk.Label(account_frame, text=f"{label}:", font=("Arial", 10, "bold")).grid(
                row=i//2, column=(i%2)*2, sticky="w", padx=5, pady=2
            )
            self.account_labels[key] = tk.Label(account_frame, text="--", font=("Arial", 10))
            self.account_labels[key].grid(
                row=i//2, column=(i%2)*2+1, sticky="w", padx=5, pady=2
            )
        
        # Current position frame
        position_frame = tk.LabelFrame(overview_frame, text="Current Positions")
        position_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Position treeview
        self.position_tree = ttk.Treeview(
            position_frame, 
            columns=("Symbol", "Type", "Lots", "Open Price", "Current Price", "P&L"),
            show="headings"
        )
        
        for col in self.position_tree["columns"]:
            self.position_tree.heading(col, text=col)
            self.position_tree.column(col, width=100)
        
        self.position_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Price chart frame
        chart_frame = tk.LabelFrame(overview_frame, text="XAUUSD Price Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create matplotlib chart
        self.price_fig, self.price_ax = plt.subplots(figsize=(8, 4))
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, chart_frame)
        self.price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_signals_tab(self):
        """Create signals monitoring tab"""
        signals_frame = ttk.Frame(self.notebook)
        self.notebook.add(signals_frame, text="AI Signals")
        
        # Current signal frame
        current_signal_frame = tk.LabelFrame(signals_frame, text="Current Signal")
        current_signal_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Signal info
        self.signal_labels = {}
        signal_fields = [
            ("Signal", "signal"),
            ("Confidence", "confidence"),
            ("Timestamp", "timestamp"),
            ("Price", "price"),
            ("Reason", "reason")
        ]
        
        for i, (label, key) in enumerate(signal_fields):
            tk.Label(current_signal_frame, text=f"{label}:", font=("Arial", 10, "bold")).grid(
                row=i, column=0, sticky="w", padx=5, pady=2
            )
            self.signal_labels[key] = tk.Label(current_signal_frame, text="--", font=("Arial", 10))
            self.signal_labels[key].grid(row=i, column=1, sticky="w", padx=5, pady=2)
        
        # Signal history frame
        history_frame = tk.LabelFrame(signals_frame, text="Signal History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Signal history treeview
        self.signal_tree = ttk.Treeview(
            history_frame,
            columns=("Time", "Signal", "Confidence", "Price", "Status"),
            show="headings"
        )
        
        for col in self.signal_tree["columns"]:
            self.signal_tree.heading(col, text=col)
        
        self.signal_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Signal statistics
        stats_frame = tk.LabelFrame(signals_frame, text="Signal Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.signal_stats_labels = {}
        stats_fields = [
            ("Total Signals", "total"),
            ("Buy Signals", "buy"),
            ("Sell Signals", "sell"),
            ("Avg Confidence", "avg_confidence")
        ]
        
        for i, (label, key) in enumerate(stats_fields):
            tk.Label(stats_frame, text=f"{label}:", font=("Arial", 10, "bold")).grid(
                row=i//2, column=(i%2)*2, sticky="w", padx=5, pady=2
            )
            self.signal_stats_labels[key] = tk.Label(stats_frame, text="--", font=("Arial", 10))
            self.signal_stats_labels[key].grid(
                row=i//2, column=(i%2)*2+1, sticky="w", padx=5, pady=2
            )
    
    def create_performance_tab(self):
        """Create performance monitoring tab"""
        performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(performance_frame, text="Performance")
        
        # Performance metrics frame
        metrics_frame = tk.LabelFrame(performance_frame, text="Trading Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.performance_labels = {}
        performance_fields = [
            ("Total Trades", "total_trades"),
            ("Win Rate", "win_rate"),
            ("Profit Factor", "profit_factor"),
            ("Total P&L", "total_pnl"),
            ("Today P&L", "today_pnl"),
            ("Max Drawdown", "max_drawdown")
        ]
        
        for i, (label, key) in enumerate(performance_fields):
            tk.Label(metrics_frame, text=f"{label}:", font=("Arial", 10, "bold")).grid(
                row=i//3, column=(i%3)*2, sticky="w", padx=5, pady=2
            )
            self.performance_labels[key] = tk.Label(metrics_frame, text="--", font=("Arial", 10))
            self.performance_labels[key].grid(
                row=i//3, column=(i%3)*2+1, sticky="w", padx=5, pady=2
            )
        
        # Equity curve frame
        equity_frame = tk.LabelFrame(performance_frame, text="Equity Curve")
        equity_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create equity chart
        self.equity_fig, self.equity_ax = plt.subplots(figsize=(8, 4))
        self.equity_canvas = FigureCanvasTkAgg(self.equity_fig, equity_frame)
        self.equity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_system_tab(self):
        """Create system monitoring tab"""
        system_frame = ttk.Frame(self.notebook)
        self.notebook.add(system_frame, text="System")
        
        # System health frame
        health_frame = tk.LabelFrame(system_frame, text="System Health")
        health_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.health_labels = {}
        health_fields = [
            ("CPU Usage", "cpu"),
            ("Memory Usage", "memory"),
            ("Disk Usage", "disk"),
            ("MT5 Status", "mt5_status"),
            ("Model Status", "model_status"),
            ("Last Update", "last_update")
        ]
        
        for i, (label, key) in enumerate(health_fields):
            tk.Label(health_frame, text=f"{label}:", font=("Arial", 10, "bold")).grid(
                row=i//2, column=(i%2)*2, sticky="w", padx=5, pady=2
            )
            self.health_labels[key] = tk.Label(health_frame, text="--", font=("Arial", 10))
            self.health_labels[key].grid(
                row=i//2, column=(i%2)*2+1, sticky="w", padx=5, pady=2
            )
        
        # Log viewer frame
        log_frame = tk.LabelFrame(system_frame, text="Recent Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Log text widget with scrollbar
        log_scroll_frame = tk.Frame(log_frame)
        log_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_scroll_frame, height=10, font=("Consolas", 9))
        log_scrollbar = tk.Scrollbar(log_scroll_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def toggle_monitoring(self):
        """Start or stop monitoring"""
        if self.monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if not self.mt5_connected:
            messagebox.showerror("Error", "MT5 not connected!")
            return
        
        self.monitoring = True
        self.start_button.config(text="Stop Monitoring", bg="red")
        self.status_label.config(text="Status: Monitoring")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("Monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.monitoring = False
        self.start_button.config(text="Start Monitoring", bg="green")
        self.status_label.config(text="Status: Stopped")
        
        print("Monitoring stopped")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Update all data
                self.update_account_data()
                self.update_position_data()
                self.update_price_data()
                self.update_signal_data()
                self.update_performance_data()
                self.update_system_health()
                self.update_logs()
                
                # Update GUI in main thread
                self.root.after(0, self.update_gui)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def update_account_data(self):
        """Update account information"""
        try:
            if self.mt5_connected:
                account_info = get_account_info()
                self.account_data = account_info
        except Exception as e:
            print(f"Account data update error: {e}")
    
    def update_position_data(self):
        """Update position information"""
        try:
            if self.mt5_connected:
                positions = mt5.positions_get(symbol="XAUUSDm")
                self.position_data = positions if positions else []
        except Exception as e:
            print(f"Position data update error: {e}")
    
    def update_price_data(self):
        """Update price data"""
        try:
            if self.mt5_connected:
                # Get recent price data
                rates = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_M1, 0, 100)
                if rates is not None:
                    self.price_data = rates
        except Exception as e:
            print(f"Price data update error: {e}")
    
    def update_signal_data(self):
        """Update signal data"""
        try:
            # Read current signal
            if os.path.exists("signals/xauusdm_signal.txt"):
                with open("signals/xauusdm_signal.txt", 'r') as f:
                    current_signal = json.load(f)
                    self.current_signal = current_signal
            
            # Load signal history
            signal_history = load_signal_history()
            self.signal_data = signal_history[-50:]  # Keep last 50 signals
            
        except Exception as e:
            print(f"Signal data update error: {e}")
    
    def update_performance_data(self):
        """Update performance metrics"""
        try:
            # Calculate performance metrics from account history
            # This is a simplified version
            if hasattr(self, 'account_data') and self.account_data:
                current_equity = self.account_data.get('equity', 0)
                current_balance = self.account_data.get('balance', 0)
                
                self.performance_data = {
                    'current_equity': current_equity,
                    'current_balance': current_balance,
                    'unrealized_pnl': current_equity - current_balance,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Performance data update error: {e}")
    
    def update_system_health(self):
        """Update system health data"""
        try:
            health = get_system_health()
            self.system_health = health
        except Exception as e:
            print(f"System health update error: {e}")
    
    def update_logs(self):
        """Update log display"""
        try:
            if os.path.exists("logs/ai_signals.log"):
                with open("logs/ai_signals.log", 'r') as f:
                    lines = f.readlines()
                    self.recent_logs = lines[-20:]  # Last 20 lines
        except Exception as e:
            print(f"Log update error: {e}")
    
    def update_gui(self):
        """Update GUI elements with new data"""
        try:
            # Update account info
            if hasattr(self, 'account_data') and self.account_data:
                for key, label in self.account_labels.items():
                    value = self.account_data.get(key, '--')
                    if isinstance(value, (int, float)):
                        if key in ['balance', 'equity', 'margin', 'free_margin']:
                            label.config(text=f"${value:,.2f}")
                        elif key == 'margin_level':
                            label.config(text=f"{value:.1f}%")
                        else:
                            label.config(text=f"{value}")
                    else:
                        label.config(text=str(value))
            
            # Update positions
            self.update_position_tree()
            
            # Update current signal
            if hasattr(self, 'current_signal'):
                signal = self.current_signal
                for key, label in self.signal_labels.items():
                    value = signal.get(key, '--')
                    if key == 'confidence' and isinstance(value, (int, float)):
                        label.config(text=f"{value*100:.1f}%")
                    elif key == 'timestamp':
                        try:
                            if isinstance(value, str):
                                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                label.config(text=dt.strftime("%H:%M:%S"))
                            else:
                                label.config(text=str(value))
                        except:
                            label.config(text=str(value))
                    else:
                        label.config(text=str(value))
            
            # Update charts
            self.update_price_chart()
            self.update_equity_chart()
            
            # Update system health
            if hasattr(self, 'system_health'):
                health = self.system_health
                system_info = health.get('system', {})
                
                self.health_labels['cpu'].config(text=f"{system_info.get('cpu_percent', 0):.1f}%")
                self.health_labels['memory'].config(text=f"{system_info.get('memory_percent', 0):.1f}%")
                self.health_labels['disk'].config(text=f"{system_info.get('disk_percent', 0):.1f}%")
                self.health_labels['mt5_status'].config(
                    text="Connected" if health.get('mt5', {}).get('connected') else "Disconnected"
                )
                self.health_labels['model_status'].config(
                    text="Available" if health.get('files', {}).get('model_exists') else "Missing"
                )
                self.health_labels['last_update'].config(text=datetime.now().strftime("%H:%M:%S"))
            
            # Update logs
            if hasattr(self, 'recent_logs'):
                self.log_text.delete(1.0, tk.END)
                for line in self.recent_logs:
                    self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                
        except Exception as e:
            print(f"GUI update error: {e}")
    
    def update_position_tree(self):
        """Update position tree view"""
        try:
            # Clear existing items
            for item in self.position_tree.get_children():
                self.position_tree.delete(item)
            
            # Add current positions
            if hasattr(self, 'position_data'):
                for position in self.position_data:
                    pos_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                    
                    # Calculate current P&L
                    if pos_type == "BUY":
                        current_price = mt5.symbol_info_tick(position.symbol).bid
                        pnl = (current_price - position.price_open) * position.volume
                    else:
                        current_price = mt5.symbol_info_tick(position.symbol).ask
                        pnl = (position.price_open - current_price) * position.volume
                    
                    self.position_tree.insert("", "end", values=(
                        position.symbol,
                        pos_type,
                        f"{position.volume:.2f}",
                        f"{position.price_open:.2f}",
                        f"{current_price:.2f}",
                        f"${pnl:.2f}"
                    ))
        except Exception as e:
            print(f"Position tree update error: {e}")
    
    def update_price_chart(self):
        """Update price chart"""
        try:
            if hasattr(self, 'price_data') and len(self.price_data) > 0:
                self.price_ax.clear()
                
                # Convert to DataFrame
                df = pd.DataFrame(self.price_data)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Plot price line
                self.price_ax.plot(df['time'], df['close'], 'b-', linewidth=1)
                self.price_ax.set_title("XAUUSD Price (M1)")
                self.price_ax.set_ylabel("Price")
                self.price_ax.grid(True, alpha=0.3)
                
                # Format x-axis
                self.price_ax.tick_params(axis='x', rotation=45)
                
                self.price_fig.tight_layout()
                self.price_canvas.draw()
        except Exception as e:
            print(f"Price chart update error: {e}")
    
    def update_equity_chart(self):
        """Update equity curve chart"""
        try:
            if hasattr(self, 'performance_data') and self.performance_data:
                # This is a simplified version - in practice you'd store equity history
                self.equity_ax.clear()
                
                # Sample equity curve (replace with actual data)
                times = [datetime.now() - timedelta(minutes=i*5) for i in range(20, 0, -1)]
                equity_values = [10000 + np.random.randn() * 100 for _ in times]
                
                self.equity_ax.plot(times, equity_values, 'g-', linewidth=2)
                self.equity_ax.set_title("Account Equity Curve")
                self.equity_ax.set_ylabel("Equity ($)")
                self.equity_ax.grid(True, alpha=0.3)
                
                self.equity_ax.tick_params(axis='x', rotation=45)
                
                self.equity_fig.tight_layout()
                self.equity_canvas.draw()
        except Exception as e:
            print(f"Equity chart update error: {e}")
    
    def refresh_data(self):
        """Refresh all data manually"""
        try:
            self.update_account_data()
            self.update_position_data()
            self.update_price_data()
            self.update_signal_data()
            self.update_performance_data()
            self.update_system_health()
            self.update_logs()
            self.update_gui()
            
            messagebox.showinfo("Refresh", "Data refreshed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Refresh failed: {e}")
    
    def run(self):
        """Run the monitoring dashboard"""
        try:
            # Initial data load
            self.refresh_data()
            
            # Start GUI main loop
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Dashboard stopped by user")
        except Exception as e:
            print(f"Dashboard error: {e}")
        finally:
            self.stop_monitoring()
            if self.mt5_connected:
                mt5.shutdown()

class SimpleMonitor:
    """Simple console-based monitor for headless systems"""
    
    def __init__(self, update_interval=30):
        self.update_interval = update_interval
        self.running = False
        
    def start(self):
        """Start console monitoring"""
        self.running = True
        print("Starting console monitor...")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                self.display_status()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped by user")
        except Exception as e:
            print(f"Monitor error: {e}")
    
    def display_status(self):
        """Display current status"""
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*60)
            print("AI SCALPER XAUUSD - LIVE MONITOR")
            print("="*60)
            print(f"Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Account info
            if mt5.initialize():
                account = get_account_info()
                if account:
                    print("ACCOUNT INFORMATION:")
                    print(f"  Balance: ${account.get('balance', 0):,.2f}")
                    print(f"  Equity:  ${account.get('equity', 0):,.2f}")
                    print(f"  Margin:  ${account.get('margin', 0):,.2f}")
                    print(f"  Free Margin: ${account.get('free_margin', 0):,.2f}")
                    print()
                
                # Positions
                positions = mt5.positions_get(symbol="XAUUSDm")
                print(f"ACTIVE POSITIONS: {len(positions) if positions else 0}")
                if positions:
                    for pos in positions:
                        pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                        print(f"  {pos.symbol} {pos_type} {pos.volume:.2f} @ {pos.price_open:.2f}")
                print()
                
                # Current price
                tick = mt5.symbol_info_tick("XAUUSDm")
                if tick:
                    print(f"XAUUSD PRICE: {tick.bid:.2f} / {tick.ask:.2f} (Spread: {tick.ask - tick.bid:.2f})")
                    print()
                
                mt5.shutdown()
            
            # Current signal
            try:
                if os.path.exists("signals/xauusdm_signal.txt"):
                    with open("signals/xauusdm_signal.txt", 'r') as f:
                        signal = json.load(f)
                    
                    print("CURRENT AI SIGNAL:")
                    print(f"  Signal: {signal.get('signal', 'NONE')}")
                    print(f"  Confidence: {signal.get('confidence', 0)*100:.1f}%")
                    print(f"  Time: {signal.get('timestamp', 'Unknown')}")
                    print()
            except Exception as e:
                print(f"Signal read error: {e}")
            
            # System health
            try:
                health = get_system_health()
                system_info = health.get('system', {})
                
                print("SYSTEM HEALTH:")
                print(f"  CPU: {system_info.get('cpu_percent', 0):.1f}%")
                print(f"  Memory: {system_info.get('memory_percent', 0):.1f}%")
                print(f"  MT5: {'Connected' if health.get('mt5', {}).get('connected') else 'Disconnected'}")
                print()
            except Exception as e:
                print(f"Health check error: {e}")
            
            print("="*60)
            
        except Exception as e:
            print(f"Display error: {e}")

def main():
    """Main function to run monitoring dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Scalper Monitoring Dashboard')
    parser.add_argument('--console', action='store_true', help='Run in console mode')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds')
    args = parser.parse_args()
    
    if args.console:
        # Run console monitor
        monitor = SimpleMonitor(args.interval)
        monitor.start()
    else:
        # Run GUI monitor
        try:
            monitor = TradingMonitor()
            monitor.run()
        except ImportError as e:
            print(f"GUI dependencies missing: {e}")
            print("Running in console mode instead...")
            monitor = SimpleMonitor(args.interval)
            monitor.start()

if __name__ == "__main__":
    main()