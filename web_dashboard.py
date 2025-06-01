#!/usr/bin/env python3
"""
Simple Web Dashboard for AI Scalper XAUUSDm System
Provides web-based monitoring and control interface
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import MetaTrader5 as mt5

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, render_template_string, jsonify, request
    from utils import get_account_info, get_symbol_info, get_system_health, load_signal_history
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install flask")
    sys.exit(1)

class WebDashboard:
    """Simple web dashboard for monitoring the AI trading system"""
    
    def __init__(self, host='127.0.0.1', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.data_cache = {}
        self.last_update = None
        
        # Initialize MT5
        self.mt5_connected = self.init_mt5()
        
        # Setup routes
        self.setup_routes()
        
        # Start data update thread
        self.start_data_updater()
    
    def init_mt5(self):
        """Initialize MT5 connection"""
        try:
            if mt5.initialize():
                print("✓ MT5 connected for web dashboard")
                return True
            else:
                print("✗ MT5 connection failed")
                return False
        except Exception as e:
            print(f"MT5 initialization error: {e}")
            return False
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(DASHBOARD_TEMPLATE)
        
        @self.app.route('/api/data')
        def get_data():
            """API endpoint for dashboard data"""
            return jsonify(self.data_cache)
        
        @self.app.route('/api/account')
        def get_account():
            """API endpoint for account info"""
            if self.mt5_connected:
                account_info = get_account_info()
                return jsonify(account_info)
            return jsonify({'error': 'MT5 not connected'})
        
        @self.app.route('/api/positions')
        def get_positions():
            """API endpoint for position info"""
            if self.mt5_connected:
                positions = mt5.positions_get(symbol="XAUUSDm")
                if positions:
                    pos_list = []
                    for pos in positions:
                        pos_list.append({
                            'ticket': pos.ticket,
                            'symbol': pos.symbol,
                            'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                            'volume': pos.volume,
                            'price_open': pos.price_open,
                            'price_current': pos.price_current,
                            'profit': pos.profit,
                            'time': pos.time
                        })
                    return jsonify(pos_list)
                return jsonify([])
            return jsonify({'error': 'MT5 not connected'})
        
        @self.app.route('/api/signal')
        def get_current_signal():
            """API endpoint for current AI signal"""
            try:
                if os.path.exists("signals/xauusdm_signal.txt"):
                    with open("signals/xauusdm_signal.txt", 'r') as f:
                        signal = json.load(f)
                    return jsonify(signal)
                return jsonify({'signal': 'NONE', 'confidence': 0, 'reason': 'No signal file'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/health')
        def get_health():
            """API endpoint for system health"""
            try:
                health = get_system_health()
                return jsonify(health)
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def control_system(action):
            """API endpoint for system control"""
            try:
                if action == 'emergency_stop':
                    # Create emergency stop file
                    with open("EMERGENCY_STOP", 'w') as f:
                        f.write(f"Emergency stop triggered from web dashboard at {datetime.now()}")
                    return jsonify({'success': True, 'message': 'Emergency stop activated'})
                
                elif action == 'clear_emergency':
                    # Remove emergency stop file
                    if os.path.exists("EMERGENCY_STOP"):
                        os.remove("EMERGENCY_STOP")
                    return jsonify({'success': True, 'message': 'Emergency stop cleared'})
                
                else:
                    return jsonify({'error': 'Unknown action'})
                    
            except Exception as e:
                return jsonify({'error': str(e)})
    
    def update_data(self):
        """Update cached data"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'mt5_connected': self.mt5_connected,
                'account': {},
                'symbol_info': {},
                'current_signal': {},
                'system_health': {},
                'emergency_stop': os.path.exists("EMERGENCY_STOP")
            }
            
            # Account info
            if self.mt5_connected:
                data['account'] = get_account_info()
                data['symbol_info'] = get_symbol_info("XAUUSDm")
            
            # Current signal
            try:
                if os.path.exists("signals/xauusdm_signal.txt"):
                    with open("signals/xauusdm_signal.txt", 'r') as f:
                        data['current_signal'] = json.load(f)
            except:
                data['current_signal'] = {'signal': 'NONE', 'confidence': 0}
            
            # System health
            try:
                data['system_health'] = get_system_health()
            except:
                data['system_health'] = {'overall_health': 'unknown'}
            
            self.data_cache = data
            self.last_update = datetime.now()
            
        except Exception as e:
            print(f"Data update error: {e}")
    
    def start_data_updater(self):
        """Start background data update thread"""
        def updater():
            while True:
                self.update_data()
                time.sleep(5)  # Update every 5 seconds
        
        thread = threading.Thread(target=updater, daemon=True)
        thread.start()
    
    def run(self, debug=False):
        """Run the web dashboard"""
        print(f"Starting web dashboard at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)

# HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Scalper XAUUSDm - Web Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .status-indicator {
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }
        
        .status-online { background-color: #27ae60; }
        .status-offline { background-color: #e74c3c; }
        .status-warning { background-color: #f39c12; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 600;
            color: #34495e;
        }
        
        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #34495e; }
        
        .signal-card {
            text-align: center;
            padding: 30px;
        }
        
        .signal-type {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .signal-buy { color: #27ae60; }
        .signal-sell { color: #e74c3c; }
        .signal-none { color: #95a5a6; }
        
        .confidence-bar {
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .controls {
            margin-top: 20px;
        }
        
        .btn {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            margin: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
        }
        
        .btn-success {
            background: linear-gradient(45deg, #27ae60, #229954);
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 20px;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Scalper XAUUSDm</h1>
            <p>
                <span id="connection-status" class="status-indicator status-offline"></span>
                <span id="status-text">Connecting...</span>
                <span class="loading" id="loading-indicator"></span>
            </p>
            <p id="last-update">Last Update: --</p>
        </div>
        
        <div class="grid">
            <!-- Account Information -->
            <div class="card">
                <h3>💼 Account Information</h3>
                <div class="metric">
                    <span class="metric-label">Balance:</span>
                    <span class="metric-value" id="balance">$--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Equity:</span>
                    <span class="metric-value" id="equity">$--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Margin:</span>
                    <span class="metric-value" id="margin">$--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Free Margin:</span>
                    <span class="metric-value" id="free-margin">$--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Margin Level:</span>
                    <span class="metric-value" id="margin-level">--%</span>
                </div>
            </div>
            
            <!-- Current Signal -->
            <div class="card signal-card">
                <h3>🎯 Current AI Signal</h3>
                <div class="signal-type" id="signal-type">NONE</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidence-fill" style="width: 0%;"></div>
                </div>
                <div id="confidence-text">Confidence: 0%</div>
                <div id="signal-time">--</div>
            </div>
            
            <!-- Market Information -->
            <div class="card">
                <h3>📊 XAUUSD Market</h3>
                <div class="metric">
                    <span class="metric-label">Bid:</span>
                    <span class="metric-value" id="bid-price">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Ask:</span>
                    <span class="metric-value" id="ask-price">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Spread:</span>
                    <span class="metric-value" id="spread">-- points</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Last Update:</span>
                    <span class="metric-value" id="price-time">--</span>
                </div>
            </div>
            
            <!-- System Health -->
            <div class="card">
                <h3>⚡ System Health</h3>
                <div class="metric">
                    <span class="metric-label">CPU Usage:</span>
                    <span class="metric-value" id="cpu-usage">--%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage:</span>
                    <span class="metric-value" id="memory-usage">--%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">MT5 Status:</span>
                    <span class="metric-value" id="mt5-status">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Model Status:</span>
                    <span class="metric-value" id="model-status">--</span>
                </div>
            </div>
        </div>
        
        <!-- Controls -->
        <div class="card controls">
            <h3>🎮 System Controls</h3>
            <button class="btn btn-danger" onclick="emergencyStop()">🛑 Emergency Stop</button>
            <button class="btn btn-success" onclick="clearEmergency()">✅ Clear Emergency</button>
            <button class="btn" onclick="refreshData()">🔄 Refresh Data</button>
        </div>
        
        <div class="footer">
            <p>AI Scalper XAUUSDm Dashboard | Real-time Trading Monitor</p>
            <p>⚠️ Trading involves substantial risk of loss</p>
        </div>
    </div>
    
    <script>
        let updateInterval;
        
        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    updateUI(data);
                })
                .catch(error => {
                    console.error('Update error:', error);
                    document.getElementById('status-text').textContent = 'Connection Error';
                    document.getElementById('connection-status').className = 'status-indicator status-offline';
                });
        }
        
        function updateUI(data) {
            // Connection status
            const isConnected = data.mt5_connected;
            const statusIndicator = document.getElementById('connection-status');
            const statusText = document.getElementById('status-text');
            
            if (isConnected) {
                statusIndicator.className = 'status-indicator status-online';
                statusText.textContent = 'Connected';
            } else {
                statusIndicator.className = 'status-indicator status-offline';
                statusText.textContent = 'Disconnected';
            }
            
            // Last update
            document.getElementById('last-update').textContent = 
                'Last Update: ' + new Date(data.timestamp).toLocaleTimeString();
            
            // Account info
            const account = data.account || {};
            document.getElementById('balance').textContent = 
                account.balance ? '$' + account.balance.toLocaleString() : '$--';
            document.getElementById('equity').textContent = 
                account.equity ? '$' + account.equity.toLocaleString() : '$--';
            document.getElementById('margin').textContent = 
                account.margin ? '$' + account.margin.toLocaleString() : '$--';
            document.getElementById('free-margin').textContent = 
                account.free_margin ? '$' + account.free_margin.toLocaleString() : '$--';
            document.getElementById('margin-level').textContent = 
                account.margin_level ? account.margin_level.toFixed(1) + '%' : '--%';
            
            // Signal info
            const signal = data.current_signal || {};
            const signalType = document.getElementById('signal-type');
            const confidenceFill = document.getElementById('confidence-fill');
            const confidenceText = document.getElementById('confidence-text');
            
            signalType.textContent = signal.signal || 'NONE';
            signalType.className = 'signal-type signal-' + (signal.signal || 'none').toLowerCase();
            
            const confidence = (signal.confidence || 0) * 100;
            confidenceFill.style.width = confidence + '%';
            confidenceText.textContent = 'Confidence: ' + confidence.toFixed(1) + '%';
            
            if (confidence > 70) {
                confidenceFill.style.backgroundColor = '#27ae60';
            } else if (confidence > 50) {
                confidenceFill.style.backgroundColor = '#f39c12';
            } else {
                confidenceFill.style.backgroundColor = '#e74c3c';
            }
            
            // Market info
            const symbolInfo = data.symbol_info || {};
            document.getElementById('bid-price').textContent = 
                symbolInfo.bid ? symbolInfo.bid.toFixed(2) : '--';
            document.getElementById('ask-price').textContent = 
                symbolInfo.ask ? symbolInfo.ask.toFixed(2) : '--';
            document.getElementById('spread').textContent = 
                symbolInfo.spread ? symbolInfo.spread + ' points' : '-- points';
            
            // System health
            const health = data.system_health || {};
            const systemInfo = health.system || {};
            
            document.getElementById('cpu-usage').textContent = 
                systemInfo.cpu_percent ? systemInfo.cpu_percent.toFixed(1) + '%' : '--%';
            document.getElementById('memory-usage').textContent = 
                systemInfo.memory_percent ? systemInfo.memory_percent.toFixed(1) + '%' : '--%';
            document.getElementById('mt5-status').textContent = 
                health.mt5 && health.mt5.connected ? 'Connected' : 'Disconnected';
            document.getElementById('model-status').textContent = 
                health.files && health.files.model_exists ? 'Available' : 'Missing';
            
            // Hide loading indicator
            document.getElementById('loading-indicator').style.display = 'none';
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to trigger an emergency stop?')) {
                fetch('/api/control/emergency_stop', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Emergency stop activated!');
                        } else {
                            alert('Error: ' + (data.error || 'Unknown error'));
                        }
                    })
                    .catch(error => {
                        alert('Network error: ' + error);
                    });
            }
        }
        
        function clearEmergency() {
            if (confirm('Clear emergency stop?')) {
                fetch('/api/control/clear_emergency', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Emergency stop cleared!');
                        } else {
                            alert('Error: ' + (data.error || 'Unknown error'));
                        }
                    })
                    .catch(error => {
                        alert('Network error: ' + error);
                    });
            }
        }
        
        function refreshData() {
            document.getElementById('loading-indicator').style.display = 'inline-block';
            updateDashboard();
        }
        
        // Start auto-update
        updateDashboard();
        updateInterval = setInterval(updateDashboard, 5000); // Update every 5 seconds
        
        // Page visibility handling
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                clearInterval(updateInterval);
            } else {
                updateDashboard();
                updateInterval = setInterval(updateDashboard, 5000);
            }
        });
    </script>
</body>
</html>
"""

def main():
    """Main function to run web dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Scalper Web Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    try:
        dashboard = WebDashboard(host=args.host, port=args.port)
        dashboard.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\nWeb dashboard stopped by user")
    except Exception as e:
        print(f"Web dashboard error: {e}")
    finally:
        if mt5.initialize():
            mt5.shutdown()

if __name__ == "__main__":
    main()