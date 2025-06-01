#!/usr/bin/env python3
"""
Backtesting Engine for AI Scalper XAUUSDm System
Tests trading strategies using historical data and AI predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_signal_generator_advanced import AdvancedAISignalGenerator
    from train_model_xgboost import XGBoostTrainer
    from utils import calculate_performance_metrics, setup_logging
except ImportError as e:
    print(f"Import warning: {e}")

class BacktestEngine:
    """Comprehensive backtesting engine for AI trading strategies"""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize the backtesting engine"""
        self.logger = setup_logging("logs/backtest.log")
        self.config = self.load_config(config_file)
        self.results = {}
        self.trades = []
        self.equity_curve = []
        
        # Initialize MT5
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5 for backtesting")
        
        self.logger.info("Backtesting engine initialized")
    
    def load_config(self, config_file: str) -> Dict:
        """Load backtesting configuration"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get('backtest_settings', {
                'start_date': '2024-01-01',
                'end_date': '2024-12-31',
                'initial_balance': 10000,
                'commission': 0.0001,
                'spread': 25,
                'slippage': 5
            })
        except Exception as e:
            self.logger.error(f"Config load error: {e}")
            return {}
    
    def get_historical_data(
        self, 
        symbol: str = "XAUUSDm", 
        timeframe: int = mt5.TIMEFRAME_M1,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Get historical data for backtesting"""
        
        try:
            # Parse dates
            if start_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
            
            self.logger.info(f"Fetching historical data from {start_dt} to {end_dt}")
            
            # Get data from MT5
            rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
            
            if rates is None or len(rates) == 0:
                raise Exception("No historical data available")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} bars of historical data")
            return df
            
        except Exception as e:
            self.logger.error(f"Historical data error: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for AI model prediction"""
        try:
            # Use the same feature calculation as in training
            trainer = XGBoostTrainer()
            df_with_indicators = trainer.calculate_technical_indicators(df.copy())
            
            # Select features that would be available during live trading
            feature_cols = [col for col in df_with_indicators.columns 
                          if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
            
            features_df = df_with_indicators[feature_cols].copy()
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature preparation error: {e}")
            return pd.DataFrame()
    
    def generate_ai_signals(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate AI signals for historical data"""
        try:
            # Load trained model and scaler
            model_path = self.config.get('model_path', 'models/xgboost_model.pkl')
            scaler_path = self.config.get('scaler_path', 'models/scaler.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise Exception("Trained model or scaler not found")
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            self.logger.info("Generating AI signals for backtest data")
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Generate predictions
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            # Convert predictions to signals
            signal_map = {0: 'SELL', 1: 'NONE', 2: 'BUY'}
            signals = pd.Series([signal_map[pred] for pred in predictions], 
                              index=features_df.index)
            
            # Add confidence scores
            confidences = pd.Series([max(prob) for prob in probabilities], 
                                  index=features_df.index)
            
            return signals, confidences
            
        except Exception as e:
            self.logger.error(f"AI signal generation error: {e}")
            return pd.Series(), pd.Series()
    
    def simulate_trading(
        self, 
        price_data: pd.DataFrame, 
        signals: pd.Series, 
        confidences: pd.Series
    ) -> Dict:
        """Simulate trading based on AI signals"""
        
        try:
            # Trading parameters
            initial_balance = self.config.get('initial_balance', 10000)
            commission = self.config.get('commission', 0.0001)
            spread_points = self.config.get('spread', 25)
            slippage_points = self.config.get('slippage', 5)
            
            # Risk management
            risk_per_trade = 0.02  # 2% risk per trade
            stop_loss_points = 200
            take_profit_points = 400
            min_confidence = 0.70
            
            # Initialize tracking variables
            balance = initial_balance
            equity = initial_balance
            open_trades = []
            closed_trades = []
            equity_history = []
            
            self.logger.info(f"Starting simulation with ${initial_balance:,.2f}")
            
            # Iterate through each bar
            for i, (timestamp, row) in enumerate(price_data.iterrows()):
                current_price = row['close']
                current_signal = signals.iloc[i] if i < len(signals) else 'NONE'
                current_confidence = confidences.iloc[i] if i < len(confidences) else 0
                
                # Close existing trades that hit SL/TP
                for trade in open_trades[:]:
                    trade_pnl = self.calculate_trade_pnl(trade, current_price)
                    
                    # Check stop loss
                    if ((trade['type'] == 'BUY' and current_price <= trade['stop_loss']) or
                        (trade['type'] == 'SELL' and current_price >= trade['stop_loss'])):
                        
                        trade['close_price'] = trade['stop_loss']
                        trade['close_time'] = timestamp
                        trade['pnl'] = self.calculate_trade_pnl(trade, trade['stop_loss'])
                        trade['close_reason'] = 'Stop Loss'
                        
                        closed_trades.append(trade)
                        open_trades.remove(trade)
                        balance += trade['pnl']
                        
                        self.logger.debug(f"SL hit: {trade['type']} @ {trade['close_price']:.2f}, P&L: ${trade['pnl']:.2f}")
                    
                    # Check take profit
                    elif ((trade['type'] == 'BUY' and current_price >= trade['take_profit']) or
                          (trade['type'] == 'SELL' and current_price <= trade['take_profit'])):
                        
                        trade['close_price'] = trade['take_profit']
                        trade['close_time'] = timestamp
                        trade['pnl'] = self.calculate_trade_pnl(trade, trade['take_profit'])
                        trade['close_reason'] = 'Take Profit'
                        
                        closed_trades.append(trade)
                        open_trades.remove(trade)
                        balance += trade['pnl']
                        
                        self.logger.debug(f"TP hit: {trade['type']} @ {trade['close_price']:.2f}, P&L: ${trade['pnl']:.2f}")
                
                # Open new trades based on signals
                if (current_signal in ['BUY', 'SELL'] and 
                    current_confidence >= min_confidence and 
                    len(open_trades) == 0):  # Only one trade at a time
                    
                    # Calculate position size
                    risk_amount = balance * risk_per_trade
                    position_size = risk_amount / stop_loss_points  # Simplified
                    position_size = max(0.01, min(1.0, position_size))  # Lot limits
                    
                    # Apply spread and slippage
                    if current_signal == 'BUY':
                        entry_price = current_price + (spread_points + slippage_points) * 0.01
                        stop_loss = entry_price - stop_loss_points * 0.01
                        take_profit = entry_price + take_profit_points * 0.01
                    else:  # SELL
                        entry_price = current_price - (spread_points + slippage_points) * 0.01
                        stop_loss = entry_price + stop_loss_points * 0.01
                        take_profit = entry_price - take_profit_points * 0.01
                    
                    # Create trade
                    trade = {
                        'type': current_signal,
                        'open_time': timestamp,
                        'open_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': current_confidence
                    }
                    
                    open_trades.append(trade)
                    
                    self.logger.debug(f"Opened {current_signal} @ {entry_price:.2f}, "
                                    f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
                                    f"Confidence: {current_confidence:.2%}")
                
                # Calculate current equity
                current_equity = balance
                for trade in open_trades:
                    current_equity += self.calculate_trade_pnl(trade, current_price)
                
                equity_history.append({
                    'time': timestamp,
                    'balance': balance,
                    'equity': current_equity,
                    'open_trades': len(open_trades)
                })
            
            # Close remaining open trades
            final_price = price_data.iloc[-1]['close']
            for trade in open_trades:
                trade['close_price'] = final_price
                trade['close_time'] = price_data.index[-1]
                trade['pnl'] = self.calculate_trade_pnl(trade, final_price)
                trade['close_reason'] = 'End of Test'
                closed_trades.append(trade)
                balance += trade['pnl']
            
            # Store results
            self.trades = closed_trades
            self.equity_curve = equity_history
            
            # Calculate performance metrics
            final_balance = balance
            total_return = (final_balance - initial_balance) / initial_balance * 100
            
            if closed_trades:
                trade_pnls = [trade['pnl'] for trade in closed_trades]
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
                
                win_rate = len(winning_trades) / len(closed_trades) * 100
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
                
                # Calculate drawdown
                equity_values = [eq['equity'] for eq in equity_history]
                max_equity = 0
                max_drawdown = 0
                for equity in equity_values:
                    max_equity = max(max_equity, equity)
                    drawdown = (max_equity - equity) / max_equity * 100
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                max_drawdown = 0
            
            results = {
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return_pct': total_return,
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades) if 'winning_trades' in locals() else 0,
                'losing_trades': len(losing_trades) if 'losing_trades' in locals() else 0,
                'win_rate_pct': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': self.calculate_sharpe_ratio(equity_history),
                'total_days': (price_data.index[-1] - price_data.index[0]).days,
                'trades_per_day': len(closed_trades) / max(1, (price_data.index[-1] - price_data.index[0]).days)
            }
            
            self.results = results
            self.logger.info(f"Simulation completed: {len(closed_trades)} trades, "
                           f"{total_return:.2f}% return, {win_rate:.1f}% win rate")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Trading simulation error: {e}")
            return {}
    
    def calculate_trade_pnl(self, trade: Dict, current_price: float) -> float:
        """Calculate P&L for a trade"""
        try:
            if trade['type'] == 'BUY':
                pnl = (current_price - trade['open_price']) * trade['size'] * 100  # Simplified
            else:  # SELL
                pnl = (trade['open_price'] - current_price) * trade['size'] * 100  # Simplified
            
            return pnl
        except:
            return 0.0
    
    def calculate_sharpe_ratio(self, equity_history: List[Dict]) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(equity_history) < 2:
                return 0
            
            # Calculate daily returns
            equity_values = [eq['equity'] for eq in equity_history]
            returns = np.diff(equity_values) / equity_values[:-1]
            
            if len(returns) == 0 or np.std(returns) == 0:
                return 0
            
            # Annualized Sharpe ratio (assuming risk-free rate = 0)
            return np.mean(returns) / np.std(returns) * np.sqrt(252)  # 252 trading days
        except:
            return 0
    
    def run_backtest(
        self, 
        start_date: str = None, 
        end_date: str = None,
        symbol: str = "XAUUSDm"
    ) -> Dict:
        """Run complete backtest"""
        
        try:
            self.logger.info("Starting backtest...")
            
            # Get historical data
            price_data = self.get_historical_data(symbol, mt5.TIMEFRAME_M1, start_date, end_date)
            if price_data.empty:
                raise Exception("No price data available")
            
            # Prepare features
            features_df = self.prepare_features(price_data)
            if features_df.empty:
                raise Exception("Feature preparation failed")
            
            # Generate AI signals
            signals, confidences = self.generate_ai_signals(features_df)
            if signals.empty:
                raise Exception("Signal generation failed")
            
            # Run simulation
            results = self.simulate_trading(price_data, signals, confidences)
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return {}
    
    def generate_report(self, save_path: str = "backtest_report.html") -> str:
        """Generate comprehensive backtest report"""
        
        try:
            if not self.results:
                return "No backtest results available"
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Scalper XAUUSDm - Backtest Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #2E86AB; color: white; padding: 20px; text-align: center; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; }}
                    .positive {{ color: green; font-weight: bold; }}
                    .negative {{ color: red; font-weight: bold; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>AI Scalper XAUUSDm Backtest Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metric">
                        <strong>Total Return:</strong> 
                        <span class="{'positive' if self.results['total_return_pct'] > 0 else 'negative'}">
                            {self.results['total_return_pct']:.2f}%
                        </span>
                    </div>
                    <div class="metric">
                        <strong>Final Balance:</strong> ${self.results['final_balance']:,.2f}
                    </div>
                    <div class="metric">
                        <strong>Total Trades:</strong> {self.results['total_trades']}
                    </div>
                    <div class="metric">
                        <strong>Win Rate:</strong> {self.results['win_rate_pct']:.1f}%
                    </div>
                    <div class="metric">
                        <strong>Profit Factor:</strong> {self.results['profit_factor']:.2f}
                    </div>
                    <div class="metric">
                        <strong>Max Drawdown:</strong> 
                        <span class="negative">{self.results['max_drawdown_pct']:.2f}%</span>
                    </div>
                    <div class="metric">
                        <strong>Sharpe Ratio:</strong> {self.results['sharpe_ratio']:.2f}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Trading Statistics</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Trades</td><td>{self.results['total_trades']}</td></tr>
                        <tr><td>Winning Trades</td><td>{self.results['winning_trades']}</td></tr>
                        <tr><td>Losing Trades</td><td>{self.results['losing_trades']}</td></tr>
                        <tr><td>Average Win</td><td>${self.results['avg_win']:.2f}</td></tr>
                        <tr><td>Average Loss</td><td>${self.results['avg_loss']:.2f}</td></tr>
                        <tr><td>Trades per Day</td><td>{self.results['trades_per_day']:.1f}</td></tr>
                        <tr><td>Test Duration</td><td>{self.results['total_days']} days</td></tr>
                    </table>
                </div>
            """
            
            # Add trade details if available
            if self.trades:
                html_content += """
                <div class="section">
                    <h2>Recent Trades</h2>
                    <table>
                        <tr>
                            <th>Open Time</th>
                            <th>Type</th>
                            <th>Open Price</th>
                            <th>Close Price</th>
                            <th>P&L</th>
                            <th>Confidence</th>
                        </tr>
                """
                
                # Show last 20 trades
                recent_trades = self.trades[-20:] if len(self.trades) > 20 else self.trades
                for trade in recent_trades:
                    pnl_class = 'positive' if trade['pnl'] > 0 else 'negative'
                    html_content += f"""
                        <tr>
                            <td>{trade['open_time'].strftime('%Y-%m-%d %H:%M')}</td>
                            <td>{trade['type']}</td>
                            <td>{trade['open_price']:.2f}</td>
                            <td>{trade.get('close_price', 0):.2f}</td>
                            <td class="{pnl_class}">${trade['pnl']:.2f}</td>
                            <td>{trade['confidence']:.1%}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Save report
            with open(save_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Report saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return ""
    
    def plot_results(self, save_path: str = "backtest_charts.png"):
        """Plot backtest results"""
        
        try:
            if not self.equity_curve:
                print("No equity curve data available")
                return
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('AI Scalper XAUUSDm - Backtest Results', fontsize=16)
            
            # Equity curve
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['time'] = pd.to_datetime(equity_df['time'])
            
            ax1.plot(equity_df['time'], equity_df['equity'], 'b-', linewidth=2)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Account Equity ($)')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            equity_values = equity_df['equity'].values
            rolling_max = pd.Series(equity_values).expanding().max()
            drawdown = (equity_values - rolling_max) / rolling_max * 100
            
            ax2.fill_between(equity_df['time'], drawdown, 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            
            # Trade P&L distribution
            if self.trades:
                pnls = [trade['pnl'] for trade in self.trades]
                ax3.hist(pnls, bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax3.set_title('Trade P&L Distribution')
                ax3.set_xlabel('P&L ($)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
            
            # Monthly returns
            equity_df['month'] = equity_df['time'].dt.to_period('M')
            monthly_equity = equity_df.groupby('month')['equity'].last()
            monthly_returns = monthly_equity.pct_change().dropna() * 100
            
            colors = ['green' if ret > 0 else 'red' for ret in monthly_returns]
            ax4.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
            ax4.set_title('Monthly Returns')
            ax4.set_ylabel('Return (%)')
            ax4.set_xlabel('Month')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Charts saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Plotting error: {e}")

def main():
    """Main function for running backtests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Scalper Backtesting Engine')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='XAUUSDm', help='Trading symbol')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    parser.add_argument('--plot', action='store_true', help='Generate charts')
    args = parser.parse_args()
    
    try:
        print("="*60)
        print("AI SCALPER XAUUSD - BACKTESTING ENGINE")
        print("="*60)
        
        # Initialize backtester
        backtester = BacktestEngine()
        
        # Run backtest
        results = backtester.run_backtest(
            start_date=args.start,
            end_date=args.end,
            symbol=args.symbol
        )
        
        if not results:
            print("Backtest failed!")
            return
        
        # Print results
        print("\nBACKTEST RESULTS:")
        print("="*40)
        print(f"Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"Final Balance:   ${results['final_balance']:,.2f}")
        print(f"Total Return:    {results['total_return_pct']:+.2f}%")
        print(f"Total Trades:    {results['total_trades']}")
        print(f"Win Rate:        {results['win_rate_pct']:.1f}%")
        print(f"Profit Factor:   {results['profit_factor']:.2f}")
        print(f"Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
        print(f"Trades/Day:      {results['trades_per_day']:.1f}")
        
        # Generate report
        if args.report:
            report_path = backtester.generate_report()
            if report_path:
                print(f"\nHTML report saved: {report_path}")
        
        # Generate charts
        if args.plot:
            backtester.plot_results()
            print("Charts displayed and saved")
        
        print("\nBacktest completed successfully!")
        
    except Exception as e:
        print(f"Backtest error: {e}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()