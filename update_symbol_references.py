#!/usr/bin/env python3
"""
Script to update all XAUUSD references to XAUUSDm for Exness
Updates all Python files, config files, and documentation
"""

import os
import re
from pathlib import Path

def update_file_content(file_path, replacements):
    """Update content in a file with given replacements"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_text, new_text in replacements:
            content = content.replace(old_text, new_text)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all symbol references"""
    print("🔄 Updating XAUUSD → XAUUSDm for Exness")
    print("=" * 50)
    
    # Define replacements
    replacements = [
        # Symbol references
        ('"XAUUSD"', '"XAUUSDm"'),
        ("'XAUUSD'", "'XAUUSDm'"),
        ('symbol="XAUUSD"', 'symbol="XAUUSDm"'),
        ("symbol='XAUUSD'", "symbol='XAUUSDm'"),
        ('= "XAUUSD"', '= "XAUUSDm"'),
        ("= 'XAUUSD'", "= 'XAUUSDm'"),
        
        # File paths
        ('signals/xau_signal.txt', 'signals/xauusdm_signal.txt'),
        ('signals\\xau_signal.txt', 'signals\\xauusdm_signal.txt'),
        
        # Comments and descriptions
        ('XAUUSD trading', 'XAUUSDm trading'),
        ('XAUUSD scalping', 'XAUUSDm scalping'),
        ('for XAUUSD', 'for XAUUSDm'),
        ('XAUUSD (Gold)', 'XAUUSDm (Gold)'),
        ('AI Scalper XAUUSD', 'AI Scalper XAUUSDm'),
        
        # MT5 specific
        ('copy_rates_from_pos("XAUUSD"', 'copy_rates_from_pos("XAUUSDm"'),
        ('positions_get(symbol="XAUUSD")', 'positions_get(symbol="XAUUSDm")'),
        ('symbol_info("XAUUSD")', 'symbol_info("XAUUSDm")'),
        ('symbol_info_tick("XAUUSD")', 'symbol_info_tick("XAUUSDm")'),
    ]
    
    # Files to update
    python_files = [
        'ai_signal_generator_advanced.py',
        'train_model_xgboost.py', 
        'run_ai_system.py',
        'utils.py',
        'monitoring.py',
        'backtest.py',
        'web_dashboard.py',
        'setup_exness_final.py'
    ]
    
    config_files = [
        'config.json',
        'requirements.txt'
    ]
    
    mq5_files = [
        'mt5_ea/AI_Scalper_Pro_XAUUSD.mq5'
    ]
    
    doc_files = [
        'README.md',
        'EXNESS_QUICKSTART.md'
    ]
    
    all_files = python_files + config_files + mq5_files + doc_files
    
    updated_count = 0
    
    # Update each file
    for file_path in all_files:
        if os.path.exists(file_path):
            if update_file_content(file_path, replacements):
                updated_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Update any remaining Python files in current directory
    for py_file in Path('.').glob('*.py'):
        if py_file.name not in [f for f in python_files if '/' not in f]:
            if py_file.name != 'update_symbol_references.py':  # Don't update self
                if update_file_content(py_file, replacements):
                    updated_count += 1
    
    # Create updated signal file if it doesn't exist
    signal_file = Path('signals/xauusdm_signal.txt')
    signal_file.parent.mkdir(exist_ok=True)
    
    if not signal_file.exists():
        default_signal = {
            "signal": "NONE",
            "confidence": 0.0,
            "price": 0.0,
            "spread": 0,
            "timestamp": "2025-01-15T00:00:00",
            "reason": "No signal generated yet"
        }
        
        import json
        with open(signal_file, 'w') as f:
            json.dump(default_signal, f, indent=2)
        print(f"✅ Created: {signal_file}")
        updated_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 SYMBOL UPDATE COMPLETE")
    print("=" * 50)
    print(f"Files updated: {updated_count}")
    print(f"Symbol changed: XAUUSD → XAUUSDm")
    print(f"Signal file: signals/xauusdm_signal.txt")
    print(f"Configured for: Exness account 247501403")
    
    print("\n📋 NEXT STEPS:")
    print("1. Verify MT5 has XAUUSDm in Market Watch")
    print("2. Recompile MT5 EA if needed")
    print("3. Update EA signal file path to: signals\\xauusdm_signal.txt")
    print("4. Test connection: python -c \"import MetaTrader5 as mt5; mt5.initialize(); print(mt5.symbol_info('XAUUSDm'))\"")
    
    # Test MT5 connection
    print("\n🔍 Testing XAUUSDm symbol availability...")
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            symbol_info = mt5.symbol_info("XAUUSDm")
            if symbol_info:
                print("✅ XAUUSDm symbol is available")
                print(f"   Contract size: {symbol_info.trade_contract_size}")
                print(f"   Min lot: {symbol_info.volume_min}")
                print(f"   Spread: {symbol_info.spread} points")
            else:
                print("❌ XAUUSDm symbol not found")
                print("💡 Add XAUUSDm to Market Watch in MT5")
            mt5.shutdown()
        else:
            print("❌ MT5 not initialized")
    except ImportError:
        print("⚠️  MetaTrader5 package not available for testing")
    
    print("\n🎉 All symbol references updated for Exness XAUUSDm!")

if __name__ == "__main__":
    main()