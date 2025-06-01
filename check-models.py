#!/usr/bin/env python3
"""
Check if AI model files exist and are properly configured
"""

import os
import json
from pathlib import Path
import pickle

def check_config():
    """Check configuration file"""
    config_file = "config.json"
    
    print("📋 Checking Configuration...")
    print("=" * 40)
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Config file found: {config_file}")
        
        # Check paths section
        if 'paths' in config:
            paths = config['paths']
            print("\n📁 Configured Paths:")
            for key, path in paths.items():
                print(f"   {key}: {path}")
            return paths
        else:
            print("❌ No 'paths' section in config")
            return None
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def check_model_files(paths):
    """Check if model files exist"""
    if not paths:
        return False
    
    print("\n🤖 Checking Model Files...")
    print("=" * 40)
    
    required_files = [
        ('model_path', 'XGBoost Model'),
        ('scaler_path', 'Feature Scaler'),
        ('anomaly_detector_path', 'Anomaly Detector')
    ]
    
    all_exist = True
    
    for path_key, description in required_files:
        if path_key in paths:
            file_path = paths[path_key]
            
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {description}: {file_path} ({file_size:,} bytes)")
                
                # Try to load the file to verify it's valid
                try:
                    with open(file_path, 'rb') as f:
                        pickle.load(f)
                    print(f"   ✅ File is loadable")
                except Exception as e:
                    print(f"   ❌ File corrupt: {e}")
                    all_exist = False
            else:
                print(f"❌ {description}: {file_path} - NOT FOUND")
                all_exist = False
        else:
            print(f"❌ {description}: Path not configured")
            all_exist = False
    
    return all_exist

def check_signal_file(paths):
    """Check signal file setup"""
    if not paths or 'signal_file' not in paths:
        return False
    
    print("\n📊 Checking Signal File...")
    print("=" * 40)
    
    signal_path = paths['signal_file']
    signal_dir = os.path.dirname(signal_path)
    
    # Check directory exists
    if not os.path.exists(signal_dir):
        print(f"❌ Signal directory missing: {signal_dir}")
        os.makedirs(signal_dir, exist_ok=True)
        print(f"✅ Created directory: {signal_dir}")
    
    # Check if signal file exists
    if os.path.exists(signal_path):
        print(f"✅ Signal file exists: {signal_path}")
        
        # Check content
        try:
            with open(signal_path, 'r') as f:
                content = json.load(f)
            print(f"   Content: {content}")
        except Exception as e:
            print(f"   ❌ Signal file corrupt: {e}")
            return False
    else:
        print(f"⚠️  Signal file will be created: {signal_path}")
        
        # Create default signal file
        default_signal = {
            "signal": "NONE",
            "confidence": 0.0,
            "price": 0.0,
            "spread": 0,
            "timestamp": "2025-05-30T18:00:00",
            "reason": "No signal generated yet"
        }
        
        try:
            with open(signal_path, 'w') as f:
                json.dump(default_signal, f, indent=2)
            print(f"✅ Created default signal file")
        except Exception as e:
            print(f"❌ Failed to create signal file: {e}")
            return False
    
    return True

def check_mt5_connection():
    """Check MT5 connection"""
    print("\n🔌 Checking MT5 Connection...")
    print("=" * 40)
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print("❌ MT5 not initialized")
            print("💡 Make sure MT5 is running")
            return False
        
        # Try to connect with your credentials
        authorized = mt5.login(247501403, password="Gold123@", server="Exness-MT5Trial")
        if not authorized:
            print("❌ Failed to connect to Exness account")
            print("💡 Check MT5 login credentials")
            mt5.shutdown()
            return False
        
        # Check account info
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Connected to account: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Balance: ${account_info.balance:,.2f}")
        
        # Check XAUUSDm symbol
        symbol_info = mt5.symbol_info("XAUUSDm")
        if symbol_info:
            print(f"✅ XAUUSDm symbol available")
            print(f"   Spread: {symbol_info.spread} points")
        else:
            print("❌ XAUUSDm symbol not found")
            print("💡 Add XAUUSDm to Market Watch in MT5")
        
        mt5.shutdown()
        return True
        
    except ImportError:
        print("❌ MetaTrader5 package not installed")
        return False
    except Exception as e:
        print(f"❌ MT5 connection error: {e}")
        return False

def create_training_script():
    """Create a simple training script to generate models"""
    print("\n🛠️  Creating Quick Training Script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Quick model training script to generate required files
"""

import json
import pickle
import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def create_dummy_models():
    """Create dummy models for testing"""
    print("Creating dummy models for testing...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Create dummy XGBoost model
    X_dummy = np.random.rand(100, 50)
    y_dummy = np.random.randint(0, 3, 100)
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=10,
        random_state=42
    )
    model.fit(X_dummy, y_dummy)
    
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Created XGBoost model")
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ Created scaler")
    
    # Create anomaly detector
    anomaly_detector = IsolationForest(random_state=42)
    anomaly_detector.fit(X_dummy)
    
    with open("models/anomaly_detector.pkl", "wb") as f:
        pickle.dump(anomaly_detector, f)
    print("✅ Created anomaly detector")
    
    print("\\n🎉 Dummy models created successfully!")
    print("These are for testing only. Train real models with:")
    print("python train_model_xgboost.py")

if __name__ == "__main__":
    create_dummy_models()
'''
    
    with open("create_dummy_models.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created: create_dummy_models.py")

def main():
    """Main check function"""
    print("🔍 AI Model Diagnostic Check")
    print("=" * 50)
    
    # Check configuration
    paths = check_config()
    
    # Check model files
    models_exist = check_model_files(paths)
    
    # Check signal file
    signal_ok = check_signal_file(paths)
    
    # Check MT5 connection
    mt5_ok = check_mt5_connection()
    
    # Summary and recommendations
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    print(f"{'✅' if paths else '❌'} Configuration: {'OK' if paths else 'Missing'}")
    print(f"{'✅' if models_exist else '❌'} Model Files: {'Found' if models_exist else 'Missing'}")
    print(f"{'✅' if signal_ok else '❌'} Signal Setup: {'OK' if signal_ok else 'Issues'}")
    print(f"{'✅' if mt5_ok else '❌'} MT5 Connection: {'OK' if mt5_ok else 'Issues'}")
    
    if not models_exist:
        print("\n🚨 MODELS NOT FOUND - This is the main issue!")
        print("\n📋 Solutions:")
        print("1. Train real models:")
        print("   python train_model_xgboost.py")
        print()
        print("2. Create dummy models for testing:")
        create_training_script()
        print("   python create_dummy_models.py")
        print()
        print("3. Then test the AI system:")
        print("   python ai_signal_generator_advanced.py")
    else:
        print("\n✅ All systems ready!")
        print("You can run: python ai_signal_generator_advanced.py")

if __name__ == "__main__":
    main()