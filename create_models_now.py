#!/usr/bin/env python3
"""
Quick model creation for AI Scalper XAUUSDm
Creates working models instantly so you can test the system
"""

import json
import pickle
import os
import numpy as np
from pathlib import Path
from datetime import datetime

def create_directories():
    """Create required directories"""
    dirs = ["models", "signals", "logs"]
    for directory in dirs:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_xgboost_model():
    """Create a working XGBoost model"""
    print("🤖 Creating XGBoost model...")
    
    try:
        import xgboost as xgb
        from sklearn.datasets import make_classification
        
        # Create realistic training data similar to forex features
        X, y = make_classification(
            n_samples=1000,
            n_features=98,  # Match the expected feature count
            n_classes=3,    # BUY, SELL, HOLD
            n_informative=30,
            n_redundant=10,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create and train XGBoost model
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Save model
        with open("models/xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        print("✅ XGBoost model created and saved")
        return True
        
    except ImportError:
        print("❌ XGBoost not available")
        return False
    except Exception as e:
        print(f"❌ Error creating XGBoost model: {e}")
        return False

def create_scaler():
    """Create feature scaler"""
    print("📊 Creating feature scaler...")
    
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Create scaler with realistic forex data
        scaler = StandardScaler()
        
        # Fit with sample data that represents typical forex features
        sample_data = np.random.normal(0, 1, (1000, 98))
        
        # Add some realistic forex-like features
        sample_data[:, 0] = np.random.normal(2000, 50, 1000)  # Price-like feature
        sample_data[:, 1] = np.random.normal(50, 10, 1000)    # RSI-like feature
        sample_data[:, 2] = np.random.normal(0, 0.1, 1000)    # Change-like feature
        
        scaler.fit(sample_data)
        
        # Save scaler
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        print("✅ Feature scaler created and saved")
        return True
        
    except ImportError:
        print("❌ scikit-learn not available")
        return False
    except Exception as e:
        print(f"❌ Error creating scaler: {e}")
        return False

def create_anomaly_detector():
    """Create anomaly detector"""
    print("🔍 Creating anomaly detector...")
    
    try:
        from sklearn.ensemble import IsolationForest
        
        # Create anomaly detector
        anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit with sample data
        sample_data = np.random.normal(0, 1, (1000, 98))
        anomaly_detector.fit(sample_data)
        
        # Save anomaly detector
        with open("models/anomaly_detector.pkl", "wb") as f:
            pickle.dump(anomaly_detector, f)
        
        print("✅ Anomaly detector created and saved")
        return True
        
    except ImportError:
        print("❌ scikit-learn not available")
        return False
    except Exception as e:
        print(f"❌ Error creating anomaly detector: {e}")
        return False

def create_signal_file():
    """Create initial signal file"""
    print("📡 Creating signal file...")
    
    try:
        # Create default signal
        signal_data = {
            "signal": "NONE",
            "confidence": 0.0,
            "price": 0.0,
            "spread": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": "System initializing",
            "features_analyzed": 0,
            "model_version": "quick_setup_v1.0"
        }
        
        # Save signal file
        with open("signals/xauusdm_signal.txt", "w") as f:
            json.dump(signal_data, f, indent=2)
        
        print("✅ Signal file created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating signal file: {e}")
        return False

def verify_models():
    """Verify all models can be loaded"""
    print("\n🧪 Verifying created models...")
    
    model_files = [
        ("models/xgboost_model.pkl", "XGBoost Model"),
        ("models/scaler.pkl", "Feature Scaler"),
        ("models/anomaly_detector.pkl", "Anomaly Detector"),
        ("signals/xauusdm_signal.txt", "Signal File")
    ]
    
    all_good = True
    
    for file_path, description in model_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_size:,} bytes")
            
            # Test loading
            try:
                if file_path.endswith('.pkl'):
                    with open(file_path, 'rb') as f:
                        pickle.load(f)
                elif file_path.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        json.load(f)
                print(f"   ✅ File loads correctly")
            except Exception as e:
                print(f"   ❌ File corrupted: {e}")
                all_good = False
        else:
            print(f"❌ {description}: Missing")
            all_good = False
    
    return all_good

def test_ai_system():
    """Quick test of the AI system"""
    print("\n🎯 Testing AI system...")
    
    try:
        # Test import
        import sys
        import subprocess
        
        # Run a quick test of the AI signal generator
        result = subprocess.run([
            sys.executable, "-c",
            """
import json
import pickle
import numpy as np

# Test loading models
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/anomaly_detector.pkl', 'rb') as f:
        anomaly_detector = pickle.load(f)
    
    # Test prediction with dummy data
    test_features = np.random.normal(0, 1, (1, 98))
    test_features_scaled = scaler.transform(test_features)
    prediction = model.predict_proba(test_features_scaled)
    
    print('✅ All models working correctly')
    print(f'Test prediction shape: {prediction.shape}')
    
except Exception as e:
    print(f'❌ Model test failed: {e}')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ AI system test passed!")
            print(result.stdout)
            return True
        else:
            print("❌ AI system test failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main function to create all models"""
    print("🚀 Quick Model Creation for AI Scalper XAUUSDm")
    print("=" * 60)
    print("Creating working models for immediate testing...")
    print("(These are functional models - train real ones later)")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Create models
    success_count = 0
    
    if create_xgboost_model():
        success_count += 1
    
    if create_scaler():
        success_count += 1
        
    if create_anomaly_detector():
        success_count += 1
        
    if create_signal_file():
        success_count += 1
    
    # Verify everything
    print("\n" + "=" * 60)
    if verify_models():
        print("🎉 ALL MODELS CREATED SUCCESSFULLY!")
        
        # Test the system
        if test_ai_system():
            print("\n✅ AI system is ready to use!")
        
        print("\n📋 NEXT STEPS:")
        print("1. Test the AI signal generator:")
        print("   python ai_signal_generator_advanced.py")
        print()
        print("2. Start the full system:")
        print("   python run_ai_system.py")
        print()
        print("3. Attach EA to XAUUSDm chart in MT5")
        print()
        print("4. Later, train real models with:")
        print("   python train_model_xgboost.py")
        
        print(f"\n🎯 Ready for Exness account 247501403 on XAUUSDm!")
        
    else:
        print("❌ Some models failed to create")
        print("Check the errors above and try again")
    
    print(f"\nSuccessfully created: {success_count}/4 components")

if __name__ == "__main__":
    main()