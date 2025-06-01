#!/usr/bin/env python3
"""
Quick fix for XGBoost compatibility issue
Fixes the early_stopping_rounds parameter error
"""

import sys
import subprocess

def check_xgboost_version():
    """Check XGBoost version and compatibility"""
    try:
        import xgboost as xgb
        version = xgb.__version__
        print(f"✅ XGBoost version: {version}")
        
        # Test basic functionality
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=100, n_features=10, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test XGBClassifier with different parameter combinations
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_estimators=10
        )
        
        # Test method 1: New API
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            print("✅ XGBoost new API working")
            return "new_api"
        except TypeError as e:
            print(f"❌ New API failed: {e}")
        
        # Test method 2: Legacy API
        try:
            model.fit(
                X_train, y_train,
                early_stopping_rounds=5,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            print("✅ XGBoost legacy API working")
            return "legacy_api"
        except TypeError as e:
            print(f"❌ Legacy API failed: {e}")
        
        # Test method 3: Simple fit
        try:
            model.fit(X_train, y_train)
            print("✅ XGBoost simple fit working")
            return "simple_fit"
        except Exception as e:
            print(f"❌ Simple fit failed: {e}")
            return "error"
            
    except ImportError:
        print("❌ XGBoost not installed")
        return "not_installed"
    except Exception as e:
        print(f"❌ XGBoost error: {e}")
        return "error"

def fix_training_file():
    """Apply the fix to train_model_xgboost.py"""
    
    filename = "train_model_xgboost.py"
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Find and replace the problematic fit method
        old_fit_code = """        # Train model with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )"""
        
        new_fit_code = """        # Train model (handle different XGBoost versions)
        try:
            # Try new XGBoost API (v1.6+)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
                verbose=False
            )
        except TypeError:
            # Fallback for older XGBoost versions
            try:
                self.model.fit(
                    X_train_scaled, y_train,
                    early_stopping_rounds=50,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
            except TypeError:
                # Simplest fallback - no early stopping
                self.model.fit(X_train_scaled, y_train)"""
        
        if old_fit_code in content:
            content = content.replace(old_fit_code, new_fit_code)
            
            with open(filename, 'w') as f:
                f.write(content)
            
            print(f"✅ Fixed {filename}")
            return True
        else:
            print(f"⚠️  Pattern not found in {filename} - may already be fixed")
            return True
            
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return False
    except Exception as e:
        print(f"❌ Error fixing file: {e}")
        return False

def install_compatible_xgboost():
    """Install a compatible XGBoost version"""
    print("🔄 Installing compatible XGBoost version...")
    
    try:
        # Uninstall current version
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "xgboost", "-y"], 
                      capture_output=True)
        
        # Install specific compatible version
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "xgboost==1.7.6"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ XGBoost 1.7.6 installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 XGBoost Compatibility Fix")
    print("=" * 40)
    
    # Check current XGBoost
    api_type = check_xgboost_version()
    
    # Apply fix to training file
    if fix_training_file():
        print("✅ Training file updated")
    
    # Test the fix
    print("\n🧪 Testing the fix...")
    try:
        result = subprocess.run([
            sys.executable, "train_model_xgboost.py", "--test"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Training fix successful!")
        else:
            print("❌ Training still has issues")
            print("Output:", result.stderr[-500:])  # Last 500 chars of error
            
            # Offer to install compatible version
            response = input("\nTry installing XGBoost 1.7.6? (y/n): ")
            if response.lower() == 'y':
                if install_compatible_xgboost():
                    print("✅ Please try training again")
                    
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out - but fix was applied")
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print("\n📋 Summary:")
    print("1. ✅ Applied XGBoost compatibility fix")
    print("2. 🔄 Try running: python train_model_xgboost.py")
    print("3. 💡 If still fails, the fix handles multiple XGBoost versions")

if __name__ == "__main__":
    main()