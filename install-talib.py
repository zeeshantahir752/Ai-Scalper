#!/usr/bin/env python3
"""
Automated TA-Lib installer for Python 3.11.6
Handles multiple installation methods and troubleshooting
"""

import sys
import subprocess
import platform
import urllib.request
import os
from pathlib import Path

def get_system_info():
    """Get system and Python information"""
    info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'python_major_minor': f"{sys.version_info.major}{sys.version_info.minor}",
        'platform': platform.system(),
        'architecture': platform.architecture()[0],
        'machine': platform.machine()
    }
    
    print("🔍 System Information:")
    print(f"   Python: {info['python_version']}")
    print(f"   Platform: {info['platform']}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Machine: {info['machine']}")
    
    return info

def test_talib():
    """Test if TA-Lib is already installed and working"""
    try:
        import talib
        print(f"✅ TA-Lib is already installed: version {talib.__version__}")
        
        # Test basic functionality
        import numpy as np
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        sma = talib.SMA(data, timeperiod=3)
        print(f"✅ TA-Lib working correctly (test SMA: {sma[-1]:.2f})")
        return True
    except ImportError:
        print("❌ TA-Lib not installed")
        return False
    except Exception as e:
        print(f"❌ TA-Lib installed but not working: {e}")
        return False

def install_windows_wheel(info):
    """Install TA-Lib using pre-compiled wheel for Windows"""
    print("\n📥 Installing TA-Lib wheel for Windows...")
    
    # Determine correct wheel for Python 3.11
    if info['architecture'] == '64bit':
        wheel_name = "TA_Lib-0.4.26-cp311-cp311-win_amd64.whl"
        wheel_url = f"https://github.com/cgohlke/talib-build/releases/download/v0.4.26/{wheel_name}"
    else:
        wheel_name = "TA_Lib-0.4.26-cp311-cp311-win32.whl"
        wheel_url = f"https://github.com/cgohlke/talib-build/releases/download/v0.4.26/{wheel_name}"
    
    try:
        print(f"Downloading {wheel_name}...")
        
        # Download wheel
        urllib.request.urlretrieve(wheel_url, wheel_name)
        print(f"✅ Downloaded {wheel_name}")
        
        # Install wheel
        print("Installing wheel...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", wheel_name], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ TA-Lib wheel installed successfully!")
            os.remove(wheel_name)  # Clean up
            return True
        else:
            print(f"❌ Wheel installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading/installing wheel: {e}")
        return False

def install_alternative_methods():
    """Try alternative installation methods"""
    methods = [
        ("pip install TA-Lib", ["pip", "install", "TA-Lib"]),
        ("pip install --no-cache-dir TA-Lib", ["pip", "install", "--no-cache-dir", "TA-Lib"]),
        ("pip install --no-binary TA-Lib TA-Lib", ["pip", "install", "--no-binary", "TA-Lib", "TA-Lib"]),
        ("pip install TA-Lib --user", ["pip", "install", "TA-Lib", "--user"]),
    ]
    
    for method_name, command in methods:
        print(f"\n🔄 Trying: {method_name}")
        try:
            result = subprocess.run([sys.executable, "-m"] + command, 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Success with: {method_name}")
                return True
            else:
                print(f"❌ Failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("❌ Timeout (5 minutes)")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return False

def install_conda():
    """Try installing via conda"""
    print("\n🐍 Trying conda installation...")
    
    try:
        # Check if conda is available
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Conda not available")
            return False
        
        print("Conda found, installing TA-Lib...")
        result = subprocess.run(["conda", "install", "-c", "conda-forge", "ta-lib", "-y"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ TA-Lib installed via conda!")
            return True
        else:
            print(f"❌ Conda installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Conda installation error: {e}")
        return False

def install_build_tools_windows():
    """Guide for installing Visual Studio Build Tools"""
    print("\n🔨 Visual Studio Build Tools Required")
    print("=" * 50)
    print("To compile TA-Lib from source, you need build tools.")
    print()
    print("Option 1 - Microsoft C++ Build Tools:")
    print("1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("2. Install 'C++ build tools' workload")
    print("3. Restart computer")
    print("4. Try installing TA-Lib again")
    print()
    print("Option 2 - Visual Studio Community:")
    print("1. Download from: https://visualstudio.microsoft.com/vs/community/")
    print("2. Install with 'Desktop development with C++' workload")
    print()
    print("Option 3 - Use pre-compiled wheel (recommended)")

def install_linux_dependencies():
    """Install TA-Lib dependencies on Linux"""
    print("\n🐧 Installing Linux dependencies...")
    
    # Detect Linux distribution
    try:
        with open('/etc/os-release', 'r') as f:
            content = f.read()
            
        if 'ubuntu' in content.lower() or 'debian' in content.lower():
            commands = [
                ["sudo", "apt-get", "update"],
                ["sudo", "apt-get", "install", "-y", "build-essential", "python3-dev"],
                ["sudo", "apt-get", "install", "-y", "libta-lib-dev"]
            ]
        elif 'centos' in content.lower() or 'rhel' in content.lower() or 'fedora' in content.lower():
            commands = [
                ["sudo", "yum", "groupinstall", "-y", "Development Tools"],
                ["sudo", "yum", "install", "-y", "python3-devel", "ta-lib-devel"]
            ]
        else:
            print("❌ Unsupported Linux distribution")
            return False
        
        for cmd in commands:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Command failed: {result.stderr}")
                return False
        
        print("✅ Linux dependencies installed")
        return True
        
    except Exception as e:
        print(f"❌ Error installing Linux dependencies: {e}")
        return False

def install_macos_dependencies():
    """Install TA-Lib dependencies on macOS"""
    print("\n🍎 Installing macOS dependencies...")
    
    try:
        # Check if Homebrew is installed
        result = subprocess.run(["brew", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Homebrew not installed")
            print("Install Homebrew first: https://brew.sh")
            return False
        
        # Install TA-Lib via Homebrew
        print("Installing TA-Lib via Homebrew...")
        result = subprocess.run(["brew", "install", "ta-lib"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ TA-Lib installed via Homebrew")
            return True
        else:
            print(f"❌ Homebrew installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ macOS installation error: {e}")
        return False

def provide_manual_solutions(info):
    """Provide manual installation solutions"""
    print("\n📖 MANUAL INSTALLATION SOLUTIONS")
    print("=" * 50)
    
    if info['platform'] == 'Windows':
        print("🪟 Windows Solutions:")
        print()
        print("1. EASIEST - Download pre-compiled wheel:")
        print("   • Go to: https://github.com/cgohlke/talib-build/releases")
        print("   • Download: TA_Lib-0.4.26-cp311-cp311-win_amd64.whl")
        print("   • Install: pip install TA_Lib-0.4.26-cp311-cp311-win_amd64.whl")
        print()
        print("2. Alternative wheels repository:")
        print("   • Go to: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("   • Find wheel for cp311 (Python 3.11)")
        print("   • Download and install with pip")
        print()
        print("3. Anaconda method:")
        print("   • Install Anaconda/Miniconda")
        print("   • conda install -c conda-forge ta-lib")
        print()
        
    elif info['platform'] == 'Linux':
        print("🐧 Linux Solutions:")
        print()
        print("1. Ubuntu/Debian:")
        print("   sudo apt-get update")
        print("   sudo apt-get install build-essential python3-dev")
        print("   sudo apt-get install libta-lib-dev")
        print("   pip install TA-Lib")
        print()
        print("2. CentOS/RHEL/Fedora:")
        print("   sudo yum groupinstall 'Development Tools'")
        print("   sudo yum install python3-devel ta-lib-devel")
        print("   pip install TA-Lib")
        print()
        print("3. Compile from source:")
        print("   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
        print("   tar -xzf ta-lib-0.4.0-src.tar.gz")
        print("   cd ta-lib/")
        print("   ./configure --prefix=/usr/local")
        print("   make && sudo make install")
        print("   pip install TA-Lib")
        
    elif info['platform'] == 'Darwin':  # macOS
        print("🍎 macOS Solutions:")
        print()
        print("1. Homebrew (recommended):")
        print("   brew install ta-lib")
        print("   pip install TA-Lib")
        print()
        print("2. MacPorts:")
        print("   sudo port install ta-lib")
        print("   pip install TA-Lib")
        print()
        print("3. Compile from source:")
        print("   Same as Linux method above")

def main():
    """Main installation function"""
    print("🔧 TA-Lib Installation Helper for Python 3.11.6")
    print("=" * 60)
    
    # Get system info
    info = get_system_info()
    
    # Test if already installed
    if test_talib():
        print("\n🎉 TA-Lib is already working! No installation needed.")
        return True
    
    print(f"\n🚀 Starting TA-Lib installation for {info['platform']}...")
    
    # Platform-specific installation
    success = False
    
    if info['platform'] == 'Windows':
        # Try wheel installation first
        success = install_windows_wheel(info)
        
        if not success:
            success = install_alternative_methods()
        
        if not success:
            success = install_conda()
        
        if not success:
            install_build_tools_windows()
            
    elif info['platform'] == 'Linux':
        # Install dependencies first
        if install_linux_dependencies():
            success = install_alternative_methods()
        
        if not success:
            success = install_conda()
            
    elif info['platform'] == 'Darwin':  # macOS
        # Install via Homebrew first
        if install_macos_dependencies():
            success = install_alternative_methods()
        
        if not success:
            success = install_conda()
    
    # Final test
    if success:
        print("\n🧪 Testing installation...")
        if test_talib():
            print("\n🎉 TA-Lib installation completed successfully!")
            return True
        else:
            print("\n❌ Installation reported success but TA-Lib not working")
    
    # If all methods failed, provide manual solutions
    if not success:
        print("\n❌ Automatic installation failed")
        provide_manual_solutions(info)
        
        print("\n💡 QUICK WORKAROUNDS:")
        print("1. Try using Python 3.10 instead of 3.11")
        print("2. Use Anaconda Python distribution")
        print("3. Use Docker with pre-built environment")
        print("4. Use cloud-based development (Google Colab, etc.)")
    
    return success

if __name__ == "__main__":
    main()