@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo  AI SCALPER XAUUSD INSTALLER (Windows)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/8] Python found
python --version

REM Check if we're in the right directory
if not exist "ai_signal_generator_advanced.py" (
    echo ERROR: Please run this script from the project directory
    echo Current directory should contain Python files
    pause
    exit /b 1
)

echo [2/8] Project structure verified

REM Create directories
echo [3/8] Creating directories...
if not exist "signals" mkdir signals
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "data" mkdir data

REM Create virtual environment
echo [4/8] Creating virtual environment...
if exist "ai_scalper_env" (
    echo Virtual environment already exists, removing...
    rmdir /s /q ai_scalper_env
)
python -m venv ai_scalper_env

REM Activate virtual environment
echo [5/8] Activating virtual environment...
call ai_scalper_env\Scripts\activate.bat

REM Upgrade pip
echo [6/8] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [7/8] Installing Python packages...
pip install -r requirements.txt

REM Check if TA-Lib installation was successful
echo [8/8] Verifying installation...
python -c "import talib" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: TA-Lib installation may have failed
    echo If you encounter TA-Lib errors:
    echo 1. Download the appropriate wheel from:
    echo    https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
    echo 2. Install using: pip install downloaded_wheel_file.whl
    echo.
)

REM Create startup script
echo Creating startup script...
(
echo @echo off
echo echo Starting AI Scalper XAUUSD System...
echo call ai_scalper_env\Scripts\activate.bat
echo python run_ai_system.py
echo pause
) > start_ai_scalper.bat

REM Create training script
(
echo @echo off
echo echo Training AI Model...
echo call ai_scalper_env\Scripts\activate.bat
echo python train_model_xgboost.py
echo pause
) > train_model.bat

echo.
echo ==========================================
echo  INSTALLATION COMPLETED SUCCESSFULLY!
echo ==========================================
echo.
echo Next steps:
echo 1. Copy AI_Scalper_Pro_XAUUSD.mq5 to your MT5 Experts folder
echo 2. Compile the EA in MetaEditor
echo 3. Train the model: train_model.bat
echo 4. Start system: start_ai_scalper.bat
echo.
echo For manual operations:
echo - Activate environment: ai_scalper_env\Scripts\activate.bat
echo - Train model: python train_model_xgboost.py
echo - Run system: python run_ai_system.py
echo.
pause