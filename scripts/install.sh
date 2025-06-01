#!/bin/bash
echo "AI Scalper XAUUSD Installer"
echo "============================"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv ai_scalper_env

echo "Installing packages..."
source ai_scalper_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Installation completed!"
