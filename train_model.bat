@echo off
echo Training AI Model...
call ai_scalper_env\Scripts\activate.bat
python train_model_xgboost.py
pause
