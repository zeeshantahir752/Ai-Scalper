import json
import pickle
import os

print("Testing model loading...")

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

paths = config["paths"]

# Test each model
print(f"Loading: {paths['model_path']}")
with open(paths["model_path"], "rb") as f:
    model = pickle.load(f)
print("✅ XGBoost model loaded")

print(f"Loading: {paths['scaler_path']}")
with open(paths["scaler_path"], "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded")

print("🎉 All models working!")