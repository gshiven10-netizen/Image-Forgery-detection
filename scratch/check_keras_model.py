import os
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "forgery_model.keras"
if not os.path.exists(model_path):
    print(f"ERROR: {model_path} not found")
else:
    try:
        print(f"Attempting to load full model from {model_path}...")
        model = load_model(model_path)
        print("✅ SUCCESS: Full model loaded successfully!")
        model.summary()
    except Exception as e:
        print(f"❌ FAILURE: Could not load full model: {e}")

weights_path = "weights.weights.h5"
if os.path.exists(weights_path):
    print(f"\nChecking weights file {weights_path}...")
    import h5py
    with h5py.File(weights_path, 'r') as f:
        print(f"Keys in weights file: {list(f.keys())}")
        if 'layers' in f:
             print(f"Sub-keys in 'layers': {list(f['layers'].keys())}")
