import os
import sys
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model

# Add current directory to path
sys.path.append(os.getcwd())

def build_model():
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = BatchNormalization(name="batch_normalization")(x)
    x = Dense(256, activation="relu", name="dense")(x)
    x = BatchNormalization(name="batch_normalization_1")(x)
    x = Dropout(0.5, name="dropout")(x)
    x = Dense(128, activation="relu", name="dense_1")(x)
    x = Dropout(0.5, name="dropout_1")(x)
    outputs = Dense(1, activation="sigmoid", name="dense_2")(x)
    return Model(inputs=base_model.input, outputs=outputs)

def manual_load_weights(model, weights_path):
    with h5py.File(weights_path, 'r') as f:
        layers_group = f['layers']
        vars_group = f['vars'] # Might be needed if layers only has groups
        
        success_count = 0
        fail_count = 0
        
        for layer in model.layers:
            if layer.name in layers_group:
                layer_data = layers_group[layer.name]
                if 'vars' in layer_data:
                    var_group = layer_data['vars']
                    # Sort keys numerically to ensure [vars/0, vars/1] -> [kernel, bias]
                    keys = sorted(var_group.keys(), key=lambda x: int(x))
                    weights = [var_group[k][()] for k in keys]
                    
                    if weights:
                        try:
                            layer.set_weights(weights)
                            success_count += 1
                        except Exception as e:
                            print(f"FAILED to set weights for {layer.name}: {e}")
                            fail_count += 1
                else:
                    # Some layers might not have vars (like dropout)
                    pass
            else:
                # Base model layers are often prefixed or under a group
                pass
                
    print(f"Manual Load Summary: {success_count} layers loaded, {fail_count} failed.")

model = build_model()
weights_path = "weights.weights.h5"
if os.path.exists(weights_path):
    manual_load_weights(model, weights_path)
    
    # Test a prediction to see if it's still near 0.5
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    pred = model.predict(dummy_input, verbose=0)
    print(f"Dummy prediction: {pred[0][0]}")
else:
    print("Weights file not found")
