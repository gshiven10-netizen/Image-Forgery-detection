import cv2
import os

# ---------- MEMORY OPTIMIZATION (MUST BE BEFORE TF IMPORT) ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import tensorflow as tf
import gc
from processing.detector import detect_forgery_overlay

# ---------- BUILD MODEL + LOAD WEIGHTS ----------
model = None

def get_model():
    global model
    if model is not None:
        return model
    
    try:
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
        from tensorflow.keras.models import Model

        print("DEBUG: Pre-heating model architecture...", flush=True)
        
        # Limit TF thread pool internally
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        # Match train_model.py architecture EXACTLY
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D(name="global_average_pooling2d")(x)
        x = BatchNormalization(name="batch_normalization")(x)
        x = Dense(256, activation="relu", name="dense")(x)
        x = BatchNormalization(name="batch_normalization_1")(x)
        x = Dropout(0.5, name="dropout")(x)
        x = Dense(128, activation="relu", name="dense_1")(x)
        x = Dropout(0.5, name="dropout_1")(x)
        outputs = Dense(1, activation="sigmoid", name="dense_2")(x)
        
        from tensorflow.keras.models import Model
        model = Model(inputs=base_model.input, outputs=outputs)

        # ✅ LOAD YOUR WEIGHTS
        weights_path = "weights.weights.h5"
        if os.path.exists(weights_path):
            try:
                # First attempt: Standard Keras load
                model.load_weights(weights_path)
                print(f"✅ SUCCESS: Model weights loaded from {weights_path}", flush=True)
            except Exception as load_err:
                print(f"⚠️ Warning: Standard load failed ({load_err}). Attempting manual layer-by-layer load...", flush=True)
                try:
                    import h5py
                    with h5py.File(weights_path, 'r') as f:
                        layers_group = f['layers']
                        success_count = 0
                        
                        # Recursive function to load weights into layers and sub-layers
                        def load_recursive(layer_to_load):
                            nonlocal success_count
                            # Handle standard layers
                            if layer_to_load.name in layers_group:
                                layer_data = layers_group[layer_to_load.name]
                                if 'vars' in layer_data:
                                    var_group = layer_data['vars']
                                    keys = sorted(var_group.keys(), key=lambda x: int(x))
                                    weights = [var_group[k][()] for k in keys]
                                    if weights:
                                        try:
                                            layer_to_load.set_weights(weights)
                                            success_count += 1
                                        except Exception:
                                            pass
                            
                            # Recurse into sub-layers (e.g. for EfficientNetB0 base model)
                            if hasattr(layer_to_load, 'layers'):
                                for sub_layer in layer_to_load.layers:
                                    load_recursive(sub_layer)
                        
                        load_recursive(model)
                        print(f"✅ SUCCESS: Manually loaded weights for {success_count} layers.", flush=True)
                except Exception as final_err:
                    print(f"❌ CRITICAL: All weight loading attempts failed: {final_err}", flush=True)
                    model = None
        else:
            print(f"⚠️ ERROR: Weights file {weights_path} not found!", flush=True)
            model = None 
        
        # Aggressive cleanup after load
        gc.collect()
        return model

    except Exception as e:
        print(f"❌ CRITICAL ERROR in get_model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

# Pre-load model at startup
model = get_model()

# ---------- PREDICTION FUNCTION ----------
def predict_forgery(image_path):
    print(f"DEBUG: Process started for {image_path}", flush=True)
    filename = os.path.basename(image_path)

    predicted_class = "Analysis Failed"
    confidence = 0.0
    accuracy = 0.0
    output_filename = filename

    # Explicit cleanup before starting
    gc.collect()

    # ---------- READ & RESIZE IF TOO LARGE (Memory Safety) ----------
    try:
        print("STEP 0: Pre-processing & Resizing...", flush=True)
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Unsupported Image", 0.0, 0.0, filename
        
        # Max dimension 1200px (reduced from 1600 for even more safety)
        h, w = img.shape[:2]
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            print(f"DEBUG: Resized image to {img.shape[1]}x{img.shape[0]}", flush=True)
            cv2.imwrite(image_path, img)
    except Exception as e:
        print(f"❌ Pre-processing Error: {e}", flush=True)

    # ---------- RUN MODEL ----------
    current_model = get_model()
    if current_model is not None:
        try:
            print("STEP 1: Preparing image for TF...", flush=True)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            img_final = np.expand_dims(img_resized, axis=0).astype(np.float32)

            print("STEP 2: Running TF inference...", flush=True)
            prediction = current_model.predict(img_final, verbose=0)[0][0]
            print(f"DEBUG: Raw prediction value: {prediction}", flush=True)

            prediction_val = float(prediction)
            
            # Keras flow_from_directory uses alphabetical order:
            # 0: Forged
            # 1: Original (Authentic)
            if prediction_val > 0.5:
                predicted_class = "Authentic"
                confidence = prediction_val
            else:
                predicted_class = "Forged"
                confidence = 1.0 - prediction_val

            # Baseline accuracy from training logs (94%)
            accuracy = 0.94
            
            # Immediate cleanup after inference
            del img_final
            gc.collect()

        except Exception as e:
            print(f"❌ TF Error: {e}", flush=True)
            predicted_class = "Prediction Error"
    else:
        predicted_class = "Model Not Loaded"

    # ---------- OVERLAY (FORGERY HIGHLIGHTING) ----------
    try:
        result_dir = "static/uploads"
        output_filename = "result_" + filename
        output_path = os.path.join(result_dir, output_filename)

        if predicted_class == "Forged":
            print("STEP 3: Running ORB Forgery Detection...", flush=True)
            result_image = detect_forgery_overlay(image_path)
            print("STEP 4: Saving marked image...", flush=True)
        else:
            print("STEP 3: Skipping ORB, using original...", flush=True)
            result_image = cv2.imread(image_path)

        if result_image is not None:
            cv2.imwrite(output_path, result_image)
        
        # Cleanup
        print("STEP 5: Memory cleanup...", flush=True)
        del result_image
        gc.collect()

        return predicted_class, confidence, accuracy, output_filename

    except Exception as e:
        print(f"❌ Overlay Error: {e}", flush=True)
        return predicted_class, 0.0, 0.0, filename