import cv2
import os
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
        # Base model
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

        # Custom head (must match training)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=outputs)

        # ✅ LOAD YOUR WEIGHTS
        weights_path = "weights.weights.h5"
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"✅ SUCCESS: Model weights loaded from {weights_path}", flush=True)
        else:
            print(f"⚠️ ERROR: Weights file {weights_path} not found!", flush=True)
            model = None 
        
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

    # ---------- READ & RESIZE IF TOO LARGE (Memory Safety) ----------
    try:
        print("STEP 0: Pre-processing & Resizing...", flush=True)
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Unsupported Image", 0.0, 0.0, filename
        
        # Max dimension 1600px to prevent OOM
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            scale = 1600 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            print(f"DEBUG: Resized image to {img.shape[1]}x{img.shape[0]}", flush=True)
            # Save the resized version back to the path so ORB uses it too
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
            
            # NOTE: Division by 255 removed because training used [0, 255] range
            img_final = np.expand_dims(img_resized, axis=0).astype(np.float32)

            print("STEP 2: Running TF inference...", flush=True)
            prediction = current_model.predict(img_final, verbose=0)[0][0]
            print(f"DEBUG: Raw prediction value: {prediction}", flush=True)

            if prediction < 0.5:
                predicted_class = "Authentic"
                confidence = float(prediction)
            else:
                predicted_class = "Forged"
                confidence = float(prediction)
            
            if predicted_class == "Forged":
                display_confidence = confidence * 100
            else:
                display_confidence = (1 - confidence) * 100
                
            accuracy = display_confidence / 100.0

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
            # detect_forgery_overlay can be slow, hence the Procfile timeout boost
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

        return predicted_class, accuracy, accuracy, output_filename

    except Exception as e:
        print(f"❌ Overlay Error: {e}", flush=True)
        return predicted_class, 0.0, 0.0, filename