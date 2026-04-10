import cv2
import os
import numpy as np
import tensorflow as tf
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

        print("DEBUG: Reconstructing model architecture...")
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
            print(f"✅ Model weights loaded successfully from {weights_path}")
        else:
            print(f"⚠️ Warning: Weights file {weights_path} not found!")
            model = None # Force it to None so we don't try to predict with random weights
        
        return model

    except Exception as e:
        print(f"❌ Critical: Model reconstruction/load failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Pre-load model at startup
model = get_model()

# ---------- PREDICTION FUNCTION ----------
def predict_forgery(image_path):
    print(f"DEBUG: Processing image: {image_path}")
    filename = os.path.basename(image_path)

    predicted_class = "Analysis Failed"
    confidence = 0.0
    accuracy = 0.0
    output_filename = filename # Default to original if processing fails

    # ---------- RUN MODEL ----------
    current_model = get_model()
    if current_model is not None:
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Error: Could not read image at {image_path}")
                return "Error: Unsupported Image", 0.0, 0.0, filename

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_final = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize if used during training

            print("DEBUG: Running model prediction...")
            prediction = current_model.predict(img_final, verbose=0)[0][0]
            print(f"DEBUG: Prediction raw value: {prediction}")

            if prediction < 0.5:
                # Assuming 0 is Forged and 1 is Authentic based on typical sigmoid outputs
                # But looking at previous code: if prediction > 0.5: Authentic
                # Let's keep consistency with previous logic but make it clearer
                predicted_class = "Authentic"
                confidence = float(prediction)
            else:
                predicted_class = "Forged"
                confidence = float(prediction) # Usually confidence is the probability of the class
            
            # Recalculate confidence for better display
            if predicted_class == "Forged":
                display_confidence = confidence * 100
            else:
                display_confidence = (1 - confidence) * 100
                
            accuracy = display_confidence / 100.0

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            predicted_class = "Prediction Error"
    else:
        print("❌ Model is None, skipping prediction logic")
        predicted_class = "Model Not Loaded"

    # ---------- OVERLAY (FORGERY HIGHLIGHTING) ----------
    try:
        result_dir = "static/uploads"
        os.makedirs(result_dir, exist_ok=True)
        output_filename = "result_" + filename
        output_path = os.path.join(result_dir, output_filename)

        if predicted_class == "Forged":
            print("DEBUG: Generating forgery overlay...")
            result_image = detect_forgery_overlay(image_path)
        else:
            result_image = cv2.imread(image_path)
            if result_image is None:
                print(f"❌ Error: Could not read image for result: {image_path}")
                return "Error: Image Load Failure", 0.0, 0.0, filename

        # Ensure result_image is valid before saving
        if result_image is not None:
            success = cv2.imwrite(output_path, result_image)
            if not success:
                print(f"❌ Error: Failed to write image to {output_path}")
                output_filename = filename # Fallback to original
        else:
            output_filename = filename

        return predicted_class, accuracy, accuracy, output_filename

    except Exception as e:
        print(f"❌ Post-processing error: {e}")
        import traceback
        traceback.print_exc()
        return predicted_class, 0.0, 0.0, filename