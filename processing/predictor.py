import cv2
import os
import numpy as np
from processing.detector import detect_forgery_overlay

# ---------- BUILD MODEL + LOAD WEIGHTS ----------
model = None

try:
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
    from tensorflow.keras.models import Model

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

    # ✅ LOAD YOUR WEIGHTS (THIS FIXES EVERYTHING)
    model.load_weights("weights.weights.h5")

    print("✅ Model loaded successfully using weights")

except Exception as e:
    print("❌ Model load failed:", e)


# ---------- PREDICTION FUNCTION ----------
def predict_forgery(image_path):
    filename = os.path.basename(image_path)

    predicted_class = "Model Not Loaded"
    confidence = 0.0
    accuracy = 0.0

    # ---------- RUN MODEL ----------
    if model is not None:
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0][0]

            if prediction > 0.5:
                predicted_class = "Authentic (Real/Original)"
                confidence = float(prediction)
            else:
                predicted_class = "Forged"
                confidence = float(1 - prediction)

            accuracy = confidence

        except Exception as e:
            print("❌ Prediction error:", e)

    # ---------- OVERLAY (NO BLUR) ----------
    if predicted_class == "Forged":
        result_image = detect_forgery_overlay(image_path)
    else:
        result_image = cv2.imread(image_path)

    output_filename = "result_" + filename
    output_path = os.path.join("static/uploads", output_filename)

    cv2.imwrite(output_path, result_image)

    return predicted_class, confidence, accuracy, output_filename