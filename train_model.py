import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATASET_DIR = "Dataset"
TRAIN_IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
MODEL_WEIGHTS_PATH = "weights.weights.h5"
CONFUSION_MATRIX_PATH = os.path.join("static", "uploads", "confusion_matrix.png")

os.makedirs(os.path.join("static", "uploads"), exist_ok=True)

train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(TRAIN_IMG_SIZE, TRAIN_IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(TRAIN_IMG_SIZE, TRAIN_IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
    ModelCheckpoint(
        MODEL_WEIGHTS_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]

print("🔥 Training started...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Training finished")

# Load best saved weights
if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_weights(MODEL_WEIGHTS_PATH)
    print(f"✅ Best weights loaded from {MODEL_WEIGHTS_PATH}")

# Predictions
val_generator.reset()
pred_probs = model.predict(val_generator, verbose=1)
y_pred = (pred_probs > 0.5).astype(int).reshape(-1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

# Confusion matrix plot
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

class_names = list(val_generator.class_indices.keys())
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.0 if cm.max() > 0 else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            ha="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close()

print(f"✅ Confusion matrix saved to {CONFUSION_MATRIX_PATH}")