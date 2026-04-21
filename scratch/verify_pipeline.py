import os
import sys
import cv2

# Add current directory to path
sys.path.append(os.getcwd())

from processing.predictor import predict_forgery

test_image = "Dataset/Original/10.jpg"
if not os.path.exists(test_image):
    # Try another one if 10.jpg isn't there
    dataset_org = "Dataset/Original"
    if os.path.exists(dataset_org):
        files = [f for f in os.listdir(dataset_org) if f.endswith(('.jpg', '.png'))]
        if files:
            test_image = os.path.join(dataset_org, files[0])
            
if not os.path.exists(test_image):
    print(f"ERROR: No test image found in Dataset/Original")
    sys.exit(1)

try:
    print(f"Running prediction on {test_image}...")
    predicted_class, confidence, accuracy, result_image = predict_forgery(test_image)
    print(f"\nResult:")
    print(f"  Class: {predicted_class}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Result Image: {result_image}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
