###to predict a single imge using the pretrained weights
import os
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from config import Config

def predict_single(image_path, model_path, class_map):
    # Load model
    model = load_model(model_path)

    # Load and preprocess image
    img = load_img(image_path, target_size=Config.IMAGE_SHAPE[:2])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class_label = class_map.get(predicted_class_idx, "Unknown")

    print(f"[RESULT] Image: {image_path}")
    print(f"Predicted class index: {predicted_class_idx}")
    print(f"Predicted label: {predicted_class_label}")
    print(f"Class probabilities: {predictions[0]}")

    return predicted_class_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.h5).")
    parser.add_argument("--class_map", type=str, required=False, help="Optional dictionary string for class labels.")

    args = parser.parse_args()
    class_map = eval(args.class_map) if args.class_map else Config.CLASS_LABELS

    predict_single(args.image_path, args.model_path, class_map)
