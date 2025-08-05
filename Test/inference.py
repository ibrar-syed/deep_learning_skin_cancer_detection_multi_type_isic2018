#to test the data, the inferencing is done using the pre-trained weights!!!!!!!!!11
import os
import numpy as np
import pandas as pd
import argparse
import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from config import Config
from data.loader import load_and_resize_dataset
from data.augmentation import augment_dataset
from data.pipeline import (
    normalize_images,
    reshape_images,
    encode_labels,
    stratified_split
)


def evaluate_model(model_path: str, normalization: str = "z-score"):
    # Load model
    print(f"[INFO] Loading model from {model_path}")
    model = load_model(model_path)

    # Prepare dataset
    df, label_map = load_and_resize_dataset(Config.DATASET_PATH, target_size=Config.IMAGE_SHAPE[:2])
    augmented_df = augment_dataset(df, max_per_class=Config.MAX_SAMPLES_PER_CLASS)

    images = reshape_images(augmented_df["image"].values, Config.IMAGE_SHAPE)
    images = normalize_images(images, method=normalization)
    labels = encode_labels(augmented_df["label"].values, num_classes=Config.NUM_CLASSES)

    _, _, x_test, _, _, y_test = stratified_split(images, labels)

    # Predict
    y_probs = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
    except:
        auc = np.nan

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f} | AUC: {auc:.4f}")

    # Save metrics
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_df = pd.DataFrame([{
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "kappa_score": kappa,
        "roc_auc": auc,
        "timestamp": timestamp,
        "model": os.path.basename(model_path)
    }])
    metrics_path = os.path.join(Config.MODEL_SAVE_PATH, f"inference_metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_path = os.path.join(Config.MODEL_SAVE_PATH, f"confusion_matrix_{timestamp}.csv")
    cm_df.to_csv(cm_path, index=False)

    print(f"[INFO] Metrics saved to: {metrics_path}")
    print(f"[INFO] Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference using trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .h5 model file")
    parser.add_argument("--norm", type=str, default="z-score", help="Normalization method (z-score|min-max|mean-std)")
    args = parser.parse_args()

    evaluate_model(args.model_path, normalization=args.norm)
