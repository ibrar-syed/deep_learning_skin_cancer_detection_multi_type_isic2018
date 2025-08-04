# preprocessing/pipeline.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Dict


def normalize_images(images: np.ndarray, method: str = "z-score") -> np.ndarray:
    """
    Normalize images using the specified method.

    Args:
        images (np.ndarray): Raw image array.
        method (str): Normalization method. Options: "z-score", "min-max", "mean-std"

    Returns:
        np.ndarray: Normalized image array.
    """
    if method == "z-score":
        mean = np.mean(images)
        std = np.std(images)
        return (images - mean) / std
    elif method == "min-max":
        return images / 255.0
    elif method == "mean-std":
        mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
        std = np.std(images, axis=(0, 1, 2), keepdims=True)
        return (images - mean) / std
    else:
        raise ValueError(f"[ERROR] Unknown normalization method: {method}")


def reshape_images(images: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Reshape a 4D tensor of images to match (N, H, W, C).

    Args:
        images (np.ndarray): Raw images.
        target_shape (Tuple[int, int, int]): Desired (H, W, C)

    Returns:
        np.ndarray: Reshaped image tensor.
    """
    return images.reshape((-1, *target_shape))


def encode_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.

    Args:
        labels (np.ndarray): Array of integer class labels.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot encoded labels.
    """
    return to_categorical(labels, num_classes=num_classes)


def stratified_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into stratified train, validation, and test sets.

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test
    """
    print("[INFO] Performing stratified train/val/test split...")
    flat_labels = np.argmax(labels, axis=1)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_idx, test_idx in sss1.split(features, flat_labels):
        x_train, x_temp = features[train_idx], features[test_idx]
        y_train, y_temp = labels[train_idx], labels[test_idx]
        flat_temp = flat_labels[test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    for val_idx, test_idx in sss2.split(x_temp, flat_temp):
        x_val, x_test = x_temp[val_idx], x_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]

    return x_train, x_val, x_test, y_train, y_val, y_test


def summarize_splits(y_train, y_val, y_test, label_map: Dict[int, str]):
    """Print class-wise distribution across splits."""
    print("\n📊 Dataset Summary by Split")
    print("-" * 80)
    print(f"{'Class ID':<10} {'Class Name':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 80)

    for class_id, class_name in label_map.items():
        train_count = int(np.sum(np.argmax(y_train, axis=1) == class_id))
        val_count = int(np.sum(np.argmax(y_val, axis=1) == class_id))
        test_count = int(np.sum(np.argmax(y_test, axis=1) == class_id))
        total = train_count + val_count + test_count

        print(f"{class_id:<10} {class_name:<25} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")

    print("-" * 80)
    print(f"{'Total':<35} {len(y_train):<10} {len(y_val):<10} {len(y_test):<10} {len(y_train) + len(y_val) + len(y_test):<10}\n")
