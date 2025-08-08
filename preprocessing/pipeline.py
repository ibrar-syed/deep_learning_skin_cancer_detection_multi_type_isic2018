# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Skin Cancer Detection Project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# preprocessing/pipeline.py,...

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from typing import Tuple

from data.loader import load_and_resize_images
from data.augmentation import augment_dataframe, summarize_augmented_data
from config import Config


def normalize_images(images: np.ndarray, method: str = "z-score") -> np.ndarray:
    """
    Normalize images using one of several standard methods.
    
    Args:
        images (np.ndarray): Image data array.
        method (str): One of ['z-score', 'min-max', 'mean-std'].
    
    Returns:
        np.ndarray: Normalized image data.
    """
    if method == "z-score":
        mean = np.mean(images)
        std = np.std(images)
        return (images - mean) / (std + 1e-7)
    elif method == "min-max":
        return images / 255.0
    elif method == "mean-std":
        mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
        std = np.std(images, axis=(0, 1, 2), keepdims=True)
        return (images - mean) / (std + 1e-7)
    else:
        raise ValueError(f"[ERROR] Unknown normalization method: {method}")


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels into one-hot encoded vectors.
    
    Args:
        labels (np.ndarray): Integer labels.
        num_classes (int): Total number of classes.
    
    Returns:
        np.ndarray: One-hot encoded labels.
    """
    return to_categorical(labels, num_classes=num_classes)


def stratified_split(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Splits X and y into stratified train/val/test sets (60/20/20).
    
    Args:
        X (np.ndarray): Feature images.
        y (np.ndarray): One-hot encoded labels.
    
    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    y_labels = np.argmax(y, axis=1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y_labels, random_state=seed)
    y_temp_labels = np.argmax(y_temp, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp_labels, random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test


def summarize_splits(y_train, y_val, y_test, label_map):
    print("\n📊 Dataset Summary by Split")
    print("-" * 80)
    print(f"{'Class ID':<10} {'Label Name':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 80)

    for class_id, class_name in label_map.items():
        train_count = int(np.sum(np.argmax(y_train, axis=1) == class_id))
        val_count = int(np.sum(np.argmax(y_val, axis=1) == class_id))
        test_count = int(np.sum(np.argmax(y_test, axis=1) == class_id))
        total = train_count + val_count + test_count

        print(f"{class_id:<10} {class_name:<25} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")

    print("-" * 80)
    print(f"{'Total':<35} {len(y_train):<10} {len(y_val):<10} {len(y_test):<10} {len(y_train) + len(y_val) + len(y_test):<10}\n")


def run_preprocessing(dataset_dir: str, max_per_class: int = 3000, norm_method: str = "z-score"):
    """
    Executes the entire preprocessing pipeline:
    - Loads image data
    - Applies augmentations
    - Normalizes images
    - Splits into train/val/test
    - One-hot encodes labels
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load & resize raw images
    df, label_map = load_and_resize_images(dataset_dir)

    # Augment underrepresented classes
    df = augment_dataframe(df, max_per_class=max_per_class)
    summarize_augmented_data(df)

    # Extract features and labels
    X = np.array(df['image'].tolist())
    y = np.array(df['label'].tolist())

    # Normalize images
    X = normalize_images(X, method=norm_method)

    # One-hot encode labels
    y = one_hot_encode(y, num_classes=Config.NUM_CLASSES)

    # Stratified split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    summarize_splits(y_train, y_val, y_test, label_map)

    return X_train, X_val, X_test, y_train, y_val, y_test
