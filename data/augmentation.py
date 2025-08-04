# data/augmentation.py

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple


def get_augmentation_pipeline() -> ImageDataGenerator:
    """
    Returns a configured Keras ImageDataGenerator with advanced transformations.
    """
    return ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def balance_classes(df: pd.DataFrame, max_per_class: int, seed: int = 42) -> pd.DataFrame:
    """
    Limits each class in the DataFrame to a maximum number of samples.
    """
    balanced_df = (
        df.groupby('label')
        .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=seed))
        .reset_index(drop=True)
    )
    return balanced_df


def augment_dataframe(df: pd.DataFrame, max_per_class: int, image_key='image') -> pd.DataFrame:
    """
    Performs class-wise augmentation to expand minority classes up to max_per_class.
    Returns a new DataFrame with augmented images and balanced classes.
    """
    augmenter = get_augmentation_pipeline()
    augmented_entries = []

    class_counts = df['label'].value_counts().to_dict()

    for label in df['label'].unique():
        class_subset = df[df['label'] == label]
        current_count = len(class_subset)

        if current_count >= max_per_class:
            continue  # Already enough data

        needed = max_per_class - current_count
        images_to_augment = class_subset.sample(min(needed, current_count), replace=True, random_state=42)

        print(f"[INFO] Augmenting class {label}: +{needed} samples")

        for idx in range(needed):
            original_image = images_to_augment.iloc[idx % len(images_to_augment)][image_key]
            image_tensor = np.expand_dims(original_image, axis=0)
            aug_iter = augmenter.flow(image_tensor, batch_size=1)

            aug_img = next(aug_iter)[0].astype(np.uint8)
            augmented_entries.append({'image_path': None, 'label': label, image_key: aug_img})

    augmented_df = pd.DataFrame(augmented_entries)
    final_df = pd.concat([df, augmented_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    return final_df


def summarize_augmented_data(df: pd.DataFrame):
    """
    Prints class distribution after augmentation.
    """
    print("\n[INFO] Final Dataset Summary:")
    class_counts = df['label'].value_counts().sort_index()
    print("-" * 50)
    print(f"{'Label':<10} {'Samples':<10}")
    print("-" * 50)
    for label, count in class_counts.items():
        print(f"{label:<10} {count:<10}")
    print("-" * 50)
    print(f"{'Total':<10} {class_counts.sum():<10}")
