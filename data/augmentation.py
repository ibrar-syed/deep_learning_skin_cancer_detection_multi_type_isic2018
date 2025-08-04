# data/augmentation.py

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Optional


class Augmentor:
    def __init__(
        self,
        rotation_range: int = 30,
        width_shift_range: float = 0.1,
        height_shift_range: float = 0.1,
        shear_range: float = 0.1,
        zoom_range: float = 0.15,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        fill_mode: str = 'nearest',
        max_samples_per_class: Optional[int] = 7500,
        seed: int = 42
    ):
        """
        Initialize the augmentation pipeline with custom transformations.
        """
        self.generator = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            brightness_range=brightness_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            fill_mode=fill_mode
        )
        self.max_samples_per_class = max_samples_per_class
        self.seed = seed

    def augment_class(self, class_images: np.ndarray, label: int, augment_count: int) -> pd.DataFrame:
        """
        Augment a specific class to reach a desired sample count.

        Args:
            class_images (np.ndarray): Array of images belonging to one class.
            label (int): The class label for the images.
            augment_count (int): Number of augmented samples to generate.

        Returns:
            pd.DataFrame: DataFrame with columns 'image', 'label'
        """
        rows = []
        image_stream = self.generator.flow(class_images, batch_size=1, seed=self.seed)
        for _ in range(augment_count):
            augmented_image = image_stream.next()[0].astype('uint8')
            rows.append({'image': augmented_image, 'label': label})
        return pd.DataFrame(rows)

    def balance_dataset(self, df: pd.DataFrame, label_column: str = 'label', image_column: str = 'image') -> pd.DataFrame:
        """
        Perform class balancing through augmentation.

        Args:
            df (pd.DataFrame): Input DataFrame with 'image' and 'label' columns.
            label_column (str): Column name for labels.
            image_column (str): Column name for image arrays.

        Returns:
            pd.DataFrame: Balanced DataFrame with augmented samples.
        """
        print("[INFO] Starting class balancing via augmentation...")
        augmented_records = []

        for label in sorted(df[label_column].unique()):
            class_subset = df[df[label_column] == label]
            num_existing = len(class_subset)
            if self.max_samples_per_class is None or num_existing >= self.max_samples_per_class:
                print(f"[INFO] Class {label} already has {num_existing} samples. Skipping augmentation.")
                continue

            images = np.stack(class_subset[image_column].values)
            augment_needed = self.max_samples_per_class - num_existing
            print(f"[INFO] Augmenting class {label} with {augment_needed} new samples...")

            aug_df = self.augment_class(images, label, augment_needed)
            augmented_records.append(aug_df)

        if augmented_records:
            augmented_df = pd.concat(augmented_records, ignore_index=True)
            balanced_df = pd.concat([df, augmented_df], ignore_index=True)
        else:
            balanced_df = df.copy()

        balanced_df = balanced_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        print(f"[INFO] Augmentation complete. Total samples after augmentation: {len(balanced_df)}")
        return balanced_df
