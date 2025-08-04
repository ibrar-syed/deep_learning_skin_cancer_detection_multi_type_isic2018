# data/loader.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import concurrent.futures
import multiprocessing
from typing import Tuple, Dict, List


def get_class_mapping(dataset_root: str) -> Dict[int, str]:
    """
    Maps numeric class indices to folder names (class labels).
    """
    class_names = sorted(os.listdir(dataset_root))
    return {idx: name for idx, name in enumerate(class_names)}


def list_images_with_labels(dataset_root: str) -> pd.DataFrame:
    """
    Generates a dataframe with image paths and numeric labels.
    """
    image_entries = []
    class_map = get_class_mapping(dataset_root)

    for label, folder in class_map.items():
        folder_path = os.path.join(dataset_root, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pgm')):
                image_path = os.path.join(folder_path, filename)
                image_entries.append({"image_path": image_path, "label": label})

    df = pd.DataFrame(image_entries)
    return df


def resize_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Loads and resizes an image to the specified target size.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize(target_size)
        return np.array(image_resized)
    except Exception as e:
        print(f"[WARN] Failed to load image {image_path}: {e}")
        return np.zeros((*target_size, 3), dtype=np.uint8)  # fallback


def parallel_image_loading(df: pd.DataFrame, num_workers: int = None) -> pd.DataFrame:
    """
    Loads and resizes all images using multithreading.
    """
    num_workers = num_workers or multiprocessing.cpu_count()
    image_paths = df["image_path"].tolist()

    print(f"[INFO] Resizing {len(image_paths)} images using {num_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        resized_images = list(executor.map(resize_image, image_paths))

    df["image"] = resized_images
    return df


def summarize_dataset(df: pd.DataFrame, label_map: Dict[int, str]):
    """
    Prints class-wise image count summary.
    """
    counts = df["label"].value_counts().sort_index()
    print("Dataset Summary")
    print("-" * 60)
    print(f"{'Class ID':<12} {'Label Name':<30} {'Count':<10}")
    print("-" * 60)
    for class_id, label_name in label_map.items():
        count = counts.get(class_id, 0)
        print(f"{class_id:<12} {label_name:<30} {count:<10}")
    print("-" * 60)
    print(f"{'Total':<44} {counts.sum():<10}")


def load_and_resize_images(dataset_path: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Full loader pipeline: list images, resize, and return dataframe with label map.
    """
    label_map = get_class_mapping(dataset_path)
    image_df = list_images_with_labels(dataset_path)
    image_df = parallel_image_loading(image_df)
    summarize_dataset(image_df, label_map)
    return image_df, label_map
