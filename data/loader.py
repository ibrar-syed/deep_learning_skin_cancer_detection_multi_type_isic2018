# data/loader.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import Tuple, Dict, List


class ImageLoader:
    def __init__(self, root_dir: str, image_size: Tuple[int, int] = (100, 75), limit_per_class: int = 7500):
        """
        Initialize the image loader.

        Args:
            root_dir (str): Root directory containing subfolders of labeled images.
            image_size (Tuple[int, int]): Size to resize each image to (width, height).
            limit_per_class (int): Max number of images to retain per class.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.limit_per_class = limit_per_class
        self.max_workers = multiprocessing.cpu_count()
        self.label_map = self._create_label_map()

    def _create_label_map(self) -> Dict[int, str]:
        """Generate a mapping of integer labels to folder names (class labels)."""
        label_names = sorted(os.listdir(self.root_dir))
        return {idx: name for idx, name in enumerate(label_names)}

    def _fetch_image_paths(self) -> pd.DataFrame:
        """Create DataFrame of image paths and integer labels."""
        records = []
        for label_idx, class_name in self.label_map.items():
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                image_path = os.path.join(class_dir, fname)
                if os.path.isfile(image_path):
                    records.append({'image_path': image_path, 'label': label_idx})
        df = pd.DataFrame(records)
        return df

    def _resize_single_image(self, path: str) -> np.ndarray:
        """Open and resize a single image, returning its array."""
        try:
            with Image.open(path) as img:
                resized = img.resize(self.image_size)
                return np.asarray(resized)
        except Exception as e:
            print(f"[WARNING] Failed to process image {path}: {e}")
            return np.zeros((*self.image_size, 3), dtype=np.uint8)  # fallback blank image

    def _parallel_resize(self, image_paths: List[str]) -> List[np.ndarray]:
        """Parallel resize using multithreading."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            resized_images = list(executor.map(self._resize_single_image, image_paths))
        return resized_images

    def load_dataset(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Loads the dataset into a DataFrame with columns:
        - image_path
        - label (int)
        - image (np.ndarray)

        Returns:
            df (pd.DataFrame): Preprocessed image DataFrame.
            label_map (Dict[int, str]): Mapping of label index to class name.
        """
        print("[INFO] Collecting image paths...")
        df = self._fetch_image_paths()

        # Limit per class
        print(f"[INFO] Limiting to {self.limit_per_class} samples per class...")
        df = df.groupby("label").apply(lambda x: x.head(self.limit_per_class)).reset_index(drop=True)

        print("[INFO] Resizing images in parallel...")
        image_arrays = self._parallel_resize(df['image_path'].tolist())
        df['image'] = image_arrays

        print("[INFO] Dataset loading complete.")
        return df, self.label_map


# Optional test runner
if __name__ == "__main__":
    data_path = "/your/dataset/path"
    loader = ImageLoader(data_path)
    df, label_map = loader.load_dataset()

    print(df.head())
    print(label_map)
