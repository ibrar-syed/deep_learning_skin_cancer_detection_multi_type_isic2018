## Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Skin_Cancer Detection Project.
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









# config.py.....


import os

class Config:
    """
    Global configuration class for all settings and constants
    used across the training, inference, and data processing pipelines.
    """

    # Dataset Paths
    BASE_DATA_DIR = "data/"
    RAW_DATA_PATH = os.path.join(BASE_DATA_DIR, "ISIC")
    IMAGE_SHAPE = (224, 224, 3)

    # Label Mapping (example for ISIC dataset with 7 classes)
    CLASS_LABELS = {
      LABEL_MAP = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Nevus",
    4: "Actinic Keratosis",
    5: "Seborrheic Keratosis",
    6: "Dermatofibroma",
    7: "Vascular Lesion",
    8: "Other"
}

    }

    NUM_CLASSES = len(CLASS_LABELS)

    # Training Parameters
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.4
    L2_REG = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42

    # Image Normalization Method: ["z-score", "min-max", "mean-std"]
    NORMALIZATION = "z-score"

    # Output Paths
    MODEL_SAVE_PATH = "saved_models/"
    METRICS_SAVE_PATH = "metrics/"
    CONFUSION_MATRIX_DIR = os.path.join(METRICS_SAVE_PATH, "confusion_matrices/")
    LOG_FILE = os.path.join(METRICS_SAVE_PATH, "training_logs.csv")

    # Model registry mapping (dynamically used in train.py)
    MODEL_REGISTRY = {
        "densenet201": "models.densenet201",
        "densenet169": "models.densenet169",
        "mobilenetv2": "models.mobilenetv2",
        "mobilenetv3large": "models.mobilenetv3large",
        "efficientnet_b7": "models.efficientnet_b7",
        "efficientnetv2_b3": "models.efficientnetv2_b3",
        "efficientnetv2_b7": "models.efficientnetv2_b7",
        "xception": "models.xception",
        "inceptionv3": "models.inceptionv3",
        "coatnet": "models.coatnet",
    }

    @staticmethod
    def ensure_directories():
        """Create required directories if not already present."""
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.METRICS_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.CONFUSION_MATRIX_DIR, exist_ok=True)

# Initialize directories at runtime
Config.ensure_directories()
