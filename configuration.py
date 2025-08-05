# config.py......

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
        0: "melanoma",
        1: "nevus",
        2: "bcc",
        3: "akiec",
        4: "vasc",
        5: "df",
        6: "bkl"
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
