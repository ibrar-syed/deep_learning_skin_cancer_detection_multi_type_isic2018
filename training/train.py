# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Skin Cancer Detection Project
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

#import os
import os
import argparse
import datetime
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from config import Config

from data.loader import load_and_resize_dataset
from data.augmentation import augment_dataset
from data.pipeline import (
    normalize_images,
    reshape_images,
    encode_labels,
    stratified_split,
    summarize_splits
)

# === Import all model builders ===
from models.densenet201 import build_densenet201
from models.densenet169 import build_densenet169
from models.densenet121 import build_densenet121
from models.nasnetmobile import build_nasnetmobile
from models.mobilenetv2 import build_mobilenetv2
from models.mobilenetv3 import build_mobilenetv3
from models.efficientnetv2_b3 import build_efficientnetv2_b3
from models.efficientnetv2_b7 import build_efficientnetv2_b7
from models.xceptionnet import build_xceptionnet
from models.inceptionv3 import build_inceptionv3
from models.coatnet import build_coatnet

# === Model selector registry ===
MODEL_REGISTRY = {
    "densenet201": build_densenet201,
    "densenet169": build_densenet169,
    "densenet121": build_densenet121,
    "nasnetmobile": build_nasnetmobile,
    "mobilenetv2": build_mobilenetv2,
    "mobilenetv3": build_mobilenetv3,
    "efficientnetv2_b3": build_efficientnetv2_b3,
    "efficientnetv2_b7": build_efficientnetv2_b7,
    "xceptionnet": build_xceptionnet,
    "inceptionv3": build_inceptionv3,
    "coatnet": build_coatnet,
}


def train_model(model_name: str, normalization: str = "z-score", freeze_base: bool = True):
    print(f"\n Training Model: {model_name}")
    print(f" Normalization Method: {normalization}")
    print(f"  Freeze Base Layers: {'Yes' if freeze_base else 'No'}")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[ERROR] Invalid model '{model_name}' specified.")

    # === Step 1: Load + Resize Images ===
    df, label_map = load_and_resize_dataset(Config.DATASET_PATH, target_size=Config.IMAGE_SHAPE[:2])

    # === Step 2: Augment Dataset ===
    augmented_df = augment_dataset(df, max_per_class=Config.MAX_SAMPLES_PER_CLASS)

    # === Step 3: Preprocess for Training ===
    images = augmented_df["image"].values
    labels = augmented_df["label"].values

    images = reshape_images(images, Config.IMAGE_SHAPE)
    images = normalize_images(images, method=normalization)
    labels = encode_labels(labels, num_classes=Config.NUM_CLASSES)

    x_train, x_val, x_test, y_train, y_val, y_test = stratified_split(
        images, labels,
        test_size=0.2,
        val_size=0.1
    )

    summarize_splits(y_train, y_val, y_test, label_map)

    # === Step 4: Build Model ===
    model_builder = MODEL_REGISTRY[model_name]
    model, lr_scheduler = model_builder(
        input_shape=Config.IMAGE_SHAPE,
        num_classes=Config.NUM_CLASSES,
        freeze_base=freeze_base,
        learning_rate=Config.LEARNING_RATE
    )

    # === Step 5: Callbacks Setup ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.h5"
    csv_log_filename = f"{model_name}_{timestamp}_log.csv"

    model_save_path = os.path.join(Config.MODEL_SAVE_PATH, model_filename)
    csv_save_path = os.path.join(Config.MODEL_SAVE_PATH, csv_log_filename)

    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    csv_logger = CSVLogger(csv_save_path, append=False)

    # === Step 6: Training ===
    print("\n Starting Training...\n")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[checkpoint_cb, csv_logger, lr_scheduler],
        verbose=1
    )

    # === Step 7: Final Evaluation ===
    print("\n Evaluating on Test Data...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\n Final Test Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")
    print(f" Best Model Saved at: {model_save_path}")
    print(f" Training Log CSV: {csv_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--model", required=True, type=str, help="Model name from MODEL_REGISTRY")
    parser.add_argument("--norm", default="z-score", type=str, help="Normalization method (z-score|min-max|mean-std)")
    parser.add_argument("--freeze", action="store_true", help="Freeze base layers of pretrained model")

    args = parser.parse_args()
    train_model(model_name=args.model, normalization=args.norm, freeze_base=args.freeze)
