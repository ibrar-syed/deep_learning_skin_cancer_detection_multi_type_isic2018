# training/train.py

import os
import argparse
from datetime import datetime

from config import Config
from preprocessing.pipeline import run_preprocessing

# Model imports (modular, extendable)
from models.densenet201 import build_densenet201
from models.densenet121 import build_densenet121  # placeholder for future file
from models.nasnetmobile import build_nasnetmobile  # placeholder for future file

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Register available models here
MODEL_REGISTRY = {
    "densenet201": build_densenet201,
    "densenet121": build_densenet121,
    "nasnetmobile": build_nasnetmobile
}


def train_model(
    model_name: str,
    epochs: int,
    batch_size: int,
    freeze_base: bool = True,
    normalization: str = "z-score"
):
    """
    Train a deep learning model on the ISIC dataset.

    Args:
        model_name (str): Model identifier in MODEL_REGISTRY
        epochs (int): Training epochs
        batch_size (int): Mini-batch size
        freeze_base (bool): Freeze base model weights or not
        normalization (str): Normalization method to apply
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[ERROR] Unsupported model: '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    print(f"\n Starting training for model: {model_name}")
    print(f" Normalization: {normalization} | Freeze base: {freeze_base}")
    print(" Loading and preprocessing data...")

    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing(
        dataset_dir=Config.DATA_ROOT,
        max_per_class=Config.MAX_PER_CLASS,
        norm_method=normalization
    )

    # Load model and learning rate scheduler
    build_fn = MODEL_REGISTRY[model_name]
    model, lr_scheduler = build_fn(
        input_shape=Config.IMAGE_SHAPE,
        num_classes=Config.NUM_CLASSES,
        freeze_base=freeze_base
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_output_path = os.path.join(Config.MODEL_SAVE_PATH, f"{model_name}_{timestamp}.h5")
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

    # Setup callbacks
    checkpoint = ModelCheckpoint(
        filepath=model_output_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max"
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print("\n Training model...\n")
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, lr_scheduler],
        verbose=1
    )

    print(f"\n Training complete. Best model saved at: {model_output_path}")
    return model, history, (X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train selected model on ISIC dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. densenet201, densenet121, nasnetmobile)")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--freeze", action='store_true', help="Freeze base model layers")
    parser.add_argument("--norm", type=str, default="z-score", choices=["z-score", "min-max", "mean-std"],
                        help="Normalization method")

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_base=args.freeze,
        normalization=args.norm
    )
