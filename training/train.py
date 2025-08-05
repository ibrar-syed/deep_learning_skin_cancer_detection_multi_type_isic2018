import os
import argparse
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
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

# === Dynamic model loading ===
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

# === Model registry ===
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
    print(f"\n Starting training with model: {model_name}")
    print(f" Normalization method: {normalization}")
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[ERROR] Unknown model: '{model_name}'. Check your model name.")

    # === 1. Load + Resize ===
    df, label_map = load_and_resize_dataset(Config.DATASET_PATH, target_size=Config.IMAGE_SHAPE[:2])

    # === 2. Augmentation ===
    augmented_df = augment_dataset(df, max_per_class=Config.MAX_SAMPLES_PER_CLASS)

    # === 3. Preprocessing ===
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

    # === 4. Load Model ===
    model_builder = MODEL_REGISTRY[model_name]
    model, lr_scheduler = model_builder(
        input_shape=Config.IMAGE_SHAPE,
        num_classes=Config.NUM_CLASSES,
        freeze_base=freeze_base,
        learning_rate=Config.LEARNING_RATE
    )

    # === 5. Setup ModelCheckpoint ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{model_name}_{timestamp}.h5"
    save_path = os.path.join(Config.MODEL_SAVE_PATH, save_name)

    checkpoint_cb = ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # === 6. Train Model ===
    print("\n Beginning training...\n")
    history = model.fit(
        x_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_cb, lr_scheduler],
        verbose=1
    )

    # === 7. Final Evaluation ===
    print("\n Final evaluation on test set...")
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f" Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

    print(f"\n Training complete. Best model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep learning model on custom dataset")
    parser.add_argument("--model", required=True, type=str, help="Model name (e.g., densenet201, coatnet, etc.)")
    parser.add_argument("--norm", default="z-score", type=str, help="Normalization method (z-score | min-max | mean-std)")
    parser.add_argument("--freeze", action="store_true", help="Freeze pretrained base layers")

    args = parser.parse_args()
    train_model(model_name=args.model, normalization=args.norm, freeze_base=args.freeze)
