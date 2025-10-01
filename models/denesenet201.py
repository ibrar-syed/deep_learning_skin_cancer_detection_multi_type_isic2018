#######models/densenet201.py
##for teh skin cancer detection with multi-layer structure
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from config import Config


def build_densenet201(input_shape=None, num_classes=None, freeze_base=True, learning_rate=1e-4):
    """
    Builds a DenseNet201 model with a progressively deep classification head.

    Args:
        input_shape (tuple): Shape of input images (H, W, C)
        num_classes (int): Total number of output classes
        freeze_base (bool): Whether to freeze the pretrained base
        learning_rate (float): Learning rate for optimizer

    Returns:
        model (tf.keras.Model): Compiled DenseNet201 model
        lr_scheduler (callback): ReduceLROnPlateau callback
    """
    input_shape = input_shape or Config.IMAGE_SHAPE
    num_classes = num_classes or Config.NUM_CLASSES

    # Load base DenseNet201
    base_model = DenseNet201(include_top=False, weights="imagenet", input_shape=input_shape)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    # Build classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Dense layers: 1024 → 512 → 256 → 128 → 64 → 32
    for units, dropout in zip([1024, 512, 256, 128], [0.5, 0.5, 0.5, 0.5]):
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output layer
    output = Dense(num_classes, activation='softmax')(x)

    # Compile model
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Learning rate callback
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
        factor=0.5,
        min_lr=1e-5
    )

    return model, lr_scheduler
