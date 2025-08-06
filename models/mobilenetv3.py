##for the skin cancer detection with multi-layer structure

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from config import Config

def build_mobilenetv3(input_shape=None, num_classes=None, freeze_base=True, learning_rate=1e-4):
    input_shape = input_shape or Config.IMAGE_SHAPE
    num_classes = num_classes or Config.NUM_CLASSES

    base_model = MobileNetV3Large(include_top=False, weights="imagenet", input_shape=input_shape)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Deep classification head
    for units in [1024, 512, 256, 128, 64, 32]:
        x = Dense(units, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
        factor=0.5,
        min_lr=1e-5
    )

    return model, lr_scheduler
