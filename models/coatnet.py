###coatnet.py, for teh skin cancer detection with multi-layer structure

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Conv2D, MultiHeadAttention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from config import Config

def build_coatnet(input_shape=None, num_classes=None, learning_rate=1e-4):
    input_shape = input_shape or Config.IMAGE_SHAPE
    num_classes = num_classes or Config.NUM_CLASSES

    inputs = Input(shape=input_shape)

    # Patch Embedding
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

    # Hybrid blocks (CNN + simplified transformer attention)
    for _ in range(3):
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)

    x = GlobalAveragePooling2D()(x)

    for units in [64, 32]:
        x = Dense(units, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

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
