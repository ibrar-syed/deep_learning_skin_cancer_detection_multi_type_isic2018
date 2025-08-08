# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Skin Cancer Detection Project.
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

# models/densenet121.py
##for teh skin cancer detection with multi-layer structure
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from config import Config


def build_densenet121(input_shape=None, num_classes=None, freeze_base=True, learning_rate=1e-4):
    """
    Builds a DenseNet121 model with custom classification head.

    Args:
        input_shape (tuple): Image input shape
        num_classes (int): Number of output classes
        freeze_base (bool): Whether to freeze base model
        learning_rate (float): Learning rate

    Returns:
        model: compiled Keras model
        lr_scheduler: ReduceLROnPlateau callback
    """
    input_shape = input_shape or Config.IMAGE_SHAPE
    num_classes = num_classes or Config.NUM_CLASSES

    base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    for units, dropout in zip([1024, 512, 256, 128], [0.5, 0.5, 0.5, 0.5]):
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    lr_scheduler = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=1e-5)

    return model, lr_scheduler
