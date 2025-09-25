# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Skin Cancer Detection Project.
###
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
