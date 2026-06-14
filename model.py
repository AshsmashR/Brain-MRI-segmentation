"""
model.py
--------
A compact 2D U-Net for binary segmentation.

This is the same encoder-decoder design as the original notebook, kept small
(16->32->64->128 filters) so it trains on modest hardware, with two practical
additions: BatchNormalization for more stable convergence, and a configurable
input size / base filter count so the architecture can be scaled up when more
compute is available.
"""

from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,
    Dropout, BatchNormalization,
)
from tensorflow.keras.models import Model


def _conv_block(x, filters, dropout):
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    if dropout:
        x = Dropout(dropout)(x)
    return x


def unet_model(input_size=(128, 128, 3), base_filters=16):
    f = base_filters
    inputs = Input(input_size)

    # Encoder
    c1 = _conv_block(inputs, f, 0.1)
    p1 = MaxPooling2D(2)(c1)
    c2 = _conv_block(p1, f * 2, 0.1)
    p2 = MaxPooling2D(2)(c2)
    c3 = _conv_block(p2, f * 4, 0.2)
    p3 = MaxPooling2D(2)(c3)

    # Bottleneck
    c4 = _conv_block(p3, f * 8, 0.3)

    # Decoder
    u5 = Conv2DTranspose(f * 4, 2, strides=2, padding="same")(c4)
    u5 = concatenate([u5, c3])
    c5 = _conv_block(u5, f * 4, 0.2)

    u6 = Conv2DTranspose(f * 2, 2, strides=2, padding="same")(c5)
    u6 = concatenate([u6, c2])
    c6 = _conv_block(u6, f * 2, 0.1)

    u7 = Conv2DTranspose(f, 2, strides=2, padding="same")(c6)
    u7 = concatenate([u7, c1])
    c7 = _conv_block(u7, f, 0.1)

    outputs = Conv2D(1, 1, activation="sigmoid")(c7)
    return Model(inputs=inputs, outputs=outputs, name="unet")
