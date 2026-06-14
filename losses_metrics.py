"""
losses_metrics.py
-----------------
Segmentation losses and metrics for binary mask prediction.

Why this file exists
====================
For segmentation, pixel "accuracy" is misleading: if a structure occupies
only ~5% of an image, a model that predicts "all background" scores ~95%
accuracy while being useless. Overlap-based measures (Dice, IoU/Jaccard)
score how well the *predicted region* matches the *true region*, which is
the quantity that actually matters in neuroimaging segmentation. These are
the standard metrics used by tools such as FreeSurfer/FastSurfer-style
evaluations of cortical and subcortical structures.

Dice  = 2|A n B| / (|A| + |B|)
IoU   = |A n B| / |A u B|        (a.k.a. Jaccard)
"""

import tensorflow as tf
from tensorflow.keras import backend as K

SMOOTH = 1e-6  # avoids division by zero when a mask is empty


def dice_coef(y_true, y_pred):
    """Soft Dice coefficient over a batch. Range [0, 1], higher is better."""
    y_true_f = K.flatten(K.cast(y_true, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred, "float32"))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + SMOOTH) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH
    )


def iou_coef(y_true, y_pred):
    """Soft IoU / Jaccard index over a batch. Range [0, 1], higher is better."""
    y_true_f = K.flatten(K.cast(y_true, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred, "float32"))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)


def dice_loss(y_true, y_pred):
    """1 - Dice. Differentiable, used directly as a training objective."""
    return 1.0 - dice_coef(y_true, y_pred)


def iou_loss(y_true, y_pred):
    """1 - IoU. Differentiable Jaccard loss."""
    return 1.0 - iou_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combined loss: binary cross-entropy + Dice.

    BCE gives stable, well-behaved gradients early in training; Dice directly
    optimises region overlap and handles class imbalance. The sum is a common,
    robust default for medical-image segmentation.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def combined_dice_iou_loss(y_true, y_pred):
    """Equal-weight Dice + IoU loss, for when overlap is the sole objective."""
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)
