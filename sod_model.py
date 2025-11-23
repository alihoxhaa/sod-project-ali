"""sod_model.py

Core model definitions for the Salient Object Detection assignment.

This file provides a compact, readable UNet-like model builder and a couple
small utilities used by training and evaluation scripts:

- `get_sod_model(...)` builds an uncompiled Keras model shaped like a
    UNet — configurable depth (`num_blocks`), starting channel width
    (`base_filters`) and optional batch-norm / dropout.
- `soft_iou`, `sod_loss`, and `iou_metric` are small helpers used to compute
    soft IoU and a combined BCE+IoU style loss used in experiments.

Notes:
- The code intentionally does not use pretrained encoders (e.g., VGG), to
    comply with the assignment requirement of building the pipeline without
    pre-trained models.
- Keep comments concise and focused on intent; implementation details are
    visible in the code.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers


def soft_iou(y_true: tf.Tensor, y_pred: tf.Tensor, eps=1e-7):
    # Compute a "soft" Intersection-over-Union between predictions and
    # targets. Inputs are expected to be probabilities in [0,1] with shape
    # [batch, height, width, 1]. This returns the mean IoU across the batch.
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = tf.reduce_sum(y_true_f + y_pred_f - y_true_f * y_pred_f, axis=1)
    iou = (intersection + eps) / (union + eps)
    return tf.reduce_mean(iou)


def sod_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    # A small combined loss: BCE plus a soft IoU term (scaled). This is
    # simple and stable for training segmentation masks — treat BCE as the
    # per-pixel objective and (1 - IoU) as a structural penalty.
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    iou = soft_iou(y_true, y_pred)
    return bce + 0.5 * (1.0 - iou)


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor):
    # Thin wrapper kept for readability in training loops: use `iou_metric`
    # when logging rather than calling `soft_iou` directly.
    return soft_iou(y_true, y_pred)


def conv_block(x, filters, kernel_size=3, use_bn=False):
    # Two consecutive conv layers with optional batchnorm and ReLU. This
    # keeps the receptive field moderate while allowing the network to
    # learn richer feature representations between pooling/upsampling.
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def get_sod_model(input_shape=(224, 224, 3), base_filters=64, use_batchnorm=False, use_dropout=False, dropout_rate=0.2, num_blocks=5):
    """Build a configurable UNet-like encoder-decoder model.

    Parameters
    - input_shape: input image size (H,W,C).
    - base_filters: number of conv channels in the first block. Each
      downsampling doubles channels; the decoder mirrors that.
    - use_batchnorm: add `BatchNormalization` after each conv when True.
    - use_dropout: apply a final dropout layer before the output.
    - dropout_rate: dropout keep probability when enabled.
    - num_blocks: number of encoder blocks (including the bottleneck).

    Returns an uncompiled `tf.keras.Model`. The model intentionally leaves
    compilation and optimizer choice to the training script so experiments
    can swap losses or optimizers without editing this file.
    """
    inputs = layers.Input(shape=input_shape)

    # Build a UNet-like architecture with `num_blocks` encoder blocks.
    # Example: num_blocks=5 produces four downsampling steps plus a
    # bottleneck layer. Keeping the code generic avoids duplicated logic
    # when the depth needs to be tuned for experiments.
    encs = []
    x = inputs
    filters = base_filters
    for i in range(num_blocks - 1):
        f = conv_block(x, filters, use_bn=use_batchnorm)
        encs.append(f)
        x = layers.MaxPooling2D((2, 2))(f)
        filters *= 2

    # bottleneck
    f = conv_block(x, filters, use_bn=use_batchnorm)
    x = f

    # Decoder: upsample and concat with corresponding encoder features
    for enc in reversed(encs):
        filters //= 2
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, enc])
        x = conv_block(x, filters, use_bn=use_batchnorm)

    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='sod_unet_like')
    return model


# Pretrained-encoder variants (e.g., VGG-based) were intentionally removed
# to comply with the assignment requirement: "Implement a full ML pipeline
# without pre-trained models." Use `get_sod_model(...)` above for all
# training and evaluation.


if __name__ == '__main__':
    # quick smoke test when run directly
    m = get_sod_model()
    m.summary()
    import numpy as np
    x = np.random.randn(1, 224, 224, 3).astype('float32')
    y = m.predict(x)
    print('Output shape:', y.shape)

#  Typical compile
# Note: model compilation should be performed in the training script (e.g. train.py)
