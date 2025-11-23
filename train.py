"""train.py

Training entry point and a compact custom training loop used for experiments.

This module keeps training logic explicit (GradientTape) so we can:

- log arbitrary per-batch metrics,\
- implement a custom LR reduction / checkpointing strategy, and\
- easily support mixed-precision or other optimizer wrappers.

Key behaviour:
- Datasets are built via `data_loader.create_dataset` and expected to yield
    `(image, mask)` pairs where images are floats in [0,1].
- The model is created by `sod_model.get_sod_model(...)`; compilation is
    deferred to the training loop to allow flexible loss choices.
- Checkpointing saves both the model weights and optimizer state. The best
    weights (by validation loss) are also written to `best_model.weights.h5`.

This file is intentionally pragmatic rather than minimal: explicit loops
make it easier to inspect gradients and plug in diagnostic code when needed.
"""
from __future__ import annotations
import os
import argparse
import time
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm

from sod_model import get_sod_model, iou_metric
import data_loader


class ThresholdedIoU(tf.keras.metrics.Metric):
    def __init__(self, name='th_iou', threshold=0.5, **kwargs):
        # A simple metric that computes IoU after thresholding predictions.
        # We keep an incremental average (sum/count) because some TF versions
        # don't always support batch-wise reductions the same way.
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        union = tf.reduce_sum(y_true_f + y_pred_f - y_true_f * y_pred_f, axis=1)
        iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))
        self.total_iou.assign_add(iou)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_iou / (self.count + 1e-7)

    def reset_states(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


def make_datasets(splits_dir: str, target_size: int, batch_size: int, augment: bool):
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv = os.path.join(splits_dir, 'val.csv')
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f'Expected CSVs not found in {splits_dir}')

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # Build tf.data.Datasets used in the training loop. The data loader handles
    # decoding, resizing and optional augmentations; the training loop simply
    # consumes the batches produced here.
    train_ds = data_loader.create_dataset(df_train['image'].tolist(), df_train['mask'].tolist(), batch_size=batch_size, target_size=target_size, shuffle=True, augment=augment)
    val_ds = data_loader.create_dataset(df_val['image'].tolist(), df_val['mask'].tolist(), batch_size=batch_size, target_size=target_size, shuffle=False, augment=False)
    return train_ds, val_ds


def train_loop(args):
    train_ds, val_ds = make_datasets(args.splits_dir, args.target_size, args.batch_size, augment=args.augment)

    # Try to enable memory growth to reduce fragmentation. This avoids TF
    # pre-allocating the entire GPU memory pool which can interfere with other
    # processes or repeated runs in the same session.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # Build UNet model for current experiment. We keep model construction in
    # the training script to allow different configs (filters, depth, etc.)
    # to be passed from CLI without editing `sod_model.py`.
    model = get_sod_model(input_shape=(args.target_size, args.target_size, 3),
                          base_filters=args.base_filters,
                          use_batchnorm=args.batchnorm,
                          use_dropout=args.dropout,
                          dropout_rate=args.dropout_rate,
                          num_blocks=args.num_blocks)

    # Mixed precision support (optional). When enabled we wrap a float32
    # optimizer in Keras' LossScaleOptimizer so gradients remain numerically
    # stable. We handle TF API differences later when scaling/unscaling.
    if getattr(args, 'mixed_precision', False):
        mixed_precision.set_global_policy('mixed_float16')
        base_opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        optimizer = mixed_precision.LossScaleOptimizer(base_opt)
        use_mixed = True
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        use_mixed = False

    # Stable combined loss: a balanced mixture of BCE and Dice loss. Dice
    # (similar to IoU) favors global alignment of masks while BCE handles
    # per-pixel probabilities. We expose `label_smoothing` so you can nudge
    # targets slightly if needed to stabilize training.
    def bce_dice_loss(bce_weight=0.5, dice_weight=0.5, epsilon=1e-6, label_smoothing=0.0):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            if label_smoothing and label_smoothing > 0.0:
                y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
            y_pred = tf.cast(y_pred, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            # BCE per-pixel averaged per-sample
            bce_per_pixel = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
            bce_per_sample = tf.reduce_mean(tf.reshape(bce_per_pixel, (tf.shape(y_pred)[0], -1)), axis=1)
            bce = tf.reduce_mean(bce_per_sample)

            # Dice loss per-sample
            y_true_f = tf.reshape(y_true, (tf.shape(y_pred)[0], -1))
            y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
            intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
            sums = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
            dice_coef = (2.0 * intersection + epsilon) / (sums + epsilon)
            dice_loss_per_sample = 1.0 - dice_coef
            dice_loss = tf.reduce_mean(dice_loss_per_sample)

            return bce_weight * bce + dice_weight * dice_loss

        return loss

    # instantiate loss function used in the training loop
    loss_fn = bce_dice_loss(bce_weight=0.5, dice_weight=0.5, epsilon=1e-6, label_smoothing=args.label_smoothing)

    # Checkpoint manager: saves optimizer state and model epoch so training
    # can be resumed. We also keep a separate `best_model.weights.h5` file
    # that stores only the model weights (handy for evaluation and lighter
    # loading during inference).
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=3)

    start_epoch = 0
    best_val_loss = float('inf')
    no_improve = 0
    patience = getattr(args, 'patience', 4)
    # Learning-rate reduction settings used by the manual scheduler below.
    lr_reduce_factor = getattr(args, 'lr_reduce_factor', 0.5)
    lr_reduce_patience = getattr(args, 'lr_reduce_patience', 3)
    # resume if requested
    if args.resume:
        latest = ckpt_manager.latest_checkpoint
        if latest:
            ckpt.restore(latest)
            start_epoch = int(ckpt.epoch.numpy())
            print(f'Resuming from checkpoint {latest}, starting at epoch {start_epoch}')

    # training loop
    for epoch in range(start_epoch, args.epochs):
        print(f'--- Epoch {epoch+1}/{args.epochs} ---')
        epoch_start = time.time()

        # training metrics
        train_loss = tf.keras.metrics.Mean()
        train_iou = tf.keras.metrics.Mean()
        train_precision = tf.keras.metrics.Precision()
        train_recall = tf.keras.metrics.Recall()

        # ----- Training step (iterate over mini-batches) -----
        for images, masks in tqdm(train_ds, desc='train', unit='batch'):
            # Compute forward and loss under GradientTape so we can get
            # gradients for backprop. We intentionally keep the loop
            # explicit for transparency and easy debugging.
            with tf.GradientTape() as tape:
                preds = model(images, training=True)
                loss_value = loss_fn(masks, preds)

                # Mixed precision: some TF versions expose helper methods on
                # the LossScaleOptimizer while others expose a `loss_scale`
                # attribute. We check both to remain compatible across TF
                # builds (safe fallback, no change in numeric intent).
                if use_mixed:
                    if hasattr(optimizer, 'get_scaled_loss'):
                        scaled_loss = optimizer.get_scaled_loss(loss_value)
                    else:
                        ls = getattr(optimizer, 'loss_scale', None)
                        if ls is not None:
                            try:
                                scaled_loss = loss_value * tf.cast(ls, loss_value.dtype)
                            except Exception:
                                scaled_loss = loss_value
                        else:
                            scaled_loss = loss_value
                else:
                    scaled_loss = loss_value

            grads = tape.gradient(scaled_loss, model.trainable_variables)
            if use_mixed:
                # Unscale gradients after backward pass. Again handle both
                # common API patterns so the code works across TF versions.
                if hasattr(optimizer, 'get_unscaled_gradients'):
                    grads = optimizer.get_unscaled_gradients(grads)
                else:
                    ls = getattr(optimizer, 'loss_scale', None)
                    if ls is not None:
                        try:
                            grads = [g / tf.cast(ls, g.dtype) if g is not None else None for g in grads]
                        except Exception:
                            pass

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.update_state(loss_value)
            train_iou.update_state(iou_metric(masks, preds))
            # threshold preds for precision/recall
            preds_bin = tf.cast(preds >= 0.5, tf.float32)
            train_precision.update_state(masks, preds_bin)
            train_recall.update_state(masks, preds_bin)

        # ----- Validation loop (evaluate on held-out batches) -----
        val_loss = tf.keras.metrics.Mean()
        val_iou = ThresholdedIoU()
        val_precision = tf.keras.metrics.Precision()
        val_recall = tf.keras.metrics.Recall()

        for images, masks in tqdm(val_ds, desc='val', unit='batch'):
            preds = model(images, training=False)
            loss_value = loss_fn(masks, preds)
            val_loss.update_state(loss_value)
            val_iou.update_state(masks, preds)
            preds_bin = tf.cast(preds >= 0.5, tf.float32)
            val_precision.update_state(masks, preds_bin)
            val_recall.update_state(masks, preds_bin)

        # Compute F1 from batch-level precision / recall metrics. We do a
        # simple harmonic mean here; for more robust reporting you could
        # compute per-image F1 and average (future improvement).
        train_f1 = 2 * (train_precision.result().numpy() * train_recall.result().numpy()) / (train_precision.result().numpy() + train_recall.result().numpy() + 1e-7)
        val_f1 = 2 * (val_precision.result().numpy() * val_recall.result().numpy()) / (val_precision.result().numpy() + val_recall.result().numpy() + 1e-7)

        print(f"Epoch {epoch+1} time: {time.time()-epoch_start:.1f}s | train_loss: {train_loss.result().numpy():.4f} | train_iou: {train_iou.result().numpy():.4f} | train_f1: {train_f1:.4f}")
        print(f"               val_loss:   {val_loss.result().numpy():.4f} | val_th_iou: {val_iou.result().numpy():.4f} | val_f1: {val_f1:.4f}")

        # checkpoint: save model+optimizer+epoch (so resume works)
        ckpt.epoch.assign(epoch + 1)
        saved_path = ckpt_manager.save()
        print('Saved checkpoint:', saved_path)

        # save best model (by val loss)
        current_val_loss = float(val_loss.result().numpy())
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improve = 0
            best_path = os.path.join(ckpt_dir, 'best_model.weights.h5')
            model.save_weights(best_path)
            print('Saved best weights to', best_path)
        else:
            no_improve += 1
            print(f'No improvement for {no_improve} epoch(s) (patience={patience})')

        # learning rate reduction on plateau (manual for custom loop)
        if no_improve >= lr_reduce_patience:
            old_lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
            new_lr = max(1e-7, old_lr * lr_reduce_factor)
            try:
                tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
                print(f'Reduced learning rate: {old_lr:.6g} -> {new_lr:.6g}')
            except Exception:
                print('Could not set optimizer learning rate programmatically')
            # reset no_improve so reductions can happen again later
            no_improve = 0

        # early stopping based on patience
        if no_improve >= patience:
            print(f'Early stopping triggered (no improvement for {patience} epochs).')
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits-dir', type=str, default='data/splits')
    parser.add_argument('--target-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training (uses LossScaleOptimizer)')
    parser.add_argument('--base-filters', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=5, help='Number of encoder blocks (e.g. 5 for deeper UNet)')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor for targets (0.0 = none)')
    parser.add_argument('--lr-reduce-factor', type=float, default=0.5, help='Factor to reduce LR by on plateau')
    parser.add_argument('--lr-reduce-patience', type=int, default=3, help='Epochs with no improvement before reducing LR')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_loop(args)
