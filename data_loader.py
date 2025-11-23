"""data_loader.py

Dataset utilities for the SOD project. This module handles three jobs:

- locating paired image/mask directories under a dataset root,\
- listing and splitting matched (image, mask) pairs into train/val/test, and\
- creating `tf.data.Dataset` objects that decode, resize and optionally
    augment image/mask pairs.

The augmentations are intentionally simple and fast (flips, color jitter,
rotations, zoom/crop and light noise) — they improve generalization without
adding much runtime overhead during training.
"""
from __future__ import annotations
import os
import glob
import argparse
from typing import List, Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def find_image_mask_dirs(root: str) -> Tuple[str, str]:
    """Find the first matching image and mask directories under `root`.

    The function searches the directory tree looking for folder names that
    contain 'Image' and 'Mask' respectively. This heuristic matches the
    DUTS dataset layout used in the assignment; if it fails we raise an
    informative error so the caller can point to the correct path.
    """
    image_dir = None
    mask_dir = None
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if 'Image' in d and image_dir is None:
                candidate = os.path.join(dirpath, d)
                image_dir = candidate
            if 'Mask' in d and mask_dir is None:
                candidate = os.path.join(dirpath, d)
                mask_dir = candidate
        if image_dir and mask_dir:
            break
    if not image_dir or not mask_dir:
        raise FileNotFoundError(f"Could not find image/mask dirs under {root}")
    return image_dir, mask_dir


def list_image_mask_pairs(image_dir: str, mask_dir: str, max_samples: int | None = None) -> List[Tuple[str, str]]:
    """Return a sorted list of (image_path, mask_path) pairs.

    We match files by basename and support common image extensions. If a
    mask does not exist for an image we skip that image. The sorting keeps
    splits deterministic, which is important for reproducible experiments.
    """
    # collect image files (common image extensions)
    img_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        img_files.extend(glob.glob(os.path.join(image_dir, ext)))
    img_files = sorted(img_files)

    pairs = []
    for img in img_files:
        base = os.path.splitext(os.path.basename(img))[0]
        # try common mask extensions
        mask_candidates = [os.path.join(mask_dir, base + ext) for ext in ('.png', '.jpg', '.bmp')]
        mask = next((m for m in mask_candidates if os.path.exists(m)), None)
        if mask:
            pairs.append((img, mask))
        # if there is no mask file, skip
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs


def split_pairs(pairs: List[Tuple[str, str]], train=0.7, val=0.15, test=0.15, seed=42):
    # Deterministic split using sklearn so results are reproducible.
    assert abs(train + val + test - 1.0) < 1e-6
    img_paths = [p for p, _ in pairs]
    mask_paths = [m for _, m in pairs]
    # Split out training set first, then partition the remainder into
    # validation and test sets according to the requested ratios.
    X_train, X_temp, y_train, y_temp = train_test_split(img_paths, mask_paths, train_size=train, random_state=seed)
    temp_ratio = val / (val + test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=temp_ratio, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _load_image(path: str, target_size: int):
    # Read and decode an image file, convert to float32 in [0,1] and resize.
    # The helper stays intentionally small — color augmentations happen later
    # in the pipeline so the raw loader remains fast and re-usable.
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [target_size, target_size])
    return img


def _load_mask(path: str, target_size: int):
    # Load mask image, resize using nearest neighbor (preserve labels)
    # and binarize so masks are returned as 0/1 floats.
    m = tf.io.read_file(path)
    m = tf.image.decode_image(m, channels=1)
    m = tf.image.convert_image_dtype(m, tf.float32)
    m = tf.image.resize(m, [target_size, target_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    m = tf.math.greater(m, 0.5)
    m = tf.cast(m, tf.float32)
    return m


def _augment(image: tf.Tensor, mask: tf.Tensor, target_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    # Small, fast augmentations applied to both image and mask. The set is
    # intentionally conservative: augmentations should change appearance but
    # preserve the underlying segmentation labels.
    if tf.random.uniform(()) > 0.5:
        # Horizontal flip with 50% probability
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Slight brightness change to vary illumination
    image = tf.image.random_brightness(image, 0.15)

    # Random zoom + crop: resize to a slightly larger size then crop back.
    # This simulates small scale changes while preserving original image
    # resolution for training.
    scale = tf.random.uniform([], 1.0, 1.15)
    new_size = tf.cast(scale * tf.cast(target_size, tf.float32), tf.int32)
    if new_size > target_size:
        image = tf.image.resize(image, [new_size, new_size])
        mask = tf.image.resize(mask, [new_size, new_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.random_crop(image, size=[target_size, target_size, 3])
        mask = tf.image.random_crop(mask, size=[target_size, target_size, 1])

    return image, mask


def create_dataset(image_paths: List[str], mask_paths: List[str], batch_size=16, target_size=224, shuffle=True, augment=False, buffer_size=1000) -> tf.data.Dataset:
    """Create a tf.data.Dataset yielding (image, mask) pairs normalized to [0,1].
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size)
    def _parse_function(img_path, m_path):
        # img_path and m_path are scalar string tensors
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [target_size, target_size])

        m = tf.io.read_file(m_path)
        m = tf.image.decode_image(m, channels=1)
        m.set_shape([None, None, 1])
        m = tf.image.convert_image_dtype(m, tf.float32)
        m = tf.image.resize(m, [target_size, target_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        m = tf.math.greater(m, 0.5)
        m = tf.cast(m, tf.float32)

        if augment:
            # random horizontal flip
            flip = tf.random.uniform(()) > 0.5
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            m = tf.cond(flip, lambda: tf.image.flip_left_right(m), lambda: m)
            # random brightness
            img = tf.image.random_brightness(img, 0.15)

            # random contrast/saturation/hue to diversify colors
            img = tf.image.random_contrast(img, 0.85, 1.15)
            img = tf.image.random_saturation(img, 0.9, 1.2)
            img = tf.image.random_hue(img, 0.02)

            # small random rotation by 0/90/180/270 deg (via rot90)
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k)
            m = tf.image.rot90(m, k)

            # random zoom/crop
            scale = tf.random.uniform([], 1.0, 1.15)
            new_size = tf.cast(scale * tf.cast(target_size, tf.float32), tf.int32)

            def _zoom_crop():
                img_r = tf.image.resize(img, [new_size, new_size])
                m_r = tf.image.resize(m, [new_size, new_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                img_c = tf.image.random_crop(img_r, size=[target_size, target_size, 3])
                m_c = tf.image.random_crop(m_r, size=[target_size, target_size, 1])
                return img_c, m_c

            def _no_op():
                return img, m

            img, m = tf.cond(new_size > target_size, _zoom_crop, _no_op)

            # add small gaussian noise (helps generalization)
            noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.01)
            img = tf.clip_by_value(img + noise, 0.0, 1.0)

        return img, m

    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_splits_csv(out_dir: str, splits: dict):
    os.makedirs(out_dir, exist_ok=True)
    for name, (imgs, masks) in splits.items():
        df = pd.DataFrame({'image': imgs, 'mask': masks})
        csv_path = os.path.join(out_dir, f'{name}.csv')
        df.to_csv(csv_path, index=False)


def main(args):
    image_dir, mask_dir = find_image_mask_dirs(args.dataset_root)
    print(f'Found image dir: {image_dir}')
    print(f'Found mask dir:  {mask_dir}')
    pairs = list_image_mask_pairs(image_dir, mask_dir, max_samples=args.max_samples)
    print(f'Collected {len(pairs)} image/mask pairs (cap {args.max_samples})')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_pairs(pairs, train=0.7, val=0.15, test=0.15, seed=args.seed)

    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
    }
    out_dir = os.path.join('data', 'splits')
    save_splits_csv(out_dir, splits)
    print(f'Saved CSV splits to {out_dir}')

    # optionally build small dataset to verify
    if args.quick_check:
        ds = create_dataset(X_train[:min(32, len(X_train))], y_train[:min(32, len(y_train))], batch_size=8, target_size=args.target_size, augment=True)
        for batch_images, batch_masks in ds.take(1):
            print('Batch images shape:', batch_images.shape)
            print('Batch masks shape: ', batch_masks.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to dataset root (e.g. data/DUTS-TR/DUTS-TR)')
    parser.add_argument('--target-size', type=int, default=224, help='Resize target (default 224)')
    parser.add_argument('--max-samples', type=int, default=2500, help='Maximum number of image/mask pairs to load (default 2500)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick-check', action='store_true', help='Build one small dataset batch to sanity check')
    args = parser.parse_args()
    main(args)
