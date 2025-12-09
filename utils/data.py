# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:59:25 2025

@author: bhejazi
"""
# Data loading utilities: collect image/label pairs and build tf.data pipelines

from typing import List, Tuple
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from .labels import parse_yolo_label_file, assign_to_grid

def discover_dataset(root_dir: str) -> List[Tuple[str, str]]:
    """Find (image_path, label_path) pairs across subfolders 0_100..9_100."""
    pairs = []
    skipped = 0
    for sub in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, sub)
        if not os.path.isdir(subdir):
            continue
        imgdir = os.path.join(subdir, "images")
        lbldir = os.path.join(subdir, "labels_with_ids")
        if not (os.path.isdir(imgdir) and os.path.isdir(lbldir)):
            continue
        for fname in os.listdir(imgdir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            image_path = os.path.join(imgdir, fname)
            label_path = os.path.join(lbldir, stem + ".txt")
            if os.path.exists(label_path):
                pairs.append((image_path, label_path))
            else:
                skipped += 1
    if skipped > 0:
        print(f"[discover_dataset] Skipped {skipped} images with no label file.")
    return pairs

def build_tf_dataset(pairs: List[Tuple[str, str]], grid_size: int = 7, batch_size: int = 8,
                     shuffle: bool = True, cache: bool = False,
                     input_size=(224, 224),
                     skip_errors: bool = True) -> tf.data.Dataset:
    """Create a tf.data.Dataset of (image_tensor, target_grid) with robust I/O handling."""

    def _py_load(image_path: bytes, label_path: bytes):
        # Always return (np.float32 HxWx3, np.float32 SxSx5)
        import warnings

        ip = image_path.decode("utf-8", errors="ignore")
        lp = label_path.decode("utf-8", errors="ignore")

        # --- Image loading ---
        try:
            img = Image.open(ip).convert("RGB")
            img = img.resize(input_size, resample=Image.BILINEAR)
            img_arr = np.asarray(img).astype(np.float32) / 255.0
            if img_arr.ndim != 3 or img_arr.shape[2] != 3:
                raise ValueError(f"Unexpected image shape {img_arr.shape} for {ip}")
        except Exception as e:
            warnings.warn(f"[data] Failed to read image '{ip}': {e}. Using zeros.")
            img_arr = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)

        # --- Label parsing ---
        try:
            if lp and os.path.exists(lp):
                boxes = parse_yolo_label_file(lp) or []  # tolerate empty/odd labels
                target = assign_to_grid(boxes, grid_size).astype(np.float32)
            else:
                # No label => negative sample
                target = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
        except Exception as e:
            warnings.warn(f"[data] Failed to parse label '{lp}': {e}. Using zero grid.")
            target = np.zeros((grid_size, grid_size, 5), dtype=np.float32)

        # Final dtype guards
        if img_arr.dtype != np.float32:
            img_arr = img_arr.astype(np.float32)
        if target.dtype != np.float32:
            target = target.astype(np.float32)

        return img_arr, target

    def _map(image_path, label_path):
        img, target = tf.numpy_function(
            _py_load, [image_path, label_path], Tout=[tf.float32, tf.float32]
        )
        img.set_shape((input_size[0], input_size[1], 3))
        target.set_shape((grid_size, grid_size, 5))
        return img, target

    if not pairs:
        raise ValueError("No data pairs provided to build_tf_dataset.")

    # Split into parallel lists to ensure two components (image_path, label_path)
    images, labels = zip(*pairs)
    images, labels = list(images), list(labels)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(images)))
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

    # Optionally skip any element that still give errors in training
    if skip_errors:
        ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


