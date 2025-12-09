# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:00:24 2025

@author: bhejazi
"""
# Minimal training script for the Macaque Detector.

import os
import argparse
import tensorflow as tf
from model.detector import build_detector, YoloLoss
from utils.data import discover_dataset, build_tf_dataset

def main():
    parser = argparse.ArgumentParser(description="Train the Macaque Detector")
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder containing 0_100 .. 9_100")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grid_size", type=int, default=7, help="Grid size S (e.g., 7)")
    parser.add_argument("--input_size", type=int, default=224, help="Square input size (e.g., 224)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder to save model and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Discovering dataset...")
    pairs = discover_dataset(args.data_dir)
    if not pairs:
        raise RuntimeError(f"No (image, label) pairs found in {args.data_dir}")
    print(f"Found {len(pairs)} samples")

    # Optionally subsample for quick runs
    # pairs = pairs[:500]

    ds_train = build_tf_dataset(
        pairs=pairs,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        shuffle=True,
        cache=False,
        input_size=(args.input_size, args.input_size),
    )
    
    # Sanity check one batch
    #for x_batch, y_batch in ds_train.take(1):
     #   print("x_batch:", x_batch.shape, x_batch.dtype)
     #   print("y_batch:", y_batch.shape, y_batch.dtype)
     #  # Optional: value range check
     #   print("y min/max:", float(tf.reduce_min(y_batch)), float(tf.reduce_max(y_batch)))

    print("Building model...")
    model = build_detector(input_shape=(args.input_size, args.input_size, 3), grid_size=args.grid_size)

    print("Compiling model...")
    loss_fn = YoloLoss(lambda_box=5.0, lambda_noobj=0.5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss_fn, run_eagerly=False)
    
    #xb, yb = next(iter(ds_train))

    #print("types:", type(xb), type(yb))
    #print("shapes:", xb.shape, yb.shape)
    #print("dtypes:", xb.dtype, yb.dtype)

    # Assert not None
    #assert xb is not None, "xb is None"
    #assert yb is not None, "yb is None"

    # No NaNs/Infs
    #tf.debugging.assert_all_finite(xb, "xb has NaNs or Infs")
    #tf.debugging.assert_all_finite(yb, "yb has NaNs or Infs")

    # Try train_on_batch first
    #model.train_on_batch(xb, yb)

    print("Training...")
    history = model.fit(ds_train, epochs=args.epochs)

    model_path = os.path.join(args.output_dir, "macaque_detector.keras")
    print(f"Saving model to {model_path}")
    model.save(model_path)

    # Save simple training log
    import json
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

if __name__ == "__main__":
    main()
