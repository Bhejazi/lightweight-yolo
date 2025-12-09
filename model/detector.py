# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:56:04 2025

@author: bhejazi
"""
# Simple YOLO-style, single-class detector using a MobileNetV2 backbone.

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models

def build_detector(input_shape: Tuple[int, int, int] = (224, 224, 3), grid_size: int = 7) -> tf.keras.Model:
    # Builds the detector model.
    # - input_shape: Input image shape.
    # - grid_size: Number of grid cells per side (S).
    # Returns: A Keras Model that outputs (S, S, 5) per image.
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = base.output  # feature map (usually 7x7 for 224x224)
    x = layers.Resizing(grid_size, grid_size, interpolation="bilinear")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(5, 1, padding="same", activation="sigmoid", name="yolo_head")(x)
    model = models.Model(inputs=base.input, outputs=x, name="macaque_detector")
    return model

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_box: float = 5.0, lambda_noobj: float = 0.5, name: str = "yolo_loss"):
        super().__init__(name=name)
        self.lambda_box = lambda_box
        self.lambda_noobj = lambda_noobj
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        # Validate inputs are tensors
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        obj_true = y_true[..., 0]
        box_true = y_true[..., 1:5]
        obj_pred = y_pred[..., 0]
        box_pred = y_pred[..., 1:5]

        # Objectness loss
        obj_loss_pos = self.bce(obj_true, obj_pred)

        # Box loss only where objectness true
        mask = tf.cast(tf.expand_dims(obj_true, axis=-1), tf.float32)  # (B,S,S,1)
        box_mse = tf.reduce_mean(tf.square((box_true - box_pred) * mask))

        # No-object penalty
        noobj_true = 1.0 - obj_true
        noobj_loss = self.bce(noobj_true, 1.0 - obj_pred)

        total = obj_loss_pos + self.lambda_box * box_mse + self.lambda_noobj * noobj_loss

        # Ensure scalar float32 tensor returned
        return tf.cast(total, tf.float32)
