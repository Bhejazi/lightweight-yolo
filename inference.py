# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:01:02 2025

@author: bhejazi
"""
# Run inference on a single image and save visualized bounding boxes

import argparse
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

class SingleClassDetector:
    """
    Wrapper to load the trained model and predict boxes on new images

    Outputs each detection as: (score, x_min, y_min, x_max, y_max) in original image pixels
    """

    def __init__(self, model_path: str, grid_size: int = 7, input_size: int = 224):
        # Load trained model
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.grid_size = grid_size       # nominal S; we will auto-detect from model output anyway
        self.input_size = input_size     # model input size used at training/inference
        self.orig_size = None            # (W, H) filled in _prepare

    def _prepare(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image:
          - EXIF transpose
          - RGB
          - Resize to (input_size, input_size)
          - Normalize to [0, 1]
        Saves original size for projection back to pixel coordinates.
        """
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")  # handle EXIF orientation
        self.orig_size = img.size  # (W, H)
        img_resized = img.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        arr = np.asarray(img_resized).astype(np.float32) / 255.0
        return arr

    def predict(self, image_path: str, score_threshold: float = 0.5) -> List[Tuple[float, float, float, float, float]]:
        """
        Run model inference on a single image.

        Args:
            image_path: path to image file.
            score_threshold: keep detections with objectness >= threshold (0..1).

        Returns:
            List of (score, x_min, y_min, x_max, y_max) in original pixel coordinates.
        """
        # Sanitize threshold
        if score_threshold < 0.0 or score_threshold > 1.0:
            score_threshold = float(np.clip(score_threshold, 0.0, 1.0))

        arr = self._prepare(image_path)
        inp = np.expand_dims(arr, axis=0)                       # (1, H, W, 3)
        pred = self.model.predict(inp, verbose=0)[0]            # (S, S, 5)
        # Auto-detect grid size S from model output to avoid mismatch
        if pred.ndim != 3 or pred.shape[2] != 5:
            raise ValueError(f"Unexpected prediction shape {pred.shape}; expected (S, S, 5).")
        S_h, S_w, C = pred.shape
        if S_h != S_w:
            raise ValueError(f"Non-square grid from model: {(S_h, S_w)}. Expected SxS.")
        S = S_h

        W, H = self.orig_size
        boxes: List[Tuple[float, float, float, float, float]] = []

        # Iterate cells
        for i in range(S):
            for j in range(S):
                score = float(pred[i, j, 0])
                if score < score_threshold:
                    continue

                # Normalized box coords in [0,1]
                x, y, w, h = pred[i, j, 1:5]
                # Clamp to [0,1] for safety
                x = float(np.clip(x, 0.0, 1.0))
                y = float(np.clip(y, 0.0, 1.0))
                w = float(np.clip(w, 0.0, 1.0))
                h = float(np.clip(h, 0.0, 1.0))

                # Convert center-format normalized box to pixel corners
                x_min = (x - w / 2.0) * W
                y_min = (y - h / 2.0) * H
                x_max = (x + w / 2.0) * W
                y_max = (y + h / 2.0) * H

                # Clip to image bounds and ensure proper ordering
                x_min = max(0.0, min(float(W - 1), x_min))
                y_min = max(0.0, min(float(H - 1), y_min))
                x_max = max(0.0, min(float(W - 1), x_max))
                y_max = max(0.0, min(float(H - 1), y_max))

                if x_max <= x_min or y_max <= y_min:
                    continue  # skip degenerate boxes

                boxes.append((score, x_min, y_min, x_max, y_max))

        # Apply NMS
        if not boxes:
            return []
        boxes = nms(boxes, iou_threshold=0.5)
        return boxes

    def visualize(self, image_path: str, boxes: List[Tuple[float, float, float, float, float]], save_path: str):
        """
        Draw predicted boxes with scores and save an image.
        """
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        draw = ImageDraw.Draw(img)

        for (score, x1, y1, x2, y2) in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            # Offset text slightly inside the rectangle
            label = f"{score:.2f}"
            tx = min(max(0, x1 + 3), img.width - 1)
            ty = min(max(0, y1 + 3), img.height - 1)
            draw.text((tx, ty), label, fill=(255, 0, 0))

        # Ensure the output directory exists
        out_dir = os.path.dirname(save_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        img.save(save_path)
        return save_path


def iou(boxA, boxB) -> float:
    """
    Intersection over Union between boxes:
      box = (score, x1, y1, x2, y2)
    """
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h

    areaA = max(0.0, boxA[3] - boxA[1]) * max(0.0, boxA[4] - boxA[2])
    areaB = max(0.0, boxB[3] - boxB[1]) * max(0.0, boxB[4] - boxB[2])

    union = areaA + areaB - inter + 1e-6  # epsilon to avoid /0
    return inter / union


def nms(boxes: List[Tuple[float, float, float, float, float]], iou_threshold: float = 0.5):
    """
    Greedy Non-Maximum Suppression:
      - Sort by score descending
      - Keep highest score and drop boxes with IoU >= threshold
    """
    boxes = sorted(boxes, key=lambda b: b[0], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_threshold]
    return kept


def cli():
    parser = argparse.ArgumentParser(description="Run Detector inference on an image")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="outputs/inference_result.jpg")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--input_size", type=int, default=224)
    args = parser.parse_args()

    det = SingleClassDetector(model_path=args.model_path, grid_size=args.grid_size, input_size=args.input_size)
    boxes = det.predict(args.image_path, score_threshold=args.score_threshold)
    out = det.visualize(args.image_path, boxes, args.save_path)
    print(f"Saved visualization to {out}")


if __name__ == "__main__":
    import os  # needed for ensure-dir in visualize
    cli()
