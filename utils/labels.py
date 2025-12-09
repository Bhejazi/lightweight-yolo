# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:58:32 2025

@author: bhejazi
"""
# Utilities to parse YOLO-format label files and to build training targets for the grid.

from typing import List, Tuple
import os

def parse_yolo_label_file(txt_path: str) -> List[Tuple[float, float, float, float]]:
    # Each line: class_id object_id x_center y_center width height (normalized [0,1]).
    # We ignore class_id and object_id and return a list of (xc, yc, w, h).
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            xc, yc, w, h = map(float, parts[2:6])
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            boxes.append((xc, yc, w, h))
    return boxes

def assign_to_grid(boxes: List[Tuple[float, float, float, float]], grid_size: int):
    # Builds a (S, S, 5) target tensor from normalized boxes.
    # For each box, mark responsible cell based on (xc, yc) and set:
    #   objness=1, x, y, w, h = normalized absolute coords.
    import numpy as np
    target = np.zeros((grid_size, grid_size, 5), dtype=float)
    for (xc, yc, w, h) in boxes:
        i = int(yc * grid_size)
        j = int(xc * grid_size)
        i = max(0, min(grid_size - 1, i))
        j = max(0, min(grid_size - 1, j))
        if target[i, j, 0] == 1.0:
            continue
        target[i, j, 0] = 1.0
        target[i, j, 1] = xc
        target[i, j, 2] = yc
        target[i, j, 3] = w
        target[i, j, 4] = h
    return target
