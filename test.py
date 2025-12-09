# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 11:57:21 2025

@author: bhejazi
"""

from inference import MacaqueDetector

detector = MacaqueDetector(model_path="outputs/macaque_detector.keras")
boxes = detector.predict("M:/codes/codes_BAM/python/Yolo/macaque_images_train/MacaqueImagePairs/7_100/images/img_00306_0.jpg", score_threshold=0.3)

for score, x1, y1, x2, y2 in boxes:
    print(f"Macaque score={score:.2f} bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

# Save visualization
detector.visualize("M:/codes/codes_BAM/python/Yolo/macaque_images_train/MacaqueImagePairs/7_100/images/img_00306_0.jpg", boxes, "M:/codes/codes_BAM/python/Yolo/outputs/detected.jpg")
