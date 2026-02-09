import argparse, os
import json, numpy as np, torch
from pathlib import Path

import cv2
from ultralytics import YOLO

def frame_to_vector(frame, K):
    """
    frame: dict for one time step
    K: max number of boxes per frame
    """
    detections = frame['bbs_list_of_keypoints']  # <-- adapt this key name

    # sort boxes by area desc
    dets_sorted = sorted(
        detections,
        key=lambda d: (d[4] - d[2]) * (d[5] - d[3]),
        reverse=True,
    )

    boxes = []
    keypoints = []
    for d in dets_sorted[:K]:
        x1, y1, x2, y2 = d[2:6]
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2 )/ 2
        cy = (y1 + y2 )/ 2

        # normalize
        '''cx /= img_w
        cy /= img_h
        w  /= img_w
        h  /= img_h'''

        boxes.extend([cx, cy, w, h]+ d[6])

    # pad if fewer than K boxes
    if len(dets_sorted) < K:

        boxes.extend([0.0] * NUM_FEATURES * (K - len(dets_sorted)))

    n_people = len(detections)

    x_t = boxes + [float(n_people)]
    return np.array(x_t, dtype=np.float32)

def json_to_sequence(path, K):
    with open(path, "r") as f:
        data = json.load(f)

    frames = data["frames"]  # adapt if needed

    features = []
    labels = []

    for frame in frames:
        x_t = frame_to_vector(frame, K)
        y_t = events_to_label(frame)

        features.append(x_t)
        labels.append(y_t)

    features = np.stack(features).astype(np.float32)  # (N, C)
    labels   = np.array(labels, dtype=np.float32)       # (N,)
    labels_ramped = add_ramp_before_events(labels)
    return features, labels_ramped
