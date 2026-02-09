import json
from pathlib import Path
import glob, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (B, C_in, T)
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        return out

class EventTCN(nn.Module):
    def __init__(self, input_dim, num_classes=1,
                 hidden_dim=32, num_layers=3):
        super().__init__()

        layers = []
        in_ch = input_dim
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                TemporalConvBlock(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                )
            )
            in_ch = hidden_dim
            dilation *= 2  # 1,2,4,...

        self.tcn = nn.Sequential(*layers)
        #self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.fc = nn.Linear(hidden_dim, 1)   # <--- ONE logit


    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        feat = self.tcn(x)             # (B, hidden_dim, T)
        #logits = self.head(feat)       # (B, num_classes, T)
        #logits = logits.transpose(1, 2)  # (B, T, num_classes)
        last_feat = feat[:, :, -1]         # (B, hidden_dim)
        logits = self.fc(last_feat).squeeze(-1)  # (B,)  raw score
        return logits

def frame_to_feature_vector(bbs, K, NUM_FEATURES):
    """
    frame: dict for one time step
    K: max number of boxes per frame
    """
    #detections = d#frame['bbs_list_of_keypoints']  # <-- adapt this key name

    # sort boxes by area desc
    dets_sorted = sorted(
        bbs,
        key=lambda d: (d[2] - d[0]) * (d[3] - d[1]),
        reverse=True,
    )

    boxes = []
    #keypoints = []
    for d in dets_sorted[:K]:
        x1, y1, x2, y2 = d
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2 )/ 2
        cy = (y1 + y2 )/ 2

        # normalize
        '''cx /= img_w
        cy /= img_h
        w  /= img_w
        h  /= img_h'''

        boxes.extend([cx, cy, w, h])#+ d[6])

    # pad if fewer than K boxes
    if len(dets_sorted) < K:
        boxes.extend([0.0] * NUM_FEATURES * (K - len(dets_sorted)))

    n_people = len(bbs)

    x_t = boxes + [float(n_people)]
    return np.array(x_t, dtype=np.float32)
         
