# scripts/infer_video_hysteresis.py
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.data.features import frame_to_vector
from src.models.tcn import EventTCN


@dataclass
class HystCfg:
    thr_on: float = 0.90
    thr_off: float = 0.70
    k_on: int = 3
    k_off: int = 4
    N: int = 8


class Hysteresis:
    def __init__(self, cfg: HystCfg):
        self.cfg = cfg
        self.buf: Deque[float] = deque(maxlen=cfg.N)
        self.on = False

    def update(self, p: float) -> bool:
        self.buf.append(float(p))
        if len(self.buf) < self.cfg.N:
            return self.on
        if not self.on:
            if sum(x >= self.cfg.thr_on for x in self.buf) >= self.cfg.k_on:
                self.on = True
        else:
            if sum(x <= self.cfg.thr_off for x in self.buf) >= self.cfg.k_off:
                self.on = False
        return self.on


def frame_to_detection_list(results, conf_thresh: float) -> List[dict]:
    det_list = []
    if results.boxes is None or results.keypoints is None:
        return det_list
    if len(results.boxes) == 0:
        return det_list

    # same logic as your JSON maker: bbox=xyxyn, key_points = (xyn + conf) flattened
    for box, kpts_norm, conf_kpts in zip(results.boxes, results.keypoints.xyn, results.keypoints.conf):
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < conf_thresh:
            continue

        # skip grouped-event classes like in your builder
        if cls_id in [3, 4]:
            continue

        x1, y1, x2, y2 = map(float, box.xyxyn[0])  # normalized bbox
        kpts_xyc = torch.cat([kpts_norm, conf_kpts.unsqueeze(-1)], dim=-1)  # (V,3)

        det_list.append({
            "class": cls_id,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            "key_points": kpts_xyc.flatten().tolist(),
        })
    return det_list


@torch.inference_mode()
def predict_prob(model: torch.nn.Module, window_btc: torch.Tensor) -> float:
    # model returns logits (B,) :contentReference[oaicite:5]{index=5}
    logits = model(window_btc)
    p = torch.sigmoid(logits)[0][-1].item()
    print(p)
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_in", required=True)
    ap.add_argument("--video_out", required=True)
    ap.add_argument("--yolo_pose", required=True)      # yolo12x-pose weights
    ap.add_argument("--ckpt", required=True)           # your TCN checkpoint (state_dict)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.5)

    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--thr_on", type=float, default=0.85)
    ap.add_argument("--thr_off", type=float, default=0.50)
    ap.add_argument("--k_on", type=int, default=6)
    ap.add_argument("--k_off", type=int, default=4)
    ap.add_argument("--N", type=int, default=12)

    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # 1) Load YOLO ONCE
    yolo = YOLO(args.yolo_pose)

    # 2) Load TCN ONCE
    # Input dim must match C=K*40+1 (1001 for K=25) :contentReference[oaicite:6]{index=6}
    C = args.K * 40 + 1
    model = EventTCN(input_dim=C, num_layers=4)  # uses BoneMLPEncoder+Pooling+TCN :contentReference[oaicite:7]{index=7}
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    hyst = Hysteresis(HystCfg(args.thr_on, args.thr_off, args.k_on, args.k_off, args.N))

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(fps)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    # rolling buffers over SAMPLED frames
    vec_buf: Deque[np.ndarray] = deque(maxlen=args.win)
    sampled_frame_ids: Deque[int] = deque(maxlen=args.win)
    next_infer_sample_idx = args.win - 1  # first time we can infer
    sample_idx = -1

    last_p: Optional[float] = None
    last_state: bool = False

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        did_infer = False

        # sample every step frames (exactly like your JSON builder) :contentReference[oaicite:8]{index=8}
        if frame_idx % args.step == 0:
            sample_idx += 1

            r = yolo(frame, conf=args.conf, verbose=False)[0]
            det_list = frame_to_detection_list(r, conf_thresh=args.conf)

            # build vector exactly like training: frame_to_vector reads detection_list
            x_t = frame_to_vector({"detection_list": det_list}, K=args.K, num_pers_features=40)  # (1001,)
            vec_buf.append(x_t)
            sampled_frame_ids.append(frame_idx)

            # infer every stride samples
            if len(vec_buf) == args.win and sample_idx >= next_infer_sample_idx:
                win_np = np.stack(list(vec_buf), axis=0)[None, ...]  # (1,16,1001)
                x = torch.from_numpy(win_np).to(device)

                # IMPORTANT: no transpose here; EventTCN expects (B,T,C) :contentReference[oaicite:9]{index=9}
                last_p = predict_prob(model, x)
                
                last_state = hyst.update(last_p)
                did_infer = True

                next_infer_sample_idx += args.stride

        # overlay
        txt1 = f"p={last_p:.3f}" if last_p is not None else "p=..."
        txt2 = "ABNORMAL=ON" if last_state else "ABNORMAL=OFF"
        txt3 = "infer" if did_infer else ""

        cv2.putText(frame, f"{txt2}  {txt1}  {txt3}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame={frame_idx}  step={args.step}  win={args.win} stride={args.stride}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        outv.write(frame)

    cap.release()
    outv.release()
    print(f"[DONE] wrote: {out_path}")


if __name__ == "__main__":
    main()