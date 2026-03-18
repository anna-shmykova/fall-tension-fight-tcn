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

from src.data.features import (
    extract_erez_motion_features,
    frame_to_vector,
    motion_feature_cfg,
    motion_feature_dim,
)
from src.models.tcn import EventTCN


@dataclass
class HystCfg:
    thr_on: float = 0.90
    thr_off: float = 0.70
    k_on: int = 3
    k_off: int = 3
    N: int = 6


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


def frame_to_detection_list(frame, W, H, results, conf_thresh: float) -> Tuple[np.ndarray, List[dict]]:
    det_list = []
    if results.boxes is None or results.keypoints is None:
        return frame, det_list
    if len(results.boxes) == 0:
        return frame, det_list

    # same logic as your JSON maker: bbox=xyxyn, key_points = (xyn + conf) flattened
    for box, kpts_norm, conf_kpts in zip(results.boxes, results.keypoints.xyn, results.keypoints.conf):
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < conf_thresh:
            continue
        for keypoint in kpts_norm:   
            cx = int(float(keypoint[0])*W)
            cy = int(float(keypoint[1])*H)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
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
    return frame, det_list


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

    ap.add_argument("--thr_on", type=float, default=0.9)
    ap.add_argument("--thr_off", type=float, default=0.7)
    ap.add_argument("--k_on", type=int, default=3)
    ap.add_argument("--k_off", type=int, default=3)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--ema_beta", type=float, default=0.85, help="EMA smoothing beta. 0 disables.")
    
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # 1) Load YOLO ONCE
    yolo = YOLO(args.yolo_pose)

    # 2) Load TCN ONCE
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    feature_cfg = ckpt_cfg.get("features", {})
    model_cfg = ckpt_cfg.get("model", {})
    motion_cfg = motion_feature_cfg(feature_cfg)
    motion_dim = motion_feature_dim(feature_cfg)

    C = args.K * 40 + 1 + motion_dim
    motion_proj_dim = model_cfg.get("motion_proj_dim", model_cfg.get("input_proj_dim", None))
    model = EventTCN(
        input_dim=C,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        mlp_out_dim=int(model_cfg.get("mlp_out_dim", 32)),
        pool_mode=str(model_cfg.get("pool_mode", "attn")),
        causal=bool(model_cfg.get("causal", True)),
        norm=str(model_cfg.get("norm", "group")),
        dropout=float(model_cfg.get("dropout", 0.1)),
        motion_dim=motion_dim,
        motion_proj_dim=int(motion_proj_dim) if motion_proj_dim is not None else None,
        tcn_input_mode=str(model_cfg.get("tcn_input_mode", "pooled_count")),
    )

    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    hyst = Hysteresis(HystCfg(args.thr_on, args.thr_off, args.k_on, args.k_off, args.N))

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print('fps', fps)
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
    prev_sample_frame: Optional[Dict[str, List[dict]]] = None

    last_p: Optional[float] = None
    last_state: bool = False
    
    ema_beta = float(args.ema_beta)
    p_smooth: Optional[float] = None
    
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

            r = yolo(frame, conf=args.conf, verbose=False, imgsz=(H,W))[0]

            frame, det_list = frame_to_detection_list(frame, W, H, r, conf_thresh=args.conf)
            frame_payload = {"detection_list": det_list}

            # build vector exactly like training: frame_to_vector reads detection_list
            x_t = frame_to_vector(frame_payload, K=args.K, num_pers_features=40, cfg=feature_cfg)
            if motion_dim > 0:
                if prev_sample_frame is None:
                    motion_vec = np.zeros((motion_dim,), dtype=np.float32)
                else:
                    motion_vec = extract_erez_motion_features(
                        [prev_sample_frame, frame_payload],
                        align=motion_cfg.get("align", "prev"),
                        j_version=feature_cfg.get("erez_json_version", 2.0),
                    )[-1]
                x_t = np.concatenate([x_t, motion_vec], axis=0).astype(np.float32)
                prev_sample_frame = frame_payload

            vec_buf.append(x_t)
            sampled_frame_ids.append(frame_idx)

            # infer every stride samples
            if len(vec_buf) == args.win and sample_idx >= next_infer_sample_idx:
                win_np = np.stack(list(vec_buf), axis=0)[None, ...]  # (1,16,1001)
                x = torch.from_numpy(win_np).to(device)

                # IMPORTANT: no transpose here; EventTCN expects (B,T,C) :contentReference[oaicite:9]{index=9}
                last_p = predict_prob(model, x)
                
                #last_state = hyst.update(last_p)
                # realization EMA on predictions before hysteresis
                if ema_beta <= 0.0:
                    p_smooth = last_p
                elif p_smooth is None:
                    p_smooth = last_p
                else:
                    p_smooth = ema_beta * p_smooth + (1.0 - ema_beta) * last_p
                
                last_state = hyst.update(p_smooth)  # <-- IMPORTANT: feed smoothed prob
                did_infer = True

                next_infer_sample_idx += args.stride

        # overlay
        #txt1 = f"p={last_p:.3f}" if last_p is not None else "p=..."
        txt1 = f"p={last_p:.3f} ps={p_smooth:.3f}" if last_p is not None else "p=..."
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
