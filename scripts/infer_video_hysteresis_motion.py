# scripts/infer_video_hysteresis_motion.py
from __future__ import annotations

import argparse
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.data.motion_sequence import MOTION_FEATURE_DIM, adapt_frame_for_erez
from src.erez_files.analyze_json_motion import extract_motion_features
from src.models.tcn import MotionTCN


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

    for box, kpts_norm, conf_kpts in zip(results.boxes, results.keypoints.xyn, results.keypoints.conf):
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < conf_thresh:
            continue

        for keypoint in kpts_norm:
            cx = int(float(keypoint[0]) * W)
            cy = int(float(keypoint[1]) * H)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        if cls_id in [3, 4]:
            continue

        x1, y1, x2, y2 = map(float, box.xyxyn[0])
        kpts_xyc = torch.cat([kpts_norm, conf_kpts.unsqueeze(-1)], dim=-1)

        det_list.append(
            {
                "class": cls_id,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "key_points": kpts_xyc.flatten().tolist(),
            }
        )
    return frame, det_list


def infer_input_dim_from_state_dict(state_dict: dict) -> int:
    if "input_proj.weight" in state_dict:
        return int(state_dict["input_proj.weight"].shape[1])

    for key in ("tcn.0.conv.conv.weight", "tcn.0.conv.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])

    return MOTION_FEATURE_DIM


def infer_kernel_size_from_state_dict(state_dict: dict) -> int:
    for key in ("tcn.0.conv.conv.weight", "tcn.0.conv.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[-1])
    return 3


def infer_num_layers_from_state_dict(state_dict: dict) -> int:
    layer_ids = set()
    for key in state_dict:
        match = re.match(r"tcn\.(\d+)\.", key)
        if match:
            layer_ids.add(int(match.group(1)))
    return max(layer_ids) + 1 if layer_ids else 1


def load_motion_model(ckpt_path: str, device: torch.device) -> MotionTCN:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
        cfg = payload.get("cfg", {})
    else:
        state_dict = payload
        cfg = {}

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model_type = str(model_cfg.get("type", "motion_tcn")).lower()
    if model_type not in {"motion_tcn", "erez_motion_tcn", ""}:
        raise ValueError(f"Checkpoint model.type={model_type!r} is not a motion_tcn checkpoint.")

    input_dim = infer_input_dim_from_state_dict(state_dict)
    input_proj_dim = int(model_cfg.get("input_proj_dim", 0))
    if input_proj_dim == 0 and "input_proj.weight" in state_dict:
        input_proj_dim = int(state_dict["input_proj.weight"].shape[0])

    model = MotionTCN(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", state_dict["head.weight"].shape[1])),
        num_layers=int(model_cfg.get("num_layers", infer_num_layers_from_state_dict(state_dict))),
        kernel_size=int(model_cfg.get("kernel_size", infer_kernel_size_from_state_dict(state_dict))),
        causal=bool(model_cfg.get("causal", True)),
        norm=str(model_cfg.get("norm", "group")),
        input_proj_dim=input_proj_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_prob(model: torch.nn.Module, window_btc: torch.Tensor) -> float:
    logits = model(window_btc)
    return float(torch.sigmoid(logits)[0, -1].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_in", required=True)
    ap.add_argument("--video_out", required=True)
    ap.add_argument("--yolo_pose", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--win", type=int, default=16, help="Number of motion vectors in the temporal window.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--erez_json_version", type=float, default=2.0)

    ap.add_argument("--thr_on", type=float, default=0.85)
    ap.add_argument("--thr_off", type=float, default=0.50)
    ap.add_argument("--k_on", type=int, default=6)
    ap.add_argument("--k_off", type=int, default=4)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--ema_beta", type=float, default=0.85, help="EMA smoothing beta. 0 disables.")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    yolo = YOLO(args.yolo_pose)
    model = load_motion_model(args.ckpt, device=device)
    hyst = Hysteresis(HystCfg(args.thr_on, args.thr_off, args.k_on, args.k_off, args.N))

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    motion_buf: Deque[np.ndarray] = deque(maxlen=args.win)
    prev_frame_erez = None
    next_infer_motion_idx = args.win - 1
    motion_idx = -1
    sample_idx = -1

    last_p: Optional[float] = None
    last_state = False
    p_smooth: Optional[float] = None
    ema_beta = float(args.ema_beta)

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        did_infer = False

        if frame_idx % args.step == 0:
            sample_idx += 1
            sample_t = sample_idx * args.step / fps

            r = yolo(frame, conf=args.conf, verbose=False)[0]
            frame, det_list = frame_to_detection_list(frame, W, H, r, conf_thresh=args.conf)

            frame_curr = {
                "f": frame_idx,
                "t": sample_t,
                "group_events": [],
                "detection_list": det_list,
            }
            frame_curr_erez = adapt_frame_for_erez(frame_curr)

            if prev_frame_erez is not None:
                motion_vec = extract_motion_features(
                    [prev_frame_erez, frame_curr_erez],
                    j_version=float(args.erez_json_version),
                )[0].astype(np.float32)
                motion_buf.append(motion_vec)
                motion_idx += 1

                if len(motion_buf) == args.win and motion_idx >= next_infer_motion_idx:
                    win_np = np.stack(list(motion_buf), axis=0)[None, ...]
                    x = torch.from_numpy(win_np).to(device)
                    last_p = predict_prob(model, x)

                    if ema_beta <= 0.0:
                        p_smooth = last_p
                    elif p_smooth is None:
                        p_smooth = last_p
                    else:
                        p_smooth = ema_beta * p_smooth + (1.0 - ema_beta) * last_p

                    last_state = hyst.update(p_smooth)
                    did_infer = True
                    next_infer_motion_idx += args.stride

            prev_frame_erez = frame_curr_erez

        txt1 = f"p={last_p:.3f} ps={p_smooth:.3f}" if last_p is not None else "p=..."
        txt2 = "ABNORMAL=ON" if last_state else "ABNORMAL=OFF"
        txt3 = "infer" if did_infer else ""

        cv2.putText(
            frame,
            f"{txt2}  {txt1}  {txt3}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame={frame_idx}  step={args.step}  win={args.win} stride={args.stride}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        outv.write(frame)

    cap.release()
    outv.release()
    print(f"[DONE] wrote: {out_path}")


if __name__ == "__main__":
    main()
