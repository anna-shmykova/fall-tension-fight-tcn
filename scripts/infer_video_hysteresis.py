# scripts/infer_video_hysteresis.py
from __future__ import annotations

import argparse
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

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
from src.data.labels import events_to_label
from src.models.tcn import EventTCN, MotionTCN


MOTION_ONLY_MODEL_TYPES = {"motion_tcn", "erez_motion_tcn"}


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


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def infer_input_dim_from_state_dict(state_dict: dict, default_dim: int) -> int:
    if "input_proj.weight" in state_dict:
        return int(state_dict["input_proj.weight"].shape[1])

    for key in ("tcn.0.conv.conv.weight", "tcn.0.conv.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])

    return default_dim


def infer_kernel_size_from_state_dict(state_dict: dict) -> int:
    for key in ("tcn.0.conv.conv.weight", "tcn.0.conv.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[-1])
    return 3


def infer_tcn_in_channels_from_state_dict(state_dict: dict) -> Optional[int]:
    for key in ("tcn.0.conv.conv.weight", "tcn.0.conv.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return None


def infer_num_layers_from_state_dict(state_dict: dict) -> int:
    layer_ids = set()
    for key in state_dict:
        match = re.match(r"tcn\.(\d+)\.", key)
        if match:
            layer_ids.add(int(match.group(1)))
    return max(layer_ids) + 1 if layer_ids else 1


def state_dict_has_prefix(state_dict: dict, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state_dict)


def infer_event_pool_mode(state_dict: dict, configured_pool_mode: Optional[str]) -> str:
    if state_dict_has_prefix(state_dict, "pool.score.") or state_dict_has_prefix(state_dict, "pool.attn.score."):
        return "attn"
    if configured_pool_mode and str(configured_pool_mode) != "attn":
        return str(configured_pool_mode)
    return "mean_max_std"


def normalize_event_state_dict(state_dict: dict) -> dict:
    if state_dict_has_prefix(state_dict, "pool.score.") and not state_dict_has_prefix(state_dict, "pool.attn.score."):
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("pool.score."):
                remapped[key.replace("pool.score.", "pool.attn.score.", 1)] = value
            else:
                remapped[key] = value
        return remapped
    return state_dict


def resolve_model_artifacts(args) -> Tuple[Path, dict, dict]:
    if not args.ckpt and not args.run_dir:
        raise ValueError("Pass --ckpt or --run_dir")

    cfg = {}
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not args.ckpt:
            args.ckpt = str(run_dir / "checkpoints" / "best.pt")
        cfg_path = run_dir / "config_resolved.yaml"
        if cfg_path.exists():
            cfg = load_yaml(cfg_path)

    ckpt_path = Path(args.ckpt).resolve()
    payload = torch.load(ckpt_path, map_location="cpu")
    payload_cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
    if payload_cfg:
        cfg = payload_cfg

    return ckpt_path, payload, cfg


def load_model_from_payload(payload: dict, cfg: dict, args, device: torch.device):
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    feature_cfg = cfg.get("features", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model_type = str(model_cfg.get("type", "tcn")).lower()

    if model_type in MOTION_ONLY_MODEL_TYPES:
        input_dim = infer_input_dim_from_state_dict(state_dict, default_dim=18)
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
    else:
        state_dict = normalize_event_state_dict(state_dict)
        feature_cfg = dict(feature_cfg)
        has_graph = state_dict_has_prefix(state_dict, "graph.")
        has_attn = state_dict_has_prefix(state_dict, "pool.score.")
        if not has_attn:
            has_attn = state_dict_has_prefix(state_dict, "pool.attn.score.")
        use_graph = has_graph
        pool_mode = infer_event_pool_mode(state_dict, model_cfg.get("pool_mode"))
        use_attention_readout = True if has_attn else False if model_cfg.get("use_attention_readout") is True else None
        mlp_out_dim = int(model_cfg.get("mlp_out_dim", 32))
        pool_mult = {"mean": 1, "max": 1, "mean_max": 2, "mean_max_std": 3, "attn": 1}
        base_tcn_in = mlp_out_dim * pool_mult[pool_mode] + 1
        actual_tcn_in = infer_tcn_in_channels_from_state_dict(state_dict)

        motion_proj_dim = None
        motion_dim = motion_feature_dim(feature_cfg)
        tcn_input_mode = str(model_cfg.get("tcn_input_mode", "pooled_count"))

        if actual_tcn_in is not None and actual_tcn_in > base_tcn_in:
            tcn_input_mode = "pooled_count_motion"
            if "motion_proj.weight" in state_dict:
                motion_dim = int(state_dict["motion_proj.weight"].shape[1])
                motion_proj_dim = int(state_dict["motion_proj.weight"].shape[0])
            else:
                motion_dim = int(actual_tcn_in - base_tcn_in)
                motion_proj_dim = None

            motion_cfg = dict(feature_cfg.get("motion", {}))
            motion_cfg["enabled"] = True
            motion_cfg.setdefault("source", "erez")
            motion_cfg.setdefault("align", "prev")
            feature_cfg["motion"] = motion_cfg
        else:
            motion_dim = 0
            tcn_input_mode = "pooled_count"

        C = args.K * 40 + 1 + motion_dim

        model = EventTCN(
            input_dim=C,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            mlp_out_dim=mlp_out_dim,
            pool_mode=pool_mode,
            use_attention_readout=use_attention_readout,
            use_graph=use_graph,
            causal=bool(model_cfg.get("causal", True)),
            norm=str(model_cfg.get("norm", "group")),
            dropout=float(model_cfg.get("dropout", 0.1)),
            motion_dim=motion_dim,
            motion_proj_dim=motion_proj_dim,
            tcn_input_mode=tcn_input_mode,
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, model_type, feature_cfg


def load_ground_truth(gt_json: Optional[str]):
    if not gt_json:
        return None, {}, {}

    path = Path(gt_json).resolve()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gt_by_frame = {}
    gt_by_time_ms = {}
    for idx, frame in enumerate(data.get("frames", [])):
        frame_id = int(frame.get("f", idx))
        time_ms = int(round(float(frame.get("t", 0.0)) * 1000))
        label = int(events_to_label(frame))
        gt_by_frame[frame_id] = label
        gt_by_time_ms[time_ms] = label

    return data, gt_by_frame, gt_by_time_ms


def gt_label_for_frame(frame_idx: int, time_sec: float, gt_by_frame: dict, gt_by_time_ms: dict):
    if frame_idx in gt_by_frame:
        return gt_by_frame[frame_idx]
    return gt_by_time_ms.get(int(round(time_sec * 1000)))


def build_intervals(records: List[dict], label_key: str) -> List[dict]:
    intervals = []
    start = None
    end = None

    for record in records:
        is_pos = bool(record.get(label_key, 0))
        if is_pos:
            if start is None:
                start = record
            end = record
        elif start is not None and end is not None:
            intervals.append(
                {
                    "start_frame": int(start["frame"]),
                    "end_frame": int(end["frame"]),
                    "start_time_sec": float(start["time_sec"]),
                    "end_time_sec": float(end["time_sec"]),
                }
            )
            start = None
            end = None

    if start is not None and end is not None:
        intervals.append(
            {
                "start_frame": int(start["frame"]),
                "end_frame": int(end["frame"]),
                "start_time_sec": float(start["time_sec"]),
                "end_time_sec": float(end["time_sec"]),
            }
        )

    return intervals


def compute_binary_stats(records: List[dict]) -> dict:
    eval_records = [record for record in records if record.get("gt_label") is not None]
    if not eval_records:
        return {"n_eval_frames": 0}

    tp = fp = tn = fn = 0
    for record in eval_records:
        pred = int(bool(record["pred_state"]))
        gt = int(record["gt_label"])
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(eval_records) if eval_records else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_eval_frames": len(eval_records),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def write_events_report(events_path: Path, predicted_events: List[dict], gt_events: List[dict], stats: dict) -> None:
    lines = ["Predicted events:"]
    if predicted_events:
        for idx, event in enumerate(predicted_events, start=1):
            lines.append(
                f"{idx}. {event['start_time_sec']:.2f}s - {event['end_time_sec']:.2f}s "
                f"(frames {event['start_frame']}..{event['end_frame']})"
            )
    else:
        lines.append("none")

    lines.append("")
    lines.append("Ground-truth events:")
    if gt_events:
        for idx, event in enumerate(gt_events, start=1):
            lines.append(
                f"{idx}. {event['start_time_sec']:.2f}s - {event['end_time_sec']:.2f}s "
                f"(frames {event['start_frame']}..{event['end_frame']})"
            )
    else:
        lines.append("none")

    lines.append("")
    lines.append("Inference stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")

    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def predict_prob(model: torch.nn.Module, window_btc: torch.Tensor) -> float:
    logits = model(window_btc)
    return float(torch.sigmoid(logits)[0, -1].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_in", required=True)
    ap.add_argument("--video_out", required=True)
    ap.add_argument("--yolo_pose", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run_dir", default=None, help="Training run folder. If set, uses checkpoints/best.pt and config_resolved.yaml.")
    ap.add_argument("--gt_json", default=None, help="Optional annotated JSON for ground-truth comparison.")
    ap.add_argument("--events_txt", default=None, help="Optional sidecar text file with predicted/GT events.")
    ap.add_argument("--stats_json", default=None, help="Optional sidecar JSON with inference-vs-GT statistics.")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.5)

    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--thr_on", type=float, default=0.90)
    ap.add_argument("--thr_off", type=float, default=0.70)
    ap.add_argument("--k_on", type=int, default=3)
    ap.add_argument("--k_off", type=int, default=3)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--ema_beta", type=float, default=0.85, help="EMA smoothing beta. 0 disables.")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    yolo = YOLO(args.yolo_pose)

    ckpt_path, payload, cfg = resolve_model_artifacts(args)
    model, model_type, feature_cfg = load_model_from_payload(payload, cfg, args, device=device)
    motion_cfg = motion_feature_cfg(feature_cfg)
    motion_dim = motion_feature_dim(feature_cfg)

    gt_data, gt_by_frame, gt_by_time_ms = load_ground_truth(args.gt_json)

    hyst = Hysteresis(HystCfg(args.thr_on, args.thr_off, args.k_on, args.k_off, args.N))

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.video_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    events_path = Path(args.events_txt).resolve() if args.events_txt else out_path.with_suffix(".events.txt")
    stats_path = Path(args.stats_json).resolve() if args.stats_json else out_path.with_suffix(".stats.json")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    vec_buf: Deque[np.ndarray] = deque(maxlen=args.win)
    prev_sample_frame: Optional[Dict[str, object]] = None
    next_infer_idx = args.win - 1
    sample_idx = -1
    motion_idx = -1

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
            sample_t = frame_idx / fps

            r = yolo(frame, conf=args.conf, verbose=False, imgsz=(H, W))[0]
            frame, det_list = frame_to_detection_list(frame, W, H, r, conf_thresh=args.conf)
            frame_payload = {
                "f": frame_idx,
                "t": sample_t,
                "group_events": [],
                "detection_list": det_list,
            }

            if model_type in MOTION_ONLY_MODEL_TYPES:
                if prev_sample_frame is not None:
                    motion_vec = extract_erez_motion_features(
                        [prev_sample_frame, frame_payload],
                        align=motion_cfg.get("align", "prev"),
                        j_version=feature_cfg.get("erez_json_version", 2.0),
                    )[-1]
                    vec_buf.append(motion_vec.astype(np.float32))
                    motion_idx += 1

                    if len(vec_buf) == args.win and motion_idx >= next_infer_idx:
                        win_np = np.stack(list(vec_buf), axis=0)[None, ...]
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
                        next_infer_idx += args.stride

                prev_sample_frame = frame_payload
            else:
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

                vec_buf.append(x_t)
                prev_sample_frame = frame_payload

                if len(vec_buf) == args.win and sample_idx >= next_infer_idx:
                    win_np = np.stack(list(vec_buf), axis=0)[None, ...]
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
                    next_infer_idx += args.stride

            if did_infer:
                gt_label = gt_label_for_frame(frame_idx, sample_t, gt_by_frame, gt_by_time_ms)
                records.append(
                    {
                        "frame": frame_idx,
                        "time_sec": sample_t,
                        "prob": last_p,
                        "prob_smooth": p_smooth,
                        "pred_state": int(last_state),
                        "gt_label": gt_label,
                    }
                )

        txt1 = f"p={last_p:.3f} ps={p_smooth:.3f}" if last_p is not None else "p=..."
        txt2 = "ABNORMAL=ON" if last_state else "ABNORMAL=OFF"
        txt3 = "infer" if did_infer else ""
        txt4 = f"model={model_type}"

        cv2.putText(frame, f"{txt2}  {txt1}  {txt3}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{txt4}  frame={frame_idx} step={args.step} win={args.win} stride={args.stride}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        outv.write(frame)

    cap.release()
    outv.release()

    predicted_events = build_intervals(records, label_key="pred_state")

    gt_events = []
    if gt_data is not None:
        gt_records = []
        for idx, frame in enumerate(gt_data.get("frames", [])):
            frame_id = int(frame.get("f", idx))
            gt_records.append(
                {
                    "frame": frame_id,
                    "time_sec": float(frame.get("t", frame_id / fps)),
                    "gt_label": int(events_to_label(frame)),
                }
            )
        gt_events = build_intervals(gt_records, label_key="gt_label")

    stats = compute_binary_stats(records)
    stats.update(
        {
            "model_type": model_type,
            "video_in": str(Path(args.video_in).resolve()),
            "video_out": str(out_path),
            "checkpoint": str(ckpt_path),
            "run_dir": str(Path(args.run_dir).resolve()) if args.run_dir else None,
            "gt_json": str(Path(args.gt_json).resolve()) if args.gt_json else None,
            "n_predictions": len(records),
        }
    )

    write_events_report(events_path, predicted_events, gt_events, stats)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] wrote video:  {out_path}")
    print(f"[DONE] wrote events: {events_path}")
    print(f"[DONE] wrote stats:  {stats_path}")


if __name__ == "__main__":
    main()
