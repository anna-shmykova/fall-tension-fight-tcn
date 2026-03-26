# scripts/infer_video_hysteresis.py
from __future__ import annotations

import argparse
import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
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
TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?\b")
INT_TOKEN_RE = re.compile(r"\b\d+\b")


@dataclass
class GroundTruthSpec:
    source_path: Optional[str] = None
    source_kind: Optional[str] = None
    positive_event_id: int = 4
    gt_by_frame: Dict[int, int] = field(default_factory=dict)
    gt_by_time_ms: Dict[int, int] = field(default_factory=dict)
    intervals: List[dict] = field(default_factory=list)

    def label_for(self, frame_idx: int, time_sec: float) -> Optional[int]:
        if frame_idx in self.gt_by_frame:
            return int(self.gt_by_frame[frame_idx])

        time_ms = int(round(time_sec * 1000))
        if time_ms in self.gt_by_time_ms:
            return int(self.gt_by_time_ms[time_ms])

        if self.source_kind == "txt":
            for interval in self.intervals:
                if interval["start_time_sec"] <= time_sec <= interval["end_time_sec"]:
                    return int(interval["gt_label"])
            return 0

        return None

    def positive_intervals(self, fps: float) -> List[dict]:
        if self.source_kind == "txt":
            src_intervals = [interval for interval in self.intervals if int(interval.get("gt_label", 0)) == 1]
        else:
            src_intervals = self.intervals

        events = []
        for interval in src_intervals:
            start_time_sec = float(interval["start_time_sec"])
            end_time_sec = float(interval["end_time_sec"])
            start_frame = interval.get("start_frame")
            end_frame = interval.get("end_frame")
            if start_frame is None:
                start_frame = int(round(start_time_sec * fps))
            if end_frame is None:
                end_frame = int(round(end_time_sec * fps))

            item = {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
            }
            if interval.get("event_ids"):
                item["event_ids"] = list(interval["event_ids"])
            events.append(item)

        return events


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


def update_state(score: float, hyst: Hysteresis, disable_hysteresis: bool, threshold: float) -> bool:
    if disable_hysteresis:
        return bool(score >= threshold)
    return hyst.update(score)


def parse_timecode(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    elif len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    else:
        raise ValueError(f"Unsupported timecode: {value!r}")
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_gt_txt_intervals(gt_txt: str, positive_event_id: int) -> List[dict]:
    path = Path(gt_txt).resolve()
    intervals = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        matches = list(TIME_TOKEN_RE.finditer(line))
        if len(matches) < 2:
            raise ValueError(f"{path}:{line_no}: expected two timecodes in line: {raw_line!r}")

        start_time_sec = parse_timecode(matches[0].group(0))
        end_time_sec = parse_timecode(matches[1].group(0))
        if end_time_sec < start_time_sec:
            raise ValueError(f"{path}:{line_no}: end time is before start time: {raw_line!r}")

        suffix = line[matches[1].end():]
        suffix = re.sub(r"(?i)\bduration\b\s*[:=]?\s*\d+(?:[.,]\d+)?", " ", suffix, count=1)
        event_ids = [int(tok.group(0)) for tok in INT_TOKEN_RE.finditer(suffix)]

        intervals.append(
            {
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "event_ids": sorted(set(event_ids)),
                "gt_label": int(int(positive_event_id) in event_ids),
                "raw_line": raw_line.strip(),
                "line_no": line_no,
            }
        )

    return intervals


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or not np.any(y_true == 1):
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def threshold_sweep_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.01,
    end: float = 0.99,
    step: float = 0.01,
) -> List[Dict[str, float]]:
    thresholds = np.arange(start, end + 1e-9, step)
    results = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        results.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return results


def confusion_stats_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true_i == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true_i == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_i == 1)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


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


def load_ground_truth(gt_json: Optional[str], gt_txt: Optional[str], positive_event_id: int) -> GroundTruthSpec:
    if gt_json and gt_txt:
        raise ValueError("Pass either --gt_json or --gt_txt, not both.")

    gt = GroundTruthSpec(positive_event_id=int(positive_event_id))
    if gt_json:
        path = Path(gt_json).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        gt.source_path = str(path)
        gt.source_kind = "json"

        gt_records = []
        for idx, frame in enumerate(data.get("frames", [])):
            frame_id = int(frame.get("f", idx))
            time_sec = float(frame.get("t", frame_id / 25.0))
            time_ms = int(round(time_sec * 1000))
            label = int(events_to_label(frame))
            gt.gt_by_frame[frame_id] = label
            gt.gt_by_time_ms[time_ms] = label
            gt_records.append({"frame": frame_id, "time_sec": time_sec, "gt_label": label})

        gt.intervals = build_intervals(gt_records, label_key="gt_label")
        return gt

    if gt_txt:
        gt.source_path = str(Path(gt_txt).resolve())
        gt.source_kind = "txt"
        gt.intervals = parse_gt_txt_intervals(gt_txt, positive_event_id=positive_event_id)

    return gt


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


def compute_probability_stats(records: List[dict], prob_key: str) -> dict:
    eval_records = [record for record in records if record.get("gt_label") is not None and record.get(prob_key) is not None]
    if not eval_records:
        return {"n_eval_frames": 0}

    y_true = np.asarray([int(record["gt_label"]) for record in eval_records], dtype=np.int32)
    y_prob = np.asarray([float(record[prob_key]) for record in eval_records], dtype=np.float32)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    sweep = threshold_sweep_binary(y_true, y_prob, start=0.01, end=0.99, step=0.01)
    best = max(sweep, key=lambda row: row["f1"])

    return {
        "n_eval_frames": int(len(eval_records)),
        "auprc": compute_auprc(y_true, y_prob),
        "auroc": compute_auroc(y_true, y_prob),
        "mean_prob_pos": float(np.mean(y_prob[pos_mask])) if np.any(pos_mask) else None,
        "mean_prob_neg": float(np.mean(y_prob[neg_mask])) if np.any(neg_mask) else None,
        "best_f1": {key: float(value) for key, value in best.items()},
        "best_threshold_stats": confusion_stats_at_threshold(y_true, y_prob, threshold=float(best["threshold"])),
        "fixed_thresholds": {
            "0.3": confusion_stats_at_threshold(y_true, y_prob, threshold=0.3),
            "0.5": confusion_stats_at_threshold(y_true, y_prob, threshold=0.5),
            "0.7": confusion_stats_at_threshold(y_true, y_prob, threshold=0.7),
        },
        "threshold_sweep": sweep,
    }


def write_events_report(events_path: Path, predicted_events: List[dict], gt_events: List[dict], stats: dict) -> None:
    def format_event_line(idx: int, event: dict) -> str:
        line = f"{idx}. {event['start_time_sec']:.2f}s - {event['end_time_sec']:.2f}s"
        if event.get("start_frame") is not None and event.get("end_frame") is not None:
            line += f" (frames {int(event['start_frame'])}..{int(event['end_frame'])})"
        if event.get("event_ids"):
            line += f" events={','.join(str(event_id) for event_id in event['event_ids'])}"
        return line

    def append_stats_block(lines: List[str], title: str, summary: dict) -> None:
        if not summary:
            return
        lines.append(title)
        for key in ("n_eval_frames", "precision", "recall", "f1", "accuracy", "auprc", "auroc", "mean_prob_pos", "mean_prob_neg"):
            if key not in summary:
                continue
            value = summary[key]
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        best = summary.get("best_threshold_stats")
        if isinstance(best, dict):
            lines.append(
                "best_threshold: "
                f"thr={best['threshold']:.2f} P={best['precision']:.3f} "
                f"R={best['recall']:.3f} F1={best['f1']:.3f} "
                f"ACC={best['accuracy']:.3f}"
            )

    lines = ["Predicted events:"]
    if predicted_events:
        for idx, event in enumerate(predicted_events, start=1):
            lines.append(format_event_line(idx, event))
    else:
        lines.append("none")

    lines.append("")
    lines.append("Ground-truth events:")
    if gt_events:
        for idx, event in enumerate(gt_events, start=1):
            lines.append(format_event_line(idx, event))
    else:
        lines.append("none")

    lines.append("")
    lines.append("Inference stats:")
    state_title = "Thresholded state:" if stats.get("state_method") == "threshold" else "Hysteresis:"
    append_stats_block(lines, state_title, stats.get("hysteresis_stats", {}))
    lines.append("")
    append_stats_block(lines, "Raw probability:", stats.get("prob_raw_stats", {}))
    lines.append("")
    append_stats_block(lines, "Smoothed probability:", stats.get("prob_smooth_stats", {}))

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
    ap.add_argument("--gt_txt", default=None, help="Optional text file with GT intervals. Supports 1.txt-style lines like '00:00:05, 00:00:15, 4'.")
    ap.add_argument("--gt_positive_event_id", type=int, default=4, help="Event id from GT annotations to treat as the positive class.")
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
    ap.add_argument("--ema_beta", type=float, default=0, help="EMA smoothing beta. 0 disables.")
    ap.add_argument("--disable_hysteresis", action="store_true", help="Threshold the score directly instead of applying hysteresis over time.")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    yolo = YOLO(args.yolo_pose)

    ckpt_path, payload, cfg = resolve_model_artifacts(args)
    model, model_type, feature_cfg = load_model_from_payload(payload, cfg, args, device=device)
    motion_cfg = motion_feature_cfg(feature_cfg)
    motion_dim = motion_feature_dim(feature_cfg)
    gt = load_ground_truth(args.gt_json, args.gt_txt, positive_event_id=args.gt_positive_event_id)

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

                        last_state = update_state(
                            score=float(p_smooth),
                            hyst=hyst,
                            disable_hysteresis=bool(args.disable_hysteresis),
                            threshold=float(args.thr_on),
                        )
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

                    last_state = update_state(
                        score=float(p_smooth),
                        hyst=hyst,
                        disable_hysteresis=bool(args.disable_hysteresis),
                        threshold=float(args.thr_on),
                    )
                    did_infer = True
                    next_infer_idx += args.stride

            if did_infer:
                gt_label = gt.label_for(frame_idx, sample_t)
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

        frame_time_sec = frame_idx / fps
        gt_now = gt.label_for(frame_idx, frame_time_sec)
        txt1 = f"p={last_p:.3f} ps={p_smooth:.3f}" if last_p is not None else "p=..."
        txt2 = "ABNORMAL=ON" if last_state else "ABNORMAL=OFF"
        txt3 = "infer" if did_infer else ""
        txt4 = f"model={model_type}"
        if gt.source_kind:
            txt5 = f"GT(event {gt.positive_event_id})={'ON' if gt_now == 1 else 'OFF' if gt_now == 0 else '?'}"
            gt_color = (0, 0, 255) if gt_now == 1 else (255, 255, 255)
        else:
            txt5 = None
            gt_color = (255, 255, 255)

        cv2.putText(frame, f"{txt2}  {txt1}  {txt3}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{txt4}  frame={frame_idx} step={args.step} win={args.win} stride={args.stride}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if txt5 is not None:
            cv2.putText(frame, txt5, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, gt_color, 2, cv2.LINE_AA)

        outv.write(frame)

    cap.release()
    outv.release()

    predicted_events = build_intervals(records, label_key="pred_state")
    gt_events = gt.positive_intervals(fps=fps)

    hyst_stats = compute_binary_stats(records)
    prob_raw_stats = compute_probability_stats(records, prob_key="prob")
    prob_smooth_stats = compute_probability_stats(records, prob_key="prob_smooth")

    stats = dict(hyst_stats)
    stats.update(
        {
            "model_type": model_type,
            "video_in": str(Path(args.video_in).resolve()),
            "video_out": str(out_path),
            "checkpoint": str(ckpt_path),
            "run_dir": str(Path(args.run_dir).resolve()) if args.run_dir else None,
            "gt_json": str(Path(args.gt_json).resolve()) if args.gt_json else None,
            "gt_txt": str(Path(args.gt_txt).resolve()) if args.gt_txt else None,
            "gt_source_kind": gt.source_kind,
            "gt_positive_event_id": int(gt.positive_event_id),
            "hysteresis_enabled": bool(not args.disable_hysteresis),
            "state_method": "threshold" if args.disable_hysteresis else "hysteresis",
            "state_threshold": float(args.thr_on),
            "n_predictions": len(records),
            "n_predicted_events": len(predicted_events),
            "n_gt_events": len(gt_events),
            "hysteresis_stats": hyst_stats,
            "prob_raw_stats": prob_raw_stats,
            "prob_smooth_stats": prob_smooth_stats,
        }
    )

    write_events_report(events_path, predicted_events, gt_events, stats)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] wrote video:  {out_path}")
    print(f"[DONE] wrote events: {events_path}")
    print(f"[DONE] wrote stats:  {stats_path}")


if __name__ == "__main__":
    main()
