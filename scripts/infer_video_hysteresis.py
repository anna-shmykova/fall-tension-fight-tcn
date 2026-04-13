# scripts/infer_video_hysteresis.py
from __future__ import annotations

import argparse
import csv
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
    EREZ_BASE_MOTION_DIM,
    EREZ_MOTION_DIM,
    extract_erez_motion_features,
    frame_to_vector,
    motion_extractor_kwargs,
    motion_feature_cfg,
    motion_feature_dim,
)
from src.data.labels import events_to_label
from src.models.tcn import (
    EventTCN,
    MotionTCN,
    infer_encoder_type,
    normalize_event_state_dict,
    state_dict_has_prefix,
)


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

        for interval in self.intervals:
            if interval["start_time_sec"] <= time_sec <= interval["end_time_sec"]:
                return int(interval.get("gt_label", 1))

        if self.source_kind:
            return 0
        return None

    def positive_intervals(
        self,
        fps: Optional[float] = None,
        frame_time_records: Optional[List[Tuple[int, float]]] = None,
    ) -> List[dict]:
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

            mapped_start_frame, mapped_end_frame = interval_time_to_frame_bounds(
                start_time_sec,
                end_time_sec,
                frame_time_records=frame_time_records,
            )
            if start_frame is None:
                start_frame = mapped_start_frame
            if end_frame is None:
                end_frame = mapped_end_frame
            if start_frame is None and fps is not None:
                start_frame = int(round(start_time_sec * float(fps)))
            if end_frame is None and fps is not None:
                end_frame = int(round(end_time_sec * float(fps)))

            item = {
                "start_frame": int(start_frame) if start_frame is not None else None,
                "end_frame": int(end_frame) if end_frame is not None else None,
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


def infer_event_pool_mode(state_dict: dict, configured_pool_mode: Optional[str]) -> str:
    if state_dict_has_prefix(state_dict, "pool.score.") or state_dict_has_prefix(state_dict, "pool.attn.score."):
        return "attn"
    if configured_pool_mode and str(configured_pool_mode) != "attn":
        return str(configured_pool_mode)
    return "mean_max_std"


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
        input_dim = infer_input_dim_from_state_dict(state_dict, default_dim=EREZ_MOTION_DIM)
        input_proj_dim = int(model_cfg.get("input_proj_dim", 0))
        if input_proj_dim == 0 and "input_proj.weight" in state_dict:
            input_proj_dim = int(state_dict["input_proj.weight"].shape[0])

        model = MotionTCN(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", state_dict["head.weight"].shape[1])),
            num_layers=int(model_cfg.get("num_layers", infer_num_layers_from_state_dict(state_dict))),
            dilations=model_cfg.get("dilations"),
            kernel_size=int(model_cfg.get("kernel_size", infer_kernel_size_from_state_dict(state_dict))),
            causal=bool(model_cfg.get("causal", True)),
            norm=str(model_cfg.get("norm", "group")),
            dropout=float(model_cfg.get("dropout", 0.1)),
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
        encoder_type = infer_encoder_type(state_dict=state_dict, configured=model_cfg.get("encoder_type"))
        person_emb_dim = int(model_cfg.get("person_emb_dim", model_cfg.get("mlp_out_dim", 32)))
        encoder_hidden_dim = int(model_cfg.get("encoder_hidden_dim", 128))
        encoder_graph_dim = model_cfg.get("encoder_graph_dim", None)
        encoder_num_layers = int(model_cfg.get("encoder_num_layers", 2))
        pool_mult = {"mean": 1, "max": 1, "mean_max": 2, "mean_max_std": 3, "attn": 1}
        base_scene_in = person_emb_dim * pool_mult[pool_mode]
        actual_tcn_in = infer_tcn_in_channels_from_state_dict(state_dict)

        motion_proj_dim = None
        motion_dim = motion_feature_dim(feature_cfg)
        tcn_input_mode = str(model_cfg.get("tcn_input_mode", "pooled_count"))
        use_person_count = bool(model_cfg.get("use_person_count", True))

        if "motion_proj.weight" in state_dict:
            motion_dim = int(state_dict["motion_proj.weight"].shape[1])
            motion_proj_dim = int(state_dict["motion_proj.weight"].shape[0])

        motion_out_dim = int(motion_proj_dim if motion_proj_dim is not None else (motion_dim if tcn_input_mode == "pooled_count_motion" else 0))

        if actual_tcn_in is not None:
            extra_tcn_in = int(actual_tcn_in - base_scene_in)
            if extra_tcn_in < 0:
                raise ValueError(f"Invalid EventTCN input width: base={base_scene_in} actual={actual_tcn_in}")

            count_dim = int(extra_tcn_in - motion_out_dim)
            if count_dim not in {0, 1}:
                if motion_out_dim == 0 and extra_tcn_in > 1:
                    motion_dim = int(extra_tcn_in - 1)
                    motion_out_dim = motion_dim
                    count_dim = 1
                    tcn_input_mode = "pooled_count_motion"
                else:
                    raise ValueError(
                        f"Could not infer use_person_count/motion layout from checkpoint: "
                        f"base_scene_in={base_scene_in} actual_tcn_in={actual_tcn_in} motion_out_dim={motion_out_dim}"
                    )

            use_person_count = bool(count_dim == 1)
            if motion_out_dim > 0:
                tcn_input_mode = "pooled_count_motion"
                motion_cfg = dict(feature_cfg.get("motion", {}))
                motion_cfg["enabled"] = True
                motion_cfg.setdefault("source", "erez")
                motion_cfg.setdefault("align", "prev")
                feature_cfg["motion"] = motion_cfg
            else:
                motion_dim = 0
                motion_proj_dim = None
                tcn_input_mode = "pooled_count"
        else:
            if motion_out_dim > 0:
                tcn_input_mode = "pooled_count_motion"
                motion_cfg = dict(feature_cfg.get("motion", {}))
                motion_cfg["enabled"] = True
                motion_cfg.setdefault("source", "erez")
                motion_cfg.setdefault("align", "prev")
                feature_cfg["motion"] = motion_cfg
            else:
                motion_dim = 0
                motion_proj_dim = None
                tcn_input_mode = "pooled_count"

        C = args.K * 40 + 1 + motion_dim

        model = EventTCN(
            input_dim=C,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dilations=model_cfg.get("dilations"),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            person_emb_dim=person_emb_dim,
            pool_mode=pool_mode,
            use_attention_readout=use_attention_readout,
            encoder_type=encoder_type,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_graph_dim=int(encoder_graph_dim) if encoder_graph_dim is not None else None,
            encoder_num_layers=encoder_num_layers,
            use_graph=use_graph,
            causal=bool(model_cfg.get("causal", True)),
            norm=str(model_cfg.get("norm", "group")),
            dropout=float(model_cfg.get("dropout", 0.1)),
            motion_dim=motion_dim,
            motion_proj_dim=motion_proj_dim,
            tcn_input_mode=tcn_input_mode,
            use_person_count=use_person_count,
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
            group_events = frame.get("group_events", []) or []
            label = int(int(positive_event_id) in group_events)
            gt.gt_by_frame[frame_id] = label
            gt.gt_by_time_ms[time_ms] = label
            gt_records.append({"frame": frame_id, "time_sec": time_sec, "gt_label": label})

        frame_dt_sec = estimate_time_step_sec(gt_records, default=1.0 / 25.0)
        attach_record_spans(gt_records, default_span_sec=frame_dt_sec, default_span_frames=1)
        gt.intervals = build_intervals(gt_records, label_key="gt_label", span_frames=1, span_sec=frame_dt_sec)
        return gt

    if gt_txt:
        gt.source_path = str(Path(gt_txt).resolve())
        gt.source_kind = "txt"
        gt.intervals = parse_gt_txt_intervals(gt_txt, positive_event_id=positive_event_id)

    return gt


def build_intervals(
    records: List[dict],
    label_key: str,
    span_frames: int = 0,
    span_sec: float = 0.0,
) -> List[dict]:
    intervals = []
    start = None
    end = None

    def close_interval(start_record: dict, end_record: dict) -> None:
        end_frame = end_record.get("record_end_frame")
        if end_frame is None and end_record.get("frame") is not None:
            end_frame = int(end_record["frame"]) + int(span_frames)

        end_time_sec = end_record.get("record_end_time_sec")
        if end_time_sec is None:
            end_time_sec = float(end_record["time_sec"]) + float(span_sec)

        interval = {
            "start_frame": int(start_record["frame"]) if start_record.get("frame") is not None else None,
            "end_frame": int(end_frame) if end_frame is not None else None,
            "start_time_sec": float(start_record["time_sec"]),
            "end_time_sec": float(end_time_sec),
        }
        if start_record.get("event_ids") or end_record.get("event_ids"):
            event_ids = sorted(set(start_record.get("event_ids", [])) | set(end_record.get("event_ids", [])))
            if event_ids:
                interval["event_ids"] = event_ids
        intervals.append(interval)

    for record in records:
        is_pos = bool(record.get(label_key, 0))
        if is_pos:
            if start is None:
                start = record
            end = record
        elif start is not None and end is not None:
            close_interval(start, end)
            start = None
            end = None

    if start is not None and end is not None:
        close_interval(start, end)

    return intervals


def estimate_time_step_sec(records: List[dict], default: float = 0.0) -> float:
    deltas = []
    prev_t = None
    for record in records:
        time_sec = record.get("time_sec")
        if time_sec is None:
            continue
        time_sec = float(time_sec)
        if prev_t is not None:
            dt = time_sec - prev_t
            if dt > 0:
                deltas.append(dt)
        prev_t = time_sec
    if deltas:
        return float(np.median(np.asarray(deltas, dtype=np.float32)))
    return float(default)


def resolve_frame_time_sec(
    cap: cv2.VideoCapture,
    frame_idx: int,
    nominal_fps: float,
    prev_time_sec: Optional[float] = None,
    time_source: str = "auto",
) -> Tuple[float, str]:
    nominal_time_sec = float(frame_idx) / float(nominal_fps) if nominal_fps > 0 else 0.0
    time_source = str(time_source).lower()

    if time_source not in {"auto", "video", "strict_video", "nominal"}:
        raise ValueError(f"Unsupported time_source: {time_source}")

    video_error = "video timestamps unavailable"
    if time_source in {"auto", "video", "strict_video"}:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec is None:
            video_error = "CAP_PROP_POS_MSEC returned None"
        else:
            try:
                video_time_sec = float(pos_msec) / 1000.0
            except (TypeError, ValueError):
                video_time_sec = None
                video_error = f"invalid CAP_PROP_POS_MSEC value: {pos_msec!r}"
            else:
                if not np.isfinite(video_time_sec) or video_time_sec < 0.0:
                    video_error = f"non-finite or negative video timestamp: {video_time_sec!r}"
                elif prev_time_sec is not None and video_time_sec < float(prev_time_sec) - 1e-3:
                    video_error = f"non-monotonic video timestamp {video_time_sec:.6f}s after {float(prev_time_sec):.6f}s"
                else:
                    return float(video_time_sec), "video"

    if time_source == "strict_video":
        raise RuntimeError(f"Failed to resolve a usable video timestamp at frame {frame_idx}: {video_error}")

    if prev_time_sec is not None:
        nominal_time_sec = max(float(nominal_time_sec), float(prev_time_sec))
    return float(nominal_time_sec), "nominal"


def resolve_training_sample_period_sec(cfg: dict) -> Optional[float]:
    if not isinstance(cfg, dict):
        return None

    data_cfg = cfg.get("data", {})
    try:
        train_fps = float(data_cfg.get("fps", 0.0))
    except (TypeError, ValueError):
        train_fps = 0.0
    try:
        train_step = int(data_cfg.get("sample_every_n_frames", 0))
    except (TypeError, ValueError):
        train_step = 0

    if train_fps > 0.0 and train_step > 0:
        return float(train_step) / float(train_fps)
    return None


def resolve_sampling_cfg(args, cfg: dict, nominal_fps: float) -> dict:
    requested_mode = str(getattr(args, "sample_mode", "auto")).lower()
    if requested_mode not in {"auto", "frames", "time"}:
        raise ValueError(f"Unsupported sample_mode: {requested_mode}")

    sample_every_n_frames = max(int(getattr(args, "step", 1)), 1)
    requested_period_sec = getattr(args, "sample_period_sec", None)
    if requested_period_sec is not None:
        requested_period_sec = float(requested_period_sec)
        if requested_period_sec <= 0.0:
            raise ValueError(f"sample_period_sec must be > 0, got {requested_period_sec}")

    training_period_sec = resolve_training_sample_period_sec(cfg)
    nominal_period_sec = float(sample_every_n_frames) / float(nominal_fps) if nominal_fps > 0 else None

    if requested_mode == "frames":
        resolved_mode = "frames"
    elif requested_mode == "time":
        resolved_mode = "time"
    else:
        resolved_mode = "time" if (requested_period_sec is not None or training_period_sec is not None) else "frames"

    if resolved_mode == "time":
        sample_period_sec = requested_period_sec if requested_period_sec is not None else training_period_sec
        if sample_period_sec is None:
            sample_period_sec = nominal_period_sec
        if sample_period_sec is None or sample_period_sec <= 0.0:
            raise ValueError("Unable to resolve a positive time-based sampling period. Pass --sample_period_sec or use --sample_mode=frames.")
    else:
        sample_period_sec = nominal_period_sec

    approx_step_frames = sample_every_n_frames
    if resolved_mode == "time" and nominal_fps > 0 and sample_period_sec is not None:
        approx_step_frames = max(int(round(float(sample_period_sec) * float(nominal_fps))), 1)

    return {
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "sample_every_n_frames": int(sample_every_n_frames),
        "requested_sample_period_sec": requested_period_sec,
        "training_sample_period_sec": training_period_sec,
        "sample_period_sec": sample_period_sec,
        "approx_step_frames": int(approx_step_frames),
    }


def should_sample_frame(
    frame_idx: int,
    frame_time_sec: float,
    sampling_cfg: dict,
    next_sample_time_sec: Optional[float],
    has_sampled_before: bool,
) -> Tuple[bool, Optional[float]]:
    if sampling_cfg["resolved_mode"] == "frames":
        step_frames = int(sampling_cfg["sample_every_n_frames"])
        return bool(frame_idx % step_frames == 0), next_sample_time_sec

    sample_period_sec = float(sampling_cfg["sample_period_sec"])
    if not has_sampled_before:
        return True, float(frame_time_sec) + sample_period_sec

    if next_sample_time_sec is None:
        next_sample_time_sec = float(frame_time_sec)

    if float(frame_time_sec) + 1e-9 < float(next_sample_time_sec):
        return False, float(next_sample_time_sec)

    while float(frame_time_sec) + 1e-9 >= float(next_sample_time_sec):
        next_sample_time_sec = float(next_sample_time_sec) + sample_period_sec
    return True, float(next_sample_time_sec)


def format_sampling_overlay(sampling_cfg: dict) -> str:
    if sampling_cfg["resolved_mode"] == "time":
        return f"sample={float(sampling_cfg['sample_period_sec']):.2f}s"
    return f"sample={int(sampling_cfg['sample_every_n_frames'])}f"


def interval_time_to_frame_bounds(
    start_time_sec: float,
    end_time_sec: float,
    frame_time_records: Optional[List[Tuple[int, float]]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    if not frame_time_records:
        return None, None

    frame_ids = np.asarray([int(frame_id) for frame_id, _ in frame_time_records], dtype=np.int64)
    time_values = np.asarray([float(time_sec) for _, time_sec in frame_time_records], dtype=np.float64)
    if time_values.size == 0:
        return None, None

    start_pos = int(np.searchsorted(time_values, float(start_time_sec), side="left"))
    end_pos = int(np.searchsorted(time_values, float(end_time_sec), side="right") - 1)

    start_pos = min(max(start_pos, 0), len(frame_ids) - 1)
    end_pos = min(max(end_pos, 0), len(frame_ids) - 1)
    if end_pos < start_pos:
        end_pos = start_pos

    return int(frame_ids[start_pos]), int(frame_ids[end_pos])


def attach_record_spans(records: List[dict], default_span_sec: float = 0.0, default_span_frames: int = 0) -> None:
    if not records:
        return

    default_span_sec = float(max(0.0, default_span_sec))
    default_span_frames = int(max(0, default_span_frames))

    for idx, record in enumerate(records):
        next_record = records[idx + 1] if idx + 1 < len(records) else None

        if next_record is not None and next_record.get("time_sec") is not None:
            record_end_time_sec = float(next_record["time_sec"])
        elif record.get("time_sec") is not None:
            record_end_time_sec = float(record["time_sec"]) + default_span_sec
        else:
            record_end_time_sec = None

        if next_record is not None and next_record.get("frame") is not None:
            record_end_frame = int(next_record["frame"])
        elif record.get("frame") is not None:
            record_end_frame = int(record["frame"]) + default_span_frames
        else:
            record_end_frame = None

        record["record_end_time_sec"] = record_end_time_sec
        record["record_end_frame"] = record_end_frame


def interval_duration_sec(interval: dict) -> float:
    return max(0.0, float(interval["end_time_sec"]) - float(interval["start_time_sec"]))


def interval_overlap_sec(a: dict, b: dict) -> float:
    return max(0.0, min(float(a["end_time_sec"]), float(b["end_time_sec"])) - max(float(a["start_time_sec"]), float(b["start_time_sec"])))


def interval_iou(a: dict, b: dict) -> float:
    overlap = interval_overlap_sec(a, b)
    if overlap <= 0.0:
        return 0.0
    union = interval_duration_sec(a) + interval_duration_sec(b) - overlap
    if union <= 0.0:
        return 0.0
    return float(overlap / union)


def normalize_intervals(intervals: List[dict]) -> List[dict]:
    return postprocess_intervals(intervals, merge_gap_sec=0.0, min_duration_sec=0.0)


def total_interval_duration_sec(intervals: List[dict]) -> float:
    normalized = normalize_intervals(intervals)
    return float(sum(interval_duration_sec(interval) for interval in normalized))


def total_overlap_between_interval_sets_sec(a_intervals: List[dict], b_intervals: List[dict]) -> float:
    a_sorted = normalize_intervals(a_intervals)
    b_sorted = normalize_intervals(b_intervals)
    if not a_sorted or not b_sorted:
        return 0.0

    a_idx = 0
    b_idx = 0
    total_overlap_sec = 0.0

    while a_idx < len(a_sorted) and b_idx < len(b_sorted):
        a_item = a_sorted[a_idx]
        b_item = b_sorted[b_idx]
        total_overlap_sec += interval_overlap_sec(a_item, b_item)

        if float(a_item["end_time_sec"]) <= float(b_item["end_time_sec"]):
            a_idx += 1
        else:
            b_idx += 1

    return float(total_overlap_sec)


def postprocess_intervals(intervals: List[dict], merge_gap_sec: float = 0.0, min_duration_sec: float = 0.0) -> List[dict]:
    if not intervals:
        return []

    merge_gap_sec = float(max(0.0, merge_gap_sec))
    min_duration_sec = float(max(0.0, min_duration_sec))

    intervals_sorted = [dict(interval) for interval in sorted(intervals, key=lambda item: (float(item["start_time_sec"]), float(item["end_time_sec"]))) ]
    merged: List[dict] = []

    for interval in intervals_sorted:
        interval.setdefault("parts", 1)
        if not merged:
            merged.append(interval)
            continue

        prev = merged[-1]
        gap_sec = float(interval["start_time_sec"]) - float(prev["end_time_sec"])
        if gap_sec <= merge_gap_sec:
            prev["end_time_sec"] = max(float(prev["end_time_sec"]), float(interval["end_time_sec"]))
            if prev.get("start_frame") is not None and interval.get("start_frame") is not None:
                prev["start_frame"] = min(int(prev["start_frame"]), int(interval["start_frame"]))
            if interval.get("end_frame") is not None:
                if prev.get("end_frame") is None:
                    prev["end_frame"] = int(interval["end_frame"])
                else:
                    prev["end_frame"] = max(int(prev["end_frame"]), int(interval["end_frame"]))
            event_ids = sorted(set(prev.get("event_ids", [])) | set(interval.get("event_ids", [])))
            if event_ids:
                prev["event_ids"] = event_ids
            prev["parts"] = int(prev.get("parts", 1)) + int(interval.get("parts", 1))
        else:
            merged.append(interval)

    if min_duration_sec <= 0.0:
        return merged
    return [interval for interval in merged if interval_duration_sec(interval) >= min_duration_sec]


def match_event_intervals(
    predicted_events: List[dict],
    gt_events: List[dict],
    min_overlap_sec: float = 0.0,
) -> Tuple[List[dict], List[int], List[int]]:
    candidates = []
    min_overlap_sec = float(max(0.0, min_overlap_sec))

    for gt_idx, gt_event in enumerate(gt_events):
        for pred_idx, pred_event in enumerate(predicted_events):
            overlap = interval_overlap_sec(pred_event, gt_event)
            if overlap <= min_overlap_sec:
                continue

            gt_duration = max(interval_duration_sec(gt_event), 1e-12)
            pred_duration = max(interval_duration_sec(pred_event), 1e-12)
            onset_delay_sec = float(pred_event["start_time_sec"]) - float(gt_event["start_time_sec"])
            match = {
                "gt_index": int(gt_idx + 1),
                "pred_index": int(pred_idx + 1),
                "overlap_sec": float(overlap),
                "iou": float(interval_iou(pred_event, gt_event)),
                "gt_coverage": float(overlap / gt_duration),
                "pred_coverage": float(overlap / pred_duration),
                "onset_delay_sec": float(onset_delay_sec),
                "detection_delay_sec": float(max(0.0, onset_delay_sec)),
                "offset_error_sec": float(pred_event["end_time_sec"] - gt_event["end_time_sec"]),
            }
            candidates.append((match["overlap_sec"], match["gt_coverage"], match["iou"], -abs(match["onset_delay_sec"]), gt_idx, pred_idx, match))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches: List[dict] = []

    for _, _, _, _, gt_idx, pred_idx, match in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append(match)

    matches.sort(key=lambda item: (item["gt_index"], item["pred_index"]))
    unmatched_gt = [idx + 1 for idx in range(len(gt_events)) if idx not in matched_gt]
    unmatched_pred = [idx + 1 for idx in range(len(predicted_events)) if idx not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def summarize_gt_fragmentation(
    predicted_events: List[dict],
    gt_events: List[dict],
    min_overlap_sec: float = 0.0,
) -> Tuple[List[int], List[dict]]:
    min_overlap_sec = float(max(0.0, min_overlap_sec))
    fragment_counts: List[int] = []
    gt_fragment_details: List[dict] = []

    for gt_idx, gt_event in enumerate(gt_events, start=1):
        overlaps = []
        for pred_idx, pred_event in enumerate(predicted_events, start=1):
            overlap_sec = interval_overlap_sec(pred_event, gt_event)
            if overlap_sec <= min_overlap_sec:
                continue
            overlaps.append(
                {
                    "pred_index": int(pred_idx),
                    "overlap_sec": float(overlap_sec),
                    "pred_start_time_sec": float(pred_event["start_time_sec"]),
                    "pred_end_time_sec": float(pred_event["end_time_sec"]),
                }
            )

        fragment_counts.append(len(overlaps))
        gt_fragment_details.append(
            {
                "gt_index": int(gt_idx),
                "gt_start_time_sec": float(gt_event["start_time_sec"]),
                "gt_end_time_sec": float(gt_event["end_time_sec"]),
                "n_fragments": int(len(overlaps)),
                "matched_pred_indices": [int(item["pred_index"]) for item in overlaps],
                "overlaps": overlaps,
            }
        )

    return fragment_counts, gt_fragment_details


def compute_event_stats(
    predicted_events: List[dict],
    gt_events: List[dict],
    video_duration_sec: float,
    min_overlap_sec: float = 0.0,
) -> dict:
    matches, unmatched_gt, unmatched_pred = match_event_intervals(
        predicted_events,
        gt_events,
        min_overlap_sec=min_overlap_sec,
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _mean_or_none(values: List[float]):
        return float(np.mean(np.asarray(values, dtype=np.float32))) if values else None

    def _median_or_none(values: List[float]):
        return float(np.median(np.asarray(values, dtype=np.float32))) if values else None

    ious = [float(match["iou"]) for match in matches]
    gt_coverages = [float(match["gt_coverage"]) for match in matches]
    onset_delays = [float(match["onset_delay_sec"]) for match in matches]
    detection_delays = [float(match["detection_delay_sec"]) for match in matches]
    offset_errors = [float(match["offset_error_sec"]) for match in matches]
    gt_fragment_counts, gt_fragment_details = summarize_gt_fragmentation(
        predicted_events,
        gt_events,
        min_overlap_sec=min_overlap_sec,
    )

    total_gt_positive_sec = total_interval_duration_sec(gt_events)
    total_pred_positive_sec = total_interval_duration_sec(predicted_events)
    total_overlap_sec = total_overlap_between_interval_sets_sec(predicted_events, gt_events)
    total_union_sec = max(total_gt_positive_sec + total_pred_positive_sec - total_overlap_sec, 0.0)

    gt_hit_count = int(sum(count > 0 for count in gt_fragment_counts))
    gt_hit_recall = float(gt_hit_count / len(gt_events)) if gt_events else 0.0
    time_coverage = float(total_overlap_sec / total_gt_positive_sec) if total_gt_positive_sec > 0.0 else 0.0
    time_iou = float(total_overlap_sec / total_union_sec) if total_union_sec > 0.0 else 0.0
    mean_fragments_per_gt = _mean_or_none([float(count) for count in gt_fragment_counts])
    median_fragments_per_gt = _median_or_none([float(count) for count in gt_fragment_counts])
    max_fragments_per_gt = int(max(gt_fragment_counts)) if gt_fragment_counts else 0

    false_alarms_per_min = None
    if video_duration_sec > 0:
        false_alarms_per_min = float(fp / (video_duration_sec / 60.0))

    return {
        "n_predicted_events": int(len(predicted_events)),
        "n_gt_events": int(len(gt_events)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarms_per_min": false_alarms_per_min,
        "gt_hit_count": int(gt_hit_count),
        "gt_hit_recall": float(gt_hit_recall),
        "total_gt_positive_sec": float(total_gt_positive_sec),
        "total_pred_positive_sec": float(total_pred_positive_sec),
        "total_overlap_sec": float(total_overlap_sec),
        "time_coverage": float(time_coverage),
        "time_iou": float(time_iou),
        "mean_fragments_per_gt": mean_fragments_per_gt,
        "median_fragments_per_gt": median_fragments_per_gt,
        "max_fragments_per_gt": int(max_fragments_per_gt),
        "mean_iou": _mean_or_none(ious),
        "median_iou": _median_or_none(ious),
        "mean_gt_coverage": _mean_or_none(gt_coverages),
        "median_gt_coverage": _median_or_none(gt_coverages),
        "mean_onset_delay_sec": _mean_or_none(onset_delays),
        "median_onset_delay_sec": _median_or_none(onset_delays),
        "mean_detection_delay_sec": _mean_or_none(detection_delays),
        "median_detection_delay_sec": _median_or_none(detection_delays),
        "mean_offset_error_sec": _mean_or_none(offset_errors),
        "median_offset_error_sec": _median_or_none(offset_errors),
        "matches": matches,
        "unmatched_gt_indices": unmatched_gt,
        "unmatched_pred_indices": unmatched_pred,
        "gt_fragment_counts": gt_fragment_details,
        "min_match_overlap_sec": float(min_overlap_sec),
    }


def compute_binary_stats(
    records: List[dict],
    label_key: str = "gt_label",
    pred_key: str = "pred_state",
    count_key: str = "n_eval_frames",
) -> dict:
    eval_records = [record for record in records if record.get(label_key) is not None and record.get(pred_key) is not None]
    if not eval_records:
        return {count_key: 0}

    tp = fp = tn = fn = 0
    for record in eval_records:
        pred = int(bool(record[pred_key]))
        gt = int(record[label_key])
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
        count_key: len(eval_records),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_probability_stats(
    records: List[dict],
    prob_key: str,
    label_key: str = "gt_label",
    count_key: str = "n_eval_frames",
) -> dict:
    eval_records = [record for record in records if record.get(label_key) is not None and record.get(prob_key) is not None]
    if not eval_records:
        return {count_key: 0}

    y_true = np.asarray([int(record[label_key]) for record in eval_records], dtype=np.int32)
    y_prob = np.asarray([float(record[prob_key]) for record in eval_records], dtype=np.float32)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    sweep = threshold_sweep_binary(y_true, y_prob, start=0.01, end=0.99, step=0.01)
    best = max(sweep, key=lambda row: row["f1"])

    return {
        count_key: int(len(eval_records)),
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


def resolve_window_target_mode(cfg: dict, requested_mode: str) -> str:
    requested_mode = str(requested_mode).lower()
    if requested_mode != "auto":
        if requested_mode not in {"last", "center"}:
            raise ValueError(f"Unsupported window_target_mode: {requested_mode}")
        return requested_mode

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    target_mode = str(
        data_cfg.get(
            "target_mode",
            "last" if bool(model_cfg.get("causal", True)) else "center",
        )
    ).lower()
    if target_mode not in {"last", "center"}:
        raise ValueError(f"Unsupported resolved window target mode: {target_mode}")
    return target_mode


def total_interval_overlap_sec(start_time_sec: float, end_time_sec: float, intervals: List[dict]) -> float:
    if end_time_sec < start_time_sec:
        end_time_sec = start_time_sec

    probe = {
        "start_time_sec": float(start_time_sec),
        "end_time_sec": float(end_time_sec),
    }
    overlap_sec = 0.0
    for interval in intervals:
        overlap_sec += interval_overlap_sec(probe, interval)
    return float(overlap_sec)


def annotate_window_labels(
    records: List[dict],
    gt: GroundTruthSpec,
    gt_positive_intervals: List[dict],
    label_rule: str,
    target_mode: str,
    positive_overlap_fraction: float,
) -> dict:
    label_rule = str(label_rule).lower()
    target_mode = str(target_mode).lower()
    positive_overlap_fraction = float(max(0.0, positive_overlap_fraction))

    summary = {
        "label_rule": label_rule,
        "target_mode": target_mode,
        "positive_overlap_fraction": positive_overlap_fraction,
        "n_candidate_windows": int(len(records)),
        "n_eval_windows": 0,
        "n_positive_windows": 0,
        "n_negative_windows": 0,
        "n_skipped_windows": 0,
    }
    if label_rule == "off" or not gt.source_kind:
        return summary

    for record in records:
        window_start_time_sec = float(record.get("window_start_time_sec", record.get("time_sec", 0.0)))
        window_end_time_sec = float(record.get("window_end_time_sec", record.get("time_sec", window_start_time_sec)))
        if window_end_time_sec < window_start_time_sec:
            window_end_time_sec = window_start_time_sec

        window_start_frame = record.get("window_start_frame")
        window_end_frame = record.get("window_end_frame", record.get("frame"))
        if window_start_frame is not None:
            window_start_frame = int(window_start_frame)
        if window_end_frame is not None:
            window_end_frame = int(window_end_frame)

        if target_mode == "center":
            target_time_sec = 0.5 * (window_start_time_sec + window_end_time_sec)
            if window_start_frame is not None and window_end_frame is not None:
                target_frame = int(round(0.5 * (window_start_frame + window_end_frame)))
            else:
                target_frame = window_end_frame if window_end_frame is not None else window_start_frame
        else:
            target_time_sec = float(record.get("time_sec", window_end_time_sec))
            target_frame = int(record["frame"]) if record.get("frame") is not None else window_end_frame

        overlap_sec = total_interval_overlap_sec(window_start_time_sec, window_end_time_sec, gt_positive_intervals)
        window_duration_sec = max(0.0, window_end_time_sec - window_start_time_sec)
        overlap_fraction = float(overlap_sec / window_duration_sec) if window_duration_sec > 0 else float(overlap_sec > 0.0)
        target_label = gt.label_for(target_frame if target_frame is not None else -1, target_time_sec)

        if label_rule == "target":
            label = target_label
        elif label_rule == "train_like":
            if target_label == 1:
                label = 1
            elif overlap_sec <= 0.0:
                label = 0
            else:
                label = None
        elif label_rule == "any_overlap":
            label = int(overlap_sec > 0.0)
        elif label_rule == "overlap_frac":
            label = int(overlap_fraction >= positive_overlap_fraction)
        else:
            raise ValueError(f"Unsupported window_eval_label_rule: {label_rule}")

        record["window_target_frame"] = target_frame
        record["window_target_time_sec"] = float(target_time_sec)
        record["window_positive_overlap_sec"] = float(overlap_sec)
        record["window_positive_overlap_frac"] = float(overlap_fraction)
        record["gt_window_target_label"] = target_label
        record["gt_window_label"] = label

        if label is None:
            summary["n_skipped_windows"] += 1
        elif int(label) == 1:
            summary["n_eval_windows"] += 1
            summary["n_positive_windows"] += 1
        else:
            summary["n_eval_windows"] += 1
            summary["n_negative_windows"] += 1

    return summary


def write_debug_records_csv(debug_path: Path, records: List[dict]) -> None:
    base_fields = [
        "frame",
        "time_sec",
        "resolved_time_source",
        "window_start_frame",
        "window_end_frame",
        "window_start_time_sec",
        "window_end_time_sec",
        "window_target_frame",
        "window_target_time_sec",
        "prob",
        "prob_smooth",
        "state_score",
        "pred_state",
        "gt_label",
        "gt_window_target_label",
        "gt_window_label",
        "window_positive_overlap_sec",
        "window_positive_overlap_frac",
        "det_count",
        "count_norm",
    ]
    max_motion_dim = max((len(record.get("motion_features") or []) for record in records), default=0)
    fieldnames = base_fields + [f"motion_{idx + 1:02d}" for idx in range(max_motion_dim)]

    with debug_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in base_fields}
            motion_features = list(record.get("motion_features") or [])
            for idx in range(max_motion_dim):
                row[f"motion_{idx + 1:02d}"] = motion_features[idx] if idx < len(motion_features) else ""
            writer.writerow(row)


def write_events_report(events_path: Path, predicted_events: List[dict], gt_events: List[dict], stats: dict) -> None:
    def format_event_line(idx: int, event: dict) -> str:
        duration_sec = interval_duration_sec(event)
        line = f"{idx}. {event['start_time_sec']:.2f}s - {event['end_time_sec']:.2f}s ({duration_sec:.2f}s)"
        if event.get("start_frame") is not None and event.get("end_frame") is not None:
            line += f" (frames {int(event['start_frame'])}..{int(event['end_frame'])})"
        if event.get("event_ids"):
            line += f" events={','.join(str(event_id) for event_id in event['event_ids'])}"
        if event.get("parts"):
            line += f" parts={int(event['parts'])}"
        return line

    def append_stats_block(lines: List[str], title: str, summary: dict) -> None:
        if not summary:
            return
        lines.append(title)
        for key in ("n_eval_frames", "n_eval_windows", "precision", "recall", "f1", "accuracy", "auprc", "auroc", "mean_prob_pos", "mean_prob_neg"):
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

    event_stats = stats.get("event_stats", {})
    if event_stats:
        lines.append("")
        lines.append("Event-level stats:")
        for key in (
            "n_predicted_events",
            "n_gt_events",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "false_alarms_per_min",
            "gt_hit_count",
            "gt_hit_recall",
            "total_gt_positive_sec",
            "total_pred_positive_sec",
            "total_overlap_sec",
            "time_coverage",
            "time_iou",
            "mean_fragments_per_gt",
            "median_fragments_per_gt",
            "max_fragments_per_gt",
            "mean_iou",
            "median_iou",
            "mean_gt_coverage",
            "median_gt_coverage",
            "mean_onset_delay_sec",
            "median_onset_delay_sec",
            "mean_detection_delay_sec",
            "median_detection_delay_sec",
            "mean_offset_error_sec",
            "median_offset_error_sec",
        ):
            if key not in event_stats:
                continue
            value = event_stats[key]
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        matches = event_stats.get("matches", [])
        if matches:
            lines.append("")
            lines.append("Matched events:")
            for match in matches:
                lines.append(
                    f"GT#{match['gt_index']} <-> Pred#{match['pred_index']}: "
                    f"overlap={match['overlap_sec']:.2f}s iou={match['iou']:.3f} "
                    f"gt_coverage={match['gt_coverage']:.3f} onset_delay={match['onset_delay_sec']:.2f}s "
                    f"detect_delay={match['detection_delay_sec']:.2f}s offset_error={match['offset_error_sec']:.2f}s"
                )

        unmatched_gt = event_stats.get("unmatched_gt_indices", [])
        if unmatched_gt:
            lines.append("")
            lines.append("Missed GT events:")
            lines.append(", ".join(f"GT#{idx}" for idx in unmatched_gt))

        unmatched_pred = event_stats.get("unmatched_pred_indices", [])
        if unmatched_pred:
            lines.append("")
            lines.append("Unmatched predicted events:")
            lines.append(", ".join(f"Pred#{idx}" for idx in unmatched_pred))

        gt_fragment_counts = event_stats.get("gt_fragment_counts", [])
        if gt_fragment_counts:
            lines.append("")
            lines.append("GT fragment coverage:")
            for item in gt_fragment_counts:
                matched_pred_indices = item.get("matched_pred_indices", [])
                matched_text = ",".join(f"Pred#{pred_idx}" for pred_idx in matched_pred_indices) if matched_pred_indices else "none"
                lines.append(
                    f"GT#{item['gt_index']}: fragments={item['n_fragments']} matched={matched_text} "
                    f"({item['gt_start_time_sec']:.2f}s-{item['gt_end_time_sec']:.2f}s)"
                )

    window_eval = stats.get("window_eval", {})
    if window_eval:
        lines.append("")
        lines.append("Window-level evaluation:")
        for key in (
            "label_rule",
            "target_mode",
            "positive_overlap_fraction",
            "n_candidate_windows",
            "n_eval_windows",
            "n_positive_windows",
            "n_negative_windows",
            "n_skipped_windows",
        ):
            if key not in window_eval:
                continue
            value = window_eval[key]
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        lines.append("")
        append_stats_block(lines, "Windowed state:", stats.get("window_state_stats", {}))
        lines.append("")
        append_stats_block(lines, "Windowed raw probability:", stats.get("window_prob_raw_stats", {}))
        lines.append("")
        append_stats_block(lines, "Windowed processed probability:", stats.get("window_prob_smooth_stats", {}))

    lines.append("")
    lines.append("Inference settings:")
    lines.append(f"state_method: {stats.get('state_method')}")
    lines.append(f"state_source: {stats.get('state_source')}")
    lines.append(f"smooth_mode: {stats.get('smooth_mode')}")
    lines.append(f"time_source: {stats.get('time_source', 'nominal')}")
    if stats.get('nominal_fps') is not None:
        lines.append(f"nominal_fps: {float(stats.get('nominal_fps')):.4f}")
    lines.append(f"merge_gap_sec: {float(stats.get('merge_gap_sec', 0.0)):.2f}")
    lines.append(f"min_event_sec: {float(stats.get('min_event_sec', 0.0)):.2f}")
    lines.append(f"min_match_overlap_sec: {float(stats.get('min_match_overlap_sec', 0.0)):.2f}")
    lines.append(f"window_eval_label_rule: {stats.get('window_eval_label_rule', 'off')}")
    lines.append(f"window_target_mode: {stats.get('window_target_mode', 'last')}")
    lines.append(f"window_positive_overlap: {float(stats.get('window_positive_overlap', 0.0)):.2f}")
    lines.append(f"n_predicted_events_raw: {stats.get('n_predicted_events_raw', 0)}")
    lines.append(f"n_gt_events_raw: {stats.get('n_gt_events_raw', 0)}")

    lines.append("")
    lines.append("Inference stats:")
    state_title = "Thresholded state:" if stats.get("state_method") == "threshold" else "Hysteresis:"
    append_stats_block(lines, state_title, stats.get("hysteresis_stats", {}))
    lines.append("")
    append_stats_block(lines, "Raw probability:", stats.get("prob_raw_stats", {}))
    lines.append("")
    smooth_label = f"Processed probability ({stats.get('smooth_mode', 'ema')}):"
    append_stats_block(lines, smooth_label, stats.get("prob_smooth_stats", {}))

    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def predict_prob(model: torch.nn.Module, window_btc: torch.Tensor) -> float:
    logits = model(window_btc)
    return float(torch.sigmoid(logits)[0, -1].item())


def update_running_probability(
    raw_prob: float,
    smooth_mode: str,
    ema_beta: float,
    mean_buffer: Deque[float],
    prev_smooth: Optional[float],
) -> float:
    mean_buffer.append(float(raw_prob))
    if smooth_mode == "mean":
        return float(np.mean(np.asarray(mean_buffer, dtype=np.float32)))
    if smooth_mode != "ema":
        raise ValueError(f"Unsupported smooth_mode: {smooth_mode}")
    if prev_smooth is None or ema_beta <= 0.0:
        return float(raw_prob)
    return float(ema_beta * prev_smooth + (1.0 - ema_beta) * raw_prob)


def align_motion_vector_dim(motion_vec: np.ndarray, target_dim: int) -> np.ndarray:
    target_dim = int(target_dim)
    motion_vec = np.asarray(motion_vec, dtype=np.float32)

    if target_dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    if motion_vec.shape[0] == target_dim:
        return motion_vec
    if motion_vec.shape[0] > target_dim:
        return motion_vec[:target_dim].astype(np.float32, copy=False)

    padded = np.zeros((target_dim,), dtype=np.float32)
    padded[: motion_vec.shape[0]] = motion_vec
    return padded


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
    ap.add_argument("--debug_csv", default=None, help="Optional sidecar CSV with one row per inference step, including scores, counts, and motion features.")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=int, default=5, help="Frame-step used when --sample_mode=frames. Also used as a fallback nominal cadence when no time-based cadence can be resolved.")
    ap.add_argument("--sample_mode", choices=["auto", "frames", "time"], default="auto", help="How to choose inference sample points. 'auto' prefers a real-time cadence from --sample_period_sec or the training config, then falls back to every --step frames.")
    ap.add_argument("--sample_period_sec", type=float, default=None, help="Real-time sampling period in seconds. Used directly when --sample_mode=time, and by --sample_mode=auto when provided.")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--time_source", choices=["auto", "video", "strict_video", "nominal"], default="auto", help="How to assign timestamps to decoded frames. 'auto' prefers per-frame video timestamps and falls back to nominal fps. 'strict_video' requires usable decoder timestamps and fails otherwise.")

    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--thr_on", type=float, default=0.90)
    ap.add_argument("--thr_off", type=float, default=0.70)
    ap.add_argument("--k_on", type=int, default=3)
    ap.add_argument("--k_off", type=int, default=3)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--ema_beta", type=float, default=0, help="EMA smoothing beta. Only used when --smooth_mode=ema.")
    ap.add_argument("--smooth_mode", choices=["ema", "mean"], default="ema", help="How to smooth raw probabilities before state thresholding/event evaluation.")
    ap.add_argument("--mean_window", type=int, default=5, help="Number of inference points for moving-mean smoothing when --smooth_mode=mean.")
    ap.add_argument("--state_source", choices=["raw", "smooth"], default="smooth", help="Probability source used for thresholding or hysteresis.")
    ap.add_argument("--disable_hysteresis", action="store_true", help="Threshold the selected state source directly instead of applying hysteresis.")
    ap.add_argument("--merge_gap_sec", type=float, default=0.0, help="Merge predicted and GT intervals separated by at most this gap for incident-level evaluation.")
    ap.add_argument("--min_event_sec", type=float, default=0.0, help="Drop predicted events shorter than this duration after merging.")
    ap.add_argument("--min_match_overlap_sec", type=float, default=0.0, help="Minimum overlap required to match a predicted incident to a GT incident.")
    ap.add_argument("--window_eval_label_rule", choices=["off", "train_like", "target", "any_overlap", "overlap_frac"], default="train_like", help="How to assign GT labels to inference windows. 'train_like' matches the training/test dataset logic by skipping mixed target-negative windows.")
    ap.add_argument("--window_target_mode", choices=["auto", "last", "center"], default="auto", help="Target position used for window labeling when --window_eval_label_rule depends on a target frame.")
    ap.add_argument("--window_positive_overlap", type=float, default=0.3, help="Positive-overlap fraction threshold when --window_eval_label_rule=overlap_frac.")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    yolo = YOLO(args.yolo_pose)

    ckpt_path, payload, cfg = resolve_model_artifacts(args)
    model, model_type, feature_cfg = load_model_from_payload(payload, cfg, args, device=device)
    motion_cfg = motion_feature_cfg(feature_cfg)
    motion_kwargs = motion_extractor_kwargs(feature_cfg)
    if model_type in MOTION_ONLY_MODEL_TYPES:
        motion_dim = int(getattr(model, "input_dim", EREZ_MOTION_DIM))
    else:
        motion_dim = int(getattr(model, "motion_dim", 0))
    gt = load_ground_truth(args.gt_json, args.gt_txt, positive_event_id=args.gt_positive_event_id)

    hyst = Hysteresis(HystCfg(args.thr_on, args.thr_off, args.k_on, args.k_off, args.N))

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nominal_fps = float(fps)
    sampling_cfg = resolve_sampling_cfg(args, cfg, nominal_fps)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.video_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    events_path = Path(args.events_txt).resolve() if args.events_txt else out_path.with_suffix(".events.txt")
    stats_path = Path(args.stats_json).resolve() if args.stats_json else out_path.with_suffix(".stats.json")
    debug_path = Path(args.debug_csv).resolve() if args.debug_csv else out_path.with_suffix(".debug.csv")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    frame_time_records: List[Tuple[int, float]] = []
    frame_time_source_counts: Dict[str, int] = {"video": 0, "nominal": 0}
    vec_buf: Deque[np.ndarray] = deque(maxlen=args.win)
    sample_meta_buf: Deque[Dict[str, float]] = deque(maxlen=args.win)
    prev_sample_frame: Optional[Dict[str, object]] = None
    next_infer_idx = args.win - 1
    sample_idx = -1
    motion_idx = -1
    next_sample_time_sec: Optional[float] = None

    last_p: Optional[float] = None
    last_state = False
    p_smooth: Optional[float] = None
    state_score: Optional[float] = None
    ema_beta = float(args.ema_beta)
    smooth_buf: Deque[float] = deque(maxlen=max(int(args.mean_window), 1))

    frame_idx = -1
    last_frame_time_sec: Optional[float] = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_time_sec, resolved_time_source = resolve_frame_time_sec(
            cap,
            frame_idx=frame_idx,
            nominal_fps=nominal_fps,
            prev_time_sec=last_frame_time_sec,
            time_source=str(args.time_source),
        )
        last_frame_time_sec = float(frame_time_sec)
        frame_time_records.append((frame_idx, float(frame_time_sec)))
        frame_time_source_counts[resolved_time_source] = frame_time_source_counts.get(resolved_time_source, 0) + 1
        did_infer = False
        do_sample, next_sample_time_sec = should_sample_frame(
            frame_idx=frame_idx,
            frame_time_sec=float(frame_time_sec),
            sampling_cfg=sampling_cfg,
            next_sample_time_sec=next_sample_time_sec,
            has_sampled_before=sample_idx >= 0,
        )

        if do_sample:
            sample_idx += 1
            sample_t = float(frame_time_sec)
            n_H = H if H % 32 == 0 else H + (32 - (H%32))
            n_W = W if W % 32 == 0 else W + (32 - (W%32))
            r = yolo(frame, conf=args.conf, verbose=False, imgsz=(n_H, n_W))[0]
            frame, det_list = frame_to_detection_list(frame, W, H, r, conf_thresh=args.conf)
            frame_payload = {
                "f": frame_idx,
                "t": sample_t,
                "group_events": [],
                "detection_list": det_list,
            }
            det_count = int(len(det_list))
            count_norm = float(min(det_count, int(args.K)) / max(int(args.K), 1))
            current_motion_vec: Optional[np.ndarray] = None

            if model_type in MOTION_ONLY_MODEL_TYPES:
                if prev_sample_frame is not None:
                    motion_vec = extract_erez_motion_features(
                        [prev_sample_frame, frame_payload],
                        align=motion_cfg.get("align", "prev"),
                        j_version=feature_cfg.get("erez_json_version", 2.0),
                        extended=(motion_dim > EREZ_BASE_MOTION_DIM),
                        **motion_kwargs,
                    )[-1]
                    motion_vec = align_motion_vector_dim(motion_vec, motion_dim)
                    current_motion_vec = motion_vec.astype(np.float32, copy=False)
                    vec_buf.append(current_motion_vec)
                    sample_meta_buf.append({"frame": int(frame_idx), "time_sec": float(sample_t)})
                    motion_idx += 1

                    if len(vec_buf) == args.win and motion_idx >= next_infer_idx:
                        win_np = np.stack(list(vec_buf), axis=0)[None, ...]
                        x = torch.from_numpy(win_np).to(device)
                        last_p = predict_prob(model, x)

                        p_smooth = update_running_probability(
                            raw_prob=float(last_p),
                            smooth_mode=str(args.smooth_mode),
                            ema_beta=ema_beta,
                            mean_buffer=smooth_buf,
                            prev_smooth=p_smooth,
                        )
                        state_score = float(last_p) if args.state_source == "raw" else float(p_smooth)

                        last_state = update_state(
                            score=float(state_score),
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
                            extended=(motion_dim > EREZ_BASE_MOTION_DIM),
                            **motion_kwargs,
                        )[-1]
                        motion_vec = align_motion_vector_dim(motion_vec, motion_dim)
                    current_motion_vec = motion_vec.astype(np.float32, copy=False)
                    x_t = np.concatenate([x_t, current_motion_vec], axis=0).astype(np.float32)

                vec_buf.append(x_t)
                sample_meta_buf.append({"frame": int(frame_idx), "time_sec": float(sample_t)})
                prev_sample_frame = frame_payload

                if len(vec_buf) == args.win and sample_idx >= next_infer_idx:
                    win_np = np.stack(list(vec_buf), axis=0)[None, ...]
                    x = torch.from_numpy(win_np).to(device)
                    last_p = predict_prob(model, x)

                    p_smooth = update_running_probability(
                        raw_prob=float(last_p),
                        smooth_mode=str(args.smooth_mode),
                        ema_beta=ema_beta,
                        mean_buffer=smooth_buf,
                        prev_smooth=p_smooth,
                    )
                    state_score = float(last_p) if args.state_source == "raw" else float(p_smooth)

                    last_state = update_state(
                        score=float(state_score),
                        hyst=hyst,
                        disable_hysteresis=bool(args.disable_hysteresis),
                        threshold=float(args.thr_on),
                    )
                    did_infer = True
                    next_infer_idx += args.stride

            if did_infer:
                gt_label = gt.label_for(frame_idx, sample_t)
                window_start_meta = sample_meta_buf[0] if sample_meta_buf else {"frame": frame_idx, "time_sec": sample_t}
                records.append(
                    {
                        "frame": frame_idx,
                        "time_sec": sample_t,
                        "resolved_time_source": resolved_time_source,
                        "window_start_frame": int(window_start_meta.get("frame", frame_idx)),
                        "window_end_frame": int(frame_idx),
                        "window_start_time_sec": float(window_start_meta.get("time_sec", sample_t)),
                        "window_end_time_sec": float(sample_t),
                        "prob": last_p,
                        "prob_smooth": p_smooth,
                        "state_score": state_score,
                        "pred_state": int(last_state),
                        "gt_label": gt_label,
                        "det_count": det_count,
                        "count_norm": count_norm,
                        "motion_features": (current_motion_vec.tolist() if current_motion_vec is not None else []),
                    }
                )

        gt_now = gt.label_for(frame_idx, frame_time_sec)
        txt1 = f"p={last_p:.3f} ps={p_smooth:.3f} ss={state_score:.3f}" if last_p is not None and p_smooth is not None and state_score is not None else "p=..."
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
        sampling_text = format_sampling_overlay(sampling_cfg)
        cv2.putText(frame, f"{txt4}  frame={frame_idx} {sampling_text} win={args.win} stride={args.stride}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if txt5 is not None:
            cv2.putText(frame, txt5, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, gt_color, 2, cv2.LINE_AA)

        outv.write(frame)

    cap.release()
    outv.release()

    resolved_sample_period_sec = sampling_cfg.get("sample_period_sec")
    infer_dt_default = float(resolved_sample_period_sec * float(args.stride)) if resolved_sample_period_sec is not None else 0.0
    infer_dt_sec = estimate_time_step_sec(records, default=infer_dt_default)
    infer_span_frames = max(int(sampling_cfg.get("approx_step_frames", max(int(args.step), 1))) * int(args.stride), 1)
    attach_record_spans(records, default_span_sec=infer_dt_sec, default_span_frames=infer_span_frames)
    predicted_events_raw = build_intervals(
        records,
        label_key="pred_state",
        span_frames=infer_span_frames,
        span_sec=infer_dt_sec,
    )
    predicted_events = postprocess_intervals(
        predicted_events_raw,
        merge_gap_sec=float(args.merge_gap_sec),
        min_duration_sec=float(args.min_event_sec),
    )
    gt_events_raw = gt.positive_intervals(fps=nominal_fps, frame_time_records=frame_time_records)
    gt_events = postprocess_intervals(
        gt_events_raw,
        merge_gap_sec=float(args.merge_gap_sec),
        min_duration_sec=0.0,
    )
    window_target_mode = resolve_window_target_mode(cfg, args.window_target_mode)
    window_eval = annotate_window_labels(
        records,
        gt=gt,
        gt_positive_intervals=postprocess_intervals(gt_events_raw, merge_gap_sec=0.0, min_duration_sec=0.0),
        label_rule=str(args.window_eval_label_rule),
        target_mode=window_target_mode,
        positive_overlap_fraction=float(args.window_positive_overlap),
    )

    hyst_stats = compute_binary_stats(records)
    prob_raw_stats = compute_probability_stats(records, prob_key="prob")
    prob_smooth_stats = compute_probability_stats(records, prob_key="prob_smooth")
    window_state_stats = compute_binary_stats(records, label_key="gt_window_label", pred_key="pred_state", count_key="n_eval_windows")
    window_prob_raw_stats = compute_probability_stats(records, prob_key="prob", label_key="gt_window_label", count_key="n_eval_windows")
    window_prob_smooth_stats = compute_probability_stats(records, prob_key="prob_smooth", label_key="gt_window_label", count_key="n_eval_windows")
    video_duration_sec = float(last_frame_time_sec) if last_frame_time_sec is not None else (float((frame_idx + 1) / nominal_fps) if frame_idx >= 0 and nominal_fps > 0 else 0.0)
    event_stats = {}
    if gt.source_kind:
        event_stats = compute_event_stats(
            predicted_events,
            gt_events,
            video_duration_sec=video_duration_sec,
            min_overlap_sec=float(args.min_match_overlap_sec),
        )

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
            "state_source": str(args.state_source),
            "smooth_mode": str(args.smooth_mode),
            "state_threshold": float(args.thr_on),
            "time_source": str(args.time_source),
            "sampling_mode_requested": str(sampling_cfg["requested_mode"]),
            "sampling_mode_resolved": str(sampling_cfg["resolved_mode"]),
            "sample_every_n_frames": int(sampling_cfg["sample_every_n_frames"]),
            "sample_period_sec": float(sampling_cfg["sample_period_sec"]) if sampling_cfg.get("sample_period_sec") is not None else None,
            "training_sample_period_sec": float(sampling_cfg["training_sample_period_sec"]) if sampling_cfg.get("training_sample_period_sec") is not None else None,
            "approx_sample_step_frames": int(sampling_cfg["approx_step_frames"]),
            "resolved_time_sources": dict(frame_time_source_counts),
            "nominal_fps": float(nominal_fps),
            "video_duration_sec": float(video_duration_sec),
            "debug_csv": str(debug_path),
            "merge_gap_sec": float(args.merge_gap_sec),
            "min_event_sec": float(args.min_event_sec),
            "min_match_overlap_sec": float(args.min_match_overlap_sec),
            "window_eval_label_rule": str(args.window_eval_label_rule),
            "window_target_mode": str(window_target_mode),
            "window_positive_overlap": float(args.window_positive_overlap),
            "inference_dt_sec": float(infer_dt_sec),
            "n_predictions": len(records),
            "n_predicted_events_raw": len(predicted_events_raw),
            "n_predicted_events": len(predicted_events),
            "n_gt_events_raw": len(gt_events_raw),
            "n_gt_events": len(gt_events),
            "hysteresis_stats": hyst_stats,
            "prob_raw_stats": prob_raw_stats,
            "prob_smooth_stats": prob_smooth_stats,
            "window_eval": window_eval,
            "window_state_stats": window_state_stats,
            "window_prob_raw_stats": window_prob_raw_stats,
            "window_prob_smooth_stats": window_prob_smooth_stats,
            "event_stats": event_stats,
        }
    )

    write_events_report(events_path, predicted_events, gt_events, stats)
    write_debug_records_csv(debug_path, records)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] wrote video:  {out_path}")
    print(f"[DONE] wrote events: {events_path}")
    print(f"[DONE] wrote stats:  {stats_path}")


if __name__ == "__main__":
    main()
