#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.build_sequence import build_sequence
from src.data.dataset import EventJsonDataset, MotionJsonDataset, label_window, resolve_target_index
from src.data.features import motion_feature_dim
from src.data.json_io import read_json_frames
from src.data.labels import events_to_label, resolve_label_cfg
from src.data.motion_sequence import build_motion_sequence
from src.data.splits import read_paths_txt
from src.models.tcn import EventTCN, MotionTCN, infer_encoder_type, normalize_event_state_dict
from src.train import (
    aggregate_window_logits,
    build_dataset,
    build_probability_calibration_report,
    resolve_window_cfg,
    build_per_video_rows,
    evaluate_binary,
    evaluate_paths_individually,
    fpr_from_stats,
    load_yaml,
    make_loader,
    save_probability_calibration_artifacts,
    safe_float,
    summarize_video_scores,
)
from src.utils.metrics import (
    apply_platt_scaling,
    apply_temperature_scaling,
    compute_pr_points,
    compute_roc_points,
    confusion_stats_at_threshold,
    save_confusion_matrix_image,
    save_event_probability_timeline_image,
    save_pr_curve_image,
    save_roc_curve_image,
    save_rows_csv,
    save_summary_csv,
    sigmoid_from_logits,
    save_threshold_sweep_csv,
)


def resolve_optional_path(path_str: str | None, base: Path) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else (base / path).resolve()


def resolve_cli_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    return path.resolve()


def resolve_device(requested: str | None, cfg_device: str | None) -> torch.device:
    choice = str(requested or cfg_device or "cpu")
    if choice.startswith("cuda") and torch.cuda.is_available():
        return torch.device(choice)
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dataset_cls(cfg: dict[str, Any]):
    model_type = str(cfg.get("model", {}).get("type", "tcn")).lower()
    return MotionJsonDataset if model_type in {"motion_tcn", "erez_motion_tcn"} else EventJsonDataset


def resolve_label_cfg_from_root(cfg: dict[str, Any]) -> dict[str, Any]:
    label_cfg = dict(cfg.get("labels", {}) or {})
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    if "mode" not in label_cfg and data_cfg.get("label_mode") is not None:
        label_cfg["mode"] = data_cfg.get("label_mode")
    return resolve_label_cfg(label_cfg)


def build_model(cfg: dict[str, Any], input_dim: int, state_dict: dict[str, Any] | None = None) -> torch.nn.Module:
    feature_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "tcn")).lower()
    motion_dim = motion_feature_dim(feature_cfg)
    state_dict = normalize_event_state_dict(state_dict or {})

    tcn_input_mode = str(model_cfg.get("tcn_input_mode", "pooled_count"))
    motion_proj_dim = model_cfg.get("motion_proj_dim", model_cfg.get("input_proj_dim", None))
    use_attention_readout = model_cfg.get("use_attention_readout", None)
    use_graph = bool(model_cfg.get("use_graph", True))
    encoder_type = infer_encoder_type(state_dict=state_dict, configured=model_cfg.get("encoder_type"))
    person_emb_dim = int(model_cfg.get("person_emb_dim", model_cfg.get("mlp_out_dim", 32)))
    encoder_hidden_dim = int(model_cfg.get("encoder_hidden_dim", 128))
    encoder_graph_dim = model_cfg.get("encoder_graph_dim", None)
    encoder_num_layers = int(model_cfg.get("encoder_num_layers", 2))

    if model_type not in {"motion_tcn", "erez_motion_tcn"} and tcn_input_mode == "pooled_count_motion" and motion_dim == 0:
        raise ValueError(
            "model.tcn_input_mode='pooled_count_motion' requires motion features. "
            "Set features.motion.enabled=true or change model.tcn_input_mode to 'pooled_count'."
        )

    if model_type in {"motion_tcn", "erez_motion_tcn"}:
        return MotionTCN(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dilations=model_cfg.get("dilations"),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            causal=bool(model_cfg.get("causal", True)),
            norm=str(model_cfg.get("norm", "group")),
            input_proj_dim=int(model_cfg.get("input_proj_dim", 0)),
        )

    return EventTCN(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        dilations=model_cfg.get("dilations"),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        person_emb_dim=person_emb_dim,
        pool_mode=str(model_cfg.get("pool_mode", "attn")),
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
        motion_proj_dim=int(motion_proj_dim) if motion_proj_dim is not None else None,
        tcn_input_mode=tcn_input_mode,
        use_person_count=bool(model_cfg.get("use_person_count", True)),
    )


def load_split(split_path: Path, data_root: Path) -> list[str]:
    if not split_path.exists():
        return []
    return read_paths_txt(split_path, base_dirs=[data_root, PROJECT_ROOT])


def finite_or_none(value: Any) -> float | None:
    value_f = safe_float(value)
    return value_f if np.isfinite(value_f) else None


def safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def timeline_stem_for_path(json_path: str) -> str:
    digest = hashlib.sha1(str(json_path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_slug(Path(json_path).stem)}_{digest}"


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float32))) if values else None


def frame_id_at(frames: list[dict[str, Any]], idx: int) -> int:
    idx = int(min(max(idx, 0), max(len(frames) - 1, 0)))
    return int(frames[idx].get("f", idx))


def frame_time_at(frames: list[dict[str, Any]], idx: int, fallback_dt_sec: float) -> float:
    idx = int(min(max(idx, 0), max(len(frames) - 1, 0)))
    value = frames[idx].get("t")
    if value is not None:
        return float(value)
    return float(idx * fallback_dt_sec)


def positive_median_step(values: list[float], default: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size >= 2:
        diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return float(default)


def positive_median_int_step(values: list[int], default: int) -> int:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size >= 2:
        diffs = np.diff(arr)
        diffs = diffs[diffs > 0]
        if diffs.size:
            return int(max(1, round(float(np.median(diffs)))))
    return int(max(1, default))


def interval_duration_sec(interval: dict[str, Any]) -> float:
    return max(0.0, float(interval["end_time_sec"]) - float(interval["start_time_sec"]))


def interval_overlap_sec(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(a["end_time_sec"]), float(b["end_time_sec"]))
        - max(float(a["start_time_sec"]), float(b["start_time_sec"])),
    )


def interval_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = interval_overlap_sec(a, b)
    if overlap <= 0.0:
        return 0.0
    union = interval_duration_sec(a) + interval_duration_sec(b) - overlap
    return float(overlap / union) if union > 0.0 else 0.0


def postprocess_intervals(
    intervals: list[dict[str, Any]],
    *,
    merge_gap_sec: float = 0.0,
    min_duration_sec: float = 0.0,
) -> list[dict[str, Any]]:
    if not intervals:
        return []

    merge_gap_sec = float(max(0.0, merge_gap_sec))
    min_duration_sec = float(max(0.0, min_duration_sec))
    intervals_sorted = [
        dict(interval)
        for interval in sorted(intervals, key=lambda item: (float(item["start_time_sec"]), float(item["end_time_sec"])))
    ]
    merged: list[dict[str, Any]] = []
    for interval in intervals_sorted:
        interval.setdefault("parts", 1)
        if not merged:
            merged.append(interval)
            continue

        prev = merged[-1]
        gap_sec = float(interval["start_time_sec"]) - float(prev["end_time_sec"])
        if gap_sec <= merge_gap_sec + 1e-9:
            prev["end_time_sec"] = max(float(prev["end_time_sec"]), float(interval["end_time_sec"]))
            prev["end_frame"] = max(int(prev.get("end_frame", 0)), int(interval.get("end_frame", 0)))
            prev["parts"] = int(prev.get("parts", 1)) + int(interval.get("parts", 1))
        else:
            merged.append(interval)

    if min_duration_sec <= 0.0:
        return merged
    return [interval for interval in merged if interval_duration_sec(interval) >= min_duration_sec]


def intervals_from_points(
    records: list[dict[str, Any]],
    *,
    label_key: str,
    span_sec: float,
    span_frames: int,
    merge_gap_sec: float,
    min_duration_sec: float,
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for record in records:
        if int(record.get(label_key, 0)) != 1:
            continue
        start_time_sec = float(record["time_sec"])
        start_frame = int(record["frame"])
        intervals.append(
            {
                "start_time_sec": start_time_sec,
                "end_time_sec": start_time_sec + float(span_sec),
                "start_frame": start_frame,
                "end_frame": start_frame + int(span_frames),
            }
        )
    # Avoid splitting contiguous sampled points because of tiny timestamp jitter.
    effective_merge_gap_sec = max(float(merge_gap_sec), float(span_sec) * 0.5)
    return postprocess_intervals(
        intervals,
        merge_gap_sec=effective_merge_gap_sec,
        min_duration_sec=float(min_duration_sec),
    )


def total_interval_duration_sec(intervals: list[dict[str, Any]]) -> float:
    normalized = postprocess_intervals(intervals, merge_gap_sec=0.0, min_duration_sec=0.0)
    return float(sum(interval_duration_sec(interval) for interval in normalized))


def total_overlap_between_interval_sets_sec(
    a_intervals: list[dict[str, Any]],
    b_intervals: list[dict[str, Any]],
) -> float:
    a_sorted = postprocess_intervals(a_intervals, merge_gap_sec=0.0, min_duration_sec=0.0)
    b_sorted = postprocess_intervals(b_intervals, merge_gap_sec=0.0, min_duration_sec=0.0)
    a_idx = b_idx = 0
    total = 0.0
    while a_idx < len(a_sorted) and b_idx < len(b_sorted):
        a_item = a_sorted[a_idx]
        b_item = b_sorted[b_idx]
        total += interval_overlap_sec(a_item, b_item)
        if float(a_item["end_time_sec"]) <= float(b_item["end_time_sec"]):
            a_idx += 1
        else:
            b_idx += 1
    return float(total)


def match_event_intervals(
    predicted_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    *,
    min_overlap_sec: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    candidates: list[tuple[float, float, float, float, int, int, dict[str, Any]]] = []
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
                "offset_error_sec": float(float(pred_event["end_time_sec"]) - float(gt_event["end_time_sec"])),
            }
            candidates.append((match["overlap_sec"], match["gt_coverage"], match["iou"], -abs(match["onset_delay_sec"]), gt_idx, pred_idx, match))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
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
    predicted_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    *,
    min_overlap_sec: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    fragment_counts: list[int] = []
    details: list[dict[str, Any]] = []
    for gt_idx, gt_event in enumerate(gt_events, start=1):
        overlaps: list[dict[str, Any]] = []
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
        details.append(
            {
                "gt_index": int(gt_idx),
                "gt_start_time_sec": float(gt_event["start_time_sec"]),
                "gt_end_time_sec": float(gt_event["end_time_sec"]),
                "n_fragments": int(len(overlaps)),
                "matched_pred_indices": ",".join(str(item["pred_index"]) for item in overlaps),
                "overlap_sec_total": float(sum(float(item["overlap_sec"]) for item in overlaps)),
            }
        )
    return fragment_counts, details


def compute_event_stats(
    predicted_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    *,
    video_duration_sec: float,
    min_overlap_sec: float,
) -> dict[str, Any]:
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
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

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

    ious = [float(match["iou"]) for match in matches]
    gt_coverages = [float(match["gt_coverage"]) for match in matches]
    onset_delays = [float(match["onset_delay_sec"]) for match in matches]
    detection_delays = [float(match["detection_delay_sec"]) for match in matches]
    offset_errors = [float(match["offset_error_sec"]) for match in matches]

    return {
        "n_predicted_events": int(len(predicted_events)),
        "n_gt_events": int(len(gt_events)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarms_per_min": float(fp / (video_duration_sec / 60.0)) if video_duration_sec > 0.0 else None,
        "gt_hit_count": int(gt_hit_count),
        "gt_hit_recall": float(gt_hit_count / len(gt_events)) if gt_events else 0.0,
        "total_gt_positive_sec": float(total_gt_positive_sec),
        "total_pred_positive_sec": float(total_pred_positive_sec),
        "total_overlap_sec": float(total_overlap_sec),
        "time_coverage": float(total_overlap_sec / total_gt_positive_sec) if total_gt_positive_sec > 0.0 else 0.0,
        "time_iou": float(total_overlap_sec / total_union_sec) if total_union_sec > 0.0 else 0.0,
        "mean_fragments_per_gt": mean_or_none([float(count) for count in gt_fragment_counts]),
        "median_fragments_per_gt": median_or_none([float(count) for count in gt_fragment_counts]),
        "max_fragments_per_gt": int(max(gt_fragment_counts)) if gt_fragment_counts else 0,
        "mean_iou": mean_or_none(ious),
        "median_iou": median_or_none(ious),
        "mean_gt_coverage": mean_or_none(gt_coverages),
        "median_gt_coverage": median_or_none(gt_coverages),
        "mean_onset_delay_sec": mean_or_none(onset_delays),
        "median_onset_delay_sec": median_or_none(onset_delays),
        "mean_detection_delay_sec": mean_or_none(detection_delays),
        "median_detection_delay_sec": median_or_none(detection_delays),
        "mean_offset_error_sec": mean_or_none(offset_errors),
        "median_offset_error_sec": median_or_none(offset_errors),
        "matches": matches,
        "unmatched_gt_indices": unmatched_gt,
        "unmatched_pred_indices": unmatched_pred,
        "gt_fragment_counts": gt_fragment_details,
        "min_match_overlap_sec": float(min_overlap_sec),
    }


def gt_intervals_from_frames(
    frames: list[dict[str, Any]],
    *,
    label_cfg: dict[str, Any],
    merge_gap_sec: float,
) -> tuple[list[dict[str, Any]], float, int, float]:
    if not frames:
        return [], 0.04, 1, 0.0

    raw_times = [frame_time_at(frames, idx, fallback_dt_sec=0.04) for idx in range(len(frames))]
    frame_ids = [frame_id_at(frames, idx) for idx in range(len(frames))]
    frame_dt_sec = positive_median_step(raw_times, default=0.04)
    frame_step = positive_median_int_step(frame_ids, default=1)
    records = [
        {
            "frame": int(frame_ids[idx]),
            "time_sec": float(raw_times[idx]),
            "gt_label": int(events_to_label(frame, cfg=label_cfg)),
        }
        for idx, frame in enumerate(frames)
    ]
    intervals = intervals_from_points(
        records,
        label_key="gt_label",
        span_sec=frame_dt_sec,
        span_frames=frame_step,
        merge_gap_sec=merge_gap_sec,
        min_duration_sec=0.0,
    )
    duration_sec = max(0.0, (max(raw_times) + frame_dt_sec) - min(raw_times)) if raw_times else 0.0
    return intervals, frame_dt_sec, frame_step, float(duration_sec)


def build_event_windows_for_path(
    dataset_cls,
    json_path: str,
    *,
    K: int,
    window_size: int,
    window_step: int,
    feature_cfg: dict[str, Any],
    label_cfg: dict[str, Any],
    window_cfg: dict[str, Any],
    target_mode: str,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], float, float, int]:
    frames = read_json_frames(json_path)
    is_motion_dataset = issubclass(dataset_cls, MotionJsonDataset)
    if is_motion_dataset:
        X_seq, y_seq = build_motion_sequence(frames, feature_cfg=feature_cfg, label_cfg=label_cfg)
        frame_offset = 1
    else:
        X_seq, y_seq = build_sequence(frames, K=K, feature_cfg=feature_cfg, label_cfg=label_cfg)
        frame_offset = 0

    if not frames:
        return [], np.zeros((0,), dtype=np.float32), [], 0.0, 0.04, 1

    gt_events, frame_dt_sec, frame_step, video_duration_sec = gt_intervals_from_frames(
        frames,
        label_cfg=label_cfg,
        merge_gap_sec=0.0,
    )
    target_index = resolve_target_index(window_size=window_size, target_mode=target_mode)
    records: list[dict[str, Any]] = []
    windows: list[np.ndarray] = []
    if len(X_seq) < window_size:
        return records, np.zeros((0,), dtype=np.float32), gt_events, video_duration_sec, frame_dt_sec, frame_step

    for start in range(0, len(X_seq) - window_size + 1, window_step):
        end = start + window_size
        y_label = label_window(y_seq[start:end], target_index=target_index, window_cfg=window_cfg)
        if y_label is None:
            continue
        target_seq_idx = start + target_index
        window_start_frame_idx = start + frame_offset
        window_end_frame_idx = end - 1 + frame_offset
        target_frame_idx = target_seq_idx + frame_offset
        windows.append(np.asarray(X_seq[start:end], dtype=np.float32))
        records.append(
            {
                "json_path": str(json_path),
                "json_name": Path(json_path).name,
                "window_index": int(len(records) + 1),
                "window_start_frame": frame_id_at(frames, window_start_frame_idx),
                "window_end_frame": frame_id_at(frames, window_end_frame_idx),
                "frame": frame_id_at(frames, target_frame_idx),
                "window_start_time_sec": frame_time_at(frames, window_start_frame_idx, fallback_dt_sec=frame_dt_sec),
                "window_end_time_sec": frame_time_at(frames, window_end_frame_idx, fallback_dt_sec=frame_dt_sec),
                "time_sec": frame_time_at(frames, target_frame_idx, fallback_dt_sec=frame_dt_sec),
                "target": float(y_label),
            }
        )

    if not windows:
        return records, np.zeros((0,), dtype=np.float32), gt_events, video_duration_sec, frame_dt_sec, frame_step
    return records, np.stack(windows).astype(np.float32), gt_events, video_duration_sec, frame_dt_sec, frame_step


@torch.no_grad()
def predict_event_records_for_path(
    model: torch.nn.Module,
    dataset_cls,
    json_path: str,
    *,
    K: int,
    window_size: int,
    window_step: int,
    feature_cfg: dict[str, Any],
    label_cfg: dict[str, Any],
    window_cfg: dict[str, Any],
    target_mode: str,
    batch_size: int,
    device: torch.device,
    agg_mode: str,
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float, int]:
    records, windows, gt_events, video_duration_sec, frame_dt_sec, frame_step = build_event_windows_for_path(
        dataset_cls,
        json_path,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
    )
    if windows.shape[0] == 0:
        return records, gt_events, video_duration_sec, frame_dt_sec, frame_step

    logits_list: list[np.ndarray] = []
    for start in range(0, len(windows), int(max(batch_size, 1))):
        batch = torch.from_numpy(windows[start : start + int(max(batch_size, 1))]).to(device)
        logits_bt = model(batch)
        win_logits = aggregate_window_logits(logits_bt, mode=agg_mode, temperature=temperature)
        logits_list.append(win_logits.detach().cpu().numpy().astype(np.float32))

    logits = np.concatenate(logits_list).astype(np.float32)
    probs = sigmoid_from_logits(logits)
    for record, logit, prob in zip(records, logits, probs):
        record["logit"] = float(logit)
        record["prob_raw"] = float(prob)
    return records, gt_events, video_duration_sec, frame_dt_sec, frame_step


def event_score_methods(calibration_report: dict[str, Any], raw_threshold: float) -> list[dict[str, Any]]:
    methods = [{"score_method": "raw", "prob_key": "prob_raw", "threshold": float(raw_threshold)}]
    temp_fit = calibration_report.get("temperature", {}).get("fit", {})
    temp_threshold = finite_or_none(calibration_report.get("temperature", {}).get("selected_threshold", float("nan")))
    if temp_fit.get("available") and temp_threshold is not None:
        methods.append(
            {
                "score_method": "temperature",
                "prob_key": "prob_temperature",
                "threshold": float(temp_threshold),
                "temperature": float(temp_fit["temperature"]),
            }
        )

    platt_fit = calibration_report.get("platt", {}).get("fit", {})
    platt_threshold = finite_or_none(calibration_report.get("platt", {}).get("selected_threshold", float("nan")))
    if platt_fit.get("available") and platt_threshold is not None:
        methods.append(
            {
                "score_method": "platt",
                "prob_key": "prob_platt",
                "threshold": float(platt_threshold),
                "slope": float(platt_fit["slope"]),
                "intercept": float(platt_fit["intercept"]),
            }
        )
    return methods


def attach_calibrated_event_probs(records: list[dict[str, Any]], methods: list[dict[str, Any]]) -> None:
    if not records:
        return
    logits = np.asarray([float(record["logit"]) for record in records], dtype=np.float32)
    for method in methods:
        if method["score_method"] == "temperature":
            probs = apply_temperature_scaling(logits, float(method["temperature"]))
        elif method["score_method"] == "platt":
            probs = apply_platt_scaling(logits, slope=float(method["slope"]), intercept=float(method["intercept"]))
        else:
            continue
        for record, prob in zip(records, probs):
            record[str(method["prob_key"])] = float(prob)


def flatten_event_stats_for_csv(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if key not in {"matches", "unmatched_gt_indices", "unmatched_pred_indices", "gt_fragment_counts"}
    }


def build_aggregate_event_stats(rows: list[dict[str, Any]], matches: list[dict[str, Any]], fragments: list[dict[str, Any]]) -> dict[str, Any]:
    tp = int(sum(int(row["tp"]) for row in rows))
    fp = int(sum(int(row["fp"]) for row in rows))
    fn = int(sum(int(row["fn"]) for row in rows))
    n_pred = int(sum(int(row["n_predicted_events"]) for row in rows))
    n_gt = int(sum(int(row["n_gt_events"]) for row in rows))
    total_duration_sec = float(sum(safe_float(row["video_duration_sec"]) for row in rows))
    total_gt_positive_sec = float(sum(safe_float(row["total_gt_positive_sec"]) for row in rows))
    total_pred_positive_sec = float(sum(safe_float(row["total_pred_positive_sec"]) for row in rows))
    total_overlap_sec = float(sum(safe_float(row["total_overlap_sec"]) for row in rows))
    total_union_sec = max(total_gt_positive_sec + total_pred_positive_sec - total_overlap_sec, 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    gt_hit_count = int(sum(1 for fragment in fragments if int(fragment["n_fragments"]) > 0))
    fragment_counts = [float(fragment["n_fragments"]) for fragment in fragments]
    method_matches = matches
    ious = [float(match["iou"]) for match in method_matches]
    gt_coverages = [float(match["gt_coverage"]) for match in method_matches]
    onset_delays = [float(match["onset_delay_sec"]) for match in method_matches]
    detection_delays = [float(match["detection_delay_sec"]) for match in method_matches]
    offset_errors = [float(match["offset_error_sec"]) for match in method_matches]
    return {
        "n_files": int(len(rows)),
        "n_predicted_events": n_pred,
        "n_gt_events": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarms_per_min": float(fp / (total_duration_sec / 60.0)) if total_duration_sec > 0.0 else None,
        "gt_hit_count": gt_hit_count,
        "gt_hit_recall": float(gt_hit_count / n_gt) if n_gt > 0 else 0.0,
        "total_duration_sec": total_duration_sec,
        "total_gt_positive_sec": total_gt_positive_sec,
        "total_pred_positive_sec": total_pred_positive_sec,
        "total_overlap_sec": total_overlap_sec,
        "time_coverage": float(total_overlap_sec / total_gt_positive_sec) if total_gt_positive_sec > 0.0 else 0.0,
        "time_iou": float(total_overlap_sec / total_union_sec) if total_union_sec > 0.0 else 0.0,
        "mean_fragments_per_gt": mean_or_none(fragment_counts),
        "median_fragments_per_gt": median_or_none(fragment_counts),
        "max_fragments_per_gt": int(max(fragment_counts)) if fragment_counts else 0,
        "mean_iou": mean_or_none(ious),
        "median_iou": median_or_none(ious),
        "mean_gt_coverage": mean_or_none(gt_coverages),
        "median_gt_coverage": median_or_none(gt_coverages),
        "mean_onset_delay_sec": mean_or_none(onset_delays),
        "median_onset_delay_sec": median_or_none(onset_delays),
        "mean_detection_delay_sec": mean_or_none(detection_delays),
        "median_detection_delay_sec": median_or_none(detection_delays),
        "mean_offset_error_sec": mean_or_none(offset_errors),
        "median_offset_error_sec": median_or_none(offset_errors),
    }


def evaluate_event_methods_for_paths(
    model: torch.nn.Module,
    dataset_cls,
    paths: list[str],
    *,
    K: int,
    window_size: int,
    window_step: int,
    feature_cfg: dict[str, Any],
    label_cfg: dict[str, Any],
    window_cfg: dict[str, Any],
    target_mode: str,
    batch_size: int,
    device: torch.device,
    agg_mode: str,
    temperature: float,
    methods: list[dict[str, Any]],
    merge_gap_sec: float,
    min_duration_sec: float,
    min_overlap_sec: float,
    timeline_dir: Path | None = None,
) -> dict[str, Any]:
    per_file_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []

    for json_path in paths:
        records, gt_events, video_duration_sec, frame_dt_sec, frame_step = predict_event_records_for_path(
            model,
            dataset_cls,
            json_path,
            K=K,
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
            window_cfg=window_cfg,
            target_mode=target_mode,
            batch_size=batch_size,
            device=device,
            agg_mode=agg_mode,
            temperature=temperature,
        )
        attach_calibrated_event_probs(records, methods)
        pred_span_sec = positive_median_step([float(record["time_sec"]) for record in records], default=frame_dt_sec * max(window_step, 1))
        pred_span_frames = positive_median_int_step([int(record["frame"]) for record in records], default=frame_step * max(window_step, 1))

        for gt_idx, event in enumerate(gt_events, start=1):
            interval_rows.append(
                {
                    "score_method": "gt",
                    "json_path": json_path,
                    "json_name": Path(json_path).name,
                    "event_index": int(gt_idx),
                    "interval_kind": "gt",
                    "threshold": "",
                    "start_time_sec": float(event["start_time_sec"]),
                    "end_time_sec": float(event["end_time_sec"]),
                    "duration_sec": interval_duration_sec(event),
                    "start_frame": int(event.get("start_frame", 0)),
                    "end_frame": int(event.get("end_frame", 0)),
                }
            )

        for method in methods:
            method_name = str(method["score_method"])
            prob_key = str(method["prob_key"])
            threshold = float(method["threshold"])
            for record in records:
                record[f"pred_{method_name}"] = int(float(record.get(prob_key, 0.0)) >= threshold)

            predicted_events = intervals_from_points(
                records,
                label_key=f"pred_{method_name}",
                span_sec=pred_span_sec,
                span_frames=pred_span_frames,
                merge_gap_sec=merge_gap_sec,
                min_duration_sec=min_duration_sec,
            )
            stats = compute_event_stats(
                predicted_events,
                gt_events,
                video_duration_sec=video_duration_sec,
                min_overlap_sec=min_overlap_sec,
            )

            per_file_rows.append(
                {
                    "score_method": method_name,
                    "json_path": json_path,
                    "json_name": Path(json_path).name,
                    "threshold": threshold,
                    "n_windows": int(len(records)),
                    "video_duration_sec": float(video_duration_sec),
                    **flatten_event_stats_for_csv(stats),
                    "unmatched_gt_indices": ",".join(str(idx) for idx in stats["unmatched_gt_indices"]),
                    "unmatched_pred_indices": ",".join(str(idx) for idx in stats["unmatched_pred_indices"]),
                }
            )
            for pred_idx, event in enumerate(predicted_events, start=1):
                interval_rows.append(
                    {
                        "score_method": method_name,
                        "json_path": json_path,
                        "json_name": Path(json_path).name,
                        "event_index": int(pred_idx),
                        "interval_kind": "predicted",
                        "threshold": threshold,
                        "start_time_sec": float(event["start_time_sec"]),
                        "end_time_sec": float(event["end_time_sec"]),
                        "duration_sec": interval_duration_sec(event),
                        "start_frame": int(event.get("start_frame", 0)),
                        "end_frame": int(event.get("end_frame", 0)),
                    }
                )
            for match in stats["matches"]:
                match_rows.append(
                    {
                        "score_method": method_name,
                        "json_path": json_path,
                        "json_name": Path(json_path).name,
                        "threshold": threshold,
                        **match,
                    }
                )
            for fragment in stats["gt_fragment_counts"]:
                fragment_rows.append(
                    {
                        "score_method": method_name,
                        "json_path": json_path,
                        "json_name": Path(json_path).name,
                        "threshold": threshold,
                        **fragment,
                    }
                )
            if timeline_dir is not None and records:
                timeline_path = timeline_dir / f"{timeline_stem_for_path(json_path)}__{method_name}.png"
                save_event_probability_timeline_image(
                    timeline_path,
                    times_sec=np.asarray([float(record["time_sec"]) for record in records], dtype=np.float32),
                    probs=np.asarray([float(record.get(prob_key, float("nan"))) for record in records], dtype=np.float32),
                    threshold=threshold,
                    gt_intervals=gt_events,
                    pred_intervals=predicted_events,
                    title=f"{Path(json_path).name} | {method_name} | thr={threshold:.2f}",
                    video_duration_sec=video_duration_sec,
                    prob_label=f"{method_name} probability",
                )
                timeline_rows.append(
                    {
                        "score_method": method_name,
                        "json_path": json_path,
                        "json_name": Path(json_path).name,
                        "threshold": threshold,
                        "timeline_png": str(timeline_path),
                        "n_windows": int(len(records)),
                        "video_duration_sec": float(video_duration_sec),
                    }
                )

    aggregate: dict[str, Any] = {}
    for method in methods:
        method_name = str(method["score_method"])
        method_rows = [row for row in per_file_rows if row["score_method"] == method_name]
        method_matches = [row for row in match_rows if row["score_method"] == method_name]
        method_fragments = [row for row in fragment_rows if row["score_method"] == method_name]
        aggregate[method_name] = {
            "threshold": float(method["threshold"]),
            **build_aggregate_event_stats(method_rows, method_matches, method_fragments),
        }

    return {
        "per_file": per_file_rows,
        "matches": match_rows,
        "fragments": fragment_rows,
        "intervals": interval_rows,
        "timelines": timeline_rows,
        "aggregate": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Training run directory with config_resolved.yaml, checkpoints/, and splits/.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to <run_dir>/checkpoints/best.pt")
    parser.add_argument("--output_dir", default=None, help="Optional output directory. Defaults to <run_dir>/final_test_standalone")
    parser.add_argument("--test_list", default=None, help="Optional text file with held-out test JSON paths. Defaults to <run_dir>/splits/test_paths.txt")
    parser.add_argument("--device", default=None, help="Optional device override, for example cpu or cuda:0")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional dataloader worker override")
    parser.add_argument("--event_merge_gap_sec", type=float, default=0.0, help="Merge predicted/GT event intervals separated by at most this gap.")
    parser.add_argument("--event_min_duration_sec", type=float, default=0.0, help="Drop predicted event intervals shorter than this duration.")
    parser.add_argument("--event_min_overlap_sec", type=float, default=0.0, help="Minimum GT/predicted overlap required to count an event match.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "config_resolved.yaml"
    splits_dir = run_dir / "splits"
    ckpt_path = resolve_optional_path(args.checkpoint, run_dir) or (run_dir / "checkpoints" / "best.pt")
    output_dir = resolve_optional_path(args.output_dir, run_dir) or (run_dir / "final_test_standalone")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml in {run_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not splits_dir.exists():
        raise FileNotFoundError(f"Missing splits directory: {splits_dir}")

    cfg = load_yaml(cfg_path)
    data_root = Path(cfg["paths"]["data_root"]).resolve()
    feature_cfg = cfg.get("features", {})
    label_cfg = resolve_label_cfg_from_root(cfg)
    train_cfg = cfg.get("train", {})
    target_mode = str(cfg.get("data", {}).get("target_mode", "last")).lower()
    window_cfg = resolve_window_cfg(cfg)
    K = int(cfg["data"].get("max_persons", 25))
    window_size = int(cfg["data"].get("window_size", 16))
    window_step = int(cfg["data"].get("window_step", 4))
    agg_mode = str(train_cfg.get("agg_mode", "logsumexp"))
    temperature = float(train_cfg.get("logsumexp_temperature", 1.0))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
    device = resolve_device(args.device, train_cfg.get("device", "cpu"))
    dataset_cls = resolve_dataset_cls(cfg)

    train_paths = load_split(splits_dir / "train_paths.txt", data_root)
    val_paths = load_split(splits_dir / "val_paths.txt", data_root)
    test_split_path = resolve_cli_path(args.test_list) or (splits_dir / "test_paths.txt")
    test_paths = load_split(test_split_path, data_root)

    if not val_paths:
        raise RuntimeError(f"No validation paths found in {splits_dir / 'val_paths.txt'}")
    if not test_paths:
        raise RuntimeError(f"No test paths found in {test_split_path}")

    preview_ds = build_dataset(
        dataset_cls,
        val_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
        verbose=False,
    )
    if len(preview_ds) == 0:
        preview_ds = build_dataset(
            dataset_cls,
            test_paths,
            K=K,
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
            window_cfg=window_cfg,
            target_mode=target_mode,
            verbose=False,
        )
    if len(preview_ds) == 0:
        raise RuntimeError("Validation and test datasets both have 0 windows after filtering.")

    payload = torch.load(ckpt_path, map_location=device)
    state_dict = normalize_event_state_dict(payload["model"])
    X0, _ = preview_ds[0]
    model = build_model(cfg, input_dim=int(X0.shape[-1]), state_dict=state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_ds = build_dataset(
        dataset_cls,
        val_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
    )
    test_ds = build_dataset(
        dataset_cls,
        test_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
    )
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset has 0 windows after filtering.")
    if len(test_ds) == 0:
        raise RuntimeError("Test dataset has 0 windows after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] run_dir:   {run_dir}")
    print(f"[INFO] ckpt:      {ckpt_path}")
    print(f"[INFO] output:    {output_dir}")
    print(f"[INFO] device:    {device}")
    print(f"[INFO] val_jsons: {len(val_paths)} | test_jsons: {len(test_paths)}")
    print(f"[INFO] val_windows: {len(val_ds)} | test_windows: {len(test_ds)}")

    val_loader = make_loader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    val_out = evaluate_binary(model, val_loader, device=device, agg_mode=agg_mode, temperature=temperature)
    test_out = evaluate_binary(model, test_loader, device=device, agg_mode=agg_mode, temperature=temperature)

    selected_threshold = float(val_out["best_f1"]["threshold"])
    val_targets = np.asarray(val_out["targets"], dtype=np.float32)
    val_logits = np.asarray(val_out["logits"], dtype=np.float32)
    test_targets = np.asarray(test_out["targets"], dtype=np.float32)
    test_binary_targets = np.asarray(test_out["binary_targets"], dtype=np.int32)
    test_logits = np.asarray(test_out["logits"], dtype=np.float32)
    test_probs = np.asarray(test_out["probs"], dtype=np.float32)
    calibration_report = build_probability_calibration_report(
        val_targets,
        val_logits,
        test_targets,
        test_logits,
    )
    selected_stats = confusion_stats_at_threshold(test_binary_targets, test_probs, threshold=selected_threshold)
    oracle_stats = test_out["best_threshold_stats"]
    roc_rows = compute_roc_points(test_binary_targets, test_probs)
    pr_rows = compute_pr_points(test_binary_targets, test_probs)

    print(f"[INFO] Evaluating {len(val_paths)} validation files for video-level threshold selection")
    val_path_rows = evaluate_paths_individually(
        model,
        dataset_cls,
        val_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        agg_mode=agg_mode,
        temperature=temperature,
        selected_threshold=selected_threshold,
    )
    print(f"[INFO] Evaluating {len(test_paths)} held-out test files for per-file/video report")
    per_file_rows = evaluate_paths_individually(
        model,
        dataset_cls,
        test_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        agg_mode=agg_mode,
        temperature=temperature,
        selected_threshold=selected_threshold,
    )

    val_video_mean = summarize_video_scores(val_path_rows, score_key="video_score_mean")
    val_video_max = summarize_video_scores(val_path_rows, score_key="video_score_max")
    test_video_mean = summarize_video_scores(
        per_file_rows,
        score_key="video_score_mean",
        selected_threshold=float(val_video_mean["selected_threshold"]),
    )
    test_video_max = summarize_video_scores(
        per_file_rows,
        score_key="video_score_max",
        selected_threshold=float(val_video_max["selected_threshold"]),
    )
    per_video_rows = build_per_video_rows(
        per_file_rows,
        threshold_mean=float(test_video_mean["selected_threshold"]),
        threshold_max=float(test_video_max["selected_threshold"]),
    )
    event_methods = event_score_methods(calibration_report, raw_threshold=selected_threshold)
    print(f"[INFO] Evaluating held-out test files for event-level report ({', '.join(method['score_method'] for method in event_methods)})")
    event_eval = evaluate_event_methods_for_paths(
        model,
        dataset_cls,
        test_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
        batch_size=batch_size,
        device=device,
        agg_mode=agg_mode,
        temperature=temperature,
        methods=event_methods,
        merge_gap_sec=float(args.event_merge_gap_sec),
        min_duration_sec=float(args.event_min_duration_sec),
        min_overlap_sec=float(args.event_min_overlap_sec),
        timeline_dir=output_dir / "timelines",
    )

    selected_fpr = fpr_from_stats(selected_stats)
    oracle_fpr = fpr_from_stats(oracle_stats)
    n_pos = int(np.sum(test_binary_targets == 1))
    n_neg = int(np.sum(test_binary_targets == 0))

    save_threshold_sweep_csv(output_dir, test_out["sweep"])
    save_rows_csv(output_dir / "per_file.csv", per_file_rows)
    save_rows_csv(output_dir / "per_video.csv", per_video_rows)
    save_rows_csv(output_dir / "event_per_file.csv", event_eval["per_file"])
    save_rows_csv(output_dir / "event_matches.csv", event_eval["matches"])
    save_rows_csv(output_dir / "event_fragments.csv", event_eval["fragments"])
    save_rows_csv(output_dir / "event_intervals.csv", event_eval["intervals"])
    save_rows_csv(output_dir / "timeline_images.csv", event_eval["timelines"])
    if roc_rows:
        save_rows_csv(output_dir / "roc.csv", roc_rows)
    if pr_rows:
        save_rows_csv(output_dir / "pr.csv", pr_rows)

    save_confusion_matrix_image(
        output_dir / "cm_norm_thr_val_selected.png",
        selected_stats,
        title=f"Final Test Confusion Matrix (normalized) @ val thr={selected_threshold:.2f}",
    )
    save_confusion_matrix_image(
        output_dir / "cm_norm_thr_test_best_f1.png",
        oracle_stats,
        title=f"Final Test Confusion Matrix (normalized) @ oracle thr={float(test_out['best_f1']['threshold']):.2f}",
    )
    save_pr_curve_image(
        output_dir / "pr_curve_test.png",
        y_true=test_binary_targets.astype(np.float32),
        y_prob=test_probs.astype(np.float32),
        title="Final Test Precision-Recall Curve",
    )
    save_roc_curve_image(
        output_dir / "roc_curve_test.png",
        y_true=test_binary_targets.astype(np.float32),
        y_prob=test_probs.astype(np.float32),
        title="Final Test ROC Curve",
    )
    save_probability_calibration_artifacts(
        output_dir,
        prefix="window_val_raw",
        y_true=val_targets.astype(np.float32),
        y_prob=np.asarray(val_out["probs"], dtype=np.float32),
        title="Validation Reliability Diagram (Raw)",
    )
    save_probability_calibration_artifacts(
        output_dir,
        prefix="window_test_raw",
        y_true=test_targets.astype(np.float32),
        y_prob=test_probs.astype(np.float32),
        title="Test Reliability Diagram (Raw)",
    )
    if calibration_report.get("temperature", {}).get("fit", {}).get("available"):
        temp = float(calibration_report["temperature"]["fit"]["temperature"])
        save_probability_calibration_artifacts(
            output_dir,
            prefix="window_val_temperature",
            y_true=val_targets.astype(np.float32),
            y_prob=apply_temperature_scaling(val_logits, temp),
            title=f"Validation Reliability Diagram (Temperature T={temp:.3f})",
        )
        save_probability_calibration_artifacts(
            output_dir,
            prefix="window_test_temperature",
            y_true=test_targets.astype(np.float32),
            y_prob=apply_temperature_scaling(test_logits, temp),
            title=f"Test Reliability Diagram (Temperature T={temp:.3f})",
        )
    if calibration_report.get("platt", {}).get("fit", {}).get("available"):
        slope = float(calibration_report["platt"]["fit"]["slope"])
        intercept = float(calibration_report["platt"]["fit"]["intercept"])
        save_probability_calibration_artifacts(
            output_dir,
            prefix="window_val_platt",
            y_true=val_targets.astype(np.float32),
            y_prob=apply_platt_scaling(val_logits, slope=slope, intercept=intercept),
            title="Validation Reliability Diagram (Platt)",
        )
        save_probability_calibration_artifacts(
            output_dir,
            prefix="window_test_platt",
            y_true=test_targets.astype(np.float32),
            y_prob=apply_platt_scaling(test_logits, slope=slope, intercept=intercept),
            title="Test Reliability Diagram (Platt)",
        )
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    temperature_fit = calibration_report.get("temperature", {}).get("fit", {})
    temperature_test = calibration_report.get("temperature", {}).get("test", {})
    temperature_selected_stats = temperature_test.get("selected_threshold_stats", {})
    platt_fit = calibration_report.get("platt", {}).get("fit", {})
    platt_test = calibration_report.get("platt", {}).get("test", {})
    platt_selected_stats = platt_test.get("selected_threshold_stats", {})
    summary_rows = [
        ("run_dir", str(run_dir)),
        ("checkpoint", str(ckpt_path)),
        ("output_dir", str(output_dir)),
        ("device", str(device)),
        ("batch_size", batch_size),
        ("num_workers", num_workers),
        ("train_jsons", len(train_paths)),
        ("val_jsons", len(val_paths)),
        ("test_jsons", len(test_paths)),
        ("val_windows", len(val_ds)),
        ("test_windows", len(test_ds)),
        ("window_threshold_source", "validation_best_f1"),
        ("window_selected_threshold", selected_threshold),
        ("test_n_samples", int(test_out["n_samples"])),
        ("test_n_pos", n_pos),
        ("test_n_neg", n_neg),
        ("test_auprc", safe_float(test_out["auprc"])),
        ("test_auroc", safe_float(test_out["auroc"])),
        ("test_brier_score", safe_float(test_out["brier_score"])),
        ("test_brier_baseline", safe_float(test_out["brier_baseline"])),
        ("test_brier_skill_score", safe_float(test_out["brier_skill_score"])),
        ("test_ece", safe_float(test_out["ece"])),
        ("test_max_calibration_error", safe_float(test_out["max_calibration_error"])),
        ("window_selected_precision", safe_float(selected_stats["precision"])),
        ("window_selected_recall", safe_float(selected_stats["recall"])),
        ("window_selected_f1", safe_float(selected_stats["f1"])),
        ("window_selected_accuracy", safe_float(selected_stats["accuracy"])),
        ("window_selected_fpr", selected_fpr),
        ("window_oracle_threshold", safe_float(test_out["best_f1"]["threshold"])),
        ("window_oracle_precision", safe_float(oracle_stats["precision"])),
        ("window_oracle_recall", safe_float(oracle_stats["recall"])),
        ("window_oracle_f1", safe_float(oracle_stats["f1"])),
        ("window_oracle_accuracy", safe_float(oracle_stats["accuracy"])),
        ("window_oracle_fpr", oracle_fpr),
        ("temperature_available", bool(temperature_fit.get("available", False))),
        ("temperature_value", safe_float(temperature_fit.get("temperature", float("nan")))),
        ("temperature_nll_before", safe_float(temperature_fit.get("nll_before", float("nan")))),
        ("temperature_nll_after", safe_float(temperature_fit.get("nll_after", float("nan")))),
        ("temperature_test_brier_score", safe_float(temperature_test.get("brier_score", float("nan")))),
        ("temperature_test_ece", safe_float(temperature_test.get("ece", float("nan")))),
        ("temperature_selected_threshold", safe_float(temperature_test.get("selected_threshold", float("nan")))),
        ("temperature_selected_precision", safe_float(temperature_selected_stats.get("precision", float("nan")))),
        ("temperature_selected_recall", safe_float(temperature_selected_stats.get("recall", float("nan")))),
        ("temperature_selected_f1", safe_float(temperature_selected_stats.get("f1", float("nan")))),
        ("platt_available", bool(platt_fit.get("available", False))),
        ("platt_slope", safe_float(platt_fit.get("slope", float("nan")))),
        ("platt_intercept", safe_float(platt_fit.get("intercept", float("nan")))),
        ("platt_nll_before", safe_float(platt_fit.get("nll_before", float("nan")))),
        ("platt_nll_after", safe_float(platt_fit.get("nll_after", float("nan")))),
        ("platt_test_brier_score", safe_float(platt_test.get("brier_score", float("nan")))),
        ("platt_test_ece", safe_float(platt_test.get("ece", float("nan")))),
        ("platt_selected_threshold", safe_float(platt_test.get("selected_threshold", float("nan")))),
        ("platt_selected_precision", safe_float(platt_selected_stats.get("precision", float("nan")))),
        ("platt_selected_recall", safe_float(platt_selected_stats.get("recall", float("nan")))),
        ("platt_selected_f1", safe_float(platt_selected_stats.get("f1", float("nan")))),
        ("video_mean_threshold_source", "validation_video_best_f1"),
        ("video_mean_selected_threshold", safe_float(test_video_mean["selected_threshold"])),
        ("video_mean_n_videos_eval", int(test_video_mean["n_videos_eval"])),
        ("video_mean_n_pos", int(test_video_mean["n_pos"])),
        ("video_mean_n_neg", int(test_video_mean["n_neg"])),
        ("video_mean_auprc", safe_float(test_video_mean["auprc"])),
        ("video_mean_auroc", safe_float(test_video_mean["auroc"])),
        ("video_mean_precision", safe_float(test_video_mean["selected_threshold_stats"]["precision"])),
        ("video_mean_recall", safe_float(test_video_mean["selected_threshold_stats"]["recall"])),
        ("video_mean_f1", safe_float(test_video_mean["selected_threshold_stats"]["f1"])),
        ("video_mean_accuracy", safe_float(test_video_mean["selected_threshold_stats"]["accuracy"])),
        ("video_max_threshold_source", "validation_video_best_f1"),
        ("video_max_selected_threshold", safe_float(test_video_max["selected_threshold"])),
        ("video_max_n_videos_eval", int(test_video_max["n_videos_eval"])),
        ("video_max_n_pos", int(test_video_max["n_pos"])),
        ("video_max_n_neg", int(test_video_max["n_neg"])),
        ("video_max_auprc", safe_float(test_video_max["auprc"])),
        ("video_max_auroc", safe_float(test_video_max["auroc"])),
        ("video_max_precision", safe_float(test_video_max["selected_threshold_stats"]["precision"])),
        ("video_max_recall", safe_float(test_video_max["selected_threshold_stats"]["recall"])),
        ("video_max_f1", safe_float(test_video_max["selected_threshold_stats"]["f1"])),
        ("video_max_accuracy", safe_float(test_video_max["selected_threshold_stats"]["accuracy"])),
    ]
    for method_name, stats in event_eval["aggregate"].items():
        prefix = f"event_{method_name}"
        summary_rows.extend(
            [
                (f"{prefix}_threshold_source", "validation_window_best_f1"),
                (f"{prefix}_threshold", safe_float(stats["threshold"])),
                (f"{prefix}_n_gt_events", int(stats["n_gt_events"])),
                (f"{prefix}_n_predicted_events", int(stats["n_predicted_events"])),
                (f"{prefix}_tp", int(stats["tp"])),
                (f"{prefix}_fp", int(stats["fp"])),
                (f"{prefix}_fn", int(stats["fn"])),
                (f"{prefix}_precision", safe_float(stats["precision"])),
                (f"{prefix}_recall", safe_float(stats["recall"])),
                (f"{prefix}_f1", safe_float(stats["f1"])),
                (f"{prefix}_false_alarms_per_min", safe_float(stats["false_alarms_per_min"])),
                (f"{prefix}_gt_hit_recall", safe_float(stats["gt_hit_recall"])),
                (f"{prefix}_time_coverage", safe_float(stats["time_coverage"])),
                (f"{prefix}_time_iou", safe_float(stats["time_iou"])),
                (f"{prefix}_mean_fragments_per_gt", safe_float(stats["mean_fragments_per_gt"])),
                (f"{prefix}_mean_iou", safe_float(stats["mean_iou"])),
                (f"{prefix}_mean_onset_delay_sec", safe_float(stats["mean_onset_delay_sec"])),
                (f"{prefix}_mean_detection_delay_sec", safe_float(stats["mean_detection_delay_sec"])),
            ]
        )
    save_summary_csv(output_dir / "summary.csv", summary_rows)

    payload_out = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "split_counts": {
            "train_jsons": len(train_paths),
            "val_jsons": len(val_paths),
            "test_jsons": len(test_paths),
            "val_windows": len(val_ds),
            "test_windows": len(test_ds),
        },
        "window_level": {
            "threshold_source": "validation_best_f1",
            "validation": {
                "n_samples": int(val_out["n_samples"]),
                "auprc": safe_float(val_out["auprc"]),
                "auroc": safe_float(val_out["auroc"]),
                "best_f1": val_out["best_f1"],
                "best_threshold_stats": val_out["best_threshold_stats"],
                "mean_prob_pos": safe_float(val_out["mean_prob_pos"]),
                "mean_prob_neg": safe_float(val_out["mean_prob_neg"]),
            },
            "test": {
                "n_samples": int(test_out["n_samples"]),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auprc": safe_float(test_out["auprc"]),
                "auroc": safe_float(test_out["auroc"]),
                "mean_prob_pos": safe_float(test_out["mean_prob_pos"]),
                "mean_prob_neg": safe_float(test_out["mean_prob_neg"]),
                "brier_score": safe_float(test_out["brier_score"]),
                "brier_baseline": safe_float(test_out["brier_baseline"]),
                "brier_skill_score": safe_float(test_out["brier_skill_score"]),
                "ece": safe_float(test_out["ece"]),
                "max_calibration_error": safe_float(test_out["max_calibration_error"]),
                "calibration": test_out["calibration"],
                "calibration_bins": test_out["calibration_bins"],
                "selected_threshold": selected_threshold,
                "selected_threshold_stats": {
                    **selected_stats,
                    "fpr": selected_fpr,
                },
                "oracle_best_f1": test_out["best_f1"],
                "oracle_best_threshold_stats": {
                    **oracle_stats,
                    "fpr": oracle_fpr,
                },
            },
        },
        "video_level": {
            "threshold_source": "validation_video_best_f1",
            "validation": {
                "mean_score": {key: value for key, value in val_video_mean.items() if key != "sweep"},
                "max_score": {key: value for key, value in val_video_max.items() if key != "sweep"},
            },
            "test": {
                "mean_score": {key: value for key, value in test_video_mean.items() if key != "sweep"},
                "max_score": {key: value for key, value in test_video_max.items() if key != "sweep"},
            },
        },
        "event_level": {
            "threshold_source": "validation_window_best_f1",
            "settings": {
                "merge_gap_sec": float(args.event_merge_gap_sec),
                "min_duration_sec": float(args.event_min_duration_sec),
                "min_overlap_sec": float(args.event_min_overlap_sec),
            },
            "test": event_eval["aggregate"],
        },
        "probability_calibration": calibration_report,
        "artifacts": {
            "summary_csv": str(output_dir / "summary.csv"),
            "per_file_csv": str(output_dir / "per_file.csv"),
            "per_video_csv": str(output_dir / "per_video.csv"),
            "event_per_file_csv": str(output_dir / "event_per_file.csv"),
            "event_matches_csv": str(output_dir / "event_matches.csv"),
            "event_fragments_csv": str(output_dir / "event_fragments.csv"),
            "event_intervals_csv": str(output_dir / "event_intervals.csv"),
            "timeline_images_csv": str(output_dir / "timeline_images.csv"),
            "timelines_dir": str(output_dir / "timelines"),
            "roc_csv": str(output_dir / "roc.csv"),
            "pr_csv": str(output_dir / "pr.csv"),
            "threshold_sweep_csv": str(output_dir / "threshold_sweep.csv"),
            "calibration_json": str(output_dir / "calibration.json"),
            "window_test_raw_reliability": str(output_dir / "window_test_raw_reliability.png"),
            "window_test_raw_calibration_bins": str(output_dir / "window_test_raw_calibration_bins.csv"),
            "window_test_temperature_reliability": str(output_dir / "window_test_temperature_reliability.png"),
            "window_test_temperature_calibration_bins": str(output_dir / "window_test_temperature_calibration_bins.csv"),
            "window_test_platt_reliability": str(output_dir / "window_test_platt_reliability.png"),
            "window_test_platt_calibration_bins": str(output_dir / "window_test_platt_calibration_bins.csv"),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[WINDOW TEST] AUPRC={test_out['auprc']:.4f} | AUROC={test_out['auroc']:.4f} "
        f"| Brier={test_out['brier_score']:.4f} | ECE={test_out['ece']:.4f} "
        f"| val_thr={selected_threshold:.2f} F1={selected_stats['f1']:.4f} "
        f"| oracle_thr={float(test_out['best_f1']['threshold']):.2f} F1={oracle_stats['f1']:.4f}"
    )
    if temperature_fit.get("available"):
        print(
            f"[WINDOW CAL] temperature={float(temperature_fit['temperature']):.4f} "
            f"| test_brier={safe_float(temperature_test.get('brier_score', float('nan'))):.4f} "
            f"| test_ece={safe_float(temperature_test.get('ece', float('nan'))):.4f}"
        )
    print(
        f"[VIDEO TEST] mean_thr={float(test_video_mean['selected_threshold']):.2f} "
        f"F1={test_video_mean['selected_threshold_stats']['f1']:.4f} | "
        f"max_thr={float(test_video_max['selected_threshold']):.2f} "
        f"F1={test_video_max['selected_threshold_stats']['f1']:.4f}"
    )
    for method_name, stats in event_eval["aggregate"].items():
        print(
            f"[EVENT TEST {method_name}] thr={float(stats['threshold']):.2f} "
            f"events={int(stats['n_predicted_events'])}/{int(stats['n_gt_events'])} pred/gt "
            f"| F1={safe_float(stats['f1']):.4f} P={safe_float(stats['precision']):.4f} R={safe_float(stats['recall']):.4f} "
            f"| hit_recall={safe_float(stats['gt_hit_recall']):.4f} time_iou={safe_float(stats['time_iou']):.4f}"
        )


if __name__ == "__main__":
    main()
