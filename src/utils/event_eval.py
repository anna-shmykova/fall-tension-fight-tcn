from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.build_sequence import build_sequence
from src.data.dataset import MotionJsonDataset, label_window, resolve_target_index
from src.data.json_io import read_json_frames
from src.data.labels import events_to_label
from src.data.motion_sequence import build_motion_sequence
from src.utils.metrics import apply_platt_scaling, apply_temperature_scaling, sigmoid_from_logits


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float32))) if values else None


def make_float_grid(values: Any = None, *, start: float = 0.01, end: float = 0.99, step: float = 0.01) -> list[float]:
    if values is not None:
        if isinstance(values, (int, float)):
            return [float(values)]
        if isinstance(values, str):
            return [float(value.strip()) for value in values.split(",") if value.strip()]
        return [float(value) for value in values]
    return [float(value) for value in np.arange(float(start), float(end) + 1e-9, float(step))]


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


def aggregate_window_logits(logits_bt: torch.Tensor, mode: str, temperature: float) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "last":
        return logits_bt[:, -1]
    if mode == "max":
        return logits_bt.max(dim=1).values

    temp = float(temperature)
    lse = torch.logsumexp(logits_bt / temp, dim=1) * temp
    if mode == "logsumexp":
        return lse
    if mode == "logmeanexp":
        return lse - math.log(logits_bt.size(1))
    raise ValueError(f"Unknown agg mode: {mode}")


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
    min_iou: float = 0.0,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    candidates: list[tuple[float, float, float, float, int, int, dict[str, Any]]] = []
    min_overlap_sec = float(max(0.0, min_overlap_sec))
    min_iou = float(max(0.0, min_iou))
    for gt_idx, gt_event in enumerate(gt_events):
        for pred_idx, pred_event in enumerate(predicted_events):
            overlap = interval_overlap_sec(pred_event, gt_event)
            if overlap <= min_overlap_sec:
                continue
            iou = interval_iou(pred_event, gt_event)
            if iou < min_iou:
                continue
            gt_duration = max(interval_duration_sec(gt_event), 1e-12)
            pred_duration = max(interval_duration_sec(pred_event), 1e-12)
            onset_delay_sec = float(pred_event["start_time_sec"]) - float(gt_event["start_time_sec"])
            match = {
                "gt_index": int(gt_idx + 1),
                "pred_index": int(pred_idx + 1),
                "overlap_sec": float(overlap),
                "iou": float(iou),
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
    min_iou: float = 0.0,
) -> dict[str, Any]:
    matches, unmatched_gt, unmatched_pred = match_event_intervals(
        predicted_events,
        gt_events,
        min_overlap_sec=min_overlap_sec,
        min_iou=min_iou,
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
        "min_match_iou": float(min_iou),
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
    if not frames:
        return [], np.zeros((0,), dtype=np.float32), [], 0.0, 0.04, 1

    is_motion_dataset = issubclass(dataset_cls, MotionJsonDataset)
    if is_motion_dataset:
        X_seq, y_seq = build_motion_sequence(frames, feature_cfg=feature_cfg, label_cfg=label_cfg)
        frame_offset = 1
    else:
        X_seq, y_seq = build_sequence(frames, K=K, feature_cfg=feature_cfg, label_cfg=label_cfg)
        frame_offset = 0

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
def predict_event_bundle_for_path(
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
) -> dict[str, Any]:
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
    pred_span_sec = positive_median_step([float(record["time_sec"]) for record in records], default=frame_dt_sec * max(window_step, 1))
    pred_span_frames = positive_median_int_step([int(record["frame"]) for record in records], default=frame_step * max(window_step, 1))

    if windows.shape[0] > 0:
        logits_list: list[np.ndarray] = []
        batch_size = int(max(batch_size, 1))
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            logits_bt = model(batch)
            win_logits = aggregate_window_logits(logits_bt, mode=agg_mode, temperature=temperature)
            logits_list.append(win_logits.detach().cpu().numpy().astype(np.float32))

        logits = np.concatenate(logits_list).astype(np.float32)
        probs = sigmoid_from_logits(logits)
        for record, logit, prob in zip(records, logits, probs):
            record["logit"] = float(logit)
            record["prob_raw"] = float(prob)

    return {
        "json_path": str(json_path),
        "json_name": Path(json_path).name,
        "records": records,
        "gt_events": gt_events,
        "video_duration_sec": float(video_duration_sec),
        "frame_dt_sec": float(frame_dt_sec),
        "frame_step": int(frame_step),
        "pred_span_sec": float(pred_span_sec),
        "pred_span_frames": int(pred_span_frames),
    }


@torch.no_grad()
def predict_event_bundles(
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
) -> list[dict[str, Any]]:
    return [
        predict_event_bundle_for_path(
            model,
            dataset_cls,
            path,
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
        for path in paths
    ]


def event_score_methods(
    calibration_report: dict[str, Any],
    *,
    raw_threshold: float,
    score_methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested = [str(item).lower() for item in (score_methods or ["raw", "temperature", "platt"])]
    methods: list[dict[str, Any]] = []

    if "raw" in requested:
        methods.append({"score_method": "raw", "prob_key": "prob_raw", "threshold": float(raw_threshold)})

    temp_fit = calibration_report.get("temperature", {}).get("fit", {})
    if "temperature" in requested and temp_fit.get("available"):
        methods.append(
            {
                "score_method": "temperature",
                "prob_key": "prob_temperature",
                "threshold": safe_float(calibration_report.get("temperature", {}).get("selected_threshold", raw_threshold)),
                "temperature": float(temp_fit["temperature"]),
            }
        )

    platt_fit = calibration_report.get("platt", {}).get("fit", {})
    if "platt" in requested and platt_fit.get("available"):
        methods.append(
            {
                "score_method": "platt",
                "prob_key": "prob_platt",
                "threshold": safe_float(calibration_report.get("platt", {}).get("selected_threshold", raw_threshold)),
                "slope": float(platt_fit["slope"]),
                "intercept": float(platt_fit["intercept"]),
            }
        )

    return methods


def attach_calibrated_probs(bundles: list[dict[str, Any]], methods: list[dict[str, Any]]) -> None:
    for bundle in bundles:
        records = bundle["records"]
        if not records:
            continue
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
    ious = [float(match["iou"]) for match in matches]
    gt_coverages = [float(match["gt_coverage"]) for match in matches]
    onset_delays = [float(match["onset_delay_sec"]) for match in matches]
    detection_delays = [float(match["detection_delay_sec"]) for match in matches]
    offset_errors = [float(match["offset_error_sec"]) for match in matches]
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


def evaluate_event_params(
    bundles: list[dict[str, Any]],
    method: dict[str, Any],
    *,
    threshold: float,
    merge_gap_sec: float,
    min_duration_sec: float,
    min_overlap_sec: float,
    min_iou: float,
    include_details: bool,
) -> dict[str, Any]:
    per_file_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    method_name = str(method["score_method"])
    prob_key = str(method["prob_key"])
    threshold = float(threshold)

    for bundle in bundles:
        records = bundle["records"]
        gt_events = bundle["gt_events"]
        json_path = bundle["json_path"]
        json_name = bundle["json_name"]
        point_records = [
            {
                "frame": int(record["frame"]),
                "time_sec": float(record["time_sec"]),
                "pred_state": int(float(record.get(prob_key, 0.0)) >= threshold),
            }
            for record in records
        ]
        predicted_events = intervals_from_points(
            point_records,
            label_key="pred_state",
            span_sec=float(bundle["pred_span_sec"]),
            span_frames=int(bundle["pred_span_frames"]),
            merge_gap_sec=float(merge_gap_sec),
            min_duration_sec=float(min_duration_sec),
        )
        stats = compute_event_stats(
            predicted_events,
            gt_events,
            video_duration_sec=float(bundle["video_duration_sec"]),
            min_overlap_sec=float(min_overlap_sec),
            min_iou=float(min_iou),
        )
        per_file_rows.append(
            {
                "score_method": method_name,
                "json_path": json_path,
                "json_name": json_name,
                "threshold": threshold,
                "merge_gap_sec": float(merge_gap_sec),
                "min_duration_sec": float(min_duration_sec),
                "min_overlap_sec": float(min_overlap_sec),
                "min_match_iou": float(min_iou),
                "n_windows": int(len(records)),
                "video_duration_sec": float(bundle["video_duration_sec"]),
                **flatten_event_stats_for_csv(stats),
                "unmatched_gt_indices": ",".join(str(idx) for idx in stats["unmatched_gt_indices"]),
                "unmatched_pred_indices": ",".join(str(idx) for idx in stats["unmatched_pred_indices"]),
            }
        )
        for match in stats["matches"]:
            match_rows.append(
                {
                    "score_method": method_name,
                    "json_path": json_path,
                    "json_name": json_name,
                    "threshold": threshold,
                    "merge_gap_sec": float(merge_gap_sec),
                    "min_duration_sec": float(min_duration_sec),
                    "min_overlap_sec": float(min_overlap_sec),
                    "min_match_iou": float(min_iou),
                    **match,
                }
            )
        for fragment in stats["gt_fragment_counts"]:
            fragment_rows.append(
                {
                    "score_method": method_name,
                    "json_path": json_path,
                    "json_name": json_name,
                    "threshold": threshold,
                    "merge_gap_sec": float(merge_gap_sec),
                    "min_duration_sec": float(min_duration_sec),
                    "min_overlap_sec": float(min_overlap_sec),
                    "min_match_iou": float(min_iou),
                    **fragment,
                }
            )

        if include_details:
            for gt_idx, event in enumerate(gt_events, start=1):
                interval_rows.append(
                    {
                        "score_method": "gt",
                        "json_path": json_path,
                        "json_name": json_name,
                        "event_index": int(gt_idx),
                        "interval_kind": "gt",
                        "threshold": "",
                        "merge_gap_sec": "",
                        "min_duration_sec": "",
                        "min_match_iou": "",
                        "start_time_sec": float(event["start_time_sec"]),
                        "end_time_sec": float(event["end_time_sec"]),
                        "duration_sec": interval_duration_sec(event),
                        "start_frame": int(event.get("start_frame", 0)),
                        "end_frame": int(event.get("end_frame", 0)),
                    }
                )
            for pred_idx, event in enumerate(predicted_events, start=1):
                interval_rows.append(
                    {
                        "score_method": method_name,
                        "json_path": json_path,
                        "json_name": json_name,
                        "event_index": int(pred_idx),
                        "interval_kind": "predicted",
                        "threshold": threshold,
                        "merge_gap_sec": float(merge_gap_sec),
                        "min_duration_sec": float(min_duration_sec),
                        "min_match_iou": float(min_iou),
                        "start_time_sec": float(event["start_time_sec"]),
                        "end_time_sec": float(event["end_time_sec"]),
                        "duration_sec": interval_duration_sec(event),
                        "start_frame": int(event.get("start_frame", 0)),
                        "end_frame": int(event.get("end_frame", 0)),
                    }
                )

    aggregate = {
        "score_method": method_name,
        "threshold": threshold,
        "merge_gap_sec": float(merge_gap_sec),
        "min_duration_sec": float(min_duration_sec),
        "min_overlap_sec": float(min_overlap_sec),
        "min_match_iou": float(min_iou),
        **build_aggregate_event_stats(per_file_rows, match_rows, fragment_rows),
    }
    return {
        "aggregate": aggregate,
        "per_file": per_file_rows,
        "matches": match_rows,
        "fragments": fragment_rows,
        "intervals": interval_rows,
    }


def select_best_event_row(
    rows: list[dict[str, Any]],
    *,
    selection_score_method: str | None,
    selection_metric: str,
    max_false_alarms_per_min: float | None,
    min_recall: float | None,
) -> dict[str, Any] | None:
    if not rows:
        return None

    candidates = list(rows)
    if selection_score_method:
        method_candidates = [row for row in candidates if str(row.get("score_method")) == str(selection_score_method)]
        if method_candidates:
            candidates = method_candidates

    if max_false_alarms_per_min is not None:
        constrained = [
            row for row in candidates
            if row.get("false_alarms_per_min") is not None and safe_float(row["false_alarms_per_min"]) <= float(max_false_alarms_per_min)
        ]
        if constrained:
            candidates = constrained

    if min_recall is not None:
        constrained = [row for row in candidates if safe_float(row.get("recall")) >= float(min_recall)]
        if constrained:
            candidates = constrained

    metric = str(selection_metric)
    return max(
        candidates,
        key=lambda row: (
            safe_float(row.get(metric)),
            safe_float(row.get("time_iou")),
            safe_float(row.get("recall")),
            safe_float(row.get("precision")),
            -safe_float(row.get("false_alarms_per_min")),
        ),
    )


def evaluate_event_validation(
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
    calibration_report: dict[str, Any],
    raw_threshold: float,
    score_methods: list[str] | None = None,
    threshold_values: list[float] | None = None,
    merge_gap_sec_values: list[float] | None = None,
    min_duration_sec_values: list[float] | None = None,
    min_overlap_sec: float = 0.0,
    min_iou: float = 0.0,
    selection_score_method: str | None = "platt",
    selection_metric: str = "f1",
    max_false_alarms_per_min: float | None = None,
    min_recall: float | None = None,
) -> dict[str, Any]:
    methods = event_score_methods(
        calibration_report,
        raw_threshold=float(raw_threshold),
        score_methods=score_methods,
    )
    threshold_values = threshold_values or make_float_grid(start=0.01, end=0.99, step=0.01)
    merge_gap_sec_values = merge_gap_sec_values or [0.0]
    min_duration_sec_values = min_duration_sec_values or [0.0]

    bundles = predict_event_bundles(
        model,
        dataset_cls,
        paths,
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
    attach_calibrated_probs(bundles, methods)

    grid_rows: list[dict[str, Any]] = []
    for method in methods:
        for threshold in threshold_values:
            for merge_gap_sec in merge_gap_sec_values:
                for min_duration_sec in min_duration_sec_values:
                    result = evaluate_event_params(
                        bundles,
                        method,
                        threshold=float(threshold),
                        merge_gap_sec=float(merge_gap_sec),
                        min_duration_sec=float(min_duration_sec),
                        min_overlap_sec=float(min_overlap_sec),
                        min_iou=float(min_iou),
                        include_details=False,
                    )
                    grid_rows.append(result["aggregate"])

    selected = select_best_event_row(
        grid_rows,
        selection_score_method=selection_score_method,
        selection_metric=selection_metric,
        max_false_alarms_per_min=max_false_alarms_per_min,
        min_recall=min_recall,
    )
    selected_details = {"aggregate": {}, "per_file": [], "matches": [], "fragments": [], "intervals": []}
    if selected:
        selected_method = next(method for method in methods if method["score_method"] == selected["score_method"])
        selected_details = evaluate_event_params(
            bundles,
            selected_method,
            threshold=float(selected["threshold"]),
            merge_gap_sec=float(selected["merge_gap_sec"]),
            min_duration_sec=float(selected["min_duration_sec"]),
            min_overlap_sec=float(selected["min_overlap_sec"]),
            min_iou=float(selected["min_match_iou"]),
            include_details=True,
        )

    return {
        "settings": {
            "paths": len(paths),
            "score_methods": [method["score_method"] for method in methods],
            "threshold_values": threshold_values,
            "merge_gap_sec_values": merge_gap_sec_values,
            "min_duration_sec_values": min_duration_sec_values,
            "min_overlap_sec": float(min_overlap_sec),
            "min_match_iou": float(min_iou),
            "selection_score_method": selection_score_method,
            "selection_metric": selection_metric,
            "max_false_alarms_per_min": max_false_alarms_per_min,
            "min_recall": min_recall,
        },
        "grid": grid_rows,
        "selected": selected or {},
        "selected_details": selected_details,
    }


def save_rows_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_event_report(out_dir: Path, report: dict[str, Any], *, prefix: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    details = report.get("selected_details", {})
    artifact_paths = {
        "grid_csv": str(out_dir / f"{prefix}_event_grid.csv"),
        "per_file_csv": str(out_dir / f"{prefix}_event_per_file.csv"),
        "matches_csv": str(out_dir / f"{prefix}_event_matches.csv"),
        "fragments_csv": str(out_dir / f"{prefix}_event_fragments.csv"),
        "intervals_csv": str(out_dir / f"{prefix}_event_intervals.csv"),
        "summary_json": str(out_dir / f"{prefix}_event_summary.json"),
    }
    save_rows_csv(Path(artifact_paths["grid_csv"]), report.get("grid", []))
    save_rows_csv(Path(artifact_paths["per_file_csv"]), details.get("per_file", []))
    save_rows_csv(Path(artifact_paths["matches_csv"]), details.get("matches", []))
    save_rows_csv(Path(artifact_paths["fragments_csv"]), details.get("fragments", []))
    save_rows_csv(Path(artifact_paths["intervals_csv"]), details.get("intervals", []))
    payload = {
        "settings": report.get("settings", {}),
        "selected": report.get("selected", {}),
        "selected_aggregate": details.get("aggregate", {}),
        "artifacts": artifact_paths,
    }
    Path(artifact_paths["summary_json"]).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact_paths
