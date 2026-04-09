# src/train.py
from __future__ import annotations
import argparse
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- your project imports ---
# Expect these modules to exist after your refactor:
from src.data.path_utils import collect_json_paths
from src.data.splits import (
    read_paths_txt,
    split_paths,
    write_paths_txt,
    write_split_files,
)
from src.data.dataset import EventJsonDataset, MotionJsonDataset, resolve_window_label_cfg
from src.data.json_io import read_json_frames
from src.data.labels import events_to_label, resolve_label_cfg
from src.data.features import motion_feature_dim
from src.models.tcn import EventTCN, MotionTCN, resolve_encoder_type
from src.utils.metrics import (
    apply_platt_scaling,
    apply_temperature_scaling,
    compute_auroc,
    compute_auprc,
    compute_pr_points,
    compute_roc_points,
    fit_platt_scaling,
    fit_temperature_scaling,
    threshold_sweep_binary,
    confusion_stats_at_threshold,
    save_reliability_diagram_image,
    save_history_csv,
    save_rows_csv,
    save_summary_csv,
    save_threshold_sweep_csv,
    save_learning_curves,
    save_confusion_matrix_image,
    save_pr_curve_image,
    save_roc_curve_image,
    sigmoid_from_logits,
    summarize_calibration,
)
from src.utils.event_eval import (
    evaluate_event_validation,
    make_float_grid,
    save_event_report,
)

#from sklearn.metrics import average_precision_score, precision_recall_curve
# ----------------------------
# Helpers
# ----------------------------

import math
import torch.nn.functional as F

def aggregate_window_logits(logits_bt: torch.Tensor, mode: str = "logsumexp", temperature: float = 1.0) -> torch.Tensor:
    """
    logits_bt: (B,T) -> (B,)
    mode:
      - last
      - max
      - logsumexp
      - logmeanexp  (recommended: logsumexp - log(T))
    """
    mode = mode.lower()
    if mode == "last":
        return logits_bt[:, -1]
    if mode == "max":
        return logits_bt.max(dim=1).values

    t = float(temperature)
    lse = torch.logsumexp(logits_bt / t, dim=1) * t
    if mode == "logsumexp":
        return lse
    if mode == "logmeanexp":
        return lse - math.log(logits_bt.size(1))
    raise ValueError(f"Unknown agg mode: {mode}")


def end_weighted_bce_loss(logits_bt: torch.Tensor, y_b: torch.Tensor, pos_weight: torch.Tensor, k_last: int = 4) -> torch.Tensor:
    """Aux BCE over last K timesteps (pushes 'now' to match the window label)."""
    B, T = logits_bt.shape
    k = int(min(max(k_last, 1), T))
    logits_end = logits_bt[:, -k:]                  # (B,k)
    y_end = y_b.view(B, 1).expand(B, k).float()     # (B,k)
    return F.binary_cross_entropy_with_logits(logits_end, y_end, pos_weight=pos_weight, reduction="mean")
def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "no_git_commit"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def project_root() -> Path:
    # src/train.py -> src -> project root
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str, base: Path) -> Path:
    # If p is absolute -> keep. If relative -> base/p.
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def make_run_dir(runs_root: Path, exp_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = runs_root / f"{ts}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def threshold_for_precision(sweep, target_precision=0.5):
    candidates = [r for r in sweep if r["precision"] >= target_precision]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["recall"])


def normalize_paths(paths: List[Path | str]) -> List[str]:
    return [str(Path(p).resolve()) for p in paths]


def require_existing_paths(paths: List[str], label: str) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing {label} paths ({len(missing)} total). First entries:\n{preview}")


def assert_disjoint(a_name: str, a_paths: List[str], b_name: str, b_paths: List[str]) -> None:
    overlap = sorted(set(a_paths) & set(b_paths))
    if overlap:
        preview = "\n".join(overlap[:5])
        raise ValueError(f"{a_name} and {b_name} overlap ({len(overlap)} files). First entries:\n{preview}")


def resolve_window_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    window_cfg = dict(cfg.get("window_labels", {}) or {})

    if "rule" not in window_cfg and data_cfg.get("window_label_rule") is not None:
        window_cfg["rule"] = data_cfg.get("window_label_rule")
    if "positive_overlap" not in window_cfg:
        if data_cfg.get("window_positive_overlap") is not None:
            window_cfg["positive_overlap"] = data_cfg.get("window_positive_overlap")
        elif data_cfg.get("window_positive_overlap_fraction") is not None:
            window_cfg["positive_overlap"] = data_cfg.get("window_positive_overlap_fraction")

    return resolve_window_label_cfg(window_cfg)


def resolve_label_cfg_from_root(cfg: Dict[str, Any]) -> Dict[str, Any]:
    label_cfg = dict(cfg.get("labels", {}) or {})
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

    if "mode" not in label_cfg and data_cfg.get("label_mode") is not None:
        label_cfg["mode"] = data_cfg.get("label_mode")

    return resolve_label_cfg(label_cfg)


def warn_ignored_cfg_fields(cfg: Dict[str, Any], model_type: str, split_mode: str) -> None:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    feature_cfg = cfg.get("features", {}) if isinstance(cfg, dict) else {}
    split_cfg = data_cfg.get("split", {}) if isinstance(data_cfg, dict) else {}

    if split_mode in {"generate", "train_test_lists"} and split_cfg.get("group_key") is not None:
        print(
            f"[WARN] data.split.group_key={split_cfg.get('group_key')!r} is currently ignored by the split implementation; "
            "generated splits are path-level.",
            flush=True,
        )

    if model_type not in {"motion_tcn", "erez_motion_tcn"}:
        ignored_feature_keys = [
            key for key in ("use_bbox", "use_keypoints", "use_quality_gates", "normalize")
            if key in feature_cfg
        ]
        if ignored_feature_keys:
            print(
                f"[WARN] EventTCN input currently ignores features.{', features.'.join(ignored_feature_keys)}; "
                "the per-person pose vector format is still fixed in frame_to_vector.",
                flush=True,
            )

        ignored_data_keys = [key for key in ("det_conf_keep", "kp_conf_keep") if key in data_cfg]
        if ignored_data_keys:
            print(
                f"[WARN] Training input currently ignores data.{', data.'.join(ignored_data_keys)}.",
                flush=True,
            )


def build_dataset(
    dataset_cls,
    paths: List[str],
    *,
    K: int,
    window_size: int,
    window_step: int,
    feature_cfg: Dict[str, Any],
    label_cfg: Dict[str, Any],
    window_cfg: Dict[str, Any],
    target_mode: str,
    verbose: bool = True,
):
    return dataset_cls(
        paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
        verbose=verbose,
    )


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def targets_for_binary_metrics(y_true: np.ndarray, positive_threshold: float = 0.0) -> np.ndarray:
    targets = np.asarray(y_true, dtype=np.float32).reshape(-1)
    finite = targets[np.isfinite(targets)]
    if finite.size > 0 and np.all(np.isin(np.unique(finite), [0.0, 1.0])):
        return targets.astype(np.int32)
    return (targets > float(positive_threshold)).astype(np.int32)


def fpr_from_stats(stats: Dict[str, float]) -> float:
    denom = float(stats["fp"] + stats["tn"])
    return float(stats["fp"] / denom) if denom > 0 else float("nan")


def infer_video_label(json_path: str, label_cfg: Dict[str, Any]) -> Dict[str, int]:
    frames = read_json_frames(json_path)
    frame_labels = np.asarray([events_to_label(frame, cfg=label_cfg) for frame in frames], dtype=np.int32)
    n_frames = int(len(frame_labels))
    n_pos_frames = int(np.sum(frame_labels == 1))
    n_neg_frames = int(np.sum(frame_labels == 0))
    return {
        "video_label": int(n_pos_frames > 0),
        "n_frames": n_frames,
        "n_pos_frames": n_pos_frames,
        "n_neg_frames": n_neg_frames,
    }


def summarize_binary_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    selected_threshold: float | None = None,
) -> Dict[str, Any]:
    y_target = np.asarray(y_true, dtype=np.float32)
    y_true = targets_for_binary_metrics(y_target)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_true.size == 0:
        empty_threshold = float("nan") if selected_threshold is None else float(selected_threshold)
        empty_stats = {
            "threshold": empty_threshold,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
        return {
            "n_samples": 0,
            "n_pos": 0,
            "n_neg": 0,
            "auprc": float("nan"),
            "auroc": float("nan"),
            "mean_prob_pos": float("nan"),
            "mean_prob_neg": float("nan"),
            "brier_score": float("nan"),
            "brier_baseline": float("nan"),
            "brier_skill_score": float("nan"),
            "ece": float("nan"),
            "max_calibration_error": float("nan"),
            "calibration": {
                "n_samples": 0,
                "n_bins": 10,
                "n_pos": 0,
                "n_neg": 0,
                "positive_rate": float("nan"),
                "brier_score": float("nan"),
                "brier_baseline": float("nan"),
                "brier_skill_score": float("nan"),
                "ece": float("nan"),
                "max_calibration_error": float("nan"),
            },
            "calibration_bins": [],
            "selected_threshold": empty_threshold,
            "selected_threshold_stats": {
                **empty_stats,
                "fpr": float("nan"),
            },
            "oracle_best_f1": {
                "threshold": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
            },
            "oracle_best_threshold_stats": {
                **empty_stats,
                "fpr": float("nan"),
            },
            "sweep": [],
        }

    auprc = compute_auprc(y_true, y_prob)
    auroc = compute_auroc(y_true, y_prob)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    sweep = threshold_sweep_binary(y_true, y_prob, start=0.01, end=0.99, step=0.01)
    best = max(sweep, key=lambda row: row["f1"])
    oracle_stats = confusion_stats_at_threshold(y_true, y_prob, threshold=float(best["threshold"]))
    calibration = summarize_calibration(y_target, y_prob, n_bins=10)
    calibration_without_bins = {key: value for key, value in calibration.items() if key != "bins"}

    resolved_threshold = selected_threshold
    if resolved_threshold is None or not np.isfinite(float(resolved_threshold)):
        resolved_threshold = float(best["threshold"])
    resolved_threshold = float(resolved_threshold)
    selected_stats = confusion_stats_at_threshold(y_true, y_prob, threshold=resolved_threshold)

    return {
        "n_samples": int(len(y_true)),
        "n_pos": int(np.sum(pos_mask)),
        "n_neg": int(np.sum(neg_mask)),
        "auprc": safe_float(auprc),
        "auroc": safe_float(auroc),
        "mean_prob_pos": float(np.mean(y_prob[pos_mask])) if np.any(pos_mask) else float("nan"),
        "mean_prob_neg": float(np.mean(y_prob[neg_mask])) if np.any(neg_mask) else float("nan"),
        "brier_score": safe_float(calibration["brier_score"]),
        "brier_baseline": safe_float(calibration["brier_baseline"]),
        "brier_skill_score": safe_float(calibration["brier_skill_score"]),
        "ece": safe_float(calibration["ece"]),
        "max_calibration_error": safe_float(calibration["max_calibration_error"]),
        "calibration": calibration_without_bins,
        "calibration_bins": calibration["bins"],
        "selected_threshold": resolved_threshold,
        "selected_threshold_stats": {
            **selected_stats,
            "fpr": fpr_from_stats(selected_stats),
        },
        "oracle_best_f1": {key: float(value) for key, value in best.items()},
        "oracle_best_threshold_stats": {
            **oracle_stats,
            "fpr": fpr_from_stats(oracle_stats),
        },
        "sweep": sweep,
    }


def summarize_video_scores(
    rows: List[Dict[str, Any]],
    *,
    score_key: str,
    selected_threshold: float | None = None,
) -> Dict[str, Any]:
    valid_rows = [row for row in rows if np.isfinite(safe_float(row.get(score_key, float("nan"))))]
    y_true = np.asarray([int(row["video_label"]) for row in valid_rows], dtype=np.int32)
    y_prob = np.asarray([float(row[score_key]) for row in valid_rows], dtype=np.float32)
    summary = summarize_binary_scores(y_true, y_prob, selected_threshold=selected_threshold)
    summary["score_key"] = score_key
    summary["n_videos_total"] = int(len(rows))
    summary["n_videos_eval"] = int(len(valid_rows))
    return summary


def compact_binary_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in summary.items() if key not in {"sweep"}}


def save_probability_calibration_artifacts(
    out_dir: Path,
    *,
    prefix: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    n_bins: int = 10,
) -> Dict[str, Any]:
    calibration = summarize_calibration(y_true, y_prob, n_bins=n_bins)
    save_rows_csv(out_dir / f"{prefix}_calibration_bins.csv", calibration["bins"])
    save_reliability_diagram_image(
        out_dir / f"{prefix}_reliability.png",
        y_true=y_true,
        y_prob=y_prob,
        title=title,
        n_bins=n_bins,
    )
    return calibration


def build_probability_calibration_report(
    val_targets: np.ndarray,
    val_logits: np.ndarray,
    test_targets: np.ndarray,
    test_logits: np.ndarray,
) -> Dict[str, Any]:
    val_targets = np.asarray(val_targets, dtype=np.float32)
    test_targets = np.asarray(test_targets, dtype=np.float32)
    val_logits = np.asarray(val_logits, dtype=np.float32)
    test_logits = np.asarray(test_logits, dtype=np.float32)
    val_raw_probs = sigmoid_from_logits(val_logits)
    test_raw_probs = sigmoid_from_logits(test_logits)

    report: Dict[str, Any] = {
        "raw": {
            "validation": compact_binary_summary(summarize_binary_scores(val_targets, val_raw_probs)),
            "test": compact_binary_summary(summarize_binary_scores(test_targets, test_raw_probs)),
        }
    }

    temperature_fit = fit_temperature_scaling(val_targets, val_logits)
    report["temperature"] = {"fit": temperature_fit}
    if temperature_fit.get("available"):
        temperature = float(temperature_fit["temperature"])
        val_temp_probs = apply_temperature_scaling(val_logits, temperature)
        test_temp_probs = apply_temperature_scaling(test_logits, temperature)
        val_temp_summary = summarize_binary_scores(val_targets, val_temp_probs)
        temp_threshold = float(val_temp_summary["oracle_best_f1"]["threshold"])
        test_temp_summary = summarize_binary_scores(test_targets, test_temp_probs, selected_threshold=temp_threshold)
        report["temperature"].update(
            {
                "selected_threshold": temp_threshold,
                "validation": compact_binary_summary(val_temp_summary),
                "test": compact_binary_summary(test_temp_summary),
            }
        )

    platt_fit = fit_platt_scaling(val_targets, val_logits)
    report["platt"] = {"fit": platt_fit}
    if platt_fit.get("available"):
        slope = float(platt_fit["slope"])
        intercept = float(platt_fit["intercept"])
        val_platt_probs = apply_platt_scaling(val_logits, slope=slope, intercept=intercept)
        test_platt_probs = apply_platt_scaling(test_logits, slope=slope, intercept=intercept)
        val_platt_summary = summarize_binary_scores(val_targets, val_platt_probs)
        platt_threshold = float(val_platt_summary["oracle_best_f1"]["threshold"])
        test_platt_summary = summarize_binary_scores(test_targets, test_platt_probs, selected_threshold=platt_threshold)
        report["platt"].update(
            {
                "selected_threshold": platt_threshold,
                "validation": compact_binary_summary(val_platt_summary),
                "test": compact_binary_summary(test_platt_summary),
            }
        )

    return report


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def list_of_floats(value: Any, default: List[float]) -> List[float]:
    if value is None:
        return [float(item) for item in default]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def list_of_strings(value: Any, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def resolve_event_validation_cfg(
    cfg: Dict[str, Any],
    *,
    root: Path,
    data_root: Path,
    val_paths: List[str],
    batch_size: int,
) -> Dict[str, Any]:
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg, dict) else {}
    event_cfg = dict(eval_cfg.get("event_validation", {}) or {}) if isinstance(eval_cfg, dict) else {}
    enabled = bool(event_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    event_paths = list(val_paths)
    list_value = event_cfg.get("list", event_cfg.get("paths_list", None))
    if list_value:
        event_list = resolve_path(str(list_value), root)
        event_paths = normalize_paths(read_paths_txt(event_list, base_dirs=[data_root, root]))
        require_existing_paths(event_paths, "event validation")

    if not event_paths:
        raise ValueError("eval.event_validation.enabled=true but no event validation paths were resolved.")

    threshold_values = make_float_grid(
        event_cfg.get("thresholds", None),
        start=float(event_cfg.get("threshold_start", 0.01)),
        end=float(event_cfg.get("threshold_end", 0.99)),
        step=float(event_cfg.get("threshold_step", 0.02)),
    )

    return {
        "enabled": True,
        "paths": event_paths,
        "every_n_epochs": max(1, int(event_cfg.get("every_n_epochs", 1))),
        "batch_size": int(event_cfg.get("batch_size", batch_size)),
        "score_methods": list_of_strings(event_cfg.get("score_methods", None), ["raw", "temperature", "platt"]),
        "threshold_values": threshold_values,
        "merge_gap_sec_values": list_of_floats(event_cfg.get("merge_gap_sec_values", None), [0.0, 0.5, 1.0, 2.0]),
        "min_duration_sec_values": list_of_floats(event_cfg.get("min_duration_sec_values", None), [0.0, 0.5, 1.0]),
        "min_overlap_sec": float(event_cfg.get("min_overlap_sec", 0.0)),
        "min_match_iou": float(event_cfg.get("min_match_iou", event_cfg.get("min_iou", 0.0))),
        "selection_score_method": event_cfg.get("selection_score_method", "platt"),
        "selection_metric": str(event_cfg.get("selection_metric", "f1")),
        "max_false_alarms_per_min": optional_float(event_cfg.get("max_false_alarms_per_min", None)),
        "min_recall": optional_float(event_cfg.get("min_recall", None)),
    }


def build_validation_calibration_report(val_out: Dict[str, Any]) -> Dict[str, Any]:
    val_targets = np.asarray(val_out["targets"], dtype=np.float32)
    val_logits = np.asarray(val_out["logits"], dtype=np.float32)
    return build_probability_calibration_report(val_targets, val_logits, val_targets, val_logits)


def empty_event_history(prefix: str = "val_event") -> Dict[str, Any]:
    return {
        f"{prefix}_score_method": "",
        f"{prefix}_threshold": float("nan"),
        f"{prefix}_merge_gap_sec": float("nan"),
        f"{prefix}_min_duration_sec": float("nan"),
        f"{prefix}_min_overlap_sec": float("nan"),
        f"{prefix}_min_match_iou": float("nan"),
        f"{prefix}_precision": float("nan"),
        f"{prefix}_recall": float("nan"),
        f"{prefix}_f1": float("nan"),
        f"{prefix}_false_alarms_per_min": float("nan"),
        f"{prefix}_gt_hit_recall": float("nan"),
        f"{prefix}_time_coverage": float("nan"),
        f"{prefix}_time_iou": float("nan"),
        f"{prefix}_n_predicted_events": float("nan"),
        f"{prefix}_n_gt_events": float("nan"),
        f"{prefix}_tp": float("nan"),
        f"{prefix}_fp": float("nan"),
        f"{prefix}_fn": float("nan"),
    }


def event_history_from_selected(selected: Dict[str, Any] | None, prefix: str = "val_event") -> Dict[str, Any]:
    row = empty_event_history(prefix=prefix)
    if not selected:
        return row

    row[f"{prefix}_score_method"] = str(selected.get("score_method", ""))
    for key in (
        "threshold",
        "merge_gap_sec",
        "min_duration_sec",
        "min_overlap_sec",
        "min_match_iou",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_min",
        "gt_hit_recall",
        "time_coverage",
        "time_iou",
        "n_predicted_events",
        "n_gt_events",
        "tp",
        "fp",
        "fn",
    ):
        row[f"{prefix}_{key}"] = safe_float(selected.get(key, float("nan")))
    return row


def compact_event_report(report: Dict[str, Any] | None, artifacts: Dict[str, str] | None = None) -> Dict[str, Any]:
    if not report:
        return {}
    payload = {
        "settings": report.get("settings", {}),
        "selected": report.get("selected", {}),
        "selected_aggregate": report.get("selected_details", {}).get("aggregate", {}),
    }
    if artifacts:
        payload["artifacts"] = artifacts
    return payload


def predict_from_score(score: Any, threshold: Any) -> int | None:
    score_f = safe_float(score)
    threshold_f = safe_float(threshold)
    if not np.isfinite(score_f) or not np.isfinite(threshold_f):
        return None
    return int(score_f >= threshold_f)


def build_per_video_rows(
    rows: List[Dict[str, Any]],
    *,
    threshold_mean: float,
    threshold_max: float,
) -> List[Dict[str, Any]]:
    per_video_rows: List[Dict[str, Any]] = []
    for row in rows:
        per_video_rows.append(
            {
                "json_path": row["json_path"],
                "json_name": row["json_name"],
                "video_label": int(row["video_label"]),
                "n_frames": int(row["n_frames"]),
                "n_pos_frames": int(row["n_pos_frames"]),
                "n_neg_frames": int(row["n_neg_frames"]),
                "n_windows": int(row["n_windows"]),
                "n_pos_windows": int(row["n_pos"]),
                "n_neg_windows": int(row["n_neg"]),
                "video_score_mean": safe_float(row["video_score_mean"]),
                "val_video_threshold_mean": safe_float(threshold_mean),
                "video_pred_mean_at_val_thr": predict_from_score(row["video_score_mean"], threshold_mean),
                "video_score_max": safe_float(row["video_score_max"]),
                "val_video_threshold_max": safe_float(threshold_max),
                "video_pred_max_at_val_thr": predict_from_score(row["video_score_max"], threshold_max),
            }
        )
    return per_video_rows


def evaluate_paths_individually(
    model: nn.Module,
    dataset_cls,
    paths: List[str],
    *,
    K: int,
    window_size: int,
    window_step: int,
    feature_cfg: Dict[str, Any],
    label_cfg: Dict[str, Any],
    window_cfg: Dict[str, Any],
    target_mode: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    agg_mode: str,
    temperature: float,
    selected_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for json_path in paths:
        rows.append(
            evaluate_single_path(
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
                num_workers=num_workers,
                device=device,
                agg_mode=agg_mode,
                temperature=temperature,
                selected_threshold=selected_threshold,
            )
        )
    return rows


def evaluate_single_path(
    model: nn.Module,
    dataset_cls,
    json_path: str,
    *,
    K: int,
    window_size: int,
    window_step: int,
    feature_cfg: Dict[str, Any],
    label_cfg: Dict[str, Any],
    window_cfg: Dict[str, Any],
    target_mode: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    agg_mode: str,
    temperature: float,
    selected_threshold: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "json_path": json_path,
        "json_name": Path(json_path).name,
        **infer_video_label(json_path, label_cfg),
    }
    ds = build_dataset(
        dataset_cls,
        [json_path],
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
        verbose=False,
    )
    row["n_windows"] = int(len(ds))
    if len(ds) == 0:
        row.update(
            {
                "n_pos": 0,
                "n_neg": 0,
                "auprc": float("nan"),
                "auroc": float("nan"),
                "val_threshold": selected_threshold,
                "val_thr_precision": float("nan"),
                "val_thr_recall": float("nan"),
                "val_thr_f1": float("nan"),
                "val_thr_accuracy": float("nan"),
                "val_thr_fpr": float("nan"),
                "oracle_threshold": float("nan"),
                "oracle_f1": float("nan"),
                "mean_prob_pos": float("nan"),
                "mean_prob_neg": float("nan"),
                "brier_score": float("nan"),
                "ece": float("nan"),
                "video_score_mean": float("nan"),
                "video_score_max": float("nan"),
            }
        )
        return row

    loader = make_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    out = evaluate_binary(model, loader, device=device, agg_mode=agg_mode, temperature=temperature)
    targets = np.asarray(out["binary_targets"], dtype=np.int32)
    probs = np.asarray(out["probs"], dtype=np.float32)
    selected_stats = confusion_stats_at_threshold(targets, probs, threshold=selected_threshold)
    row.update(
        {
            "n_pos": int(np.sum(targets == 1)),
            "n_neg": int(np.sum(targets == 0)),
            "auprc": safe_float(out["auprc"]),
            "auroc": safe_float(out["auroc"]),
            "val_threshold": selected_threshold,
            "val_thr_precision": safe_float(selected_stats["precision"]),
            "val_thr_recall": safe_float(selected_stats["recall"]),
            "val_thr_f1": safe_float(selected_stats["f1"]),
            "val_thr_accuracy": safe_float(selected_stats["accuracy"]),
            "val_thr_fpr": fpr_from_stats(selected_stats),
            "oracle_threshold": safe_float(out["best_f1"]["threshold"]),
            "oracle_f1": safe_float(out["best_f1"]["f1"]),
            "mean_prob_pos": safe_float(out["mean_prob_pos"]),
            "mean_prob_neg": safe_float(out["mean_prob_neg"]),
            "brier_score": safe_float(out["brier_score"]),
            "ece": safe_float(out["ece"]),
            "video_score_mean": safe_float(np.mean(probs)),
            "video_score_max": safe_float(np.max(probs)),
        }
    )
    return row

# ----------------------------
# Training / Eval loops
# ----------------------------
@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device, agg_mode="logsumexp", temperature=1.0,) -> Dict[str, Any]:
    model.eval()
    all_logits = []
    all_targets = []

    for X, y in loader:
        X = X.to(device)  # (B,T,C)
        y = y.to(device).float()  # (B,)
        #logits = model(X).view(-1)  # (B,)
        logits_bt = model(X)  # (B,T)
        win_logits = aggregate_window_logits(logits_bt, mode=agg_mode, temperature=temperature)  # (B,) 
        all_logits.append(win_logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy().astype(np.float32)
    binary_targets = targets_for_binary_metrics(targets)

    probs = sigmoid_from_logits(logits)
    auprc = compute_auprc(binary_targets, probs)
    auroc = compute_auroc(binary_targets, probs)
    calibration = summarize_calibration(targets, probs, n_bins=10)

    pos_mask = binary_targets == 1
    neg_mask = binary_targets == 0
    mean_prob_pos = float(np.mean(probs[pos_mask])) if np.any(pos_mask) else float("nan")
    mean_prob_neg = float(np.mean(probs[neg_mask])) if np.any(neg_mask) else float("nan")

    sweep = threshold_sweep_binary(binary_targets, probs, start=0.01, end=0.99, step=0.01)
    best = max(sweep, key=lambda r: r["f1"])
    best_threshold_stats = confusion_stats_at_threshold(binary_targets, probs, threshold=float(best["threshold"]))
    fixed_thresholds = {
        "0.3": confusion_stats_at_threshold(binary_targets, probs, threshold=0.3),
        "0.5": confusion_stats_at_threshold(binary_targets, probs, threshold=0.5),
        "0.7": confusion_stats_at_threshold(binary_targets, probs, threshold=0.7),
    }

    return {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "best_f1": {k: float(v) for k, v in best.items()},
        "best_threshold_stats": best_threshold_stats,
        "mean_prob_pos": mean_prob_pos,
        "mean_prob_neg": mean_prob_neg,
        "brier_score": safe_float(calibration["brier_score"]),
        "brier_baseline": safe_float(calibration["brier_baseline"]),
        "brier_skill_score": safe_float(calibration["brier_skill_score"]),
        "ece": safe_float(calibration["ece"]),
        "max_calibration_error": safe_float(calibration["max_calibration_error"]),
        "calibration": {key: value for key, value in calibration.items() if key != "bins"},
        "calibration_bins": calibration["bins"],
        "fixed_thresholds": fixed_thresholds,
        "targets": targets.tolist(),
        "binary_targets": binary_targets.tolist(),
        "logits": logits.tolist(),
        "probs": probs.tolist(),
        "n_samples": int(len(targets)),
        "sweep": sweep,  # list of dicts
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
    agg_mode="logsumexp",
    temperature=1.0,
    end_loss_alpha=0.5,
    end_loss_k=4
) -> float:
    model.train()
    losses = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float()  # BCE expects float targets

        optim.zero_grad(set_to_none=True)
        logits_bt = model(X)  # (B,T)

        #win_logits = aggregate_window_logits(logits_bt)  # (B,)mode=agg_mode,,  temperature=temperature
        win_logits = aggregate_window_logits(
                                                logits_bt,
                                                mode=agg_mode,
                                                temperature=temperature,
                                            )
        loss_main = criterion(win_logits, y)
        
        if end_loss_alpha > 0:
            pos_weight = getattr(criterion, "pos_weight", torch.tensor(1.0, device=device))
            loss_end = end_weighted_bce_loss(logits_bt, y, pos_weight=pos_weight, k_last=end_loss_k)
            loss = (1 - end_loss_alpha) * loss_main + end_loss_alpha * loss_end
        else:
            loss = loss_main
                #logits = model(X).view(-1)  # (B,)
                #loss = criterion(logits, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optim.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    preview_data_cfg = cfg.get("data", {})
    preview_model_cfg = cfg.get("model", {})
    preview_train_cfg = cfg.get("train", {})
    target_mode = str(
        preview_data_cfg.get(
            "target_mode",
            "last" if bool(preview_model_cfg.get("causal", True)) else "center",
        )
    ).lower()
    save_all_checkpoints = bool(preview_train_cfg.get("save_all_checkpoints", True))
    if target_mode not in {"center", "last"}:
        raise ValueError(f"Unsupported data.target_mode: {target_mode}")

    # Resolve base roots
    root = project_root()

    # 1) Resolve data_root / runs_root from config (simple mode)
    # You can later extend this to merge with local_paths.yaml.
    data_root = resolve_path(cfg["paths"]["data_root"], root)
    runs_root = resolve_path(cfg["paths"]["runs_root"], root)
    runs_root.mkdir(parents=True, exist_ok=True)

    # 2) Seed
    seed = int(cfg.get("exp", {}).get("seed", 42))
    set_seed(seed)

    # 3) Create run folder + save resolved config
    exp_name = cfg.get("exp", {}).get("name", "exp")
    run_dir = make_run_dir(runs_root, exp_name)
    
    checkpoints_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    reports_dir = run_dir / "reports"
    splits_dir = run_dir / "splits"

    for d in [checkpoints_dir, metrics_dir, reports_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    cfg_resolved = dict(cfg)
    cfg_resolved["paths"] = dict(cfg_resolved.get("paths", {}))
    cfg_resolved["paths"]["data_root"] = str(data_root)
    cfg_resolved["paths"]["runs_root"] = str(runs_root)
    cfg_resolved["data"] = dict(cfg_resolved.get("data", {}))
    cfg_resolved["data"]["target_mode"] = target_mode
    window_cfg = resolve_window_cfg(cfg)
    cfg_resolved["window_labels"] = dict(window_cfg)
    label_cfg = resolve_label_cfg_from_root(cfg)
    cfg_resolved["labels"] = dict(label_cfg)
    cfg_resolved["train"] = dict(cfg_resolved.get("train", {}))
    cfg_resolved["train"]["save_all_checkpoints"] = save_all_checkpoints
    cfg_resolved["run_dir"] = str(run_dir)

    save_yaml(cfg_resolved, run_dir / "config_resolved.yaml")
    #(run_dir / "git_commit.txt").write_text(get_git_commit() + "\n", encoding="utf-8")

    # 4) Collect json paths
    # Expect jsons under data_root (possibly nested).
    all_jsons = normalize_paths(collect_json_paths(data_root))
    if len(all_jsons) == 0:
        raise RuntimeError(f"No jsons found under {data_root}")
    print(f"[INFO] data_root: {data_root}", flush=True)
    print(f"[INFO] found {len(all_jsons)} json files under data_root", flush=True)

    # 5) Make or load split
    split_cfg = cfg.get("data", {}).get("split", {"mode": "generate"})
    split_mode = str(split_cfg.get("mode", "generate")).lower()
    split_seed = int(split_cfg.get("seed", seed))
    test_paths: List[str] = []

    if split_mode == "fixed":
        train_list = resolve_path(split_cfg["train_list"], root)
        val_list = resolve_path(split_cfg["val_list"], root)
        train_paths = normalize_paths(read_paths_txt(train_list, base_dirs=[data_root, root]))
        val_paths = normalize_paths(read_paths_txt(val_list, base_dirs=[data_root, root]))
        test_list_value = split_cfg.get("test_list", None)
        if test_list_value:
            test_list = resolve_path(test_list_value, root)
            test_paths = normalize_paths(read_paths_txt(test_list, base_dirs=[data_root, root]))

        require_existing_paths(train_paths, "train")
        require_existing_paths(val_paths, "val")
        if test_paths:
            require_existing_paths(test_paths, "test")

        assert_disjoint("train", train_paths, "val", val_paths)
        if test_paths:
            assert_disjoint("train", train_paths, "test", test_paths)
            assert_disjoint("val", val_paths, "test", test_paths)

        write_split_files(
            train_paths,
            val_paths,
            out_dir=splits_dir,
            train_name="train_paths.txt",
            val_name="val_paths.txt",
        )
        if test_paths:
            write_paths_txt(test_paths, splits_dir / "test_paths.txt")
    elif split_mode == "train_test_lists":
        train_list = resolve_path(split_cfg["train_list"], root)
        test_list = resolve_path(split_cfg["test_list"], root)
        train_pool_paths = normalize_paths(read_paths_txt(train_list, base_dirs=[data_root, root]))
        test_paths = normalize_paths(read_paths_txt(test_list, base_dirs=[data_root, root]))
        require_existing_paths(train_pool_paths, "train_pool")
        require_existing_paths(test_paths, "test")
        assert_disjoint("train_pool", train_pool_paths, "test", test_paths)

        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        train_paths, val_paths = split_paths(train_pool_paths, val_size=val_ratio, seed=split_seed)
        train_paths = normalize_paths(train_paths)
        val_paths = normalize_paths(val_paths)
        assert_disjoint("train", train_paths, "val", val_paths)

        write_split_files(
            train_paths,
            val_paths,
            out_dir=splits_dir,
            train_name="train_paths.txt",
            val_name="val_paths.txt",
        )
        write_paths_txt(test_paths, splits_dir / "test_paths.txt")
    else:
        # generate per run
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        train_paths, val_paths = split_paths(
            all_jsons, val_size=val_ratio, seed=split_seed
        )
        train_paths = normalize_paths(train_paths)
        val_paths = normalize_paths(val_paths)
        write_split_files(
            train_paths, val_paths,
            out_dir=splits_dir,
            train_name="train_paths.txt",
            val_name="val_paths.txt",
            #base=data_root,  # write relative paths to data_root
        )

    print(f"[INFO] run_dir:   {run_dir}", flush=True)
    print(
        f"[INFO] split_mode: {split_mode} | {len(train_paths)} train_jsons | {len(val_paths)} val_jsons"
        + (f" | {len(test_paths)} test_jsons" if test_paths else ""),
        flush=True,
    )

    # 6) Build datasets
    K = int(cfg["data"].get("max_persons", 25))
    window_size = int(cfg["data"].get("window_size", 16))
    window_step = int(cfg["data"].get("window_step", 4))

    feature_cfg = cfg.get("features", {})
    label_cfg = resolve_label_cfg_from_root(cfg)
    window_cfg = resolve_window_cfg(cfg)
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "tcn")).lower()
    motion_dim = motion_feature_dim(feature_cfg)
    dataset_cls = MotionJsonDataset if model_type in {"motion_tcn", "erez_motion_tcn"} else EventJsonDataset
    warn_ignored_cfg_fields(cfg, model_type=model_type, split_mode=split_mode)

    print("[INFO] Building training dataset...", flush=True)
    train_ds = build_dataset(
        dataset_cls,
        train_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        window_cfg=window_cfg,
        target_mode=target_mode,
    )
    print("[INFO] Building validation dataset...", flush=True)
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
    test_ds = None
    if test_paths:
        print("[INFO] Building test dataset...", flush=True)
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

    # Quick sanity prints
    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] run_dir:   {run_dir}")
    json_counts = f"{len(all_jsons)} total | {len(train_paths)} train_jsons | {len(val_paths)} val_jsons"
    window_counts = f"{len(train_ds)} train_windows | {len(val_ds)} val_windows"
    if test_ds is not None:
        json_counts += f" | {len(test_paths)} test_jsons"
        window_counts += f" | {len(test_ds)} test_windows"
    print(f"[INFO] split_mode: {split_mode}")
    print(f"[INFO] jsons:     {json_counts}")
    print(f"[INFO] windows:   {window_counts}")
    if len(train_ds) == 0:
        raise RuntimeError("Train dataset has 0 windows after filtering.")
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset has 0 windows after filtering.")

    # 7) Dataloaders
    bs = int(cfg["train"].get("batch_size", 64))
    num_workers = int(cfg["train"].get("num_workers", 2))
    train_loader = make_loader(train_ds, batch_size=bs, num_workers=num_workers, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=bs, num_workers=num_workers, shuffle=False)

    # 8) Model
    # You must know input dim C. Best is to infer from one sample:
    X0, y0 = train_ds[0]
    print(y0)
    C = int(X0.shape[-1])
    print(C)
    print(f"[INFO] data_shape X, y: {X0.shape}, {y0.shape}")

    tcn_input_mode = str(model_cfg.get("tcn_input_mode", "pooled_count"))
    motion_proj_dim = model_cfg.get("motion_proj_dim", model_cfg.get("input_proj_dim", None))
    use_attention_readout = model_cfg.get("use_attention_readout", None)
    use_graph = bool(model_cfg.get("use_graph", True))
    encoder_type = resolve_encoder_type(model_cfg.get("encoder_type"))
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
        model = MotionTCN(
            input_dim=C,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dilations=model_cfg.get("dilations"),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            causal=bool(model_cfg.get("causal", True)),
            norm=str(model_cfg.get("norm", "group")),
            input_proj_dim=int(model_cfg.get("input_proj_dim", 0)),
        )
    else:
        model = EventTCN(
            input_dim=C,
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
    
    device_str = cfg["train"].get("device", "cpu")
    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    # 9) Optim + loss
    lr = float(cfg["train"].get("lr", 5e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",      # AUPRC higher is better
        factor=0.5,      # halve LR
        patience=2,      # wait 2 epochs without improvement
        threshold=1e-3,  # min improvement to count
        min_lr=1e-6,
        verbose=True,
    )
    pos_weight = float(cfg["train"].get("pos_weight", 1.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    epochs = int(cfg["train"].get("epochs", 20))
    grad_clip = cfg["train"].get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None
    agg_mode = cfg["train"].get("agg_mode", 'logsumexp')
    temperature = cfg["train"].get("logsumexp_temperature", 1.0)
    end_loss_alpha = cfg["train"].get("end_loss_alpha", 0.5)
    end_loss_k = cfg["train"].get("end_loss_k", 4)
    early_stop_patience = cfg["train"].get("early_stop_patience", None)
    early_stop_patience = int(early_stop_patience) if early_stop_patience is not None else None
    event_val_cfg = resolve_event_validation_cfg(
        cfg,
        root=root,
        data_root=data_root,
        val_paths=val_paths,
        batch_size=bs,
    )
    event_val_enabled = bool(event_val_cfg.get("enabled", False))

    print(f"[INFO] target_mode: {target_mode}")
    print(f"[INFO] window_label_rule: {window_cfg['rule']} (positive_overlap={window_cfg['positive_overlap']:.2f})")
    print(f"[INFO] labels: {label_cfg}")
    if str(window_cfg.get("rule")) == "soft_last_k" and float(end_loss_alpha) > 0.0:
        print(
            "[WARN] window_labels.rule=soft_last_k with train.end_loss_alpha>0 repeats the soft window "
            "target over the last end_loss_k logits. Set train.end_loss_alpha=0.0 to disable that auxiliary loss."
        )
    if bool(model_cfg.get("causal", True)) and target_mode != "last":
        print("[WARN] model.causal=true but data.target_mode is not 'last'.")
    if target_mode == "last" and str(agg_mode).lower() != "last":
        print("[WARN] data.target_mode='last' but train.agg_mode is not 'last'; validation remains window-level.")
    if event_val_enabled:
        print(
            f"[INFO] event validation: {len(event_val_cfg['paths'])} validation files | "
            f"{len(event_val_cfg['threshold_values'])} thresholds | "
            f"merge_gap_sec={event_val_cfg['merge_gap_sec_values']} | "
            f"min_duration_sec={event_val_cfg['min_duration_sec_values']}",
            flush=True,
        )
    print(f"[INFO] save_all_checkpoints: {save_all_checkpoints}")
    
    # 10) Train
    best_auprc = -1.0
    best_epoch = 0
    best_val_threshold = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
                                        model, train_loader, optim, criterion, device, grad_clip=grad_clip, 
                                        end_loss_alpha=end_loss_alpha, end_loss_k=end_loss_k, 
                                        agg_mode=agg_mode,
                                        temperature=temperature,
                                    )

        val_out = evaluate_binary(model, val_loader, device=device, agg_mode=agg_mode, temperature=temperature)
        val_auprc = val_out["auprc"]
        val_auroc = val_out["auroc"]
        is_best_candidate = val_auprc > best_auprc

        # Step scheduler (ReduceLROnPlateau expects metric)
        scheduler.step(val_auprc)

        # IMPORTANT: read LR AFTER scheduler.step
        lr = optim.param_groups[0]["lr"]

        fixed03 = val_out["fixed_thresholds"]["0.3"]
        fixed05 = val_out["fixed_thresholds"]["0.5"]
        fixed07 = val_out["fixed_thresholds"]["0.7"]
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_auprc={val_auprc:.4f} "
            f"| val_auroc={val_auroc:.4f} "
            f"| bestF1={val_out['best_f1']} | mean_p(pos)={val_out['mean_prob_pos']:.3f} "
            f"| mean_p(neg)={val_out['mean_prob_neg']:.3f} | brier={val_out['brier_score']:.4f} "
            f"| ece={val_out['ece']:.4f} | thr0.5 F1={fixed05['f1']:.3f} "
            f"(P={fixed05['precision']:.3f}, R={fixed05['recall']:.3f}) | lr={lr:.2e}"
        )

        event_val_report = None
        if event_val_enabled and (is_best_candidate or epoch % int(event_val_cfg["every_n_epochs"]) == 0):
            print(
                f"[INFO] Event validation epoch {epoch:03d}: evaluating {len(event_val_cfg['paths'])} files "
                f"and sweeping {len(event_val_cfg['threshold_values'])} thresholds.",
                flush=True,
            )
            event_val_report = evaluate_event_validation(
                model,
                dataset_cls,
                event_val_cfg["paths"],
                K=K,
                window_size=window_size,
                window_step=window_step,
                feature_cfg=feature_cfg,
                label_cfg=label_cfg,
                window_cfg=window_cfg,
                target_mode=target_mode,
                batch_size=int(event_val_cfg["batch_size"]),
                device=device,
                agg_mode=agg_mode,
                temperature=temperature,
                calibration_report=build_validation_calibration_report(val_out),
                raw_threshold=float(val_out["best_f1"]["threshold"]),
                score_methods=event_val_cfg["score_methods"],
                threshold_values=event_val_cfg["threshold_values"],
                merge_gap_sec_values=event_val_cfg["merge_gap_sec_values"],
                min_duration_sec_values=event_val_cfg["min_duration_sec_values"],
                min_overlap_sec=float(event_val_cfg["min_overlap_sec"]),
                min_iou=float(event_val_cfg["min_match_iou"]),
                selection_score_method=event_val_cfg["selection_score_method"],
                selection_metric=event_val_cfg["selection_metric"],
                max_false_alarms_per_min=event_val_cfg["max_false_alarms_per_min"],
                min_recall=event_val_cfg["min_recall"],
            )
            selected_event = event_val_report.get("selected", {})
            print(
                f"[VAL EVENT] method={selected_event.get('score_method', '')} "
                f"thr={safe_float(selected_event.get('threshold', float('nan'))):.2f} "
                f"merge={safe_float(selected_event.get('merge_gap_sec', float('nan'))):.2f}s "
                f"min_dur={safe_float(selected_event.get('min_duration_sec', float('nan'))):.2f}s "
                f"F1={safe_float(selected_event.get('f1', float('nan'))):.4f} "
                f"P={safe_float(selected_event.get('precision', float('nan'))):.4f} "
                f"R={safe_float(selected_event.get('recall', float('nan'))):.4f} "
                f"FA/min={safe_float(selected_event.get('false_alarms_per_min', float('nan'))):.4f} "
                f"timeIoU={safe_float(selected_event.get('time_iou', float('nan'))):.4f}",
                flush=True,
            )
        
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "best_f1": val_out["best_f1"],
            "mean_prob_pos": float(val_out["mean_prob_pos"]),
            "mean_prob_neg": float(val_out["mean_prob_neg"]),
            "brier_score": float(val_out["brier_score"]),
            "ece": float(val_out["ece"]),
            "thr_03_f1": float(fixed03["f1"]),
            "thr_03_precision": float(fixed03["precision"]),
            "thr_03_recall": float(fixed03["recall"]),
            "thr_05_f1": float(fixed05["f1"]),
            "thr_05_precision": float(fixed05["precision"]),
            "thr_05_recall": float(fixed05["recall"]),
            "thr_07_f1": float(fixed07["f1"]),
            "thr_07_precision": float(fixed07["precision"]),
            "thr_07_recall": float(fixed07["recall"]),
            "lr": float(lr),
        }
        if event_val_enabled:
            history_row.update(event_history_from_selected(event_val_report.get("selected", {}) if event_val_report else None))
        history.append(history_row)

        # Save metrics.csv every epoch (safer than only on "best")
        save_history_csv(reports_dir, history)

        ckpt_payload = {
            "model": model.state_dict(),
            "cfg": cfg_resolved,
            "epoch": epoch,
            "val_auprc": float(val_auprc),
            "val_auroc": float(val_auroc),
            "best_auprc": float(max(best_auprc, val_auprc)),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        latest_ckpt_path = checkpoints_dir / "latest.pt"
        torch.save(ckpt_payload, latest_ckpt_path)

        if save_all_checkpoints:
            epoch_ckpt_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt_payload, epoch_ckpt_path)
        
        # Save best checkpoint + best-epoch artifacts
        if is_best_candidate:
            best_auprc = val_auprc
            best_epoch = epoch
            best_val_threshold = float(val_out["best_f1"]["threshold"])
            epochs_without_improvement = 0
            ckpt_path = checkpoints_dir / "best.pt"
            ckpt_payload["best_auprc"] = float(best_auprc)
            torch.save(ckpt_payload, ckpt_path)
            event_val_artifacts = None
            if event_val_report:
                event_val_artifacts = save_event_report(reports_dir, event_val_report, prefix="val_best")
            # Save full metrics for best epoch
            ( metrics_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "best_epoch": epoch,
                        "best_auprc": best_auprc,
                        "history": history,
                        "val_details": {
                            "auprc": val_out["auprc"],
                            "auroc": val_out["auroc"],
                            "best_f1": val_out["best_f1"],
                            "best_threshold_stats": val_out["best_threshold_stats"],
                            "mean_prob_pos": val_out["mean_prob_pos"],
                            "mean_prob_neg": val_out["mean_prob_neg"],
                            "brier_score": val_out["brier_score"],
                            "brier_baseline": val_out["brier_baseline"],
                            "brier_skill_score": val_out["brier_skill_score"],
                            "ece": val_out["ece"],
                            "max_calibration_error": val_out["max_calibration_error"],
                            "calibration": val_out["calibration"],
                            "calibration_bins": val_out["calibration_bins"],
                            "fixed_thresholds": val_out["fixed_thresholds"],
                            "n_samples": val_out["n_samples"],
                        },
                        "val_event": compact_event_report(event_val_report, event_val_artifacts),
                    },
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            
            # Threshold sweep csv (only if provided)
            sweep = val_out.get("sweep", None)
            if sweep:
                save_threshold_sweep_csv(metrics_dir, sweep)

            save_confusion_matrix_image(
                reports_dir / "cm_norm_thr_0.5.png",
                val_out["fixed_thresholds"]["0.5"],
                title="Validation Confusion Matrix (normalized) @ thr=0.5",
            )
            best_thr = float(val_out["best_f1"]["threshold"])
            save_confusion_matrix_image(
                reports_dir / "cm_norm_thr_best_f1.png",
                val_out["best_threshold_stats"],
                title=f"Validation Confusion Matrix (normalized) @ thr={best_thr:.2f}",
            )
            save_pr_curve_image(
                reports_dir / "pr_curve_val.png",
                y_true=np.asarray(val_out["binary_targets"], dtype=np.float32),
                y_prob=np.asarray(val_out["probs"], dtype=np.float32),
                title="Validation Precision-Recall Curve",
            )
            save_probability_calibration_artifacts(
                reports_dir,
                prefix="window_val_raw",
                y_true=np.asarray(val_out["targets"], dtype=np.float32),
                y_prob=np.asarray(val_out["probs"], dtype=np.float32),
                title="Validation Reliability Diagram (Raw)",
            )
            
            # Plots for best checkpoint moment (optional)
            save_learning_curves(reports_dir, history)
        else:
            epochs_without_improvement += 1
            if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
                print(
                    f"[EARLY STOP] No validation AUPRC improvement for {epochs_without_improvement} "
                    f"epoch(s); patience={early_stop_patience}."
                )
                break

    if test_ds is not None:
        final_test_dir = run_dir / "final_test"
        final_test_dir.mkdir(parents=True, exist_ok=True)

        if len(test_ds) == 0:
            print("[WARN] Test dataset has 0 windows after filtering. Skipping final test evaluation.")
        else:
            best_ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device)
            model.load_state_dict(best_ckpt["model"])
            model.to(device)

            val_loader_final = make_loader(val_ds, batch_size=bs, num_workers=num_workers, shuffle=False)
            test_loader = make_loader(test_ds, batch_size=bs, num_workers=num_workers, shuffle=False)
            val_final_out = evaluate_binary(model, val_loader_final, device=device, agg_mode=agg_mode, temperature=temperature)
            test_out = evaluate_binary(model, test_loader, device=device, agg_mode=agg_mode, temperature=temperature)
            val_final_targets = np.asarray(val_final_out["targets"], dtype=np.float32)
            val_final_logits = np.asarray(val_final_out["logits"], dtype=np.float32)
            test_targets = np.asarray(test_out["targets"], dtype=np.float32)
            test_binary_targets = np.asarray(test_out["binary_targets"], dtype=np.int32)
            test_logits = np.asarray(test_out["logits"], dtype=np.float32)
            test_probs = np.asarray(test_out["probs"], dtype=np.float32)
            calibration_report = build_probability_calibration_report(
                val_final_targets,
                val_final_logits,
                test_targets,
                test_logits,
            )
            selected_threshold = float(best_val_threshold if best_val_threshold is not None else test_out["best_f1"]["threshold"])
            selected_stats = confusion_stats_at_threshold(test_binary_targets, test_probs, threshold=selected_threshold)
            oracle_stats = test_out["best_threshold_stats"]
            roc_rows = compute_roc_points(test_binary_targets, test_probs)
            pr_rows = compute_pr_points(test_binary_targets, test_probs)

            event_val_final_report = None
            event_test_report = None
            event_val_final_artifacts = None
            event_test_artifacts = None
            if event_val_enabled:
                print(
                    f"[INFO] Selecting event parameters on {len(event_val_cfg['paths'])} validation files "
                    f"with the best checkpoint",
                    flush=True,
                )
                event_val_final_report = evaluate_event_validation(
                    model,
                    dataset_cls,
                    event_val_cfg["paths"],
                    K=K,
                    window_size=window_size,
                    window_step=window_step,
                    feature_cfg=feature_cfg,
                    label_cfg=label_cfg,
                    window_cfg=window_cfg,
                    target_mode=target_mode,
                    batch_size=int(event_val_cfg["batch_size"]),
                    device=device,
                    agg_mode=agg_mode,
                    temperature=temperature,
                    calibration_report=build_probability_calibration_report(
                        val_final_targets,
                        val_final_logits,
                        val_final_targets,
                        val_final_logits,
                    ),
                    raw_threshold=float(val_final_out["best_f1"]["threshold"]),
                    score_methods=event_val_cfg["score_methods"],
                    threshold_values=event_val_cfg["threshold_values"],
                    merge_gap_sec_values=event_val_cfg["merge_gap_sec_values"],
                    min_duration_sec_values=event_val_cfg["min_duration_sec_values"],
                    min_overlap_sec=float(event_val_cfg["min_overlap_sec"]),
                    min_iou=float(event_val_cfg["min_match_iou"]),
                    selection_score_method=event_val_cfg["selection_score_method"],
                    selection_metric=event_val_cfg["selection_metric"],
                    max_false_alarms_per_min=event_val_cfg["max_false_alarms_per_min"],
                    min_recall=event_val_cfg["min_recall"],
                )
                event_val_final_artifacts = save_event_report(final_test_dir, event_val_final_report, prefix="val_selected")
                selected_event_params = event_val_final_report.get("selected", {})
                if selected_event_params:
                    event_score_method = str(selected_event_params["score_method"])
                    print(
                        f"[INFO] Applying validation-selected event parameters to {len(test_paths)} test files: "
                        f"method={event_score_method}, threshold={safe_float(selected_event_params.get('threshold')):.2f}, "
                        f"merge={safe_float(selected_event_params.get('merge_gap_sec')):.2f}s, "
                        f"min_dur={safe_float(selected_event_params.get('min_duration_sec')):.2f}s",
                        flush=True,
                    )
                    event_test_report = evaluate_event_validation(
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
                        batch_size=int(event_val_cfg["batch_size"]),
                        device=device,
                        agg_mode=agg_mode,
                        temperature=temperature,
                        calibration_report=calibration_report,
                        raw_threshold=selected_threshold,
                        score_methods=[event_score_method],
                        threshold_values=[float(selected_event_params["threshold"])],
                        merge_gap_sec_values=[float(selected_event_params["merge_gap_sec"])],
                        min_duration_sec_values=[float(selected_event_params["min_duration_sec"])],
                        min_overlap_sec=float(selected_event_params["min_overlap_sec"]),
                        min_iou=float(selected_event_params.get("min_match_iou", event_val_cfg["min_match_iou"])),
                        selection_score_method=event_score_method,
                        selection_metric=event_val_cfg["selection_metric"],
                        max_false_alarms_per_min=None,
                        min_recall=None,
                    )
                    event_test_artifacts = save_event_report(final_test_dir, event_test_report, prefix="test_selected")

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
                batch_size=bs,
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
                batch_size=bs,
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

            n_pos = int(np.sum(test_binary_targets == 1))
            n_neg = int(np.sum(test_binary_targets == 0))
            selected_fpr = fpr_from_stats(selected_stats)
            oracle_fpr = fpr_from_stats(oracle_stats)

            save_threshold_sweep_csv(final_test_dir, test_out["sweep"])
            save_rows_csv(final_test_dir / "per_file.csv", per_file_rows)
            save_rows_csv(final_test_dir / "per_video.csv", per_video_rows)
            if roc_rows:
                save_rows_csv(final_test_dir / "roc.csv", roc_rows)
            if pr_rows:
                save_rows_csv(final_test_dir / "pr.csv", pr_rows)

            save_confusion_matrix_image(
                final_test_dir / "cm_norm_thr_val_selected.png",
                selected_stats,
                title=f"Final Test Confusion Matrix (normalized) @ val thr={selected_threshold:.2f}",
            )
            save_confusion_matrix_image(
                final_test_dir / "cm_norm_thr_test_best_f1.png",
                oracle_stats,
                title=f"Final Test Confusion Matrix (normalized) @ oracle thr={float(test_out['best_f1']['threshold']):.2f}",
            )
            save_pr_curve_image(
                final_test_dir / "pr_curve_test.png",
                y_true=test_binary_targets.astype(np.float32),
                y_prob=test_probs.astype(np.float32),
                title="Final Test Precision-Recall Curve",
            )
            save_roc_curve_image(
                final_test_dir / "roc_curve_test.png",
                y_true=test_binary_targets.astype(np.float32),
                y_prob=test_probs.astype(np.float32),
                title="Final Test ROC Curve",
            )
            save_probability_calibration_artifacts(
                final_test_dir,
                prefix="window_val_raw",
                y_true=val_final_targets.astype(np.float32),
                y_prob=np.asarray(val_final_out["probs"], dtype=np.float32),
                title="Final Validation Reliability Diagram (Raw)",
            )
            save_probability_calibration_artifacts(
                final_test_dir,
                prefix="window_test_raw",
                y_true=test_targets.astype(np.float32),
                y_prob=test_probs.astype(np.float32),
                title="Final Test Reliability Diagram (Raw)",
            )
            if calibration_report.get("temperature", {}).get("fit", {}).get("available"):
                temp = float(calibration_report["temperature"]["fit"]["temperature"])
                save_probability_calibration_artifacts(
                    final_test_dir,
                    prefix="window_val_temperature",
                    y_true=val_final_targets.astype(np.float32),
                    y_prob=apply_temperature_scaling(val_final_logits, temp),
                    title=f"Final Validation Reliability Diagram (Temperature T={temp:.3f})",
                )
                save_probability_calibration_artifacts(
                    final_test_dir,
                    prefix="window_test_temperature",
                    y_true=test_targets.astype(np.float32),
                    y_prob=apply_temperature_scaling(test_logits, temp),
                    title=f"Final Test Reliability Diagram (Temperature T={temp:.3f})",
                )
            if calibration_report.get("platt", {}).get("fit", {}).get("available"):
                slope = float(calibration_report["platt"]["fit"]["slope"])
                intercept = float(calibration_report["platt"]["fit"]["intercept"])
                save_probability_calibration_artifacts(
                    final_test_dir,
                    prefix="window_val_platt",
                    y_true=val_final_targets.astype(np.float32),
                    y_prob=apply_platt_scaling(val_final_logits, slope=slope, intercept=intercept),
                    title="Final Validation Reliability Diagram (Platt)",
                )
                save_probability_calibration_artifacts(
                    final_test_dir,
                    prefix="window_test_platt",
                    y_true=test_targets.astype(np.float32),
                    y_prob=apply_platt_scaling(test_logits, slope=slope, intercept=intercept),
                    title="Final Test Reliability Diagram (Platt)",
                )
            (final_test_dir / "calibration.json").write_text(
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
                ("best_checkpoint", str(checkpoints_dir / "best.pt")),
                ("best_epoch", best_epoch),
                ("split_mode", split_mode),
                ("train_jsons", len(train_paths)),
                ("val_jsons", len(val_paths)),
                ("test_jsons", len(test_paths)),
                ("train_windows", len(train_ds)),
                ("val_windows", len(val_ds)),
                ("test_windows", len(test_ds)),
                ("threshold_source", "validation_best_f1"),
                ("selected_threshold", selected_threshold),
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
                ("selected_precision", safe_float(selected_stats["precision"])),
                ("selected_recall", safe_float(selected_stats["recall"])),
                ("selected_f1", safe_float(selected_stats["f1"])),
                ("selected_accuracy", safe_float(selected_stats["accuracy"])),
                ("selected_fpr", selected_fpr),
                ("oracle_threshold", safe_float(test_out["best_f1"]["threshold"])),
                ("oracle_precision", safe_float(oracle_stats["precision"])),
                ("oracle_recall", safe_float(oracle_stats["recall"])),
                ("oracle_f1", safe_float(oracle_stats["f1"])),
                ("oracle_accuracy", safe_float(oracle_stats["accuracy"])),
                ("oracle_fpr", oracle_fpr),
                ("mean_prob_pos", safe_float(test_out["mean_prob_pos"])),
                ("mean_prob_neg", safe_float(test_out["mean_prob_neg"])),
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
            if event_val_final_report:
                event_val_selected = event_val_final_report.get("selected", {})
                summary_rows.extend(
                    [
                        ("event_threshold_source", "validation_event_grid"),
                        ("event_val_score_method", str(event_val_selected.get("score_method", ""))),
                        ("event_val_selected_threshold", safe_float(event_val_selected.get("threshold", float("nan")))),
                        ("event_val_merge_gap_sec", safe_float(event_val_selected.get("merge_gap_sec", float("nan")))),
                        ("event_val_min_duration_sec", safe_float(event_val_selected.get("min_duration_sec", float("nan")))),
                        ("event_val_min_match_iou", safe_float(event_val_selected.get("min_match_iou", float("nan")))),
                        ("event_val_precision", safe_float(event_val_selected.get("precision", float("nan")))),
                        ("event_val_recall", safe_float(event_val_selected.get("recall", float("nan")))),
                        ("event_val_f1", safe_float(event_val_selected.get("f1", float("nan")))),
                        ("event_val_false_alarms_per_min", safe_float(event_val_selected.get("false_alarms_per_min", float("nan")))),
                        ("event_val_time_iou", safe_float(event_val_selected.get("time_iou", float("nan")))),
                    ]
                )
            if event_test_report:
                event_test_selected = event_test_report.get("selected_details", {}).get(
                    "aggregate",
                    event_test_report.get("selected", {}),
                )
                summary_rows.extend(
                    [
                        ("event_test_score_method", str(event_test_selected.get("score_method", ""))),
                        ("event_test_selected_threshold", safe_float(event_test_selected.get("threshold", float("nan")))),
                        ("event_test_merge_gap_sec", safe_float(event_test_selected.get("merge_gap_sec", float("nan")))),
                        ("event_test_min_duration_sec", safe_float(event_test_selected.get("min_duration_sec", float("nan")))),
                        ("event_test_min_match_iou", safe_float(event_test_selected.get("min_match_iou", float("nan")))),
                        ("event_test_n_gt_events", safe_float(event_test_selected.get("n_gt_events", float("nan")))),
                        ("event_test_n_predicted_events", safe_float(event_test_selected.get("n_predicted_events", float("nan")))),
                        ("event_test_tp", safe_float(event_test_selected.get("tp", float("nan")))),
                        ("event_test_fp", safe_float(event_test_selected.get("fp", float("nan")))),
                        ("event_test_fn", safe_float(event_test_selected.get("fn", float("nan")))),
                        ("event_test_precision", safe_float(event_test_selected.get("precision", float("nan")))),
                        ("event_test_recall", safe_float(event_test_selected.get("recall", float("nan")))),
                        ("event_test_f1", safe_float(event_test_selected.get("f1", float("nan")))),
                        ("event_test_false_alarms_per_min", safe_float(event_test_selected.get("false_alarms_per_min", float("nan")))),
                        ("event_test_gt_hit_recall", safe_float(event_test_selected.get("gt_hit_recall", float("nan")))),
                        ("event_test_time_coverage", safe_float(event_test_selected.get("time_coverage", float("nan")))),
                        ("event_test_time_iou", safe_float(event_test_selected.get("time_iou", float("nan")))),
                    ]
                )

            save_summary_csv(final_test_dir / "summary.csv", summary_rows)

            final_test_payload = {
                "best_epoch": best_epoch,
                "best_checkpoint": str(checkpoints_dir / "best.pt"),
                "split_mode": split_mode,
                "threshold_source": "validation_best_f1",
                "selected_threshold": selected_threshold,
                "split_counts": {
                    "train_jsons": len(train_paths),
                    "val_jsons": len(val_paths),
                    "test_jsons": len(test_paths),
                    "train_windows": len(train_ds),
                    "val_windows": len(val_ds),
                    "test_windows": len(test_ds),
                },
                "test_summary": {
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
                "video_level": {
                    "threshold_source": "validation_video_best_f1",
                    "validation": {
                        "mean_score": {
                            key: value for key, value in val_video_mean.items() if key != "sweep"
                        },
                        "max_score": {
                            key: value for key, value in val_video_max.items() if key != "sweep"
                        },
                    },
                    "test": {
                        "mean_score": {
                            key: value for key, value in test_video_mean.items() if key != "sweep"
                        },
                        "max_score": {
                            key: value for key, value in test_video_max.items() if key != "sweep"
                        },
                    },
                },
                "probability_calibration": calibration_report,
                "artifacts": {
                    "summary_csv": str(final_test_dir / "summary.csv"),
                    "per_file_csv": str(final_test_dir / "per_file.csv"),
                    "per_video_csv": str(final_test_dir / "per_video.csv"),
                    "roc_csv": str(final_test_dir / "roc.csv"),
                    "pr_csv": str(final_test_dir / "pr.csv"),
                    "threshold_sweep_csv": str(final_test_dir / "threshold_sweep.csv"),
                    "calibration_json": str(final_test_dir / "calibration.json"),
                    "window_test_raw_reliability": str(final_test_dir / "window_test_raw_reliability.png"),
                    "window_test_raw_calibration_bins": str(final_test_dir / "window_test_raw_calibration_bins.csv"),
                    "window_test_temperature_reliability": str(final_test_dir / "window_test_temperature_reliability.png"),
                    "window_test_temperature_calibration_bins": str(final_test_dir / "window_test_temperature_calibration_bins.csv"),
                    "window_test_platt_reliability": str(final_test_dir / "window_test_platt_reliability.png"),
                    "window_test_platt_calibration_bins": str(final_test_dir / "window_test_platt_calibration_bins.csv"),
                },
            }
            if event_val_final_report or event_test_report:
                final_test_payload["event_level"] = {
                    "threshold_source": "validation_event_grid",
                    "validation": compact_event_report(event_val_final_report, event_val_final_artifacts),
                    "test": compact_event_report(event_test_report, event_test_artifacts),
                }
                final_test_payload["artifacts"].update(
                    {
                        "event_val_grid_csv": str(final_test_dir / "val_selected_event_grid.csv"),
                        "event_val_summary_json": str(final_test_dir / "val_selected_event_summary.json"),
                        "event_test_per_file_csv": str(final_test_dir / "test_selected_event_per_file.csv"),
                        "event_test_matches_csv": str(final_test_dir / "test_selected_event_matches.csv"),
                        "event_test_fragments_csv": str(final_test_dir / "test_selected_event_fragments.csv"),
                        "event_test_intervals_csv": str(final_test_dir / "test_selected_event_intervals.csv"),
                        "event_test_summary_json": str(final_test_dir / "test_selected_event_summary.json"),
                    }
                )
            (final_test_dir / "metrics.json").write_text(
                json.dumps(final_test_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"[FINAL TEST] AUPRC={test_out['auprc']:.4f} | AUROC={test_out['auroc']:.4f} "
                f"| Brier={test_out['brier_score']:.4f} | ECE={test_out['ece']:.4f} "
                f"| val_thr={selected_threshold:.2f} F1={selected_stats['f1']:.4f} "
                f"| oracle_thr={float(test_out['best_f1']['threshold']):.2f} F1={oracle_stats['f1']:.4f}"
            )
            if temperature_fit.get("available"):
                print(
                    f"[FINAL TEST CAL] temperature={float(temperature_fit['temperature']):.4f} "
                    f"| test_brier={safe_float(temperature_test.get('brier_score', float('nan'))):.4f} "
                    f"| test_ece={safe_float(temperature_test.get('ece', float('nan'))):.4f}"
                )
            print(
                f"[FINAL TEST VIDEO] mean_thr={float(test_video_mean['selected_threshold']):.2f} "
                f"F1={test_video_mean['selected_threshold_stats']['f1']:.4f} | "
                f"max_thr={float(test_video_max['selected_threshold']):.2f} "
                f"F1={test_video_max['selected_threshold_stats']['f1']:.4f}"
            )
            if event_test_report:
                event_test_selected = event_test_report.get("selected_details", {}).get(
                    "aggregate",
                    event_test_report.get("selected", {}),
                )
                print(
                    f"[FINAL TEST EVENT] method={event_test_selected.get('score_method', '')} "
                    f"thr={safe_float(event_test_selected.get('threshold', float('nan'))):.2f} "
                    f"merge={safe_float(event_test_selected.get('merge_gap_sec', float('nan'))):.2f}s "
                    f"min_dur={safe_float(event_test_selected.get('min_duration_sec', float('nan'))):.2f}s "
                    f"F1={safe_float(event_test_selected.get('f1', float('nan'))):.4f} "
                    f"P={safe_float(event_test_selected.get('precision', float('nan'))):.4f} "
                    f"R={safe_float(event_test_selected.get('recall', float('nan'))):.4f} "
                    f"FA/min={safe_float(event_test_selected.get('false_alarms_per_min', float('nan'))):.4f} "
                    f"timeIoU={safe_float(event_test_selected.get('time_iou', float('nan'))):.4f}"
                )
    print(f"[DONE] Best AUPRC: {best_auprc:.4f}")
    print(f"[DONE] Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
