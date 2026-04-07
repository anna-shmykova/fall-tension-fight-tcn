#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.dataset import EventJsonDataset, MotionJsonDataset
from src.data.features import motion_feature_dim
from src.data.labels import resolve_label_cfg
from src.data.splits import read_paths_txt
from src.models.tcn import EventTCN, MotionTCN
from src.train import (
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
    save_pr_curve_image,
    save_roc_curve_image,
    save_rows_csv,
    save_summary_csv,
    save_threshold_sweep_csv,
)


def resolve_optional_path(path_str: str | None, base: Path) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else (base / path).resolve()


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


def build_model(cfg: dict[str, Any], input_dim: int) -> torch.nn.Module:
    feature_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "tcn")).lower()
    motion_dim = motion_feature_dim(feature_cfg)

    tcn_input_mode = str(model_cfg.get("tcn_input_mode", "pooled_count"))
    motion_proj_dim = model_cfg.get("motion_proj_dim", model_cfg.get("input_proj_dim", None))
    use_attention_readout = model_cfg.get("use_attention_readout", None)
    use_graph = bool(model_cfg.get("use_graph", True))

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
        mlp_out_dim=int(model_cfg.get("mlp_out_dim", 32)),
        pool_mode=str(model_cfg.get("pool_mode", "attn")),
        use_attention_readout=use_attention_readout,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Training run directory with config_resolved.yaml, checkpoints/, and splits/.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to <run_dir>/checkpoints/best.pt")
    parser.add_argument("--output_dir", default=None, help="Optional output directory. Defaults to <run_dir>/final_test_standalone")
    parser.add_argument("--device", default=None, help="Optional device override, for example cpu or cuda:0")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional dataloader worker override")
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
    test_paths = load_split(splits_dir / "test_paths.txt", data_root)

    if not val_paths:
        raise RuntimeError(f"No validation paths found in {splits_dir / 'val_paths.txt'}")
    if not test_paths:
        raise RuntimeError(f"No test paths found in {splits_dir / 'test_paths.txt'}")

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

    X0, _ = preview_ds[0]
    model = build_model(cfg, input_dim=int(X0.shape[-1]))
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
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

    selected_fpr = fpr_from_stats(selected_stats)
    oracle_fpr = fpr_from_stats(oracle_stats)
    n_pos = int(np.sum(test_binary_targets == 1))
    n_neg = int(np.sum(test_binary_targets == 0))

    save_threshold_sweep_csv(output_dir, test_out["sweep"])
    save_rows_csv(output_dir / "per_file.csv", per_file_rows)
    save_rows_csv(output_dir / "per_video.csv", per_video_rows)
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
        "probability_calibration": calibration_report,
        "artifacts": {
            "summary_csv": str(output_dir / "summary.csv"),
            "per_file_csv": str(output_dir / "per_file.csv"),
            "per_video_csv": str(output_dir / "per_video.csv"),
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


if __name__ == "__main__":
    main()
