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
from src.data.dataset import EventJsonDataset, MotionJsonDataset
from src.data.features import motion_feature_dim
from src.models.tcn import EventTCN, MotionTCN
from src.utils.metrics import (
    compute_auroc,
    compute_auprc,
    compute_pr_points,
    compute_roc_points,
    threshold_sweep_binary,
    confusion_stats_at_threshold,
    save_history_csv,
    save_rows_csv,
    save_summary_csv,
    save_threshold_sweep_csv,
    save_learning_curves,
    save_confusion_matrix_image,
    save_pr_curve_image,
    save_roc_curve_image,
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


def build_dataset(
    dataset_cls,
    paths: List[str],
    *,
    K: int,
    window_size: int,
    window_step: int,
    feature_cfg: Dict[str, Any],
    label_cfg: Dict[str, Any],
    target_mode: str,
):
    return dataset_cls(
        paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        target_mode=target_mode,
    )


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fpr_from_stats(stats: Dict[str, float]) -> float:
    denom = float(stats["fp"] + stats["tn"])
    return float(stats["fp"] / denom) if denom > 0 else float("nan")


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
    target_mode: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    agg_mode: str,
    temperature: float,
    selected_threshold: float,
) -> Dict[str, Any]:
    ds = build_dataset(
        dataset_cls,
        [json_path],
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        target_mode=target_mode,
    )
    row: Dict[str, Any] = {
        "json_path": json_path,
        "json_name": Path(json_path).name,
        "n_windows": int(len(ds)),
    }
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
            }
        )
        return row

    loader = make_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    out = evaluate_binary(model, loader, device=device, agg_mode=agg_mode, temperature=temperature)
    targets = np.asarray(out["targets"], dtype=np.int32)
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
    targets = torch.cat(all_targets).numpy()

    probs = 1 / (1 + np.exp(-logits))
    auprc = compute_auprc(targets, probs)
    auroc = compute_auroc(targets, probs)

    pos_mask = targets == 1
    neg_mask = targets == 0
    mean_prob_pos = float(np.mean(probs[pos_mask])) if np.any(pos_mask) else float("nan")
    mean_prob_neg = float(np.mean(probs[neg_mask])) if np.any(neg_mask) else float("nan")

    sweep = threshold_sweep_binary(targets, probs, start=0.01, end=0.99, step=0.01)
    best = max(sweep, key=lambda r: r["f1"])
    best_threshold_stats = confusion_stats_at_threshold(targets, probs, threshold=float(best["threshold"]))
    fixed_thresholds = {
        "0.3": confusion_stats_at_threshold(targets, probs, threshold=0.3),
        "0.5": confusion_stats_at_threshold(targets, probs, threshold=0.5),
        "0.7": confusion_stats_at_threshold(targets, probs, threshold=0.7),
    }

    return {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "best_f1": {k: float(v) for k, v in best.items()},
        "best_threshold_stats": best_threshold_stats,
        "mean_prob_pos": mean_prob_pos,
        "mean_prob_neg": mean_prob_neg,
        "fixed_thresholds": fixed_thresholds,
        "targets": targets.tolist(),
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

    # 6) Build datasets
    K = int(cfg["data"].get("max_persons", 25))
    window_size = int(cfg["data"].get("window_size", 16))
    window_step = int(cfg["data"].get("window_step", 4))

    feature_cfg = cfg.get("features", {})
    label_cfg = cfg.get("labels", {})
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "tcn")).lower()
    motion_dim = motion_feature_dim(feature_cfg)
    dataset_cls = MotionJsonDataset if model_type in {"motion_tcn", "erez_motion_tcn"} else EventJsonDataset

    train_ds = build_dataset(
        dataset_cls,
        train_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        target_mode=target_mode,
    )
    val_ds = build_dataset(
        dataset_cls,
        val_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        target_mode=target_mode,
    )
    test_ds = None
    if test_paths:
        test_ds = build_dataset(
            dataset_cls,
            test_paths,
            K=K,
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
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

    print(f"[INFO] target_mode: {target_mode}")
    if bool(model_cfg.get("causal", True)) and target_mode != "last":
        print("[WARN] model.causal=true but data.target_mode is not 'last'.")
    if target_mode == "last" and str(agg_mode).lower() != "last":
        print("[WARN] data.target_mode='last' but train.agg_mode is not 'last'; validation remains window-level.")
    print(f"[INFO] save_all_checkpoints: {save_all_checkpoints}")
    
    # 10) Train
    best_auprc = -1.0
    best_epoch = 0
    best_val_threshold = None
    history = []

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
            f"| mean_p(neg)={val_out['mean_prob_neg']:.3f} | thr0.5 F1={fixed05['f1']:.3f} "
            f"(P={fixed05['precision']:.3f}, R={fixed05['recall']:.3f}) | lr={lr:.2e}"
        )
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "best_f1": val_out["best_f1"],
            "mean_prob_pos": float(val_out["mean_prob_pos"]),
            "mean_prob_neg": float(val_out["mean_prob_neg"]),
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
        })

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
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            best_val_threshold = float(val_out["best_f1"]["threshold"])
            ckpt_path = checkpoints_dir / "best.pt"
            ckpt_payload["best_auprc"] = float(best_auprc)
            torch.save(ckpt_payload, ckpt_path)
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
                            "fixed_thresholds": val_out["fixed_thresholds"],
                            "n_samples": val_out["n_samples"],
                        },
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
                y_true=np.asarray(val_out["targets"], dtype=np.float32),
                y_prob=np.asarray(val_out["probs"], dtype=np.float32),
                title="Validation Precision-Recall Curve",
            )
    
            # Plots for best checkpoint moment (optional)
            save_learning_curves(reports_dir, history)

    if test_ds is not None:
        final_test_dir = run_dir / "final_test"
        final_test_dir.mkdir(parents=True, exist_ok=True)

        if len(test_ds) == 0:
            print("[WARN] Test dataset has 0 windows after filtering. Skipping final test evaluation.")
        else:
            best_ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device)
            model.load_state_dict(best_ckpt["model"])
            model.to(device)

            test_loader = make_loader(test_ds, batch_size=bs, num_workers=num_workers, shuffle=False)
            test_out = evaluate_binary(model, test_loader, device=device, agg_mode=agg_mode, temperature=temperature)
            test_targets = np.asarray(test_out["targets"], dtype=np.int32)
            test_probs = np.asarray(test_out["probs"], dtype=np.float32)
            selected_threshold = float(best_val_threshold if best_val_threshold is not None else test_out["best_f1"]["threshold"])
            selected_stats = confusion_stats_at_threshold(test_targets, test_probs, threshold=selected_threshold)
            oracle_stats = test_out["best_threshold_stats"]
            roc_rows = compute_roc_points(test_targets, test_probs)
            pr_rows = compute_pr_points(test_targets, test_probs)

            per_file_rows: List[Dict[str, Any]] = []
            print(f"[INFO] Evaluating {len(test_paths)} held-out test files for per-file report")
            for json_path in test_paths:
                per_file_rows.append(
                    evaluate_single_path(
                        model,
                        dataset_cls,
                        json_path,
                        K=K,
                        window_size=window_size,
                        window_step=window_step,
                        feature_cfg=feature_cfg,
                        label_cfg=label_cfg,
                        target_mode=target_mode,
                        batch_size=bs,
                        num_workers=num_workers,
                        device=device,
                        agg_mode=agg_mode,
                        temperature=temperature,
                        selected_threshold=selected_threshold,
                    )
                )

            n_pos = int(np.sum(test_targets == 1))
            n_neg = int(np.sum(test_targets == 0))
            selected_fpr = fpr_from_stats(selected_stats)
            oracle_fpr = fpr_from_stats(oracle_stats)

            save_threshold_sweep_csv(final_test_dir, test_out["sweep"])
            save_rows_csv(final_test_dir / "per_file.csv", per_file_rows)
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
                y_true=test_targets.astype(np.float32),
                y_prob=test_probs.astype(np.float32),
                title="Final Test Precision-Recall Curve",
            )
            save_roc_curve_image(
                final_test_dir / "roc_curve_test.png",
                y_true=test_targets.astype(np.float32),
                y_prob=test_probs.astype(np.float32),
                title="Final Test ROC Curve",
            )

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
            ]

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
                "artifacts": {
                    "summary_csv": str(final_test_dir / "summary.csv"),
                    "per_file_csv": str(final_test_dir / "per_file.csv"),
                    "roc_csv": str(final_test_dir / "roc.csv"),
                    "pr_csv": str(final_test_dir / "pr.csv"),
                    "threshold_sweep_csv": str(final_test_dir / "threshold_sweep.csv"),
                },
            }
            (final_test_dir / "metrics.json").write_text(
                json.dumps(final_test_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"[FINAL TEST] AUPRC={test_out['auprc']:.4f} | AUROC={test_out['auroc']:.4f} "
                f"| val_thr={selected_threshold:.2f} F1={selected_stats['f1']:.4f} "
                f"| oracle_thr={float(test_out['best_f1']['threshold']):.2f} F1={oracle_stats['f1']:.4f}"
            )
    print(f"[DONE] Best AUPRC: {best_auprc:.4f}")
    print(f"[DONE] Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
