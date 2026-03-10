# src/train.py
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
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
    split_paths,
    write_split_files
    )
from src.data.dataset import EventJsonDataset
from src.models.tcn import MLP_POOL
from src.utils.metrics import (
    compute_auprc,
    threshold_sweep_binary,
    confusion_stats_at_threshold,
    save_history_csv,
    save_threshold_sweep_csv,
    save_learning_curves,
    save_confusion_matrix_image,
    save_pr_curve_image,
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

# ----------------------------
# Training / Eval loops
# ----------------------------
@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_logits = []
    all_targets = []

    for X, y in loader:
        X = X.to(device)  # (B,T,C)
        y = y.to(device).float()  # (B,)
        #logits = model(X).view(-1)  # (B,)
        logits_bt = model(X)  # (B,T)
        #win_logits = aggregate_window_logits(logits_bt)  # (B,) mode=agg_mode,  temperature=temperature
        all_logits.append(logits_bt.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()

    probs = 1 / (1 + np.exp(-logits))
    auprc = compute_auprc(targets, probs)

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
        loss = criterion(logits_bt, y)
        
        '''if end_loss_alpha > 0:
            pos_weight = getattr(criterion, "pos_weight", torch.tensor(1.0, device=device))
            loss_end = end_weighted_bce_loss(logits_bt, y, pos_weight=pos_weight, k_last=end_loss_k)
            loss = (1 - end_loss_alpha) * loss_main + end_loss_alpha * loss_end
        else:
            loss = loss_main
                #logits = model(X).view(-1)  # (B,)
                #loss = criterion(logits, y)'''
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
    cfg_resolved["run_dir"] = str(run_dir)

    save_yaml(cfg_resolved, run_dir / "config_resolved.yaml")
    #(run_dir / "git_commit.txt").write_text(get_git_commit() + "\n", encoding="utf-8")

    # 4) Collect json paths
    # Expect jsons under data_root (possibly nested).
    all_jsons = collect_json_paths(data_root)
    if len(all_jsons) == 0:
        raise RuntimeError(f"No jsons found under {data_root}")

    # 5) Make or load split
    split_cfg = cfg.get("data", {}).get("split", {"mode": "generate"})
    split_mode = split_cfg.get("mode", "generate")

    if split_mode == "fixed":
        train_list = resolve_path(split_cfg["train_list"], root)
        val_list = resolve_path(split_cfg["val_list"], root)
        #train_paths = read_paths_txt(train_list, base=data_root)
        #val_paths = read_paths_txt(val_list, base=data_root)
        with open(train_list, 'r') as f:
            lines = f.readlines()
            train_paths = [line.rstrip('\n') for line in lines]

        with open(val_list, 'r') as f:
            lines = f.readlines()
            val_paths = [line.rstrip('\n') for line in lines]
            
    else:
        # generate per run
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        split_seed = int(split_cfg.get("seed", seed))
        train_paths, val_paths = split_paths(
            all_jsons, val_size=val_ratio, seed=split_seed
        )
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

    train_ds = EventJsonDataset(
        train_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
    )
    val_ds = EventJsonDataset(
        val_paths,
        K=K,
        window_size=window_size,
        window_step=window_step,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
    )

    # Quick sanity prints
    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] run_dir:   {run_dir}")
    print(f"[INFO] jsons:     {len(all_jsons)} total | {len(train_paths)} train_jsons | {len(val_paths)} val_jsons")
    print(f"[INFO] windows:   {len(train_ds)} train_windows | {len(val_ds)} val_windows")

    # 7) Dataloaders
    bs = int(cfg["train"].get("batch_size", 64))
    num_workers = int(cfg["train"].get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 8) Model
    # You must know input dim C. Best is to infer from one sample:
    X0, y0 = train_ds[0]
    print(y0)
    C = int(X0.shape[-1])
    print(C)
    print(f"[INFO] data_shape X, y: {X0.shape}, {y0.shape}")

    model_cfg = cfg.get("model", {})
    model = MLP_POOL(
        #input_dim=C,
        #hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        #num_layers=int(model_cfg.get("num_layers", 4)),
        #dilations=list(model_cfg.get("dilations", [1, 2, 4, 8])),
        #kernel_size=int(model_cfg.get("kernel_size", 3)),
        #dropout=float(model_cfg.get("dropout", 0.2)),
        #causal=bool(model_cfg.get("causal", True)),
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
    
    # 10) Train
    best_auprc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device, grad_clip=grad_clip, end_loss_alpha=end_loss_alpha, end_loss_k=end_loss_k)

        val_out = evaluate_binary(model, val_loader, device=device)
        val_auprc = val_out["auprc"]

        # Step scheduler (ReduceLROnPlateau expects metric)
        scheduler.step(val_auprc)

        # IMPORTANT: read LR AFTER scheduler.step
        lr = optim.param_groups[0]["lr"]

        fixed03 = val_out["fixed_thresholds"]["0.3"]
        fixed05 = val_out["fixed_thresholds"]["0.5"]
        fixed07 = val_out["fixed_thresholds"]["0.7"]
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_auprc={val_auprc:.4f} "
            f"| bestF1={val_out['best_f1']} | mean_p(pos)={val_out['mean_prob_pos']:.3f} "
            f"| mean_p(neg)={val_out['mean_prob_neg']:.3f} | thr0.5 F1={fixed05['f1']:.3f} "
            f"(P={fixed05['precision']:.3f}, R={fixed05['recall']:.3f}) | lr={lr:.2e}"
        )
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auprc": val_auprc,
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
        
        # Save best checkpoint + best-epoch artifacts
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            ckpt_path = checkpoints_dir / "best.pt"
            #torch.save({"model": model.state_dict(), "cfg": cfg_resolved}, ckpt_path)

            torch.save(
            {
                "model": model.state_dict(),
                "cfg": cfg_resolved,
                "epoch": epoch,
                "best_auprc": best_auprc,
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_path
            )            
            # Save full metrics for best epoch
            ( metrics_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "best_epoch": epoch,
                        "best_auprc": best_auprc,
                        "history": history,
                        "val_details": {
                            "auprc": val_out["auprc"],
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
    print(f"[DONE] Best AUPRC: {best_auprc:.4f}")
    print(f"[DONE] Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
