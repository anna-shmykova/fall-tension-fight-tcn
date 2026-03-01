# src/train.py
from __future__ import annotations
import matplotlib.pyplot as plt
import argparse
import json
import csv
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
from src.models.tcn import EventTCN
from src.utils.metrics import (
    compute_auprc,
    threshold_sweep_binary,
)

#from sklearn.metrics import average_precision_score, precision_recall_curve
# ----------------------------
# Helpers
# ----------------------------
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
    
def save_history_csv(out_dir: Path, history: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

def save_threshold_sweep_csv(metrics_dir: Path, sweep: list):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
        
def save_learning_curves(out_dir: Path, history: list):
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_list = [h["epoch"] for h in history]
    train_loss_list = [h["train_loss"] for h in history]
    val_auprc_list = [h["val_auprc"] for h in history]
    lr_list = [h["lr"] for h in history]

    # Train loss
    plt.figure()
    plt.plot(epoch_list, train_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(out_dir / "train_loss.png", dpi=150)
    plt.close()

    # Val AUPRC
    plt.figure()
    plt.plot(epoch_list, val_auprc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Val AUPRC")
    plt.title("Validation AUPRC")
    plt.grid(True)
    plt.savefig(out_dir / "val_auprc.png", dpi=150)
    plt.close()

    # LR
    plt.figure()
    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate")
    plt.grid(True)
    plt.savefig(out_dir / "lr.png", dpi=150)
    plt.close()

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
        logits = model(X).view(-1)  # (B,)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()

    probs = 1 / (1 + np.exp(-logits))
    auprc = compute_auprc(targets, probs)

    sweep = threshold_sweep_binary(targets, probs, start=0.01, end=0.99, step=0.01)
    best = max(sweep, key=lambda r: r["f1"])

    return {
        "auprc": float(auprc),
        "best_f1": {k: float(v) for k, v in best.items()},
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
) -> float:
    model.train()
    losses = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float()  # BCE expects float targets

        optim.zero_grad(set_to_none=True)
        logits = model(X).view(-1)  # (B,)
        loss = criterion(logits, y)
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
    model = EventTCN(
        input_dim=C,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        #dilations=list(model_cfg.get("dilations", [1, 2, 4, 8])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
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

    # 10) Train
    best_auprc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device, grad_clip=grad_clip)

        val_out = evaluate_binary(model, val_loader, device=device)
        val_auprc = val_out["auprc"]

        # Step scheduler (ReduceLROnPlateau expects metric)
        scheduler.step(val_auprc)

        # IMPORTANT: read LR AFTER scheduler.step
        lr = optim.param_groups[0]["lr"]
        
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_auprc={val_auprc:.4f} | bestF1={val_out['best_f1']} | lr={lr:.2e}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auprc": val_auprc,
            "best_f1": val_out["best_f1"],
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
    
            # Plots for best checkpoint moment (optional)
            save_learning_curves(reports_dir, history)
    print(f"[DONE] Best AUPRC: {best_auprc:.4f}")
    print(f"[DONE] Run saved to: {run_dir}")


if __name__ == "__main__":
    main()