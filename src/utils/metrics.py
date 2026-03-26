import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score


# ---------------------------------------------------------
# AUPRC (Area Under Precision-Recall Curve)
# ---------------------------------------------------------
def _compute_pr_curve(y_true: np.ndarray, y_prob: np.ndarray):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or not np.any(y_true == 1):
        return None
    return precision_recall_curve(y_true, y_prob)


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute AUPRC for binary classification.

    Parameters
    ----------
    y_true : (N,) array of {0,1}
    y_prob : (N,) array of probabilities (after sigmoid)

    Returns
    -------
    float : area under precision-recall curve
    """
    pr_curve = _compute_pr_curve(y_true, y_prob)
    if pr_curve is None:
        return float("nan")

    precision, recall, _ = pr_curve
    return auc(recall, precision)


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_pr_points(y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict[str, float]]:
    pr_curve = _compute_pr_curve(y_true, y_prob)
    if pr_curve is None:
        return []

    precision, recall, thresholds = pr_curve

    rows: List[Dict[str, float]] = []
    for idx, (p, r) in enumerate(zip(precision, recall)):
        thr = float("nan") if idx == 0 else float(thresholds[idx - 1])
        rows.append(
            {
                "precision": float(p),
                "recall": float(r),
                "threshold": thr,
            }
        )
    return rows


def compute_roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict[str, float]]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.unique(y_true).size < 2:
        return []

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    rows: List[Dict[str, float]] = []
    for fp_rate, tp_rate, thr in zip(fpr, tpr, thresholds):
        rows.append(
            {
                "fpr": float(fp_rate),
                "tpr": float(tp_rate),
                "threshold": float(thr),
            }
        )
    return rows


# ---------------------------------------------------------
# Threshold sweep for binary classification
# ---------------------------------------------------------
def threshold_sweep_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.01,
    end: float = 0.99,
    step: float = 0.01,
) -> List[Dict[str, float]]:
    """
    Sweep thresholds and compute precision/recall/F1.

    Returns a list of dicts:
        [
            {
              "threshold": ...,
              "precision": ...,
              "recall": ...,
              "f1": ...,
              "tp": ...,
              "fp": ...,
              "fn": ...
            },
            ...
        ]
    """
    results = []

    thresholds = np.arange(start, end + 1e-9, step)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results.append({
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

    return results


def confusion_stats_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true_i == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true_i == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_i == 1)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def save_history_csv(out_dir: Path, history: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_threshold_sweep_csv(metrics_dir: Path, sweep: List[Dict[str, Any]]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if not sweep:
        return
    csv_path = metrics_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)


def save_rows_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(out_path: Path, rows: List[Tuple[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def save_learning_curves(out_dir: Path, history: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    epoch_list = [h["epoch"] for h in history]
    train_loss_list = [h["train_loss"] for h in history]
    val_auprc_list = [h["val_auprc"] for h in history]
    val_auroc_list = [h.get("val_auroc", float("nan")) for h in history]
    lr_list = [h["lr"] for h in history]

    plt.figure()
    plt.plot(epoch_list, train_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(out_dir / "train_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epoch_list, val_auprc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Val AUPRC")
    plt.title("Validation AUPRC")
    plt.grid(True)
    plt.savefig(out_dir / "val_auprc.png", dpi=150)
    plt.close()

    if any("val_auroc" in h for h in history):
        plt.figure()
        plt.plot(epoch_list, val_auroc_list)
        plt.xlabel("Epoch")
        plt.ylabel("Val AUROC")
        plt.title("Validation AUROC")
        plt.grid(True)
        plt.savefig(out_dir / "val_auroc.png", dpi=150)
        plt.close()

    plt.figure()
    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate")
    plt.grid(True)
    plt.savefig(out_dir / "lr.png", dpi=150)
    plt.close()


def save_confusion_matrix_image(out_path: Path, stats: Dict[str, float], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.array(
        [
            [stats["tn"], stats["fp"]],
            [stats["fn"], stats["tp"]],
        ],
        dtype=np.float64,
    )
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.3f}\n(n={int(cm[i, j])})",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pr_curve_image(out_path: Path, y_true: np.ndarray, y_prob: np.ndarray, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pr_curve = _compute_pr_curve(y_true, y_prob)
    if pr_curve is None:
        return

    precision, recall, _ = pr_curve
    auprc = compute_auprc(y_true, y_prob)

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC={auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_roc_curve_image(out_path: Path, y_true: np.ndarray, y_prob: np.ndarray, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if np.unique(np.asarray(y_true)).size < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = compute_auroc(y_true, y_prob)

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC={auroc:.4f}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
