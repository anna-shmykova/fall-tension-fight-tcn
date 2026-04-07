import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
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


def sigmoid_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(logits, dtype=np.float64)
    pos_mask = logits >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-logits[pos_mask]))
    exp_logits = np.exp(logits[~pos_mask])
    out[~pos_mask] = exp_logits / (1.0 + exp_logits)
    return out.astype(np.float32)


def _binary_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
    y_true_arr = np.clip(y_true_arr[mask], 0.0, 1.0)
    y_prob_arr = np.clip(y_prob_arr[mask], 0.0, 1.0)
    return y_true_arr, y_prob_arr


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true_arr, y_prob_arr = _binary_arrays(y_true, y_prob)
    if y_true_arr.size == 0:
        return float("nan")
    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def compute_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    y_true_arr, y_prob_arr = _binary_arrays(y_true, y_prob)
    n_bins = max(int(n_bins), 1)
    if y_true_arr.size == 0:
        return []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bin_edges[1:-1], right=False)
    rows: List[Dict[str, float]] = []

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(np.sum(mask))
        lower = float(bin_edges[bin_idx])
        upper = float(bin_edges[bin_idx + 1])
        center = float((lower + upper) / 2.0)

        if count == 0:
            rows.append(
                {
                    "bin": int(bin_idx),
                    "lower": lower,
                    "upper": upper,
                    "center": center,
                    "count": 0,
                    "mean_confidence": float("nan"),
                    "fraction_positive": float("nan"),
                    "abs_gap": float("nan"),
                }
            )
            continue

        mean_confidence = float(np.mean(y_prob_arr[mask]))
        fraction_positive = float(np.mean(y_true_arr[mask]))
        rows.append(
            {
                "bin": int(bin_idx),
                "lower": lower,
                "upper": upper,
                "center": center,
                "count": count,
                "mean_confidence": mean_confidence,
                "fraction_positive": fraction_positive,
                "abs_gap": float(abs(fraction_positive - mean_confidence)),
            }
        )

    return rows


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = compute_calibration_bins(y_true, y_prob, n_bins=n_bins)
    total = sum(int(row["count"]) for row in bins)
    if total == 0:
        return float("nan")
    return float(sum((int(row["count"]) / total) * float(row["abs_gap"]) for row in bins if int(row["count"]) > 0))


def summarize_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    y_true_arr, y_prob_arr = _binary_arrays(y_true, y_prob)
    bins = compute_calibration_bins(y_true_arr, y_prob_arr, n_bins=n_bins)
    total = int(y_true_arr.size)
    n_pos = int(np.sum(y_true_arr > 0.0))
    positive_rate = float(np.mean(y_true_arr)) if total > 0 else float("nan")
    brier = compute_brier_score(y_true_arr, y_prob_arr)
    brier_baseline = float(np.mean((positive_rate - y_true_arr) ** 2)) if total > 0 else float("nan")
    brier_skill_score = float(1.0 - brier / brier_baseline) if brier_baseline > 0.0 else float("nan")
    ece = compute_ece(y_true_arr, y_prob_arr, n_bins=n_bins)
    populated_gaps = [float(row["abs_gap"]) for row in bins if int(row["count"]) > 0]

    return {
        "n_samples": total,
        "n_bins": int(max(int(n_bins), 1)),
        "n_pos": n_pos,
        "n_neg": int(np.sum(y_true_arr <= 0.0)),
        "positive_rate": positive_rate,
        "brier_score": brier,
        "brier_baseline": brier_baseline,
        "brier_skill_score": brier_skill_score,
        "ece": ece,
        "max_calibration_error": float(max(populated_gaps)) if populated_gaps else float("nan"),
        "bins": bins,
    }


def binary_nll_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true_arr) & np.isfinite(logits_arr)
    y_true_arr = y_true_arr[mask]
    logits_arr = logits_arr[mask]
    if y_true_arr.size == 0:
        return float("nan")
    losses = np.maximum(logits_arr, 0.0) - logits_arr * y_true_arr + np.log1p(np.exp(-np.abs(logits_arr)))
    return float(np.mean(losses))


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
    return sigmoid_from_logits(np.asarray(logits, dtype=np.float64) / temperature)


def fit_temperature_scaling(
    y_true: np.ndarray,
    logits: np.ndarray,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    n_iter: int = 80,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true_arr) & np.isfinite(logits_arr)
    y_true_arr = y_true_arr[mask]
    logits_arr = logits_arr[mask]
    if y_true_arr.size == 0 or np.unique(y_true_arr).size < 2:
        return {
            "method": "temperature",
            "available": False,
            "temperature": float("nan"),
            "nll_before": binary_nll_from_logits(y_true_arr, logits_arr),
            "nll_after": float("nan"),
            "reason": "need at least one positive and one negative label",
        }

    lo = math.log(max(float(min_temperature), 1e-6))
    hi = math.log(max(float(max_temperature), math.exp(lo) * 1.0001))
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        return binary_nll_from_logits(y_true_arr, logits_arr / temperature)

    h = hi - lo
    c = lo + inv_phi_sq * h
    d = lo + inv_phi * h
    yc = objective(c)
    yd = objective(d)

    for _ in range(max(int(n_iter), 1)):
        if yc < yd:
            hi = d
            d = c
            yd = yc
            h = inv_phi * h
            c = lo + inv_phi_sq * h
            yc = objective(c)
        else:
            lo = c
            c = d
            yc = yd
            h = inv_phi * h
            d = lo + inv_phi * h
            yd = objective(d)

    best_log_temperature = (lo + hi) / 2.0
    temperature = float(math.exp(best_log_temperature))
    return {
        "method": "temperature",
        "available": True,
        "temperature": temperature,
        "nll_before": binary_nll_from_logits(y_true_arr, logits_arr),
        "nll_after": binary_nll_from_logits(y_true_arr, logits_arr / temperature),
    }


def apply_platt_scaling(logits: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return sigmoid_from_logits(float(slope) * np.asarray(logits, dtype=np.float64) + float(intercept))


def fit_platt_scaling(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true_arr) & np.isfinite(logits_arr)
    y_true_arr = y_true_arr[mask]
    logits_arr = logits_arr[mask]
    if y_true_arr.size > 0 and not np.all(np.isin(np.unique(y_true_arr), [0.0, 1.0])):
        y_true_arr = (y_true_arr > 0.0).astype(np.int32)
    if y_true_arr.size == 0 or np.unique(y_true_arr).size < 2:
        return {
            "method": "platt",
            "available": False,
            "slope": float("nan"),
            "intercept": float("nan"),
            "nll_before": binary_nll_from_logits(y_true_arr, logits_arr),
            "nll_after": float("nan"),
            "reason": "need at least one positive and one negative label",
        }
    y_true_arr = y_true_arr.astype(np.int32)

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        clf.fit(logits_arr.reshape(-1, 1), y_true_arr)
    except Exception as exc:
        return {
            "method": "platt",
            "available": False,
            "slope": float("nan"),
            "intercept": float("nan"),
            "nll_before": binary_nll_from_logits(y_true_arr, logits_arr),
            "nll_after": float("nan"),
            "reason": str(exc),
        }

    slope = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])
    calibrated_logits = slope * logits_arr + intercept
    return {
        "method": "platt",
        "available": True,
        "slope": slope,
        "intercept": intercept,
        "nll_before": binary_nll_from_logits(y_true_arr, logits_arr),
        "nll_after": binary_nll_from_logits(y_true_arr, calibrated_logits),
    }


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


def save_reliability_diagram_image(
    out_path: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    n_bins: int = 10,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_calibration(y_true, y_prob, n_bins=n_bins)
    bins = summary.get("bins", [])
    populated = [row for row in bins if int(row["count"]) > 0]
    if not populated:
        return

    mean_confidence = [float(row["mean_confidence"]) for row in populated]
    fraction_positive = [float(row["fraction_positive"]) for row in populated]
    centers = [float(row["center"]) for row in bins]
    counts = [int(row["count"]) for row in bins]
    n_bins = max(int(summary["n_bins"]), 1)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="gray", label="perfect")
    ax.plot(mean_confidence, fraction_positive, marker="o", linewidth=2.0, label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(
        f"{title}\nECE={summary['ece']:.4f} | Brier={summary['brier_score']:.4f} | n={summary['n_samples']}"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=0.9 / n_bins, color="tab:gray", alpha=0.18)
    ax2.set_ylabel("Bin count")
    ax2.set_ylim(0.0, max(max(counts), 1) * 1.2)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
