from src.data.build_sequence import build_sequence
from src.data.json_io import read_json_frames
from src.data.motion_sequence import build_motion_sequence
from torch.utils.data import Dataset
import numpy as np
import torch


WINDOW_LABEL_RULES = {"train_like", "target", "any_overlap", "overlap_frac", "soft_last_k"}


def resolve_target_index(window_size: int, target_mode: str) -> int:
    target_mode = str(target_mode).lower()
    if target_mode == "center":
        return window_size // 2
    if target_mode == "last":
        return window_size - 1
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def resolve_window_label_cfg(window_cfg=None):
    cfg = dict(window_cfg or {})
    rule = str(cfg.get("rule", cfg.get("label_rule", "train_like"))).lower()
    if rule not in WINDOW_LABEL_RULES:
        raise ValueError(f"Unsupported window label rule: {rule}")

    positive_overlap = float(cfg.get("positive_overlap", cfg.get("positive_overlap_fraction", 0.3)))
    if not (0.0 <= positive_overlap <= 1.0):
        raise ValueError(f"window positive overlap must be in [0, 1], got {positive_overlap}")

    soft_k = int(cfg.get("soft_k", cfg.get("soft_last_k", 4)))
    if soft_k < 1:
        raise ValueError(f"window soft_k must be >= 1, got {soft_k}")

    return {"rule": rule, "positive_overlap": positive_overlap, "soft_k": soft_k}


def label_window(y_win: np.ndarray, target_index: int, window_cfg=None):
    cfg = resolve_window_label_cfg(window_cfg)
    y_win = np.asarray(y_win, dtype=np.int32)
    if y_win.size == 0:
        return None

    y_target = int(y_win[target_index])
    positive_fraction = float(np.mean(y_win == 1))
    has_any_pos = bool(positive_fraction > 0.0)
    rule = cfg["rule"]

    if rule == "train_like":
        if y_target == 1 or not has_any_pos:
            return y_target
        return None
    if rule == "target":
        return y_target
    if rule == "any_overlap":
        return int(has_any_pos)
    if rule == "overlap_frac":
        return int(positive_fraction >= float(cfg["positive_overlap"]))
    if rule == "soft_last_k":
        k = int(min(max(int(cfg["soft_k"]), 1), target_index + 1))
        start = int(target_index - k + 1)
        return float(np.mean(y_win[start : target_index + 1] == 1))
    raise ValueError(f"Unsupported window label rule: {rule}")


class _BaseWindowDataset(Dataset):
    def __init__(self, json_paths, K, sequence_builder,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None, target_mode="center", verbose=True):
        self.samples = []
        self.window_size = window_size
        self.window_step = window_step
        self.K = K
        self.feature_cfg = feature_cfg or {}
        self.label_cfg = label_cfg or {}
        self.window_cfg = resolve_window_label_cfg(window_cfg)
        self.verbose = bool(verbose)
        self.target_mode = str(target_mode).lower()
        self.target_index = resolve_target_index(window_size=self.window_size, target_mode=self.target_mode)
        total_paths = len(json_paths)
        report_every = 100

        if self.verbose:
            print(
                f"[INFO] Building windows from {total_paths} files "
                f"(window_size={self.window_size}, step={self.window_step}, target={self.target_mode})",
                flush=True,
            )

        for index, path in enumerate(json_paths, start=1):
            frame = read_json_frames(path)
            X_seq, y_seq = sequence_builder(frame, K, feature_cfg=self.feature_cfg, label_cfg=self.label_cfg)
            N = len(X_seq)
            if N < window_size:
                if self.verbose and (index == total_paths or index % report_every == 0):
                    print(
                        f"[INFO] Processed {index}/{total_paths} files | windows={len(self.samples)}",
                        flush=True,
                    )
                continue

            for start in range(0, N - window_size + 1, window_step):
                end = start + window_size
                X_win = X_seq[start:end]  # (T, C)
                y_win = y_seq[start:end]  # (T,)
                y_label = label_window(y_win, target_index=self.target_index, window_cfg=self.window_cfg)
                if y_label is None:
                    continue
                self.samples.append((X_win, float(y_label)))

            if self.verbose and (index == total_paths or index % report_every == 0):
                print(
                    f"[INFO] Processed {index}/{total_paths} files | windows={len(self.samples)}",
                    flush=True,
                )

        if self.verbose:
            print(
                f"Total windows ({self.target_mode} target, {self.window_cfg['rule']} rule):",
                len(self.samples),
                flush=True,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_win, y_label = self.samples[idx]
        X_win = torch.from_numpy(X_win)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        return X_win, y_label


class EventJsonDataset(_BaseWindowDataset):
    def __init__(self, json_paths, K,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None, target_mode="center", verbose=True):
        super().__init__(
            json_paths=json_paths,
            K=K,
            sequence_builder=lambda frames, K, feature_cfg, label_cfg: build_sequence(
                frames,
                K,
                feature_cfg=feature_cfg,
                label_cfg=label_cfg,
            ),
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
            window_cfg=window_cfg,
            target_mode=target_mode,
            verbose=verbose,
        )


class MotionJsonDataset(_BaseWindowDataset):
    def __init__(self, json_paths, K,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None, target_mode="center", verbose=True):
        super().__init__(
            json_paths=json_paths,
            K=K,
            sequence_builder=lambda frames, K, feature_cfg, label_cfg: build_motion_sequence(
                frames,
                feature_cfg=feature_cfg,
                label_cfg=label_cfg,
            ),
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
            window_cfg=window_cfg,
            target_mode=target_mode,
            verbose=verbose,
        )
