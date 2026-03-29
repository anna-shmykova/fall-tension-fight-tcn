from src.data.build_sequence import build_sequence
from src.data.json_io import read_json_frames
from src.data.motion_sequence import build_motion_sequence
from torch.utils.data import Dataset
import numpy as np
import torch


def resolve_target_index(window_size: int, target_mode: str) -> int:
    target_mode = str(target_mode).lower()
    if target_mode == "center":
        return window_size // 2
    if target_mode == "last":
        return window_size - 1
    raise ValueError(f"Unsupported target_mode: {target_mode}")


class _BaseWindowDataset(Dataset):
    def __init__(self, json_paths, K, sequence_builder,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None, target_mode="center", verbose=True):
        self.samples = []
        self.window_size = window_size
        self.window_step = window_step
        self.K = K
        self.feature_cfg = feature_cfg or {}
        self.verbose = bool(verbose)
        self.target_mode = str(target_mode).lower()
        self.target_index = resolve_target_index(window_size=self.window_size, target_mode=self.target_mode)

        for path in json_paths:
            frame = read_json_frames(path)
            X_seq, y_seq = sequence_builder(frame, K, feature_cfg=self.feature_cfg)
            N = len(X_seq)
            if N < window_size:
                continue

            for start in range(0, N - window_size + 1, window_step):
                end = start + window_size
                X_win = X_seq[start:end] # (T, C)
                y_win = y_seq[start:end]    # (T,)
                y_target = y_win[self.target_index]
                has_any_pos = bool(np.any(y_win == 1))

                # Keep windows whose target frame is positive, or windows that are entirely negative.
                # Drop mixed windows where the event is elsewhere in the window but not at the target frame.
                if y_target == 1 or not has_any_pos:
                    self.samples.append((X_win, y_target))
                else:
                    continue
                
        if self.verbose:
            print(f"Total windows ({self.target_mode} target):", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_win, y_last = self.samples[idx]
        X_win = torch.from_numpy(X_win)        # (T, C)
        y_last = torch.tensor(y_last)        # (1,)
        return X_win, y_last


class EventJsonDataset(_BaseWindowDataset):
    def __init__(self, json_paths, K,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None, target_mode="center", verbose=True):
        super().__init__(
            json_paths=json_paths,
            K=K,
            sequence_builder=lambda frames, K, feature_cfg: build_sequence(frames, K, feature_cfg=feature_cfg),
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
            sequence_builder=lambda frames, K, feature_cfg: build_motion_sequence(frames, feature_cfg=feature_cfg),
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
            window_cfg=window_cfg,
            target_mode=target_mode,
            verbose=verbose,
        )
