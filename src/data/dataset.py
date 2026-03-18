from src.data.build_sequence import build_sequence
from src.data.json_io import read_json_frames
from src.data.motion_sequence import build_motion_sequence
from torch.utils.data import Dataset
import numpy as np
import torch

class _BaseWindowDataset(Dataset):
    def __init__(self, json_paths, K, sequence_builder,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None):
        self.samples = []
        self.window_size = window_size
        self.window_step = window_step
        self.K = K
        self.feature_cfg = feature_cfg or {}

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
                #ignore examples of dataset where events are at the very beginning and at the very end
                count_pos_beg = np.sum(y_win[:window_size//2] == 1)
                count_pos_end = np.sum(y_win[window_size//2:] == 1)
                if y_win[window_size//2] == 1 or (y_win[window_size//2] == 0 and (count_pos_beg == 0 and count_pos_end == 0)):
                    y_central = y_win[window_size//2]
                    #print(y_central)
                
                    # if window has only 'none', keep only a fraction
                    #if np.all(y_win == 0):
                        #if np.random.rand() > 0.2:  # keep ~20% of pure-none windows
                            #continue
    
                    self.samples.append((X_win, y_central))
                else:
                    continue
                
        print("Total windows:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_win, y_last = self.samples[idx]
        X_win = torch.from_numpy(X_win)        # (T, C)
        y_last = torch.tensor(y_last)        # (1,)
        return X_win, y_last


class EventJsonDataset(_BaseWindowDataset):
    def __init__(self, json_paths, K,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None):
        super().__init__(
            json_paths=json_paths,
            K=K,
            sequence_builder=lambda frames, K, feature_cfg: build_sequence(frames, K, feature_cfg=feature_cfg),
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
            window_cfg=window_cfg,
        )


class MotionJsonDataset(_BaseWindowDataset):
    def __init__(self, json_paths, K,
                 window_size=16, window_step=4, feature_cfg=None, label_cfg=None, window_cfg=None):
        super().__init__(
            json_paths=json_paths,
            K=K,
            sequence_builder=lambda frames, K, feature_cfg: build_motion_sequence(frames, feature_cfg=feature_cfg),
            window_size=window_size,
            window_step=window_step,
            feature_cfg=feature_cfg,
            label_cfg=label_cfg,
            window_cfg=window_cfg,
        )
