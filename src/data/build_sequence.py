import numpy as np
from src.data.features import (
    extract_erez_motion_features,
    frame_to_vector,
    motion_feature_cfg,
    motion_feature_dim,
    select_motion_features,
)
from src.data.labels import events_to_label


def build_sequence(frames, K: int, feature_cfg=None):  # , label_cfg: dict):
    X, Y = [], []
    for frame in frames:
        x_t = frame_to_vector(frame, K, cfg=feature_cfg)
        y_t = events_to_label(frame)  # , label_cfg)
        X.append(x_t)
        Y.append(y_t)

    X = np.stack(X).astype(np.float32)  # (T, C) or (T, K, C)
    motion_cfg = motion_feature_cfg(feature_cfg)
    if motion_cfg.get("enabled", False):
        motion_seq = extract_erez_motion_features(
            frames,
            align=motion_cfg.get("align", "prev"),
            j_version=(feature_cfg or {}).get("erez_json_version", 2.0),
        )
        motion_seq = select_motion_features(motion_seq, target_dim=motion_feature_dim(feature_cfg))
        X = np.concatenate([X, motion_seq], axis=-1)

    Y = np.asarray(Y, dtype=np.float32)  # (T,)
    return X, Y
