from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.data.labels import events_to_label
from src.erez_files.analyze_json_motion import extract_motion_features


MOTION_FEATURE_DIM = 21


def adapt_frame_for_erez(frame: Dict[str, Any]) -> Dict[str, Any]:
    detections = []
    for det in frame.get("detection_list", []):
        key_pts = det.get("key_points")
        if key_pts is None:
            key_pts = det.get("key_pts", [])

        detections.append(
            {
                "class": det.get("class", 0),
                "conf": det.get("conf", 0.0),
                "bbox": det.get("bbox", []),
                "key_pts": key_pts,
            }
        )

    return {
        "f": frame.get("f"),
        "t": frame.get("t", 0.0),
        "group_events": frame.get("group_events", []),
        "detections_list": detections,
    }


def adapt_frames_for_erez(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [adapt_frame_for_erez(frame) for frame in frames]


def build_motion_sequence(frames, feature_cfg: Dict[str, Any] | None = None):
    feature_cfg = feature_cfg or {}
    j_version = float(feature_cfg.get("erez_json_version", 2.0))

    if len(frames) < 2:
        return (
            np.zeros((0, MOTION_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    frames_erez = adapt_frames_for_erez(frames)
    X = extract_motion_features(frames_erez, j_version=j_version).astype(np.float32)

    # Motion feature x[t] describes the transition from frames[t] -> frames[t+1].
    # Align it with the later frame label so the sequence stays causal.
    y_full = np.asarray([events_to_label(frame) for frame in frames], dtype=np.float32)
    y = y_full[1:]

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Motion features/labels length mismatch: X={X.shape[0]} y={y.shape[0]}")

    return X, y
