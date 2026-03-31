import numpy as np
from src.erez_files.analyze_json_motion import (
    DEFAULT_VERSION as EREZ_DEFAULT_VERSION,
    extract_motion_features as erez_extract_motion_features,
)


EREZ_MOTION_DIM = 25


def _get_detections(frame):
    return frame.get("detection_list") or frame.get("detections_list") or []


def _get_keypoints(det):
    return det.get("key_points") or det.get("key_pts") or []


def motion_feature_cfg(feature_cfg=None):
    feature_cfg = feature_cfg or {}
    motion_cfg = dict(feature_cfg.get("motion", {}))

    # Legacy configs used `features.type: erez_motion`.
    if feature_cfg.get("type") == "erez_motion" and "enabled" not in motion_cfg:
        motion_cfg["enabled"] = True
    if feature_cfg.get("erez_json_version") is not None and "source" not in motion_cfg:
        motion_cfg["source"] = "erez"

    motion_cfg.setdefault("enabled", False)
    motion_cfg.setdefault("source", "erez")
    motion_cfg.setdefault("align", "prev")
    return motion_cfg


def motion_feature_dim(feature_cfg=None):
    motion_cfg = motion_feature_cfg(feature_cfg)
    if not motion_cfg.get("enabled", False):
        return 0
    if motion_cfg.get("source") != "erez":
        raise ValueError(f"Unsupported motion feature source: {motion_cfg.get('source')}")

    requested_dim = motion_cfg.get("num_features", motion_cfg.get("first_n", EREZ_MOTION_DIM))
    requested_dim = int(requested_dim)
    if requested_dim <= 0 or requested_dim > EREZ_MOTION_DIM:
        raise ValueError(
            f"features.motion.num_features must be in [1, {EREZ_MOTION_DIM}], got {requested_dim}"
        )
    return requested_dim


def _to_erez_frame(frame):
    detections = []
    for det in _get_detections(frame):
        detections.append(
            {
                "class": det.get("class", 0),
                "conf": det.get("conf", 0.0),
                "bbox": det.get("bbox", []),
                "key_pts": _get_keypoints(det),
            }
        )

    return {
        "f": frame.get("f", 0),
        "t": frame.get("t", 0.0),
        "group_events": frame.get("group_events", []),
        "detections_list": detections,
    }


def norm_kps_xy_flat_with_mask(kps_xy_flat, cx, cy, w, h, eps=1e-6):
    inv_w = 1.0 / (w + eps)
    inv_h = 1.0 / (h + eps)

    kps_norm = []

    for j in range(0, len(kps_xy_flat), 3):
        x = kps_xy_flat[j]
        y = kps_xy_flat[j + 1]

        visible = not (x == 0 and y == 0)
        kps_norm.extend([(x - cx) * inv_w, (y - cy) * inv_h, 1.0 if visible else 0.0])

    return kps_norm


def frame_to_vector(frame, K, num_pers_features=40, cfg=None):
    """
    frame: dict for one time step
    K: max number of boxes per frame
    """
    detections = _get_detections(frame)

    dets_sorted = sorted(
        detections,
        key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        reverse=True,
    )

    boxes = []
    for d in dets_sorted[:K]:
        x1, y1, x2, y2 = d["bbox"]
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        kps = _get_keypoints(d)[15:]
        kps_norm = norm_kps_xy_flat_with_mask(kps, cx, cy, w, h)
        boxes.extend([cx, cy, w, h] + kps_norm)

    if len(dets_sorted) < K:
        boxes.extend([0.0] * num_pers_features * (K - len(dets_sorted)))

    n_people_norm = len(detections) / 25
    x_t = boxes + [float(n_people_norm)]
    return np.array(x_t, dtype=np.float32)


def select_motion_features(motion_values, target_dim: int):
    motion_values = np.asarray(motion_values, dtype=np.float32)
    target_dim = int(target_dim)
    if target_dim < 0 or target_dim > EREZ_MOTION_DIM:
        raise ValueError(f"target_dim must be in [0, {EREZ_MOTION_DIM}], got {target_dim}")
    if motion_values.shape[-1] < target_dim:
        raise ValueError(
            f"Motion feature tensor has width {motion_values.shape[-1]}, cannot keep first {target_dim} features"
        )
    if motion_values.shape[-1] == target_dim:
        return motion_values.astype(np.float32, copy=False)
    return motion_values[..., :target_dim].astype(np.float32, copy=False)


def extract_erez_motion_features(frames, align="prev", j_version=EREZ_DEFAULT_VERSION):
    if align not in {"prev", "next"}:
        raise ValueError(f"Unsupported motion alignment: {align}")

    num_frames = len(frames)
    if num_frames == 0:
        return np.zeros((0, EREZ_MOTION_DIM), dtype=np.float32)
    motion_feats = np.zeros((num_frames, EREZ_MOTION_DIM), dtype=np.float32)

    if num_frames == 1:
        return motion_feats

    erez_frames = [_to_erez_frame(frame) for frame in frames]
    erez_motion = erez_extract_motion_features(erez_frames, j_version=float(j_version)).astype(np.float32)

    if erez_motion.shape[1] != EREZ_MOTION_DIM:
        raise ValueError(
            f"Unexpected erez motion dimension: expected {EREZ_MOTION_DIM}, got {erez_motion.shape[1]}"
        )

    if align == "prev":
        motion_feats[1:] = erez_motion
    else:
        motion_feats[:-1] = erez_motion

    return motion_feats
