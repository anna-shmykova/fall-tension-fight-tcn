"""     Extract order-free motion features from JSON frame sequences.
    This file converts per-frame detections/keypoints into a clip-level motion
    representation. The main entry point is `extract_motion_features(...)`, which
    builds one 25-dim feature vector for each pair of consecutive frames by default,
    or 48 dims when extended=True.
    Feature order:
    1.  d_mean_center_x      : change in mean bbox center (x,y); global  crowd motion
    2.  d_mean_center_y      :
    3.  d_var_center_x       : change in bbox-center variance;  spread/compression
    4.  d_var_center_y       :
    5.  d_mean_size_w        : change in mean bbox size (width/height); average scale/depth change
    6.  d_mean_size_h        :
    7.  d_max_size_w         : change in largest bbox size; strongest scale change
    8.  d_max_size_h         :
    9.  d_mean_pairwise      : change in mean pairwise bbox-center distance; crowd density cue
    10. d_var_pairwise       : change in variance of pairwise distances; spacing heterogeneity
    11. d_kp_mean_x          : change in mean keypoint(x, y); global articulated  motion
    12. d_kp_mean_y          :
    13. d_kp_var_x           : change in keypoint variance; pose spread
    14. d_kp_var_y           :
    15. bb_nn_mean           : mean nearest-neighbor bbox-center motion; average object motion
    16. bb_nn_max            : max nearest-neighbor bbox-center motion; strongest mover
    17. kp_nn_mean           : mean nearest-neighbor keypoint motion; average articulated motion
    18. kp_nn_max            : max nearest-neighbor keypoint motion; strongest articulated motion
    * Added later (14/03/26)
    19. d_union_coverage     : change in union area covered by all bboxes
    20. d_mean_pairwise_iou  : change in mean pairwise bbox IoU
    21. d_max_pairwise_iou   : change in max pairwise bbox IoU
    * Static features
    22. union_coverage       : union area covered by all bboxes in frame t+1
    23. mean_pairwise_iou    : mean pairwise bbox IoU in frame t+1
    24. max_pairwise_iou     : max pairwise bbox IoU in frame t+1
    25. overlap_ratio        : redundant bbox overlap ratio in frame t+1
    * Optional extended features, enabled with extract_motion_features(..., extended=True)
    26. min_pairwise_bbox_dist         : nearest bbox-center distance in frame t+1
    27. d_min_pairwise_bbox_dist       : change in nearest bbox-center distance; negative means closing
    28. close_pair_ratio               : fraction of bbox-center pairs closer than a threshold in frame t+1
    29. d_close_pair_ratio             : change in close_pair_ratio
    30. wrist_rel_motion_mean          : matched wrist motion after subtracting bbox-center motion
    31. wrist_rel_motion_p90           : robust high wrist relative motion
    32. elbow_rel_motion_mean          : matched elbow motion after subtracting bbox-center motion
    33. upper_limb_rel_motion_energy   : matched shoulder/elbow/wrist relative motion mean
    34. upper_limb_visibility_ratio    : visible/confident upper-limb keypoint ratio in frame t+1
    35. lower_limb_rel_motion_mean     : matched hip/knee/ankle relative motion mean
    36. lower_limb_visibility_ratio    : visible/confident lower-limb keypoint ratio in frame t+1
    37. matched_person_ratio           : accepted bbox-based matches from t to t+1
    38. ambiguous_match_ratio          : rejected ambiguous source-person matches
    39. wrist_to_other_bbox_dist_min   : closest visible wrist to another person's bbox in frame t+1
    40. wrist_near_other_bbox_ratio    : fraction of visible wrists close to another person's bbox
    41. upper_limb_overlap_motion      : upper_limb_rel_motion_energy weighted by bbox overlap
    42. match_quality_mean             : mean accepted bbox-match quality
    43. body_anchor_dist_min           : nearest torso/body-anchor distance in frame t+1
    44. d_body_anchor_dist_min         : change in nearest torso/body-anchor distance; negative means closing
    45. head_to_head_dist_min          : nearest visible head-anchor distance in frame t+1
    46. wrist_to_other_body_anchor_min : closest visible wrist to another person's torso/body anchor
    47. pair_rel_speed_mean_abs        : mean absolute matched pairwise body-anchor distance change
    48. pair_closing_speed_min         : strongest matched pairwise body-anchor closing speed
"""

import numpy as np

N_Keypoints = 17
DEFAULT_VERSION = 2.0
BASE_MOTION_FEATURE_DIM = 25
EXTENDED_MOTION_FEATURE_DIM = 48

WRIST_KP = (9, 10)
ELBOW_KP = (7, 8)
UPPER_LIMB_KP = (5, 6, 7, 8, 9, 10)
LOWER_LIMB_KP = (11, 12, 13, 14, 15, 16)
HEAD_KP = (0, 1, 2, 3, 4)
SHOULDER_KP = (5, 6)
HIP_KP = (11, 12)
TORSO_KP = (5, 6, 11, 12)

DEFAULT_KP_CONF_MIN = 0.3
DEFAULT_MATCH_MIN_IOU = 0.2
DEFAULT_MATCH_MAX_CENTER_DIST = 0.15
DEFAULT_MATCH_MAX_SIZE_RATIO = 2.5
DEFAULT_MATCH_AMBIGUITY_MARGIN = 0.03
DEFAULT_CLOSE_PAIR_DIST_THRESH = 0.15
DEFAULT_WRIST_NEAR_BBOX_THRESH = 0.08
DEFAULT_MISSING_PAIR_DIST = 1.0

# --------------------------------------------------
# * public fucntion, to be used by other units
# --------------------------------------------------

def extract_motion_features(frames, j_version:float=DEFAULT_VERSION, **kwargs):
    """Convert frames into a T x C motion sequence.
    Kwargs:
        extended: if True, append upper-limb and bbox interaction features (42 dims total).
        pure_motion: if True, drop the static overlap features (22-25).
        legacy: if True, return only the original 18 features.
    """
    frame_feats = []
    raw_points = []
    raw_persons = []

    kp_conf = True if j_version >= 2.0 else False
    vec_kp = 3 if kp_conf else 2
    extended = bool(kwargs.get('extended', kwargs.get('include_extended', False)))

    for frm in frames:
        if len(frm['detections_list']) > 0:
            pass
            # print_color(frm)
        bb_centers, bb_sizes, keypoints, bboxes = extract_frame_geometry(frm, kp_conf=kp_conf)
        agg = frame_aggregates(bb_centers, bb_sizes, keypoints, bboxes)
        frame_feats.append(agg)
        raw_points.append((bb_centers, keypoints))
        if extended:
            raw_persons.append(extract_person_geometry(
                frm,
                vec_kp=vec_kp,
                kp_min_conf=kwargs.get('kp_min_conf', DEFAULT_KP_CONF_MIN),
            ))

    frame_feats = np.stack(frame_feats)  #* TxC

    motion_feats = []
    for t in range(len(frames) - 1):
        delta_agg = frame_feats[t + 1] - frame_feats[t]
        curr_agg = frame_feats[t + 1]

        bb_c_t, kp_t = raw_points[t]
        bb_c_tp1, kp_tp1 = raw_points[t + 1]

        #* Nearest Neighbor Motion
        bb_mean, bb_max = nearest_neighbor_motion(bb_c_t, bb_c_tp1)
        kp_mean, kp_max = nearest_neighbor_motion(kp_t, kp_tp1)
        nnm_features = np.array([bb_mean, bb_max, kp_mean, kp_max], dtype=np.float32)

        motion_vec = np.concatenate([delta_agg[:14],
                                    nnm_features,
                                    delta_agg[14:17],   #* d_union_coverage, d_mean_pairwise_iou, d_max_pairwise_iou
                                    curr_agg[14:17],    #* union_coverage, mean_pairwise_iou, max_pairwise_iou
                                    curr_agg[17:18],    #* overlap_ratio
                                     ])
        if extended:
            motion_vec = np.concatenate([
                motion_vec,
                extended_limb_bbox_motion_features(raw_persons[t], raw_persons[t + 1], curr_agg, **kwargs),
            ])
        motion_feats.append(motion_vec)

    motion_feats = np.stack(motion_feats)
    if kwargs.get('legacy', False):      #* The original 18 features, for backward compatibility with old models
        return motion_feats[:, :18]
    if kwargs.get('pure_motion', False): #* Exclude static features
        return motion_feats[:, :21]
    return motion_feats  # (T-1) x C'


# -------------------------------------------------
# * private function; local helpers for compute_motion_sequence
# --------------------------------------------------

# ***** Step 1:  extract frame-level geometry ****
def extract_frame_geometry(frame, **kwargs):
    """  From a single frame dict, extract bounding-box and keypoint geometry.
    kwargs : vec_kp kp_conf: p
    Returns:  bb_centers: (N, 2)
              bb_sizes:   (N, 2)
              keypoints:  (M, 2)  -- flattened over all people
              bboxes:     (N, 4)  -- [x1, y1, x2, y2]
    """
    #* vec_kp - key point vector length, may have 2 elements v = [x, y] or 3 v = [x, y, conf]
    #* There are 2 options to set KP vector length (i) pass the value directly by vec_kp
    #* or to pass a flag kp_conf
    # vec_kp = 3 if kwargs.get('kp_conf', True) else 2
    vec_kp = kwargs.get('vec_kp', 3 if kwargs.get('kp_conf', True) else 2)
    # vec_kp = 3 if kwargs.get('vec_kp', DEFAULT_VERSION) > 2.0 else 2

    bb_centers, bb_sizes, keypoints, bboxes = [], [], [], []

    # for bb in frame.get('bbs_list_of_keypoints', []):
    #     # * bounding box
    #     x1, y1, x2, y2 = bb[2], bb[3], bb[4], bb[5]
    for det in frame.get('detections_list', []):
        x1, y1, x2, y2 = det['bbox']
        w, h = (x2 - x1), (y2 - y1)
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2

        bb_sizes.append([w, h])
        bb_centers.append([cx, cy])
        bboxes.append([x1, y1, x2, y2])

        # kps = bb[6]  #* keypoints (13 points, aligned)
        kps = det['key_pts']
        for i in range(0, len(kps), vec_kp):
            x, y = kps[i], kps[i + 1]
            #* JS:0 is treated as missing
            if x == 0 and y == 0:
                continue
            keypoints.append([x, y])

    return (np.asarray(bb_centers, dtype=np.float32),
            np.asarray(bb_sizes  , dtype=np.float32),
            np.asarray(keypoints , dtype=np.float32),
            np.asarray(bboxes    , dtype=np.float32),)


def _bbox_union_area(bboxes):
    """ Compute exact union area of axis-aligned boxes clipped to [0, 1]."""
    if len(bboxes) == 0:
        return 0.0

    boxes = np.asarray(bboxes, dtype=np.float64).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, 1.0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, 1.0)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if len(boxes) == 0:
        return 0.0

    xs = np.unique(boxes[:, [0, 2]])
    area = 0.0
    for i in range(len(xs) - 1):
        x_l, x_r = xs[i], xs[i + 1]
        if x_r <= x_l:
            continue
        active = boxes[(boxes[:, 0] < x_r) & (boxes[:, 2] > x_l)]
        if len(active) == 0:
            continue

        ys = sorted((y1, y2) for _, y1, _, y2 in active)
        covered = 0.0
        cur_y1, cur_y2 = ys[0]
        for y1, y2 in ys[1:]:
            if y1 <= cur_y2:
                cur_y2 = max(cur_y2, y2)
            else:
                covered += cur_y2 - cur_y1
                cur_y1, cur_y2 = y1, y2
        covered += cur_y2 - cur_y1
        area += (x_r - x_l) * covered

    return float(area)


def _pairwise_iou_stats(bboxes):
    """Return mean/max pairwise IoU for a set of axis-aligned boxes."""
    if len(bboxes) < 2:
        return 0.0, 0.0

    vals = []
    boxes = np.asarray(bboxes, dtype=np.float64)
    for i in range(len(boxes)):
        x1a, y1a, x2a, y2a = boxes[i]
        area_a = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
        for j in range(i + 1, len(boxes)):
            x1b, y1b, x2b, y2b = boxes[j]
            area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
            ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
            ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = area_a + area_b - inter
            vals.append(0.0 if union <= 0 else inter / union)

    vals = np.asarray(vals, dtype=np.float32)
    return float(vals.mean()), float(vals.max())


def _overlap_ratio(bboxes):
    """Return redundant overlap ratio: (sum_area - union_area) / sum_area."""
    if len(bboxes) == 0:
        return 0.0

    boxes = np.asarray(bboxes, dtype=np.float64).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, 1.0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, 1.0)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if len(boxes) == 0:
        return 0.0

    sum_area = np.sum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    if sum_area <= 0:
        return 0.0

    union_area = _bbox_union_area(boxes)
    return float(max(0.0, sum_area - union_area) / sum_area)


# *****  Step 2: compute per-frame aggregate descriptors
def frame_aggregates(bb_centers, bb_sizes, keypoints, bboxes):
    """ Compute order-free, tracking-free aggregates for one frame.
        Returns a 1D feature vector.
    """
    feats = []

    #* bounding boxes
    if len(bb_centers) > 0:
        mean_center = bb_centers.mean(axis=0)
        var_center = bb_centers.var(axis=0)
        mean_size = bb_sizes.mean(axis=0)
        max_size = bb_sizes.max(axis=0)

        #* pairwise distances between centers
        if len(bb_centers) > 1:
            diffs = bb_centers[:, None, :] - bb_centers[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            #* take upper triangle without diagonal
            iu = np.triu_indices(len(bb_centers), k=1)
            pairwise = dists[iu]
            mean_pairwise = pairwise.mean()
            var_pairwise = pairwise.var()
        else:
            mean_pairwise = 0.0
            var_pairwise = 0.0
        union_coverage = _bbox_union_area(bboxes)
        mean_pairwise_iou, max_pairwise_iou = _pairwise_iou_stats(bboxes)
        overlap_ratio = _overlap_ratio(bboxes)
    else:
        mean_center = var_center = mean_size = max_size = np.zeros(2)
        mean_pairwise = var_pairwise = 0.0
        union_coverage = mean_pairwise_iou = max_pairwise_iou = overlap_ratio = 0.0

    feats.extend(mean_center)
    feats.extend(var_center)
    feats.extend(mean_size)
    feats.extend(max_size)
    feats.append(mean_pairwise)
    feats.append(var_pairwise)
    #* keypoints
    if len(keypoints) > 0:
        kp_mean = keypoints.mean(axis=0)
        kp_var = keypoints.var(axis=0)
    else:
        kp_mean = kp_var = np.zeros(2)
    feats.extend(kp_mean)
    feats.extend(kp_var)
    feats.append(union_coverage)
    feats.append(mean_pairwise_iou)
    feats.append(max_pairwise_iou)
    feats.append(overlap_ratio)

    return np.asarray(feats, dtype=np.float32)


# * Step 3: tracking-free motion between frames
def nearest_neighbor_motion(A, B):
    """ A, B: (N, 2) point sets from consecutive frames
        Returns mean and max nearest-neighbor distance.
    """
    if len(A) == 0 or len(B) == 0:
        return 0.0, 0.0

    dists = []
    for a in A:
        dist = np.linalg.norm(B - a, axis=1)
        dists.append(dist.min())

    dists = np.asarray(dists)
    return float(dists.mean()), float(dists.max())


def extract_person_geometry(frame, vec_kp=3, kp_min_conf=DEFAULT_KP_CONF_MIN):
    """Return per-detection geometry with confidence-gated keypoints."""
    persons = []
    kp_min_conf = float(kp_min_conf)

    for det in frame.get('detections_list', []):
        bbox = det.get('bbox', [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = (x2 - x1), (y2 - y1)
        if w <= 0 or h <= 0:
            continue

        kps = det.get('key_pts', [])
        kps_xy = np.full((N_Keypoints, 2), np.nan, dtype=np.float32)
        kps_conf = np.zeros((N_Keypoints,), dtype=np.float32)
        kps_valid = np.zeros((N_Keypoints,), dtype=bool)

        for kp_idx in range(N_Keypoints):
            base = kp_idx * vec_kp
            if base + 1 >= len(kps):
                break

            x = float(kps[base])
            y = float(kps[base + 1])
            conf = float(kps[base + 2]) if vec_kp >= 3 and base + 2 < len(kps) else 1.0
            visible = not (x == 0.0 and y == 0.0)
            valid = visible and conf >= kp_min_conf

            if valid:
                kps_xy[kp_idx] = [x, y]
            kps_conf[kp_idx] = conf
            kps_valid[kp_idx] = valid

        persons.append({
            'bbox': np.asarray([x1, y1, x2, y2], dtype=np.float32),
            'center': np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32),
            'size': np.asarray([w, h], dtype=np.float32),
            'keypoints': kps_xy,
            'kp_conf': kps_conf,
            'kp_valid': kps_valid,
        })

    return persons


def _safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.asarray(values, dtype=np.float32).mean())


def _safe_percentile(values, q):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _bbox_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def _bbox_size_log_delta(prev_person, curr_person, eps=1e-6):
    prev_size = np.maximum(prev_person['size'].astype(np.float64), eps)
    curr_size = np.maximum(curr_person['size'].astype(np.float64), eps)
    return float(np.max(np.abs(np.log(curr_size / prev_size))))


def _pairwise_center_distances(persons):
    if len(persons) < 2:
        return np.zeros((0,), dtype=np.float32)

    centers = np.stack([p['center'] for p in persons]).astype(np.float32)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(len(persons), k=1)
    return dists[iu].astype(np.float32)


def _pairwise_point_distances(points):
    if len(points) < 2:
        return np.zeros((0,), dtype=np.float32)

    coords = np.asarray(points, dtype=np.float32)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(len(coords), k=1)
    return dists[iu].astype(np.float32)


def _bbox_proximity_features(persons, close_dist_thresh, missing_pair_dist):
    dists = _pairwise_center_distances(persons)
    if len(dists) == 0:
        return float(missing_pair_dist), 0.0

    return float(dists.min()), float(np.mean(dists <= close_dist_thresh))


def _mean_valid_keypoints(person, kp_indices):
    idx = np.asarray(tuple(kp_indices), dtype=np.int64)
    valid = person['kp_valid'][idx]
    if not np.any(valid):
        return None
    return person['keypoints'][idx[valid]].mean(axis=0).astype(np.float32)


def _person_body_anchor(person):
    hips = np.asarray(HIP_KP, dtype=np.int64)
    if np.all(person['kp_valid'][hips]):
        return person['keypoints'][hips].mean(axis=0).astype(np.float32)

    shoulders = np.asarray(SHOULDER_KP, dtype=np.int64)
    if np.all(person['kp_valid'][shoulders]):
        return person['keypoints'][shoulders].mean(axis=0).astype(np.float32)

    torso_anchor = _mean_valid_keypoints(person, TORSO_KP)
    if torso_anchor is not None:
        return torso_anchor

    all_valid = person['kp_valid']
    if np.any(all_valid):
        return person['keypoints'][all_valid].mean(axis=0).astype(np.float32)

    return person['center'].astype(np.float32)


def _person_head_anchor(person):
    return _mean_valid_keypoints(person, HEAD_KP)


def _body_anchor_proximity_features(persons, missing_pair_dist):
    anchors = [_person_body_anchor(person) for person in persons]
    dists = _pairwise_point_distances(anchors)
    if len(dists) == 0:
        return float(missing_pair_dist)
    return float(dists.min())


def _head_anchor_proximity_features(persons, missing_pair_dist):
    anchors = []
    for person in persons:
        head_anchor = _person_head_anchor(person)
        if head_anchor is not None:
            anchors.append(head_anchor)
    dists = _pairwise_point_distances(anchors)
    if len(dists) == 0:
        return float(missing_pair_dist)
    return float(dists.min())


def _match_persons_by_bbox(prev_persons, curr_persons, **kwargs):
    if len(prev_persons) == 0 or len(curr_persons) == 0:
        return [], 0.0, 0.0, 0.0

    min_iou = float(kwargs.get('match_min_iou', DEFAULT_MATCH_MIN_IOU))
    max_center_dist = float(kwargs.get('match_max_center_dist', DEFAULT_MATCH_MAX_CENTER_DIST))
    max_size_ratio = float(kwargs.get('match_max_size_ratio', DEFAULT_MATCH_MAX_SIZE_RATIO))
    max_size_delta = float(np.log(max(max_size_ratio, 1.0 + 1e-6)))
    ambiguity_margin = float(kwargs.get('match_ambiguity_margin', DEFAULT_MATCH_AMBIGUITY_MARGIN))
    iou_weight = float(kwargs.get('match_iou_weight', 0.25))
    size_weight = float(kwargs.get('match_size_weight', 0.05))

    candidates = []
    ambiguous_sources = 0

    for i, prev_person in enumerate(prev_persons):
        row = []
        for j, curr_person in enumerate(curr_persons):
            center_dist = float(np.linalg.norm(curr_person['center'] - prev_person['center']))
            iou = _bbox_iou(prev_person['bbox'], curr_person['bbox'])
            size_delta = _bbox_size_log_delta(prev_person, curr_person)
            valid = (iou >= min_iou or center_dist <= max_center_dist) and size_delta <= max_size_delta
            if not valid:
                continue

            cost = center_dist + iou_weight * (1.0 - iou) + size_weight * size_delta
            row.append((cost, j, iou, center_dist, size_delta))

        if not row:
            continue

        row.sort(key=lambda item: item[0])
        is_ambiguous = len(row) > 1 and (row[1][0] - row[0][0]) < ambiguity_margin
        if is_ambiguous:
            ambiguous_sources += 1
            continue

        cost, j, iou, center_dist, size_delta = row[0]
        candidates.append((cost, i, j, iou, center_dist, size_delta))

    candidates.sort(key=lambda item: item[0])
    matches = []
    used_prev = set()
    used_curr = set()
    qualities = []

    for cost, i, j, iou, center_dist, size_delta in candidates:
        if i in used_prev or j in used_curr:
            continue

        used_prev.add(i)
        used_curr.add(j)
        quality = 1.0 / (1.0 + float(cost))
        matches.append((i, j, quality))
        qualities.append(quality)

    denom = max(len(prev_persons), 1)
    matched_ratio = len(matches) / denom
    ambiguous_ratio = ambiguous_sources / denom
    match_quality_mean = _safe_mean(qualities)

    return matches, matched_ratio, ambiguous_ratio, match_quality_mean


def _relative_keypoint_motion(prev_persons, curr_persons, matches, kp_indices):
    values = []

    for prev_idx, curr_idx, _quality in matches:
        prev_person = prev_persons[prev_idx]
        curr_person = curr_persons[curr_idx]
        body_delta = curr_person['center'] - prev_person['center']

        for kp_idx in kp_indices:
            if prev_person['kp_valid'][kp_idx] and curr_person['kp_valid'][kp_idx]:
                kp_delta = curr_person['keypoints'][kp_idx] - prev_person['keypoints'][kp_idx]
                values.append(float(np.linalg.norm(kp_delta - body_delta)))

    return values


def _keypoint_visibility_ratio(persons, kp_indices):
    denom = len(persons) * len(kp_indices)
    if denom == 0:
        return 0.0

    visible = 0
    for person in persons:
        visible += int(np.sum(person['kp_valid'][list(kp_indices)]))
    return float(visible / denom)


def _point_to_bbox_distance(point, bbox):
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def _wrist_to_other_bbox_features(persons, near_thresh, missing_pair_dist):
    distances = []

    for person_idx, person in enumerate(persons):
        for kp_idx in WRIST_KP:
            if not person['kp_valid'][kp_idx]:
                continue

            wrist = person['keypoints'][kp_idx]
            for other_idx, other in enumerate(persons):
                if other_idx == person_idx:
                    continue
                distances.append(_point_to_bbox_distance(wrist, other['bbox']))

    if len(distances) == 0:
        return float(missing_pair_dist), 0.0

    distances = np.asarray(distances, dtype=np.float32)
    return float(distances.min()), float(np.mean(distances <= near_thresh))


def _wrist_to_other_body_anchor_min(persons, missing_pair_dist):
    distances = []

    for person_idx, person in enumerate(persons):
        for kp_idx in WRIST_KP:
            if not person['kp_valid'][kp_idx]:
                continue

            wrist = person['keypoints'][kp_idx]
            for other_idx, other in enumerate(persons):
                if other_idx == person_idx:
                    continue
                other_anchor = _person_body_anchor(other)
                distances.append(float(np.linalg.norm(wrist - other_anchor)))

    if len(distances) == 0:
        return float(missing_pair_dist)

    return float(np.min(np.asarray(distances, dtype=np.float32)))


def _pairwise_body_anchor_distance_deltas(prev_persons, curr_persons, matches):
    if len(matches) < 2:
        return []

    matched = []
    for prev_idx, curr_idx, _quality in matches:
        matched.append((
            _person_body_anchor(prev_persons[prev_idx]),
            _person_body_anchor(curr_persons[curr_idx]),
        ))

    deltas = []
    for i in range(len(matched)):
        prev_a, curr_a = matched[i]
        for j in range(i + 1, len(matched)):
            prev_b, curr_b = matched[j]
            prev_dist = float(np.linalg.norm(prev_a - prev_b))
            curr_dist = float(np.linalg.norm(curr_a - curr_b))
            deltas.append(curr_dist - prev_dist)

    return deltas


def extended_limb_bbox_motion_features(prev_persons, curr_persons, curr_agg, **kwargs):
    """Extra robust features after the original 25 Erez motion dimensions."""
    close_dist_thresh = float(kwargs.get('close_pair_dist_thresh', DEFAULT_CLOSE_PAIR_DIST_THRESH))
    wrist_near_thresh = float(kwargs.get('wrist_near_bbox_thresh', DEFAULT_WRIST_NEAR_BBOX_THRESH))
    missing_pair_dist = float(kwargs.get('missing_pair_dist', DEFAULT_MISSING_PAIR_DIST))

    prev_min_dist, prev_close_ratio = _bbox_proximity_features(
        prev_persons,
        close_dist_thresh=close_dist_thresh,
        missing_pair_dist=missing_pair_dist,
    )
    curr_min_dist, curr_close_ratio = _bbox_proximity_features(
        curr_persons,
        close_dist_thresh=close_dist_thresh,
        missing_pair_dist=missing_pair_dist,
    )

    matches, matched_ratio, ambiguous_ratio, match_quality_mean = _match_persons_by_bbox(
        prev_persons,
        curr_persons,
        **kwargs,
    )

    wrist_motion = _relative_keypoint_motion(prev_persons, curr_persons, matches, WRIST_KP)
    elbow_motion = _relative_keypoint_motion(prev_persons, curr_persons, matches, ELBOW_KP)
    upper_motion = _relative_keypoint_motion(prev_persons, curr_persons, matches, UPPER_LIMB_KP)
    lower_motion = _relative_keypoint_motion(prev_persons, curr_persons, matches, LOWER_LIMB_KP)
    body_pair_distance_deltas = _pairwise_body_anchor_distance_deltas(prev_persons, curr_persons, matches)

    upper_energy = _safe_mean(upper_motion)
    upper_visibility = _keypoint_visibility_ratio(curr_persons, UPPER_LIMB_KP)
    lower_visibility = _keypoint_visibility_ratio(curr_persons, LOWER_LIMB_KP)
    prev_body_anchor_dist = _body_anchor_proximity_features(prev_persons, missing_pair_dist=missing_pair_dist)
    curr_body_anchor_dist = _body_anchor_proximity_features(curr_persons, missing_pair_dist=missing_pair_dist)
    curr_head_dist = _head_anchor_proximity_features(curr_persons, missing_pair_dist=missing_pair_dist)
    wrist_other_min_dist, wrist_near_other_ratio = _wrist_to_other_bbox_features(
        curr_persons,
        near_thresh=wrist_near_thresh,
        missing_pair_dist=missing_pair_dist,
    )
    wrist_other_body_anchor_min = _wrist_to_other_body_anchor_min(
        curr_persons,
        missing_pair_dist=missing_pair_dist,
    )

    max_pairwise_iou = float(curr_agg[16]) if len(curr_agg) > 16 else 0.0
    overlap_ratio = float(curr_agg[17]) if len(curr_agg) > 17 else 0.0
    overlap_score = max(max_pairwise_iou, overlap_ratio)

    return np.asarray([
        curr_min_dist,
        curr_min_dist - prev_min_dist,
        curr_close_ratio,
        curr_close_ratio - prev_close_ratio,
        _safe_mean(wrist_motion),
        _safe_percentile(wrist_motion, 90),
        _safe_mean(elbow_motion),
        upper_energy,
        upper_visibility,
        _safe_mean(lower_motion),
        lower_visibility,
        matched_ratio,
        ambiguous_ratio,
        wrist_other_min_dist,
        wrist_near_other_ratio,
        upper_energy * overlap_score,
        match_quality_mean,
        curr_body_anchor_dist,
        curr_body_anchor_dist - prev_body_anchor_dist,
        curr_head_dist,
        wrist_other_body_anchor_min,
        _safe_mean(np.abs(np.asarray(body_pair_distance_deltas, dtype=np.float32))),
        float(np.min(np.asarray(body_pair_distance_deltas, dtype=np.float32))) if body_pair_distance_deltas else 0.0,
    ], dtype=np.float32)


# ***** Steps 4 & 5 : reduce variable-length clips to fixed size
def _clip_pooling(motion_seq, mode='max', **kwargs):
    """ Reduce a (T-1) x C motion sequence to a single C-dimensional vector.
        Pooling is order-free over time.
    """
    if len(motion_seq) == 0:
        return None

    if mode == 'max':
        return motion_seq.max(axis=0)
    elif mode == 'mean':
        return motion_seq.mean(axis=0)
    elif mode == 'lse':
        alpha = kwargs.get('alpha', 5.0)
        return (1.0/alpha) * np.log(np.exp(alpha*motion_seq).sum(axis=0))
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


#* light smoothing: applies short sliding temporal window
def _temporal_conv_1d(motion_seq, kernel_size=3):
    """ Very simple temporal modeling using a fixed 1D convolution kernel.
        Returns a filtered motion sequence of the same shape.
    """
    if len(motion_seq) < kernel_size:
        return motion_seq

    pad = kernel_size // 2
    padded = np.pad(motion_seq, ((pad, pad), (0, 0)), mode='edge')

    out = []
    for t in range(len(motion_seq)):
        window = padded[t:t + kernel_size]
        out.append(window.mean(axis=0))

    return np.stack(out)


# --------------------------------------------------
# ***  Sanity Testing
# --------------------------------------------------+
import random

#* set the type of key points that will be used for the test
if DEFAULT_VERSION >= 2.0:
    KP_VEC_TST, KP_CONF_TST  = 3, True
else:
    KP_VEC_TST, KP_CONF_TST  = 2, False

def get_empty_frame():
    return {'f': 0, 't': 0.0, 'individual_events': [], 'group_events': [], 'detections_list': []}

def generate_random_frame(n_bb=10, version=DEFAULT_VERSION):
    """     Generate a synthetic frame in the unified dict format.  """

    kp_conf = True if float(version) >= 2.0 else False
    C_min = 0.3 #* minimal confidence for keypoint
    detections = []
    for _ in range(n_bb):
        # x1 = random.uniform(0.1, 0.6)
        # y1 = random.uniform(0.1, 0.6)
        # w  = random.uniform(0.1, 0.2)
        # h  = random.uniform(0.1, 0.2)
        x1, y1 = np.random.rand(2)*0.5 + 0.1
        w , h  = np.random.rand(2)*0.1 + 0.1
        x2 = min(x1 + w, 0.95)
        y2 = min(y1 + h, 0.95)

        key_pts = []
        for _ in range(N_Keypoints):
            key_pts +=[random.uniform(x1, x2), random.uniform(y1, y2)]
            if kp_conf:
                key_pts+= [random.uniform(C_min, .9)]
        # * (0) Individual annotation (ignored), (1) confidence
        detections += [ {'class': 0, 'conf': 1.0,
                         'bbox': [float(x1), float(y1), float(x2), float(y2)],
                         'key_pts': key_pts,}]

    return {'f': 0, 't': 0.0, 'group_events': [], 'detections_list': detections }

def shift_frame(frame, dx=0.1, dy=0.1, vec_kp=(3 if DEFAULT_VERSION >= 2.0  else 2)):
    """ Shift all bounding boxes and keypoints by (dx, dy).  """

    shifted = {'f':frame['f'], 't':frame['t'], 'group_events': frame['group_events'], 'detections_list':[]}

    for det in frame.get('detections_list', []):
        x1, y1, x2, y2 = det['bbox']
        new_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

        kps = det['key_pts']
        new_kps = []
        for i in range(0, len(kps), vec_kp):
            new_kps += [kps[i] + dx, kps[i + 1] + dy]
            if vec_kp == 3:
                new_kps += [kps[i+2]]
        shifted['detections_list'].append({'class': det['class'],
                                           'conf': det['conf'],
                                           'bbox': new_bbox,
                                           'key_pts': new_kps,})
    return shifted


def controlled_motion_test(dx=0.1, dy=0.1, eps=1e-5, **kwargs):

    frame_0 = generate_random_frame(n_bb=kwargs.get('n_bb',10))
    frame_1 = shift_frame(frame_0, dx=dx, dy=dy)

    #* --- geometry extraction ---
    bb0, _, _, boxes0 = extract_frame_geometry(frame_0, kp_conf=KP_CONF_TST)
    bb1, _, _, boxes1 = extract_frame_geometry(frame_1, kp_conf=KP_CONF_TST)
    #* --- aggregate check ---
    agg0 = frame_aggregates(bb0, np.zeros_like(bb0), np.zeros((0, 2)), boxes0)
    agg1 = frame_aggregates(bb1, np.zeros_like(bb1), np.zeros((0, 2)), boxes1)
    delta = agg1 - agg0
    #* --- indices depend on feature order:
    delta_mean_center = delta[0:2]    #* mean_center = feats[0:2]
    #*  pairwise distance stats
    mean_pairwise_idx = 8
    var_pairwise_idx = 9
    #* --- nearest neighbor motion ---
    nn_mean, nn_max = nearest_neighbor_motion(bb0, bb1)

    print("* Controlled_Motion_Test")
    print("Δmean_center:", delta_mean_center)
    print("pairwise mean delta:", delta[mean_pairwise_idx])
    print("pairwise var delta :", delta[var_pairwise_idx])
    print("NN mean motion     :", nn_mean)
    print("NN max motion      :", nn_max)

    #* --- assertions ---
    expected_mag = np.sqrt(dx*dx + dy*dy)
    assert np.allclose(delta_mean_center, [dx, dy], atol=eps)
    assert abs(delta[mean_pairwise_idx]) < eps, "pairwise mean should stay constant"
    assert abs(delta[var_pairwise_idx] ) < eps, "pairwise var should stay constant"
    assert abs(nn_max - expected_mag) < eps
    assert nn_mean <= expected_mag + eps

    print("✔ Controlled motion test PASSED\n")

def crowd_compression_test(scl=0.5, **kwargs): #72
    """ Sanity check for crowd compression (no global translation).
        People move toward their centroid, reducing pairwise distances.
    """

    frame_0 = generate_random_frame(n_bb=kwargs.get('n_bb', 10))
    #* extract centers
    # bb0, _, _ = extract_frame_geometry(frame_0)
    bb0, bb_sizes0, kp0, boxes0 = extract_frame_geometry(frame_0)
    centroid = bb0.mean(axis=0)

    #* build compressed frame
    compressed = get_empty_frame()
    compressed['f'] =  frame_0.get('f', 0)
    compressed['t'] =  frame_0.get('t', 0)

    for det in frame_0['detections_list']:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2

        #* move center toward centroid
        new_cx = centroid[0] + scl*(cx - centroid[0])
        new_cy = centroid[1] + scl*(cy - centroid[1])
        w = x2 - x1
        h = y2 - y1

        nx1, ny1 = (new_cx - w/2), (new_cy - h/2)
        nx2, ny2 = (new_cx + w/2), (new_cy + h/2)

        kps_shift = []
        kps = det.get('key_pts', [])
        for i in range(0, len(kps), KP_VEC_TST):
            kx, ky = kps[i], kps[i + 1]
            dkx = kx - cx
            dky = ky - cy
            kps_shift.extend([new_cx + scl*dkx, new_cy + scl*dky,])
            if KP_CONF_TST:
                kps_shift += [kps[i+1]]

        compressed['detections_list']+= [{'class': det['class'], 'conf': det['conf'],
                                          'bbox': [nx1, ny1, nx2, ny2],
                                          'key_pts': kps_shift,}]

    bb1, bb_sizes1, kp1, boxes1 = extract_frame_geometry(compressed, vec_kp=KP_VEC_TST)

    agg0 = frame_aggregates(bb0, bb_sizes0, kp0, boxes0)
    agg1 = frame_aggregates(bb1, bb_sizes1, kp1, boxes1)

    mean_pairwise_idx = 8      # var_pairwise_idx = 9
    delta_mean_center = agg1[0:2] - agg0[0:2]

    print(f"* Crowd Compression Test\n"
          f"Δmean_center:{delta_mean_center}\n"
          f"pairwise mean before:{agg0[mean_pairwise_idx]}\n"
          f"pairwise mean after :{agg1[mean_pairwise_idx]}\n")

    #* --- assertions ---
    assert np.linalg.norm(delta_mean_center) < kwargs.get('eps', 1e-5)
    assert agg1[mean_pairwise_idx] < agg0[mean_pairwise_idx]

    print("✔ Crowd compression test PASSED\n")


def generate_static_json(n_frm=100, n_bb=10): #55
    """  Generate a JSON-like dict where all frames are identical.
         Useful for zero-motion sanity checks.
    """

    frame_template = generate_random_frame(n_bb=n_bb)
    frames = []
    for i in range(n_frm):
        frame = frame_template.copy()
        frame['f'] = 5*i
        frame['t'] = i/3  #* 0.33*i
        frames.append(frame)

    return {'video_file':"static_test", 'fps':15, 'sampling': 5, 'version': '2.0',
            'frames': frames}

def test_motion_sequence(tst_json:dict, eps= 1e-5):

    # info = tst_json['header']
    frames = tst_json['frames']
    j_version = float(tst_json['header']['version'])
    motion_seq = extract_motion_features(frames, j_version=j_version)

    # * check (1): shape is stable
    print(f"Number of frames: {len(frames)}\nMotion Shape: {motion_seq.shape}")
    print(f"Correct shape:",  len(frames) - 1 ==  motion_seq.shape[0])
    # * check (2): calculation  consistency
    ms2 = extract_motion_features(frames, j_version=j_version)
    assert np.allclose(motion_seq, ms2)
    # * check (3): Static (Zero motion)
    ms_0 = extract_motion_features(generate_static_json()['frames'])
    print(f"\n* Static motion tensor: mean = {ms_0.mean()}, max = {abs(ms_0.max())}, sum = {ms_0.sum()}\n")
    assert ms_0.max() < eps

    controlled_motion_test()
    crowd_compression_test()


if __name__ == '__main__':
    try:
        from .json_utils import load_json_data
        from .my_local_utils import print_color
    except ImportError:
        from json_utils import load_json_data
        from my_local_utils import print_color

    # json_example = "/mnt/local-data/Projects/Wesmart/data/usual_jsons_from_events/event_18.json"
    # json_example = "data/json_data/full_ann_w_keys/new_21_1_keypoints.json"
    # json_newfrmt = "data/json_data/jsons_nf/cam3_5_4.json"
    json_example = "data/json_files/tst_conv/try_03 (tms)/Russian_Road_Rage- Micky_Mouse_&_Sponge_Bob.json"
    # static_json_ = "data/json_data/static_clip.json"

    test_motion_sequence(load_json_data(json_example), eps=1e-5)
    print_color("✔✔✔ Old Format OK !\n", 'g')
    test_motion_sequence(load_json_data(json_example, j_type='2'), eps=1e-5)
    print_color("✔✔✔ New Format OK !\n", 'g')

    pass

#480(,8,4)-> 417(,,3)
