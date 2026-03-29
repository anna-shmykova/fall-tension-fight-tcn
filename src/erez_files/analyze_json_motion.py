"""     Extract order-free motion features from JSON frame sequences.
    This file converts per-frame detections/keypoints into a clip-level motion
    representation. The main entry point is `extract_motion_features(...)`, which
    builds one 21-dim feature vector for each pair of consecutive frames.
    Feature order:
    1.  d_mean_center_x      : change in mean bbox center x; global horizontal crowd motion
    2.  d_mean_center_y      : change in mean bbox center y; global vertical crowd motion
    3.  d_var_center_x       : change in bbox-center variance on x; horizontal spread/compression
    4.  d_var_center_y       : change in bbox-center variance on y; vertical spread/compression
    5.  d_mean_size_w        : change in mean bbox width; average scale/depth change
    6.  d_mean_size_h        : change in mean bbox height; average scale/depth change
    7.  d_max_size_w         : change in largest bbox width; strongest scale change
    8.  d_max_size_h         : change in largest bbox height; strongest scale change
    9.  d_mean_pairwise      : change in mean pairwise bbox-center distance; crowd density cue
    10. d_var_pairwise       : change in variance of pairwise distances; spacing heterogeneity
    11. d_kp_mean_x          : change in mean keypoint x; global articulated horizontal motion
    12. d_kp_mean_y          : change in mean keypoint y; global articulated vertical motion
    13. d_kp_var_x           : change in keypoint variance on x; pose spread horizontally
    14. d_kp_var_y           : change in keypoint variance on y; pose spread vertically
    15. bb_nn_mean           : mean nearest-neighbor bbox-center motion; average object motion
    16. bb_nn_max            : max nearest-neighbor bbox-center motion; strongest mover
    17. kp_nn_mean           : mean nearest-neighbor keypoint motion; average articulated motion
    18. kp_nn_max            : max nearest-neighbor keypoint motion; strongest articulated motion
    #*  added later (14/03/26)
    19. d_union_coverage     : change in union area covered by all bboxes
    20. d_mean_pairwise_iou  : change in mean pairwise bbox IoU
    21. d_max_pairwise_iou   : change in max pairwise bbox IoU
"""

import numpy as np

N_Keypoints = 17
DEFAULT_VERSION = 2.0

# --------------------------------------------------
# * public fucntion, to be used by other units
# --------------------------------------------------

def extract_motion_features(frames, j_version:float=DEFAULT_VERSION): # **kwargs):
    """    Converts a list of frames into a T x C motion feature sequence.
         former  compute_motion_sequence"""
    frame_feats = []
    raw_points = []

    kp_conf = True if j_version >= 2.0 else False

    for frm in frames:
        if len(frm['detections_list']) > 0:
            pass
            # print_color(frm)
        bb_centers, bb_sizes, keypoints, bboxes = extract_frame_geometry(frm, kp_conf=kp_conf)
        agg = frame_aggregates(bb_centers, bb_sizes, keypoints, bboxes)
        frame_feats.append(agg)
        raw_points.append((bb_centers, keypoints))

    frame_feats = np.stack(frame_feats)  #* TxC

    motion_feats = []
    for t in range(len(frames) - 1):
        delta_agg = frame_feats[t + 1] - frame_feats[t]

        bb_c_t, kp_t = raw_points[t]
        bb_c_tp1, kp_tp1 = raw_points[t + 1]

        bb_mean, bb_max = nearest_neighbor_motion(bb_c_t, bb_c_tp1)
        kp_mean, kp_max = nearest_neighbor_motion(kp_t, kp_tp1)

        motion_vec = np.concatenate([delta_agg,
                                     np.array([bb_mean, bb_max, kp_mean, kp_max], dtype=np.float32),])
        motion_feats.append(motion_vec)

    return np.stack(motion_feats)  # (T-1) x C'


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
    else:
        mean_center = var_center = mean_size = max_size = np.zeros(2)
        mean_pairwise = var_pairwise = 0.0
        union_coverage = mean_pairwise_iou = max_pairwise_iou = 0.0

    feats.extend(mean_center)
    feats.extend(var_center)
    feats.extend(mean_size)
    feats.extend(max_size)
    feats.append(mean_pairwise)
    feats.append(var_pairwise)
    feats.append(union_coverage)
    feats.append(mean_pairwise_iou)
    feats.append(max_pairwise_iou)

    #* keypoints
    if len(keypoints) > 0:
        kp_mean = keypoints.mean(axis=0)
        kp_var = keypoints.var(axis=0)
    else:
        kp_mean = kp_var = np.zeros(2)
    feats.extend(kp_mean)
    feats.extend(kp_var)

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
    json_example = "data/json_data/full_ann_w_keys/new_21_1_keypoints.json"
    json_newfrmt = "data/json_data/jsons_nf/cam3_5_4.json"
    static_json_ = "/mnt/local-data/Python/Projects/weSmart/data/json_data/static_clip.json"

    test_motion_sequence(load_json_data(json_example), eps=1e-5)
    print_color("✔✔✔ Old Format OK !\n", 'g')
    test_motion_sequence(load_json_data(json_newfrmt, j_type='2'), eps=1e-5)
    print_color("✔✔✔ New Format OK !\n", 'g')

    pass

#480(,8,4)-> 417(,,3)
