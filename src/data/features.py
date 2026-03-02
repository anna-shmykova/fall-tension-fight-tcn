import numpy as np

def norm_kps_xy_flat_with_mask(kps_xy_flat, cx, cy, w, h, eps=1e-6):
    inv_w = 1.0 / (w + eps)
    inv_h = 1.0 / (h + eps)

    kps_norm = []
    joint_mask = []

    for j in range(0, len(kps_xy_flat), 3):
        x = kps_xy_flat[j]
        y = kps_xy_flat[j+1]

        visible = not (x == 0 and y == 0)
        #joint_mask.append(1 if visible else 0)

        #if visible:
        kps_norm.extend([(x - cx) * inv_w, (y - cy) * inv_h, 1.00 if visible else 0.00])
        #else:
            # keep zeros for missing (but mask tells the truth)
            #kps_norm.extend([0.0, 0.0])

    return kps_norm#, joint_mask

def frame_to_vector(frame, K, num_pers_features=40, cfg=None):
    """
    frame: dict for one time step
    K: max number of boxes per frame
    """
    detections = frame['detection_list']  # <-- adapt this key name

    # sort boxes by area desc
    dets_sorted = sorted(
        detections,
        key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]),
        reverse=True,
    )

    boxes = []
    keypoints = []
    for d in dets_sorted[:K]:
        x1, y1, x2, y2 = d['bbox']
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2 )/ 2
        cy = (y1 + y2 )/ 2

        # normalize
        '''cx /= img_w
        cy /= img_h
        w  /= img_w
        h  /= img_h'''
        kps = d['key_points'][15:]
        # choose the correct normalizer for your keypoint format:
        kps_norm = norm_kps_xy_flat_with_mask(kps, cx, cy, w, h)          # xy only + mask
        #print(len(kps_norm))
        # kps_norm = norm_kps_xyc_flat(kps, cx, cy, w, h)       # if x,y,conf triplets
        boxes.extend([cx, cy, w, h]+ kps_norm)

    # pad if fewer than K boxes
    if len(dets_sorted) < K:
        boxes.extend([0.0] * num_pers_features * (K - len(dets_sorted)))

    n_people = len(detections)
    n_people_norm = len(detections)/25

    x_t = boxes + [float(n_people_norm)]
    return np.array(x_t, dtype=np.float32)