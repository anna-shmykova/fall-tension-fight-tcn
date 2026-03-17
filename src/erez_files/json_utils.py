"""  json_utils
Utilities for loading JSON annotation files into a unified,
firm internal structure.

Public API:
    load_json_data(file: str | Path, format: str = 'type1') -> dict

Returned structure:
    json_data = {'header': {'video_file': ...,
                            'fps': ...,
                            'sampling':
                             'version'...},
                 'frames': [{'f': ...,
                            't': ...,
                            'group_events': ...,
                            'detections_list': [{'class': ...,
                                                 'conf': ...,
                                                 'bbox': [...],
                                                 'key_pts': [17 pts x 3]},
                                                ...]
                            }, ...]
                 }
"""

import json
from pathlib import Path
#* local imports
try:
    from .my_local_utils import print_color, _make_unique_dir
except ImportError:
    from my_local_utils import print_color, _make_unique_dir

# --------------------------------------------------
# * Public loader
# --------------------------------------------------

def load_json_data(file:str|Path, j_type='type_1'):
    """ Load JSON file and normalize into firm internal structure.
        :param str|Path file : Path to JSON file
        :param str j_type: Loader type ('type1', 'type2', ...)
        :returns dict json_data: Unified structured representation
    """
    file = Path(file)

    try:
        if j_type == 'type_1':
            return load_type_1(file)
        elif j_type in ['type_2', '2', 2]:
            return load_type_2(file)
        else:
            print_color(f"Warning: Unknown Json format: {j_type}", 'y')
            return None
    except :
        raise ValueError(f"Error: Failed to load {file.name};  format: {j_type}")



# * Type 1 loader (current format)
def load_type_1(file: Path):
    """   Load JSON type 1 (old) files  """
    # * old format had 2 variations for detection_list
    DET_LIST = 'bbs_list_of_keypoints'  # 'list_of_bbs_keypoints'

    with open(file, 'r') as f:
        raw = json.load(f)

    header = {'video_file': raw.get('video'),
              'fps': raw.get('fps'),
              'sampling': raw.get('step'),
              'version': '1.0'}

    frames_out = []
    for frame in raw.get('frames', []):
        detections = []

        for bb in frame.get(DET_LIST, []):
            det = {'class': bb[0],'conf': bb[1],
                   'bbox': bb[2:6], 'key_pts': bb[6]}
            detections.append(det)

        frame_struct = {'f': frame.get('f'),
                        't': frame.get('t'),
                        'group_events': frame.get('group_events', []),
                        'detections_list': detections,
                        }
        frames_out.append(frame_struct)

    return {'header': header, 'frames': frames_out}


#* Type 2 loader (future format placeholder)
def load_type_2(file: Path):
    """ Load JSON files in the new format (type_2).
    updates for type 2:
    - 'detection_list' instead of 'list_of_bbs_keypoints'
    - Each detection is already a dict
    - 'key_points' contains 17 keypoints * 3 values (x, y, conf) flattened as [x0, y0, c0, x1, y1, c1, ..., x16, y16, c16]
    """
    with open(file, 'r') as f:
        raw = json.load(f)

    header = {'video_file': raw['video'], 'fps': raw['fps'], 'sampling': raw['step'], 'version': '2.0'}

    frames_out = []
    for frame in raw.get('frames', []):
        detections = []

        for det in frame.get('detection_list', []):
            key_pts_raw = det.get('key_points',[])
            #* Ensure list length consistency (should be 51 = 17 * 3)
            if key_pts_raw and len(key_pts_raw) % 3 != 0:
                print_color(f"[WARN] key_points length not divisible by 3 in frame {frame.get('f')}",'y')


            detections.append({'class': det['class'],
                               'conf': det['conf'],
                               'bbox': det.get('bbox', []),
                               'key_pts': key_pts_raw, #* keep flattened format for compatibility with other code
                               })

        frame_struct = {'f': frame.get('f'),
                        't': frame.get('t'),
                        'group_events': frame.get('group_events', []),
                        'detections_list': detections,
                        }
        frames_out += [frame_struct]

    return {'header': header, 'frames': frames_out}

#130(,1,2) -> 88(,,2)


FRAME_H, FRAME_W = 1080, 1920
def json_to_box_frames( json_path, out_root, H:int=FRAME_H, W:int=FRAME_W, **kwargs):
    """  Convert detector JSON into a sequence of box-frame images. #176- 146
    Each frame is rendered on an H×W monochrome canvas using the
    normalized TL/BR bounding boxes in data["frames"][i]["bbs"].
    :param json_path: Path to detector JSON.
    :param out_root: Root directory for the output clip folder.
    :param H, W: Output frame height and width in pixels.
    Returns:  (clip_name, clip_info)
    """

    json_path = Path(json_path)
    out_root = Path(out_root)

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    title = data.get("video", "")

    # If no frames, return empty info
    if not frames:
        return None, {'Duration': 0.0, 'Frames': 0, 'Events': 0, 'T_evn': 0.0, 'stat': {},}

    #* Derive clip name if not given (replace '.' to be safe in folder names)
    if kwargs.get("clip_name", None) is not None:
        clip_name = kwargs.get("clip_name", None)
    elif  kwargs.get("use_file_name", False):
        # Use the JSON filename (no extension), sanitized
        clip_name = json_path.stem.replace(".", "_")
    elif isinstance(title, str) and len(title) > 0:
        clip_name = Path(title).stem
    else:
        clip_name = json_path.stem.replace(".", "_")

    #* Create unique directory for this clip
    clip_dir, clip_name = _make_unique_dir(out_root, clip_name, no_space=True)

    #* collect logs lines
    event_lines = []
    #* Collect clip-level stats
    times_s: list[float] = []
    event_flags: list[bool] = []
    stat: dict[int, int] = {}

    for idx, fr in enumerate(frames, start=1):
        # White background
        img = np.full((H, W), 255, dtype=np.uint8)
        bb_ls = fr.get('bbs',[]) or fr.get('bbs_list_of_keypoints', [])
        for bb in bb_ls:
            if len(bb) < 6:
                continue
            cls_id, cls_conf, tl_x, tl_y, br_x, br_y = bb[0:6]

            #* Optional vertical flip if y is measured from bottom
            if kwargs.get('y_from_bottom',):
                tl_y = 1.0 - tl_y
                br_y = 1.0 - br_y
            #* Convert normalized coords to pixel coords
            x1 = int(tl_x*W)
            y1 = int(tl_y*H)
            x2 = int(br_x*W)
            y2 = int(br_y*H)
            # Clip to image bounds
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Intensity mapping (for now: black box outlines).
            intensity = 0
            cv2.rectangle(img, (x1, y1), (x2, y2), int(intensity), thickness=1)

            if kwargs.get("draw_center", False):
                bb_w = x2 - x1
                bb_h = y2 - y1
                if bb_w > 0 and bb_h > 0:
                    cx = x1 + bb_w//2
                    cy = y1 + bb_h//2
                    diameter = max(5, int(CENTER_SIZE*min(bb_w, bb_h)))
                    radius = max(1, diameter // 2)
                    cv2.circle(img, (cx, cy), radius, int(intensity), thickness=-1)

        frame_num = fr.get('f', idx)
        # time_ms = int(fr.get("t", 0) * 1000)
        t = float(fr.get('t', 0.0))
        time_ms = int(t*1000)
        times_s += [t]

        #* Extract group events for this frame
        raw_events = fr.get('event_grouped') or fr.get('group_events') or []
        pairs = []
        is_event = False
        for ev in raw_events:
            if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                flag = int(ev[0])
                conf = ev[1]
            elif isinstance(ev, (int, float)):
                flag = int(ev)
                conf = None
            else:
                continue

            is_event = True
            #* Per-flag counts (by frame occurrences)
            stat[flag] = stat.get(flag, 0) + 1

            conf_str = "NA" if conf is None else f"{float(conf):.4f}"
            pairs.append(f"{flag}:{conf_str}")

        event_flags += [is_event]
        if pairs:
            events_str = "[" + "; ".join(pairs) + "]"
            event_lines.append(f"{time_ms},{frame_num},{events_str}")

        out_name = clip_dir/f"frm_{frame_num:05d}_{time_ms:08d}.jpg"
        cv2.imwrite(str(out_name), img)

    if event_lines:  #* Write event log if any
        _save_log(event_lines, clip_dir / f"{clip_name}_events.log")


    #*** crate clip_info ***
    if times_s:
        duration = max(times_s) - min(times_s)
        dt = duration/(len(times_s) - 1) if duration > 0 else 0.0
        t_event = sum(event_flags)*dt
    else:
        duration = 0.0
        t_event = 0.0

    #* Count contiguous event segments (any event flag)
    events_count, prev = 0, False
    for cur in event_flags:
        if cur and not prev:
            events_count += 1
        prev = cur

    clip_info = {'Duration': max(times_s),  # float(total_time),
                 'Frames': len(frames),
                 'Events': events_count,
                 'T_evn': float(t_event),
                 'stat': stat,}

    return clip_name, clip_info

