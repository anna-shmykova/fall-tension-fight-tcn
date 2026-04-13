"""
    * The unit converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default tags.

    * The unit is wrapped into CLI in convert_json.py via process_video function.

    * JSON structure:
           {'video' : "...",           # video/file name
            'fps': 25.0,
            'step': 5,                # step size
            'frames':[
                    {'f': 15,               #* frame index
                     't': 0.6,              #* time from start in seconds
                     'individual_events': [],
                     'group_events': [2, 3],
                     'detection_list':[     #* list of YOLO detections (human)
                              {'class':     #* normal,  fall ...
                               'conf':      #* confidence of 'class'
                               'bbox':      #* [x1, y1, x2, y2] in normalized unit
                               'key_points': [x1, y1, c1, x2, y2, c2, ....  x17, y17, c17]
                                            #* flatten vector of 17 key points (x,y) and it's confidence (c)
                                       ]
                    },]
             `      ...
"""

import csv, cv2, json, re, shutil, statistics, subprocess, torch
from pathlib import Path
from ultralytics import YOLO
#* import from my utils
from my_local_utils import get_unique_name, print_color

#* Events Thresholds  -------------------------------------------------------------------
# SINGLE_THRESHOLDS = { 0: 0.5,   #* normal
#                       1: 0.9,   #* abnormal
#                       2: 0.7,   #* fall
#                       5: 0.9,}  #* kick
# GROUPED_THRESHOLDS = {3: 0.7,   #* tension
#                       4: 0.7,}  #* violence
DETECTION_THRESHOLD = 0.5
DEFAULT_SAMPELING = 5
# STEP = 5

#* Events flags
TAG_NO_EVENT = 0
TAG_FALL     = 2
TAG_TENSION  = 3
TAG_FIGHT    = 4

MODELS_DIR = 'models/'
DEFAULT_YOLO = MODELS_DIR + "yolo26x-pose.pt"
#* "yolo26x-pose.pt" / "yolo11x-pose.pt" / "yolov8s.pt"
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv', '.mts', '.m2ts'}


def parse_rational(value):
    """Convert ffprobe rational strings like '30000/1001' to float."""
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        num, den = value.split("/", 1)
        den = float(den)
        if den == 0:
            return None
        return float(num) / den
    except (TypeError, ValueError):
        return None


def summarize_frame_times(frame_times: list[float]):
    """Build timing summary from true per-frame timestamps.

    Works with non-integer fps such as 11.999 and with videos whose cadence
    changes inside the file. All values are derived from frame timestamps,
    not from container/header fps.
    """
    if not frame_times:
        return {
            "effective_fps": None,
            "min_frame_delta": None,
            "max_frame_delta": None,
            "per_second_frame_counts": {},
            "fps_segments": [],
        }

    nondecreasing_times = [frame_times[0]]
    for t in frame_times[1:]:
        if t >= nondecreasing_times[-1]:
            nondecreasing_times.append(t)

    deltas = [curr - prev for prev, curr in zip(nondecreasing_times, nondecreasing_times[1:])]
    duration = max(nondecreasing_times[-1] - nondecreasing_times[0], 1e-9)
    effective_fps = len(nondecreasing_times) / duration if len(nondecreasing_times) > 1 else None

    per_second_counts = {}
    for t in nondecreasing_times:
        sec = int(t)
        per_second_counts[str(sec)] = per_second_counts.get(str(sec), 0) + 1

    fps_segments = []
    items = sorted((int(k), v) for k, v in per_second_counts.items())
    if items:
        seg_start, prev_sec, cur_count = items[0][0], items[0][0], items[0][1]
        for sec, count in items[1:]:
            if sec != prev_sec + 1 or count != cur_count:
                fps_segments.append({
                    "start_sec": seg_start,
                    "end_sec": prev_sec + 1,
                    "frames_per_sec": cur_count,
                })
                seg_start, cur_count = sec, count
            prev_sec = sec
        fps_segments.append({
            "start_sec": seg_start,
            "end_sec": prev_sec + 1,
            "frames_per_sec": cur_count,
        })

    return {
        "effective_fps": effective_fps,
        "min_frame_delta": min(deltas) if deltas else None,
        "max_frame_delta": max(deltas) if deltas else None,
        "per_second_frame_counts": per_second_counts,
        "fps_segments": fps_segments,
    }


def probe_video_timing_ffprobe(video_path: Path):
    """Read stream timing metadata and per-frame timestamps with ffprobe."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return None

    stream_cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,time_base,nb_frames,duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        stream_proc = subprocess.run(stream_cmd, check=True, capture_output=True, text=True)
        stream_payload = json.loads(stream_proc.stdout or "{}")
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"[WARN] ffprobe stream metadata failed for {video_path}: {exc}")
        return None

    frame_cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    frame_times = []
    try:
        with subprocess.Popen(frame_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    frame_times.append(float(line))
                except ValueError:
                    continue
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            return_code = proc.wait()
        if return_code != 0:
            print(f"[WARN] ffprobe frame timestamps failed for {video_path}: {stderr.strip()}")
            return None
    except OSError as exc:
        print(f"[WARN] Could not launch ffprobe for {video_path}: {exc}")
        return None

    stream_info = (stream_payload.get("streams") or [{}])[0]
    avg_frame_rate = parse_rational(stream_info.get("avg_frame_rate"))
    nominal_frame_rate = parse_rational(stream_info.get("r_frame_rate"))
    deltas = [curr - prev for prev, curr in zip(frame_times, frame_times[1:]) if curr >= prev]
    median_delta = statistics.median(deltas) if deltas else None
    jitter_tolerance = max(0.002, 0.10 * median_delta) if median_delta else None
    variable_timing = bool(
        median_delta
        and jitter_tolerance
        and any(abs(delta - median_delta) > jitter_tolerance for delta in deltas)
    )
    return {
        "frame_times": frame_times,
        "frame_timestamp_count": len(frame_times),
        "avg_frame_rate": avg_frame_rate,
        "nominal_frame_rate": nominal_frame_rate,
        "stream_time_base": stream_info.get("time_base"),
        "stream_duration_sec": float(stream_info["duration"]) if stream_info.get("duration") not in (None, "N/A") else None,
        "stream_nb_frames": int(stream_info["nb_frames"]) if stream_info.get("nb_frames", "").isdigit() else None,
        "first_frame_time_sec": frame_times[0] if frame_times else None,
        "last_frame_time_sec": frame_times[-1] if frame_times else None,
        "median_frame_delta": median_delta,
        "variable_frame_timing": variable_timing,
    }


def is_video_file(path: Path | str) -> bool:
    """Return True when the path looks like a supported video file."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES

def parse_sec_str(t):
    """ Convert 'HH:MM:SS' or 'MM:SS' or 'SS' to seconds"""
    if t is None:
        return None
    t = t.strip()
    if t == '':
        return None

    parts = t.split(':')
    parts = [int(p) for p in parts]
    if   len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:  #* seconds only
        h,m = 0, 0
        s = parts[0]
    return h * 3600 + m * 60 + s


def parse_interval(s):
    """ Parses 'start-end' where start or end can be empty.
    Examples:
        '00:01:00-00:02:00'
        '00:03:00-'
        '-00:00:30'
        '00:01:00'   -> (start, None)
    Returns (start_sec, end_sec) where any can be None.
    """
    if s is None:
        return None, None

    s = s.strip()
    if s == "":
        # Completely empty → ignore this interval later
        return None, None

    # No '-' → interpret as single time = from that time until end
    if "-" not in s:
        start = parse_sec_str(s)
        end = None
        return start, end

    start_str, end_str = s.split("-", 1)

    start = parse_sec_str(start_str) if start_str.strip() != '' else None
    end   = parse_sec_str(end_str)   if end_str.strip()   != '' else None
    return start, end


def parse_annotation_file(ann_file):
    """ Parse an annotation text file into event intervals by flag.
        Expected row format:
        start_time,\t   end_time,\t event_flag
        '#' Lines starting with '#' are ignored.
    """
    event_intervals = {TAG_FALL: [], TAG_TENSION: [], TAG_FIGHT: []}
    ann_path     = Path(ann_file)

    with ann_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",\t")]
            if len(parts) != 3:
                parts = [part.strip() for part in line.split("\t")]

            if len(parts) != 3:
                print(f"[WARN] Invalid annotation row at {ann_path}:{line_no}: {raw_line.rstrip()}")
                continue

            start_str, end_str, flag_str = parts
            try:
                event_flag = int(flag_str)
            except ValueError:
                print(f"[WARN] Invalid event flag at {ann_path}:{line_no}: {flag_str}")
                continue

            if event_flag not in event_intervals:
                continue

            event_intervals[event_flag].append((parse_sec_str(start_str), parse_sec_str(end_str)))

    return event_intervals


def parse_frame_label_csv(ann_file):
    """Parse same-stem CSV annotations with one frame label per row.

    UBI_FIGHTS stores a binary label per original video frame. The parser accepts:
    - a single numeric value per row, or
    - a row with multiple columns where the first numeric field is the label.
    Non-numeric header rows are skipped.
    """
    frame_labels = []
    ann_path = Path(ann_file)

    with ann_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row_no, row in enumerate(reader, start=1):
            cells = [cell.strip() for cell in row if cell.strip() != ""]
            if not cells:
                continue

            parsed_label = None
            for cell in cells:
                try:
                    parsed_label = int(float(cell))
                    break
                except ValueError:
                    continue

            if parsed_label is None:
                if row_no == 1:
                    continue
                print(f"[WARN] Invalid CSV annotation row at {ann_path}:{row_no}: {row}")
                continue

            frame_labels.append(1 if parsed_label > 0 else 0)

    return frame_labels


def load_annotation_data(ann_file):
    """Load either interval annotations or per-frame CSV labels."""
    ann_path = Path(ann_file)
    if ann_path.suffix.lower() == ".csv":
        frame_labels = parse_frame_label_csv(ann_path)
        return {
            'intervals': {TAG_FALL: [], TAG_TENSION: [], TAG_FIGHT: []},
            'frame_labels': frame_labels,
            'source_type': 'frame_csv',
            'source_path': str(ann_path),
        }

    return {
        'intervals': parse_annotation_file(ann_path),
        'frame_labels': None,
        'source_type': 'interval_text',
        'source_path': str(ann_path),
    }

#*  --- Main function, to be used from other unit ---------------------------------------
def process_video(input_path: Path|str,
                  output_path:Path|str=None,
                  sample_rate=DEFAULT_SAMPELING,
                  conf_thresh=DETECTION_THRESHOLD,
                  ann_file=None,
                  default_group_tag=None,
                  tension_intervals=None, fight_intervals=None, fall_intervals=None,
                  **kwargs):
    """ Converts one or more videos into structured JSON annotations.
    A YOLO detector is used to perform per-frame person detection and
    extract bounding boxes and keypoints. Optional time-based labeling
    can be applied at frame level using predefined intervals or default
    tags. Process video file/dir of files.

    The function always writes JSON for decodable videos and also writes
    one dataset_manifest.csv file summarizing per-video QA results.
    """

    def find_annotation_file(video_path):
        """Find a sibling annotation file matching the video stem."""
        video_path = Path(video_path)
        candidates = (video_path.with_suffix('.csv'),
                      video_path.with_suffix('.txt'),
                      video_path.with_suffix('.ann'),
                      video_path.with_suffix(''))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def find_txt_annotation_file(video_path):
        """Find a sibling .txt annotation file matching the video stem."""
        candidate = Path(video_path).with_suffix('.txt')
        return candidate if candidate.is_file() else None

    def sanitize_name_part(part):
        sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', str(part).strip())
        return sanitized.strip('_') or 'unnamed'

    def build_output_stem(video_path):
        video_path = Path(video_path)
        if json_name is not None:
            return json_name
        if not name_from_video_path:
            return sanitize_name_part(video_path.stem)

        stem_path = video_path.with_suffix('')
        parts = list(stem_path.parts)
        anchor_idx = None
        for idx, part in enumerate(parts):
            if part.lower() in {'video', 'videos'}:
                anchor_idx = idx
        if anchor_idx is not None and anchor_idx + 1 < len(parts):
            raw_parts = parts[anchor_idx + 1:]
        elif stem_path.parent.name:
            raw_parts = [stem_path.parent.name, stem_path.name]
        else:
            raw_parts = [stem_path.name]

        clean_parts = [sanitize_name_part(part) for part in raw_parts]
        return '__'.join(part for part in clean_parts if part) or sanitize_name_part(video_path.stem)

    def as_event_list(tags):
        if tags is None:
            return []
        if isinstance(tags, (list, tuple, set)):
            return list(tags)
        return [tags]

    def in_any_interval(t, intervals):
        """Return True if time_sec falls inside any valid interval."""
        for start, stop in intervals:
            if start is None and stop is None:
                continue
            if start is None:
                start = 0.0
            if stop is None:
                stop = video_duration_sec
            if start <= t <= stop:
                return True
        return False

    def collect_interval_issues(intervals, label, duration_sec):
        issues = []
        for idx, (start, stop) in enumerate(intervals):
            start_chk = 0.0 if start is None else start
            stop_chk = duration_sec if stop is None else stop
            if stop_chk < start_chk:
                issues.append(f'{label}[{idx}] stop_before_start')
                continue
            if start_chk < -1e-6 or stop_chk > duration_sec + 1e-3:
                issues.append(f'{label}[{idx}] out_of_range')
        return issues

    def determine_qa_status(flags):
        reject_flags = {
            'cannot_open_video',
            'no_decodable_frames',
            'ffprobe_failed',
            'timing_fallback',
            'timestamp_nonmonotonic',
            'ffprobe_timestamp_short',
            'decoded_short_vs_timestamps',
            'annotation_out_of_range',
            'sampling_above_trusted_fps',
            'no_sampled_frames',
        }
        review_flags = {
            'variable_timing',
            'fps_inconsistent',
            'large_timestamp_gap',
            'metadata_count_mismatch',
            'annotation_frame_count_mismatch',
        }
        flag_set = set(flags)
        if flag_set & reject_flags:
            return 'reject'
        if flag_set & review_flags:
            return 'review'
        return 'accept'

    def manifest_row_base(video_path, json_path=''):
        return {
            'video': str(video_path),
            'json_path': json_path,
            'status': 'reject',
            'flags': '',
            'time_source': '',
            'decoded_frame_count': '',
            'sampled_frame_count': '',
            'metadata_frame_count': '',
            'opencv_fps': '',
            'ffprobe_nominal_fps': '',
            'ffprobe_avg_fps': '',
            'effective_fps': '',
            'duration_sec': '',
            'notes': '',
        }

    tension_intervals = tension_intervals or []
    fight_intervals = fight_intervals or []
    fall_intervals = fall_intervals or []
    requested_time_source = str(kwargs.get('time_source', 'ffprobe')).lower()
    only_with_txt_ann = bool(kwargs.get('only_with_txt_ann', False))
    name_from_video_path = bool(kwargs.get('name_from_video_path', False))
    if requested_time_source not in {'fps', 'ffprobe'}:
        raise ValueError(f'Invalid time_source: {requested_time_source}')

    model_path = kwargs.get('model_path', None) or DEFAULT_YOLO
    model = YOLO(model_path if Path(model_path).is_file() else DEFAULT_YOLO)
    show = kwargs.get('show', False)

    input_path = Path(input_path)
    if input_path.is_dir():
        all_files = [p for p in input_path.iterdir() if p.is_file()]
        vid_list = [p for p in all_files if is_video_file(p)]
        ignored_non_video_count = len(all_files) - len(vid_list)
        if ignored_non_video_count:
            print(f'[INFO] Ignoring {ignored_non_video_count} non-video files in {input_path}')
        if only_with_txt_ann and ann_file is None:
            filtered_vid_list = []
            for vid_path in vid_list:
                if find_txt_annotation_file(vid_path) is not None:
                    filtered_vid_list.append(vid_path)
                else:
                    print(f'[INFO] Skipping {vid_path.name}: no sibling .txt annotation file')
            vid_list = filtered_vid_list
    else:
        vid_list = [input_path]

    output_path = Path(output_path) if output_path else None
    if output_path is None:
        json_dir = input_path if input_path.is_dir() else input_path.parent
        json_name = None
    elif output_path.suffix == '.json':
        json_dir = output_path.parent
        json_name = output_path.stem
    elif output_path.is_dir():
        json_dir = output_path
        json_name = None
    else:
        json_dir = output_path
        json_name = None

    json_dir.mkdir(parents=True, exist_ok=True)

    detector_info = {'model': Path(model_path).stem, 'version': model.ckpt['version'], 'threshold': conf_thresh}
    print_color(f"YOLO:\nversion - {detector_info['version']}\nthreshold = {detector_info['threshold']}", 'b')
    print(f'Default group event: {default_group_tag}\n')
    if name_from_video_path:
        print("[INFO] Output naming mode: path-based names after the last 'video'/'videos' folder")

    explicit_ann_data = load_annotation_data(ann_file) if ann_file else None
    manifest_rows = []

    for vid_path in vid_list:
        output_stem = build_output_stem(vid_path)
        qa_flags = []
        qa_notes = []
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f'[WARN] Cannot open video (probably corrupted): {vid_path}')
            row = manifest_row_base(vid_path)
            row.update({'status': 'reject', 'flags': 'cannot_open_video', 'notes': 'cv2.VideoCapture failed'})
            manifest_rows.append(row)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        meta_frame_count = int(round(frame_count)) if frame_count > 0 else None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w != 1920 and w > 0:
            h = int(w * 1080 / 1920)
        if fps <= 0:
            fps = 25.0
            qa_notes.append('opencv fps missing; using 25.0 fallback for metadata-based calculations')

        timing_info = {
            'requested_source': requested_time_source,
            'source': 'fps',
            'opencv_fps': fps,
        }
        ffprobe_timing = probe_video_timing_ffprobe(vid_path)
        raw_frame_times = None
        zero_based_frame_times = None
        timestamp_nonmonotonic = False
        if ffprobe_timing is not None and ffprobe_timing['frame_times']:
            raw_frame_times = ffprobe_timing['frame_times']
            timestamp_nonmonotonic = any(curr < prev for prev, curr in zip(raw_frame_times, raw_frame_times[1:]))
            if timestamp_nonmonotonic:
                qa_flags.append('timestamp_nonmonotonic')
                qa_notes.append('ffprobe frame timestamps go backwards')
            first_pts_abs = raw_frame_times[0]
            zero_based_frame_times = [max(0.0, t - first_pts_abs) for t in raw_frame_times]
            timing_info.update({k: v for k, v in ffprobe_timing.items() if k != 'frame_times'})
            timing_info['source_first_frame_time_sec_abs'] = first_pts_abs
            timing_info['first_frame_time_sec'] = 0.0
            timing_info['last_frame_time_sec'] = zero_based_frame_times[-1]
            timing_info['frame_times_zero_based'] = True
            timing_info.update(summarize_frame_times(zero_based_frame_times))
        else:
            if requested_time_source == 'ffprobe':
                qa_flags.extend(['ffprobe_failed', 'timing_fallback'])
                qa_notes.append('ffprobe timestamps unavailable; fell back to fps timing')

        if requested_time_source == 'ffprobe' and zero_based_frame_times is not None:
            frame_times = zero_based_frame_times
            timing_info['source'] = 'ffprobe'
        else:
            frame_times = None
            timing_info['source'] = 'fps'

        if frame_times:
            if timing_info.get('stream_duration_sec'):
                video_duration_sec = float(timing_info['stream_duration_sec'])
            else:
                tail_delta = timing_info.get('median_frame_delta') or (1.0 / max((timing_info.get('effective_fps') or timing_info.get('avg_frame_rate') or timing_info.get('nominal_frame_rate') or fps), 1e-9))
                video_duration_sec = frame_times[-1] + tail_delta
        elif timing_info.get('stream_duration_sec'):
            video_duration_sec = float(timing_info['stream_duration_sec'])
        elif frame_count > 0 and fps > 0:
            video_duration_sec = frame_count / fps
        else:
            video_duration_sec = 0.0

        trusted_fps = (
            timing_info.get('effective_fps')
            or timing_info.get('avg_frame_rate')
            or timing_info.get('nominal_frame_rate')
            or fps
        )
        nominal_candidates = [x for x in (timing_info.get('nominal_frame_rate'), timing_info.get('avg_frame_rate'), fps) if x]
        for nominal in nominal_candidates:
            if trusted_fps and nominal and abs(nominal - trusted_fps) / max(trusted_fps, nominal, 1e-9) > 0.20:
                if 'fps_inconsistent' not in qa_flags:
                    qa_flags.append('fps_inconsistent')
                    qa_notes.append(f'nominal/header fps differs from effective fps ({nominal:.3f} vs {trusted_fps:.3f})')
                break

        if timing_info.get('variable_frame_timing'):
            qa_flags.append('variable_timing')
            qa_notes.append('frame timestamps show mixed cadence / variable timing')

        median_delta = timing_info.get('median_frame_delta')
        max_delta = timing_info.get('max_frame_delta')
        if median_delta and max_delta and max_delta > (3.0 * median_delta + 1e-9):
            qa_flags.append('large_timestamp_gap')
            qa_notes.append(f'largest timestamp gap {max_delta:.6f}s exceeds 3x median delta {median_delta:.6f}s')

        print(f"*** Converting {vid_path.name}, {(w, h)}p {video_duration_sec} s")
        print(
            f"Timing source: {timing_info['source']}; "
            f"opencv fps={fps:.3f}; "
            f"ffprobe avg fps={timing_info.get('avg_frame_rate')}; "
            f"effective fps={timing_info.get('effective_fps')}"
        )

        video_ann_data = explicit_ann_data
        if video_ann_data is None:
            auto_ann_file = find_txt_annotation_file(vid_path) if only_with_txt_ann else find_annotation_file(vid_path)
            if auto_ann_file is not None:
                print(f'Using annotation file: {auto_ann_file}')
                video_ann_data = load_annotation_data(auto_ann_file)

        video_ann_intervals = video_ann_data['intervals'] if video_ann_data is not None else None
        frame_annotation_labels = video_ann_data['frame_labels'] if video_ann_data is not None else None
        if frame_annotation_labels is not None:
            positive_frame_count = sum(frame_annotation_labels)
            qa_notes.append(
                f"frame-label csv loaded: {len(frame_annotation_labels)} frames, positives={positive_frame_count}"
            )
            if meta_frame_count is not None and abs(len(frame_annotation_labels) - meta_frame_count) > 1:
                qa_flags.append('annotation_frame_count_mismatch')
                qa_notes.append(
                    f'annotation csv length differs from metadata frame count: ann={len(frame_annotation_labels)}, metadata={meta_frame_count}'
                )

        tension_intervals_sec = list(video_ann_intervals[TAG_TENSION]) if video_ann_intervals is not None else []
        fight_intervals_sec = list(video_ann_intervals[TAG_FIGHT]) if video_ann_intervals is not None else []
        fall_intervals_sec = list(video_ann_intervals[TAG_FALL]) if video_ann_intervals is not None else []
        tension_intervals_sec.extend(parse_interval(s) for s in tension_intervals if s and s.strip())
        fight_intervals_sec.extend(parse_interval(s) for s in fight_intervals if s and s.strip())
        fall_intervals_sec.extend(parse_interval(s) for s in fall_intervals if s and s.strip())

        interval_issues = (
            collect_interval_issues(tension_intervals_sec, 'tension', video_duration_sec)
            + collect_interval_issues(fight_intervals_sec, 'fight', video_duration_sec)
            + collect_interval_issues(fall_intervals_sec, 'fall', video_duration_sec)
        )
        if interval_issues:
            qa_flags.append('annotation_out_of_range')
            qa_notes.append('; '.join(interval_issues[:5]))

        target_sampling = float(kwargs.get('sample_rate', sample_rate))
        if target_sampling <= 0:
            raise ValueError(f'Invalid sampling rate: {target_sampling} Hz')
        if trusted_fps is not None and target_sampling > trusted_fps + 1e-6:
            qa_flags.append('sampling_above_trusted_fps')
            qa_notes.append(f'target sampling {target_sampling:.6f} Hz exceeds trusted fps {trusted_fps:.6f} Hz')
            print(
                f"[WARN] target_sampling={target_sampling:.6f} Hz is higher than trusted fps={trusted_fps:.6f} Hz. "
                f"Some target sampling points may not have a nearby decoded frame."
            )

        sample_period = 1.0 / target_sampling
        legacy_step = max(1, int(round((trusted_fps or fps) / target_sampling)))
        frame_interval_sec = timing_info.get('median_frame_delta') or (1.0 / max(trusted_fps or fps, 1e-9))
        frame_time_tolerance = 0.5 * frame_interval_sec
        print_color(
            f"target sampling = {target_sampling:.3f} Hz -> sample period = {sample_period:.3f} s "
            f"(legacy approx step = {legacy_step})"
        )

        frames = []
        frame_idx = 0
        decode_stop_idx = None
        decode_stop_pos = None
        next_sample_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                decode_stop_idx = frame_idx
                decode_stop_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f'[INFO] cv2.VideoCapture.read() returned False at frame_idx={frame_idx}, cap_pos={decode_stop_pos}')
                break
            if frame is None:
                decode_stop_idx = frame_idx
                decode_stop_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f'[WARN] Decoder returned an empty frame at frame_idx={frame_idx}, cap_pos={decode_stop_pos}')
                break

            frame = cv2.resize(frame, (w, h))
            if frame_times is not None:
                if frame_idx >= len(frame_times):
                    decode_stop_idx = frame_idx
                    decode_stop_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    qa_flags.append('ffprobe_timestamp_short')
                    qa_notes.append(f'ffprobe timestamps ended at frame {len(frame_times)}')
                    print(
                        f"[WARN] ffprobe timestamp list ended at frame_idx={frame_idx} for {vid_path.name}; "
                        f"stopping instead of falling back to fps timing"
                    )
                    break
                time_sec = frame_times[frame_idx]
            else:
                time_sec = frame_idx / fps

            if time_sec + frame_time_tolerance < next_sample_time:
                frame_idx += 1
                continue

            try:
                n_h = h if h % 32 == 0 else h + (32 - (h%32))
                n_w = w if w % 32 == 0 else w + (32 - (w%32))
                results = model(frame, conf=conf_thresh, verbose=False, imgsz=(n_h, n_w))[0]
            except Exception as exc:
                print(f'[ERROR] YOLO inference failed at frame_idx={frame_idx}: {exc}')
                cap.release()
                if show:
                    cv2.destroyAllWindows()
                raise

            detection_list = []
            if results.boxes:
                for box, kpts_norm, conf_kpts in zip(results.boxes, results.keypoints.xyn, results.keypoints.conf):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    kpts_xyc = torch.cat([kpts_norm, conf_kpts.unsqueeze(-1)], dim=-1)
                    if cls_id not in [3, 4]:
                        x1, y1, x2, y2 = map(float, box.xyxyn[0])
                        if show:
                            for kp_pair in kpts_xyc:
                                cx = int(float(kp_pair[0]) * w)
                                cy = int(float(kp_pair[1]) * h)
                                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                        detection_list.append({
                            'class': cls_id,
                            'conf': conf,
                            'bbox': [x1, y1, x2, y2],
                            'key_points': kpts_xyc[:].flatten().tolist(),
                        })

            group_events = []
            if in_any_interval(time_sec, fall_intervals_sec):
                group_events.append(TAG_FALL)
            if in_any_interval(time_sec, tension_intervals_sec):
                group_events.append(TAG_TENSION)
            if in_any_interval(time_sec, fight_intervals_sec):
                group_events.append(TAG_FIGHT)
            if frame_annotation_labels is not None and frame_idx < len(frame_annotation_labels) and frame_annotation_labels[frame_idx] > 0:
                group_events.append(TAG_FIGHT)

            default_tags = as_event_list(default_group_tag)
            group_tags = group_events if group_events else default_tags
            frames.append({
                'f': frame_idx,
                't': time_sec,
                'individual_events': [],
                'group_events': sorted(set(group_tags), reverse=True),
                'detection_list': detection_list,
            })

            next_sample_time += sample_period
            while next_sample_time <= time_sec + frame_time_tolerance:
                next_sample_time += sample_period

            if show:
                cv2.imshow('head_center_debug', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            frame_idx += 1

        if frame_idx == 0:
            print(f'[WARN] Skipping video with no decodable frames: {vid_path} (unsupported codec/container or corrupted file)')
            cap.release()
            if show:
                cv2.destroyAllWindows()
            row = manifest_row_base(vid_path)
            row.update({
                'status': 'reject',
                'flags': 'no_decodable_frames',
                'notes': 'cv2.VideoCapture opened but no frames decoded',
                'metadata_frame_count': meta_frame_count if meta_frame_count is not None else '',
                'opencv_fps': f'{fps:.6f}',
            })
            manifest_rows.append(row)
            continue

        decoded_frame_count = frame_idx
        timestamps_match_decode = frame_times is not None and len(frame_times) == decoded_frame_count
        metadata_count_mismatch = (
            meta_frame_count is not None
            and abs(meta_frame_count - decoded_frame_count) > 1
        )
        if metadata_count_mismatch:
            qa_flags.append('metadata_count_mismatch')
            qa_notes.append(
                f'metadata frame count differs from decoded frames: metadata={meta_frame_count}, decoded={decoded_frame_count}'
            )
            print('[WARN] Metadata frame count differs from decoded frames. Treating this as a soft warning because timestamps are the primary truth.')

        decoded_short_vs_timestamps = (
            frame_times is not None
            and decoded_frame_count + 1 < len(frame_times)
        )
        if decoded_short_vs_timestamps:
            qa_flags.append('decoded_short_vs_timestamps')
            qa_notes.append(
                f'cv2 decoded fewer frames than ffprobe timestamps: decoded={decoded_frame_count}, ffprobe={len(frame_times)}'
            )
            print('[WARN] cv2 stopped before the ffprobe timestamp timeline ended. This file is not safe for precise timestamp-based labeling.')

        decode_complete = (
            timing_info.get('source') == 'ffprobe'
            and not any(flag in qa_flags for flag in ('ffprobe_failed', 'timing_fallback', 'timestamp_nonmonotonic', 'ffprobe_timestamp_short', 'decoded_short_vs_timestamps'))
        )

        if not frames:
            qa_flags.append('no_sampled_frames')
            qa_notes.append('no frames matched the requested sampling schedule')
            status = determine_qa_status(qa_flags)
            row = manifest_row_base(vid_path)
            row.update({
                'status': status,
                'flags': ';'.join(sorted(set(qa_flags))),
                'time_source': timing_info.get('source', ''),
                'decoded_frame_count': decoded_frame_count,
                'sampled_frame_count': 0,
                'metadata_frame_count': meta_frame_count if meta_frame_count is not None else '',
                'opencv_fps': f'{fps:.6f}',
                'ffprobe_nominal_fps': '' if timing_info.get('nominal_frame_rate') is None else f"{timing_info['nominal_frame_rate']:.6f}",
                'ffprobe_avg_fps': '' if timing_info.get('avg_frame_rate') is None else f"{timing_info['avg_frame_rate']:.6f}",
                'effective_fps': '' if timing_info.get('effective_fps') is None else f"{timing_info['effective_fps']:.6f}",
                'duration_sec': f'{video_duration_sec:.6f}',
                'notes': ' | '.join(dict.fromkeys(qa_notes)),
            })
            manifest_rows.append(row)
            cap.release()
            if show:
                cv2.destroyAllWindows()
            continue

        print(
            f"Decoded frames: {decoded_frame_count}; sampled frames saved: {len(frames)}; "
            f"metadata frame count: {meta_frame_count if meta_frame_count is not None else 'unknown'}"
        )

        event_intervals = {
            'tension': {'raw': tension_intervals, 'sec': tension_intervals_sec},
            'fight': {'raw': fight_intervals, 'sec': fight_intervals_sec},
            'fall': {'raw': fall_intervals, 'sec': fall_intervals_sec},
        }

        qa_flags = sorted(set(qa_flags))
        qa_status = determine_qa_status(qa_flags)
        data = {
            'video': str(vid_path),
            'fps': timing_info.get('effective_fps') or timing_info.get('avg_frame_rate') or fps,
            'decoded_frame_count': decoded_frame_count,
            'decode_complete': decode_complete,
            'qa': {
                'status': qa_status,
                'flags': qa_flags,
                'time_source': timing_info.get('source', 'fps'),
            },
            'annotation': {
                'source_type': '' if video_ann_data is None else video_ann_data.get('source_type', ''),
                'source_path': '' if video_ann_data is None else video_ann_data.get('source_path', ''),
                'frame_label_count': 0 if frame_annotation_labels is None else len(frame_annotation_labels),
            },
            'event_intervals': event_intervals,
            'frames': frames,
        }

        json_path = get_unique_name(json_dir / f"{output_stem}.json", 4)
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        row = manifest_row_base(vid_path, str(json_path))
        row.update({
            'status': qa_status,
            'flags': ';'.join(qa_flags),
            'time_source': timing_info.get('source', ''),
            'decoded_frame_count': decoded_frame_count,
            'sampled_frame_count': len(frames),
            'metadata_frame_count': meta_frame_count if meta_frame_count is not None else '',
            'opencv_fps': f'{fps:.6f}',
            'ffprobe_nominal_fps': '' if timing_info.get('nominal_frame_rate') is None else f"{timing_info['nominal_frame_rate']:.6f}",
            'ffprobe_avg_fps': '' if timing_info.get('avg_frame_rate') is None else f"{timing_info['avg_frame_rate']:.6f}",
            'effective_fps': '' if timing_info.get('effective_fps') is None else f"{timing_info['effective_fps']:.6f}",
            'duration_sec': f'{video_duration_sec:.6f}',
            'notes': ' | '.join(dict.fromkeys(qa_notes)),
        })
        manifest_rows.append(row)

        print_color(f"Saved::{len(frames)} sampled frames to {json_path} | QA={qa_status} flags={qa_flags}\n----------------\n", 'b')

        cap.release()
        if show:
            cv2.destroyAllWindows()

    manifest_path = json_dir / 'dataset_manifest.csv'
    fieldnames = [
        'video', 'json_path', 'status', 'flags', 'time_source',
        'decoded_frame_count', 'sampled_frame_count', 'metadata_frame_count',
        'opencv_fps', 'ffprobe_nominal_fps', 'ffprobe_avg_fps', 'effective_fps',
        'duration_sec', 'notes'
    ]
    import csv
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    print_color(f'Manifest saved to {manifest_path}', 'b')

#* ------  local runner (wrapper) -----------------------------
#* CLI interface moved to designated wrapper convert_to_json.py
def local_runner(tst_path, **kwargs):
    process_video(tst_path, **kwargs)


if __name__ == "__main__":
    local_runner("/mnt/local-data/Projects/Wesmart/Video-datasets/test_ds/tst_conv",
                 output_path="/mnt/local-data/Python/Projects/weSmart/data/json_files/tst_conv/try_05",
                 default_group_tag = 0,
                 )

#534(5,19,27) -> 333(5,7,6)- 460(4,8,8)/
# 460(5,9,8)->412(4,4,2)
