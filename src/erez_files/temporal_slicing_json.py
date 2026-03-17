""" * Temporal slicing of continuous JSON streams """
from pathlib import Path
from colorama import Fore

#* Local imports
try:
    from .my_local_utils import print_color
    from .json_utils import load_json_data
except ImportError:
    from my_local_utils import print_color
    from json_utils import load_json_data

# --------------------------------------------------
# * Unit-level defaults (ToDo: later to be loaded from config)
# * Option (i): 1s window, 0.5s stride
WINDOW_SEC = 1.0   #* clip duration in seconds
STRIDE_SEC = 0.5   #* stride between clips in seconds
MIN_EVENTS = 2     #* minimum number of non-zero group_events to mark clip positive

# --------------------------------------------------
# * main function to be called by other units
# --------------------------------------------------
def slice_json_stream(# json_path:str|Path,
                      data:dict,
                      window_sec:float=WINDOW_SEC,
                      stride_sec:float=STRIDE_SEC,
                      min_events:int=MIN_EVENTS,
                      allow_empty_lbl:bool = True):
    """  Slice a continuous JSON stream into overlapping temporal clips.
    :param data: json raw data
    window_sec, stride_sec, min_events : slicing parameters
    allow_empty_lbl: if True , label = None (neither positive nor negative)
                     if False, label = 0 or 1 (fully annotated negatives)
    Return:
    clips : list of dicts with keys:
        - frames : list of frame dicts belonging to the clip
        - label  : 1 - violent motion,
                   0 - non-violent motion
                   None - unknown/ignore
        - t_start, t_end : temporal bounds of the clip
    """

    frames = data.get('frames', [])
    if not frames:
        print_color("Error - in slice_json_stream",Fore.CYAN)
        return []

    #* extract timestamps
    times = [frm['t'] for frm in frames]
    t_0, t_max = times[0], times[-1]

    clips = []
    t = t_0
    while t + window_sec <= t_max:
        t_first = t
        t_last  = t + window_sec
        #* select frames inside window
        clip_frames = [frm for frm in frames if t_first <= frm['t'] < t_last]

        if len(clip_frames) >= 2:
            #* count group events (non-zero)
            event_count = 0
            for frm in clip_frames:
                ge = frm.get('group_events', [])
                if not ge:  continue
                #* group_events may contain multiple values
                has_zero = any(e == 0 for e in ge)
                has_nonzero = any(e != 0 for e in ge)
                #* 0 together with non-zero is considered an error (log and continue)
                if has_zero and has_nonzero:
                    fidx = frm.get('f', '?')
                    print_color(f"[WARNING] mixed zero/non-zero group_events at frame #{fidx}: {ge}",Fore.YELLOW)
                # * treat any non-zero value as positive
                if has_nonzero:
                    event_count += 1

            if event_count >= min_events:
                label = 1
            else:
                label = None if allow_empty_lbl else 0

            clips.append({'frames':clip_frames, 'label':label, 't_start':t_first, 't_end':t_last})

        t += stride_sec

    return clips


def find_events(clips):
    """  Find merged events from a list of clips.
         Consecutive clips with label == 1 are merged into a single event.
    Returns: events : dict { idx : { 'begin': t_begin, 'end': t_end } }
    """
    events = {}
    if not clips:
        return events

    event_idx = 0
    in_event = False
    event_start = None

    for clip in clips:
        is_pos = (clip.get('label') == 1)

        if is_pos and not in_event:
            in_event = True
            event_idx += 1
            event_start = clip['t_start']

        elif not is_pos and in_event:
            events[event_idx] = {'begin': event_start, 'end': clip['t_start']}
            in_event = False
            event_start = None

    if in_event:
        events[event_idx] = {'begin': event_start, 'end': clips[-1]['t_end']}

    return events

# --------------------------------------------------
# * testing and inspection tools
# --------------------------------------------------

def inspect_clips(clips):
    """  Inspect a list of temporal clips and print basic statistics.
    Prints:
    - total number of clips
    - total clip duration (sum over clips)
    - effective FPS of JSON frames
    - total number of True (label == 1) clips
    - number of events (consecutive True clips counted as one)
    """
    if not clips:
        print("No clips to inspect")
        return

    n_clips = len(clips) #* total number of clips

    #* total video duration
    # total_duration = sum(c["t_end"] - c["t_start"] for c in clips)
    #* original stream duration
    #*               last frame of the last clip  - first frame of the first clip
    stream_duration = clips[-1]['frames'][-1]['t'] - clips[0]['frames'][0]['t']


    #* estimate actual FPS from timestamps (use first clip)
    frames = clips[0]['frames']
    if len(frames) >= 2:
        times = [f['t'] for f in frames]
        dt = [(t2 - t1) for t1, t2 in zip(times[:-1], times[1:])]
        mean_dt = sum(dt) / len(dt)
        fps = 1.0 / mean_dt if mean_dt > 0 else 0.0
    else:
        fps = 0.0

    #* count True labels
    labels = [c["label"] == 1 for c in clips]
    n_true = sum(labels)

    #* count events (consecutive True = one event)
    n_events = 0
    prev = False
    for cur in labels:
        if cur and not prev:
            n_events += 1
        prev = cur

    print("=== Clips info ===")
    print(f"Stream duration : {stream_duration:.2f} s")
    print(f"Window/ Stride  : {WINDOW_SEC:.1f}s / {STRIDE_SEC:.1f}s ")
    print(f"Total clips     : {n_clips}")
    print(f"Effective FPS   : {fps:.2f}")
    print(f"Positive clips  : {n_true}")
    print(f"Events          : {n_events}")


def print_events(events:dict):
    for i, e in events.items():
        print('\t ', i, f": from\t {e['begin']:3.1f} to {e['end']:6.1f} sec")

def inspect_dir(j_dir:Path):
    for j in j_dir.glob("*.json"):
        print(f"== File ==========================\n{j.name:s} - {j.stat().st_size:,}")
        sjs = slice_json_stream(load_json_data(j), allow_empty_lbl=False)
        # inspect_clips(slice_json_stream(sjs, allow_empty_lbl=False))
        inspect_clips(sjs)
        # print_events(find_events(slice_json_stream(j)))
        print_events(find_events(sjs))
        print("\n") # print("===================================")


if __name__ == "__main__":
    data_path = Path("./data/")
    json_example = data_path/"json_data/full_ann_w_keys/new_21_1_keypoints.json"
    # inspect_dir(json_example.parent)
    inspect_dir(data_path/"json_data/full_ann_w_keys")

    pass
#202(,6,)->195(,,)
