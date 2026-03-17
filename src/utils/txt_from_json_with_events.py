#!/usr/bin/env python3
"""
group_events_json_to_txt.py

Extract event intervals from JSON with:
  top-level: video, fps, step, frames[]
  frames[i]: t (seconds), f (frame index), group_events (list[int])

Event IDs (your mapping):
  2 = fall, 3 = tension, 4 = fight

Default behavior: EXCLUSIVE timeline (no overlaps) using priority:
  fight > fall > tension
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

EVENT_MAP: Dict[int, str] = {2: "fall", 3: "tension", 4: "fight"}
PRIORITY: List[int] = [4, 2, 3]  # fight > fall > tension


@dataclass
class Segment:
    start_t: float
    end_t: float
    start_f: int
    end_f: int


def iter_json_paths(inp: Path) -> Iterable[Path]:
    if inp.is_file():
        yield inp
    else:
        yield from sorted(inp.rglob("*.json"))


def pick_exclusive_label(group_events: List[int]) -> Optional[int]:
    s = set(group_events)
    for eid in PRIORITY:
        if eid in s:
            return eid
    return None


def format_time(t: float, timefmt: str, decimals: int) -> str:
    if timefmt == "sec":
        return f"{t:.{decimals}f}"

    # mmss / hh:mm:ss
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60

    if decimals == 0:
        # round to nearest second
        s_i = int(round(s))
        if s_i == 60:
            s_i = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
        if h > 0:
            return f"{h}:{m:02d}:{s_i:02d}"
        return f"{m}:{s_i:02d}"

    width = 2 + 1 + decimals  # "SS.xxx"
    s_str = f"{s:0{width}.{decimals}f}"
    if h > 0:
        return f"{h}:{m:02d}:{s_str}"
    return f"{m}:{s_str}"


def segments_from_labels(
    labels: List[Tuple[float, int, Optional[int]]],
    fps: float,
    step: int,
) -> Dict[str, List[Segment]]:
    """
    labels: list of (t, f, chosen_event_id or None)
    Build segments by label changes. End time is the boundary time where label changes,
    so segments do NOT overlap in exclusive mode.
    """
    out: Dict[str, List[Segment]] = {name: [] for name in EVENT_MAP.values()}
    dt = step / fps

    active_eid: Optional[int] = None
    start_t = start_f = None  # type: ignore

    for (t, f, eid) in labels:
        if active_eid is None:
            if eid is not None:
                active_eid = eid
                start_t, start_f = t, f
        else:
            if eid != active_eid:
                # close active at boundary time t (no widening)
                out[EVENT_MAP[active_eid]].append(
                    Segment(start_t=float(start_t), end_t=float(t), start_f=int(start_f), end_f=int(f))
                )
                active_eid = None
                if eid is not None:
                    active_eid = eid
                    start_t, start_f = t, f

    # close tail if still active: extend by one dt just to cover last sampled step
    if active_eid is not None:
        last_t, last_f, _ = labels[-1]
        out[EVENT_MAP[active_eid]].append(
            Segment(start_t=float(start_t), end_t=float(last_t + dt), start_f=int(start_f), end_f=int(last_f + step))
        )

    # drop empty
    return {k: v for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="Input JSON file or directory of JSONs")
    ap.add_argument("--out", dest="out", type=Path, required=True, help="Output TXT file")
    ap.add_argument("--timefmt", choices=["sec", "mmss"], default="sec", help="Time format for output")
    ap.add_argument("--decimals", type=int, default=1, help="Decimals (used for sec and mmss)")
    ap.add_argument("--pretty", action="store_true", help="Multi-line per video instead of one TSV line")
    ap.add_argument("--show-frames", action="store_true", help="Also print frame ranges")
    ap.add_argument("--keep-overlap", action="store_true",
                    help="Do NOT force exclusivity; treat all IDs in group_events as active (can overlap)")
    args = ap.parse_args()

    lines: List[str] = []

    for jp in iter_json_paths(args.inp):
        data = json.loads(jp.read_text(encoding="utf-8"))

        video = str(data.get("video", jp.stem))
        fps = float(data["fps"])
        step = int(data["step"])
        frames = list(data["frames"])
        frames.sort(key=lambda x: (x["t"], x.get("f", 0)))

        # Build (t,f,label)
        labels: List[Tuple[float, int, Optional[int]]] = []
        if args.keep_overlap:
            # In overlap mode we still produce per-event segments separately,
            # but you asked for "no overlap", so default is exclusive mode.
            # Here, we encode eid as None and handle below.
            # (Kept for completeness.)
            pass

        if not args.keep_overlap:
            for fr in frames:
                t = float(fr["t"])
                f = int(fr["f"])
                ge = fr.get("group_events", [])
                eid = pick_exclusive_label(ge)
                labels.append((t, f, eid))

            segs = segments_from_labels(labels, fps, step)

        else:
            # Overlap mode: build segments per eid directly (can overlap).
            # Uses the same "boundary close at first absent frame time".
            dt = step / fps
            active: Dict[int, Optional[Tuple[float, int]]] = {eid: None for eid in EVENT_MAP.keys()}
            segs = {name: [] for name in EVENT_MAP.values()}

            for fr in frames:
                t = float(fr["t"])
                f = int(fr["f"])
                ge = set(fr.get("group_events", []))

                for eid, name in EVENT_MAP.items():
                    if eid in ge:
                        if active[eid] is None:
                            active[eid] = (t, f)
                    else:
                        if active[eid] is not None:
                            st, sf = active[eid]
                            segs[name].append(Segment(st, t, sf, f))
                            active[eid] = None

            # close tails
            if frames:
                last_t = float(frames[-1]["t"])
                last_f = int(frames[-1]["f"])
                for eid, name in EVENT_MAP.items():
                    if active[eid] is not None:
                        st, sf = active[eid]
                        segs[name].append(Segment(st, last_t + dt, sf, last_f + step))

            segs = {k: v for k, v in segs.items() if v}

        def fmt_seg(s: Segment) -> str:
            a = format_time(s.start_t, args.timefmt, args.decimals)
            b = format_time(s.end_t, args.timefmt, args.decimals)
            if args.show_frames:
                return f"{a}-{b} ({s.start_f}-{s.end_f})"
            return f"{a}-{b}"

        if args.pretty:
            lines.append(f"VIDEO: {video}")
            lines.append(f"FPS: {fps}  STEP: {step}")
            for name in ["tension", "fight", "fall"]:
                if name in segs:
                    lines.append(f"  {name}: " + ",".join(fmt_seg(s) for s in segs[name]))
            lines.append("")
        else:
            fields = [video]
            for name in ["tension", "fight", "fall"]:
                if name in segs:
                    fields.append(f"{name}:" + ",".join(fmt_seg(s) for s in segs[name]))
            lines.append("\t".join(fields))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()