#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np


FRAME_COLUMN_CANDIDATES = (
    "frame",
    "frame_idx",
    "frame_id",
    "frame_number",
    "frame_num",
    "frame_no",
    "f",
)
LABEL_COLUMN_CANDIDATES = (
    "fight",
    "fight_event",
    "is_fight",
    "is_fighting",
    "fight_label",
    "label",
    "gt",
    "target",
    "event",
    "class",
)
POSITIVE_VALUES = {"1", "true", "yes", "y", "fight", "fighting", "positive", "pos"}
NEGATIVE_VALUES = {"0", "false", "no", "n", "no_fight", "nonfight", "negative", "neg", "normal"}


@dataclass
class AnnotationTable:
    frame_to_label: Dict[int, int]
    frame_column: str
    label_column: str
    num_rows: int


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def auto_select_column(fieldnames: Iterable[str], candidates: Iterable[str], kind: str) -> str:
    normalized_map = {normalize_column_name(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]
    available = ", ".join(fieldnames)
    raise ValueError(f"Could not auto-detect {kind} column. Available CSV columns: {available}")


def parse_frame_index(value: object, line_no: int) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"CSV line {line_no}: frame value is empty")
    try:
        frame_idx = int(float(text))
    except ValueError as exc:
        raise ValueError(f"CSV line {line_no}: invalid frame value {text!r}") from exc
    if frame_idx < 0:
        raise ValueError(f"CSV line {line_no}: frame index must be >= 0, got {frame_idx}")
    return frame_idx


def parse_binary_label(value: object, line_no: int) -> int:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"CSV line {line_no}: label value is empty")
    if text in POSITIVE_VALUES:
        return 1
    if text in NEGATIVE_VALUES:
        return 0

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(
            f"CSV line {line_no}: unsupported label value {value!r}. "
            "Use 0/1, true/false, yes/no, or fight/no_fight."
        ) from exc

    return int(numeric != 0.0)


def looks_like_plain_label_lines(csv_path: Path, sample_size: int = 20) -> bool:
    checked = 0
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            text = raw_line.strip()
            if not text:
                continue
            if any(sep in text for sep in (",", ";", "\t")):
                return False
            try:
                parse_binary_label(text, line_no)
            except ValueError:
                return False
            checked += 1
            if checked >= sample_size:
                break
    return checked > 0


def load_plain_label_annotations(csv_path: Path) -> AnnotationTable:
    frame_to_label: Dict[int, int] = {}
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for frame_idx, raw_line in enumerate(f):
            text = raw_line.strip()
            if not text:
                raise ValueError(
                    f"{csv_path}: empty annotation line at frame {frame_idx}. "
                    "UBI_FIGHTS-style CSVs should contain one 0/1 label per line."
                )
            label = parse_binary_label(text, frame_idx + 1)
            frame_to_label[frame_idx] = label

    if not frame_to_label:
        raise ValueError(f"{csv_path}: no annotation rows were loaded")

    return AnnotationTable(
        frame_to_label=frame_to_label,
        frame_column="line_index",
        label_column="value",
        num_rows=len(frame_to_label),
    )


def load_dict_annotations(csv_path: Path, frame_column: Optional[str], label_column: Optional[str]) -> AnnotationTable:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: CSV file has no header row")

        resolved_frame_column = frame_column or auto_select_column(reader.fieldnames, FRAME_COLUMN_CANDIDATES, "frame")
        resolved_label_column = label_column or auto_select_column(reader.fieldnames, LABEL_COLUMN_CANDIDATES, "label")

        frame_to_label: Dict[int, int] = {}
        num_rows = 0
        for line_no, row in enumerate(reader, start=2):
            if row is None:
                continue
            if not any(str(value).strip() for value in row.values() if value is not None):
                continue

            frame_idx = parse_frame_index(row.get(resolved_frame_column), line_no)
            label = parse_binary_label(row.get(resolved_label_column), line_no)
            frame_to_label[frame_idx] = label
            num_rows += 1

    if not frame_to_label:
        raise ValueError(f"{csv_path}: no annotation rows were loaded")

    return AnnotationTable(
        frame_to_label=frame_to_label,
        frame_column=resolved_frame_column,
        label_column=resolved_label_column,
        num_rows=num_rows,
    )


def load_annotations(csv_path: Path, frame_column: Optional[str], label_column: Optional[str]) -> AnnotationTable:
    if frame_column is None and label_column is None and looks_like_plain_label_lines(csv_path):
        return load_plain_label_annotations(csv_path)
    return load_dict_annotations(csv_path, frame_column, label_column)


def format_time_sec(time_sec: float) -> str:
    total_ms = int(round(max(float(time_sec), 0.0) * 1000.0))
    minutes, rem_ms = divmod(total_ms, 60_000)
    seconds, millis = divmod(rem_ms, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def label_to_text(label: Optional[int]) -> str:
    if label is None:
        return "UNKNOWN"
    return "FIGHT" if int(label) == 1 else "NO FIGHT"


def label_to_color(label: Optional[int]) -> tuple[int, int, int]:
    if label is None:
        return (160, 160, 160)
    return (0, 0, 255) if int(label) == 1 else (0, 180, 0)


def build_timeline(width: int, total_frames: int, annotations: Dict[int, int], height: int) -> np.ndarray:
    timeline = np.full((height, width, 3), 32, dtype=np.uint8)
    if width <= 0 or total_frames <= 0:
        return timeline

    denom = max(total_frames - 1, 1)
    for frame_idx, label in annotations.items():
        if frame_idx < 0 or frame_idx >= total_frames:
            continue
        x = int(round(frame_idx * (width - 1) / denom))
        x0 = max(x - 1, 0)
        x1 = min(x + 2, width)
        timeline[:, x0:x1] = label_to_color(label)

    cv2.rectangle(timeline, (0, 0), (width - 1, height - 1), (90, 90, 90), 1)
    return timeline


def draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    label: Optional[int],
    label_source: str,
    timeline_base: np.ndarray,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    header_h = 88
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0, out)

    state_text = label_to_text(label)
    state_color = label_to_color(label)
    current_time_sec = float(frame_idx) / float(fps) if fps > 0 else 0.0

    cv2.putText(out, f"Frame: {frame_idx}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if total_frames > 0:
        cv2.putText(
            out,
            f"Total: {total_frames} | Time: {format_time_sec(current_time_sec)}",
            (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out,
            f"Time: {format_time_sec(current_time_sec)}",
            (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        f"Annotation: {state_text} ({label_source})",
        (16, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        state_color,
        2,
        cv2.LINE_AA,
    )

    timeline = timeline_base.copy()
    if total_frames > 0 and timeline.shape[1] > 0:
        x = int(round(frame_idx * (timeline.shape[1] - 1) / max(total_frames - 1, 1)))
        cv2.line(timeline, (x, 0), (x, timeline.shape[0] - 1), (255, 255, 255), 2)

    return np.vstack([out, timeline])


def resolve_label_for_frame(
    frame_idx: int,
    annotations: Dict[int, int],
    missing_label: str,
    last_known_label: Optional[int],
) -> tuple[Optional[int], str, Optional[int]]:
    if frame_idx in annotations:
        label = int(annotations[frame_idx])
        return label, "csv", label

    if missing_label == "keep-last" and last_known_label is not None:
        return last_known_label, "carry", last_known_label

    if missing_label == "0":
        return 0, "default", last_known_label

    if missing_label == "1":
        return 1, "default", last_known_label

    return None, "missing", last_known_label


def open_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {path}")
    return writer


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize per-frame fight annotations from a CSV on top of a video.")
    ap.add_argument("--video", required=True, help="Path to the source video.")
    ap.add_argument(
        "--annotations_csv",
        default=None,
        help="CSV file with frame numbers and binary fight labels. Default: same-stem .csv next to the video.",
    )
    ap.add_argument("--frame_column", default=None, help="Optional CSV column name for frame index.")
    ap.add_argument("--label_column", default=None, help="Optional CSV column name for fight label.")
    ap.add_argument(
        "--missing_label",
        choices=("unknown", "0", "1", "keep-last"),
        default="unknown",
        help="How to treat video frames missing from the CSV.",
    )
    ap.add_argument("--start_frame", type=int, default=0, help="First video frame to render.")
    ap.add_argument("--end_frame", type=int, default=None, help="Last video frame to render, inclusive.")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor for the displayed/saved frames.")
    ap.add_argument("--timeline_height", type=int, default=28, help="Height in pixels for the annotation timeline.")
    ap.add_argument("--playback_fps", type=float, default=0.0, help="Preview speed. Default 0 uses the video FPS.")
    ap.add_argument("--output_video", default=None, help="Optional path to save an annotated MP4.")
    ap.add_argument("--no_display", action="store_true", help="Disable the OpenCV preview window.")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    csv_path = Path(args.annotations_csv).resolve() if args.annotations_csv else video_path.with_suffix(".csv")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if args.scale <= 0:
        raise ValueError(f"--scale must be > 0, got {args.scale}")
    if args.timeline_height < 0:
        raise ValueError(f"--timeline_height must be >= 0, got {args.timeline_height}")

    display_enabled = not bool(args.no_display)
    output_path = Path(args.output_video).resolve() if args.output_video else None
    if not display_enabled and output_path is None:
        raise ValueError("Nothing to do: enable display or pass --output_video.")

    annotations = load_annotations(csv_path, args.frame_column, args.label_column)
    pos_count = sum(1 for value in annotations.frame_to_label.values() if int(value) == 1)
    neg_count = sum(1 for value in annotations.frame_to_label.values() if int(value) == 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    start_frame = max(int(args.start_frame), 0)
    if total_frames > 0 and start_frame >= total_frames:
        raise ValueError(f"--start_frame={start_frame} is outside the video range [0, {total_frames - 1}]")

    if args.end_frame is None:
        end_frame = total_frames - 1 if total_frames > 0 else None
    else:
        end_frame = int(args.end_frame)
        if end_frame < start_frame:
            raise ValueError(f"--end_frame={end_frame} must be >= --start_frame={start_frame}")
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    ret, probe_frame = cap.read()
    if not ret or probe_frame is None:
        raise RuntimeError(f"Could not read the first requested frame from {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

    if args.scale != 1.0:
        display_width = max(int(round(probe_frame.shape[1] * args.scale)), 1)
        display_height = max(int(round(probe_frame.shape[0] * args.scale)), 1)
    else:
        display_width = int(probe_frame.shape[1])
        display_height = int(probe_frame.shape[0])

    timeline = build_timeline(display_width, total_frames, annotations.frame_to_label, args.timeline_height)
    output_height = display_height + timeline.shape[0]

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = open_video_writer(output_path, fps, (display_width, output_height))

    preview_fps = float(args.playback_fps) if args.playback_fps and args.playback_fps > 0 else float(fps)
    delay_ms = max(int(round(1000.0 / preview_fps)), 1)

    print(f"[INFO] video: {video_path}")
    print(f"[INFO] csv:   {csv_path}")
    print(f"[INFO] detected columns: frame={annotations.frame_column!r}, label={annotations.label_column!r}")
    print(f"[INFO] annotation rows: {annotations.num_rows} | positive={pos_count} | negative={neg_count}")
    print(f"[INFO] video fps={fps:.3f} | total_frames={total_frames}")
    if end_frame is not None:
        print(f"[INFO] rendering frames {start_frame}..{end_frame}")
    else:
        print(f"[INFO] rendering frames starting at {start_frame}")
    if output_path is not None:
        print(f"[INFO] output video: {output_path}")

    paused = False
    should_quit = False
    last_known_label: Optional[int] = None
    frame_idx = start_frame

    try:
        while True:
            if end_frame is not None and frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if args.scale != 1.0:
                frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)

            label, label_source, last_known_label = resolve_label_for_frame(
                frame_idx,
                annotations.frame_to_label,
                args.missing_label,
                last_known_label,
            )
            rendered = draw_overlay(frame, frame_idx, total_frames, fps, label, label_source, timeline)

            if writer is not None:
                writer.write(rendered)

            if display_enabled:
                cv2.imshow("Video Annotation Review", rendered)
                key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
                if key in (27, ord("q")):
                    should_quit = True
                elif key == ord(" "):
                    paused = not paused

                while paused and not should_quit:
                    key = cv2.waitKey(0) & 0xFF
                    if key in (27, ord("q")):
                        should_quit = True
                        break
                    if key == ord(" "):
                        paused = False
                        break

            if should_quit:
                break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
