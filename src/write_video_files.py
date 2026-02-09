#!/usr/bin/env python3
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# ------------- EDIT HERE -------------
STREAMS = {
    "cam02": "rtsp://admin:a7638231@172.26.168.9:554/Streaming/Channels/201",
    "cam03": "rtsp://admin:a7638231@172.26.168.9:554/Streaming/Channels/301",
}
# -------------------------------------


def safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120] if len(s) > 120 else s


def build_ffmpeg_cmd(url: str, out_path: str, duration_sec: int,
                     encoder: str, crf: int, preset: str, rtsp_transport: str) -> list[str]:
    # Longest side = 720, keep aspect ratio
    scale = "scale='if(gt(iw,ih),720,-2)':'if(gt(iw,ih),-2,720)'"
    # Keep every 5th frame
    select = "select='not(mod(n\\,5))'"
    vf = f"{select},{scale}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-rtsp_transport", rtsp_transport,
        "-stimeout", "10000000",   # 10s open
        "-rw_timeout", "15000000", # 15s I/O
        "-i", url,
        "-t", str(duration_sec),
        "-an",  # no audio (recommended when dropping frames)
        "-sn",
        "-vf", vf,
        "-fps_mode", "vfr",
        "-c:v", encoder,
    ]

    if encoder == "libx264":
        cmd += ["-preset", preset, "-crf", str(crf)]
    elif encoder in ("h264_nvenc", "hevc_nvenc"):
        cmd += ["-preset", preset, "-cq", str(max(1, min(51, crf))), "-b:v", "0"]
    else:
        cmd += ["-preset", preset, "-crf", str(crf)]

    cmd += ["-movflags", "+faststart", out_path]
    return cmd


def wait_until(dt: datetime):
    while True:
        now = datetime.now(dt.tzinfo)
        remaining = (dt - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


def main():
    # Defaults match your request: tomorrow 17:00-23:00 Asia/Jerusalem
    tz = ZoneInfo("Asia/Jerusalem")
    start_hour, end_hour = 17, 23
    duration_hours = end_hour - start_hour

    encoder = "libx264"        # or "h264_nvenc"
    crf = 23                   # quality
    preset = "veryfast"        # encoding speed/quality tradeoff
    rtsp_transport = "tcp"     # tcp is usually more stable than udp
    out_root = "recordings"

    now = datetime.now(tz)
    tomorrow = (now + timedelta(days=1)).date()
    start_dt = datetime(tomorrow.year, tomorrow.month, tomorrow.day, start_hour, 0, 0, tzinfo=tz)
    end_dt   = datetime(tomorrow.year, tomorrow.month, tomorrow.day, end_hour, 0, 0, tzinfo=tz)

    if end_dt <= start_dt:
        raise ValueError("end_hour must be > start_hour")

    planned_duration = int((end_dt - start_dt).total_seconds())
    tag = f"{tomorrow:%Y%m%d}_{start_hour:02d}00-{end_hour:02d}00"
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Now:       {now.isoformat()}")
    print(f"Window:    {start_dt.isoformat()} -> {end_dt.isoformat()} ({duration_hours}h)")
    print(f"Out dir:   {out_dir}")
    print(f"Streams:   {len(STREAMS)}")

    # If launched during window, record remaining time
    if now < start_dt:
        wait_until(start_dt)
        actual_duration = planned_duration
    elif start_dt <= now < end_dt:
        actual_duration = int((end_dt - now).total_seconds())
        print(f"Already inside window; recording remaining {actual_duration/3600:.2f}h")
    else:
        print("The requested window is already in the past relative to current time.")
        sys.exit(1)

    procs: dict[str, subprocess.Popen] = {}

    def terminate_all():
        for p in procs.values():
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except Exception:
                    pass

    def handle_sig(sig, _frame):
        print(f"\nSignal {sig} received; terminating...")
        terminate_all()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # Start ffmpeg per stream
    for name, url in STREAMS.items():
        name = safe_name(name)
        out_path = os.path.join(out_dir, f"{name}_{tag}.mp4")

        cmd = build_ffmpeg_cmd(
            url=url,
            out_path=out_path,
            duration_sec=actual_duration,
            encoder=encoder,
            crf=crf,
            preset=preset,
            rtsp_transport=rtsp_transport,
        )

        print(f"\nStarting {name} -> {out_path}")
        print("CMD:", " ".join(shlex.quote(x) for x in cmd))
        procs[name] = subprocess.Popen(cmd)

    # Wait for all
    exit_codes = {name: p.wait() for name, p in procs.items()}

    print("\nDone. Exit codes:")
    for name, code in exit_codes.items():
        print(f"  {name}: {code}")

    failed = [n for n, c in exit_codes.items() if c != 0]
    if failed:
        print("\nSome streams failed:", ", ".join(failed))
        sys.exit(2)


if __name__ == "__main__":
    main()
