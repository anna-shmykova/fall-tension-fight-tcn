""" CLI interface for the timestamp-truth video_to_json converter with QA manifest
    Converts one or more videos into structured JSON dict.
    YOLO detector used for people detection and extracting
    bound-boxes and key-points at frame level.
    If needed, time-based labeling at frame level can be performed

    Usage:
    >> convert_to_json.py [-h] VIDEO [-m MODEL] [-o OUT] [-s SAMPLE_RATE] [-c CONF]
                              [-g GROUP_ANN [GROUP_ANN ...]] [-a ANN_FILE]
                              [--time-source {fps,ffprobe}]
                              [-t TENSION] [-f FIGHT] [-fa FALL]
    Options:
      -h, --help                  show this help message and exit
      VIDEO                       Path to input video or dir of videos (.mp4, .mkv, ...)
      -m/--model MODEL            Path to YOLO___.pt model
      -o/--out OUT                Path to output JSON file or output directory
      -s/--sample-rate RATE       Target saved frames per second (Hz)
      -c/--conf CONF              YOLO detection confidence threshold
      --time-source               How frame timestamps are derived
      -g/--group-ann [GROUP_ANN ...]  Default annotation for group event
      -a/--ann-file ANN_FILE      Optional annotation text file
      -t/--tension TENSION        Tension interval(s) in format START-END, e.g. 00:01:00-00:01:30, -00:00:40, 00:05:00-
      -f/--fight FIGHT            Fight interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-
      -fa/--fall FALL             Fall interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-
"""
import argparse
from pathlib import Path
from video_to_json_bb_keypoints_folder_timestamp_truth_qa import process_video
from my_local_utils import print_color

def main():
    parser = argparse.ArgumentParser(description="Convert video to JSON data file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', type=Path, help="Path to input video or dir of videos ( .mp4, .mkv, ...)")
    parser.add_argument( '-m', '--model',type=Path, help="Path to YOLO___.pt model")
    parser.add_argument( '-o', '--out',  type=Path, help="Path to output JSON file or output directory")
    parser.add_argument( '-s', '--sample-rate', type=float, help="Target saved frames per second (Hz), e.g. 5 keeps ~5 JSON frames/sec")
    parser.add_argument( '-c', '--conf', type=float, help="YOLO detection confidence threshold")
    parser.add_argument( '-g', '--group-ann', type=int, nargs='+', help="Default annotation for group event")
    parser.add_argument( '-a', '--ann-file', type=Path, help="Optional annotation text file")
    parser.add_argument( '-t', '--tension', action="append", help="Tension interval(s) in format START-END, e.g. 00:01:00-00:01:30, -00:00:40, 00:05:00-")
    parser.add_argument( '-f', '--fight', action="append", help="Fight interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-")
    parser.add_argument( '-fa','--fall', action="append", help="Fall interval(s) in format START-END, e.g. 00:02:10-00:02:40, 00:03:00-")
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    parser.add_argument('--time-source', choices=('fps', 'ffprobe'), default='ffprobe',
                        help="How to derive frame time: metadata fps or ffprobe per-frame timestamps")
    parser.add_argument('--only-with-txt-ann', action='store_true',
                        help='When VIDEO is a directory and --ann-file is not set, process only videos that have a sibling <stem>.txt file')
    parser.add_argument('--name-from-video-path', action='store_true',
                        help="Build JSON names from path parts after the last 'video'/'videos' folder, e.g. 3_6__1.json")

    args = parser.parse_args()
    if args.conf is not None:
        print_color(args.conf)

    process_kwargs = dict(input_path=args.video,
                          output_path=args.out,
                          ann_file=args.ann_file,
                          default_group_tag=args.group_ann,
                          tension_intervals=args.tension,
                          fight_intervals=args.fight,
                          fall_intervals=args.fall,
                          model_path=args.model,
                          show=args.show,
                          time_source=args.time_source,
                          only_with_txt_ann=args.only_with_txt_ann,
                          name_from_video_path=args.name_from_video_path)
    if args.sample_rate is not None:
        process_kwargs['sample_rate'] = args.sample_rate
    if args.conf is not None:
        process_kwargs['conf_thresh'] = args.conf

    process_video(**process_kwargs)

if __name__ == "__main__":
    main()
