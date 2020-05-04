import argparse
import os
import shlex
import subprocess


ffmpeg_tpl = (
    'ffmpeg -i {input_path} -acodec copy -f segment -vcodec copy '
    '-segment_time {duration} -reset_timestamps 1 {output_path}'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("input_path")
    parser.add_argument("output_dir")

    return parser.parse_args()


def process_file(input_path, output_path, duration=5):
    ffmpeg_cmd = shlex.split(
        ffmpeg_tpl.format(
            input_path=input_path,
            duration=duration,
            output_path=output_path,
        )
    )
    print(f"Running {ffmpeg_cmd}")
    pipe = subprocess.Popen(ffmpeg_cmd)
    pipe.communicate()


def get_output_filename(input_path):
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    # Insert "%d" in file template
    return f'{name}%d{ext}'


def is_video_file(name):
    _, ext = os.path.splitext(name)
    return ext.lower() in {'.mp4', '.mov', '.m4a', '.avi', '.mkv'}


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_dir):
        # Create
        os.makedirs(args.output_dir)
    if os.path.isfile(args.output_dir):
        raise RuntimeError('Please specify output_dir to refer to directory, not file.')

    if os.path.isfile(args.input_path):
        output_name = get_output_filename(args.input_path)
        output_path = os.path.join(args.output_dir, output_name)
        process_file(args.input_path, output_path, args.duration)
    elif os.path.isdir(args.input_path):
        files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if is_video_file(f)]
        if not files:
            raise RuntimeError('Please specify input source to file or directory containing video.')

        for f in files:
            output_name = get_output_filename(f)
            output_path = os.path.join(args.output_dir, output_name)
            process_file(args.input_path, output_path, args.duration)
    else:
        raise RuntimeError('Please specify input source to file or directory containing video.')
