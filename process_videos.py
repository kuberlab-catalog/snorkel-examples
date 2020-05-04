import argparse
import os
import re
import shlex
import subprocess


cut_tpl = (
    'ffmpeg -i {input_path} -acodec copy -f segment -vcodec copy '
    '-segment_time {duration} -reset_timestamps 1 -y {output_path}'
)
resize_tpl = (
    'ffmpeg -i {input_path} -acodec copy '
    '-s {resolution} -maxrate {maxrate} -y {output_path}'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("input_path")
    parser.add_argument("output_dir")
    parser.add_argument("--mode", choices=['cut', 'resize'], default='cut')
    parser.add_argument("--resize-res", default='original')
    parser.add_argument("--resize-maxrate", default='original')

    return parser.parse_args()


def parse_resolution_ffprobe(output):
    groups = re.findall('([1-9][0-9]+x[0-9]+)', output)
    if groups and len(groups) > 0:
        resolution = groups[0]
        splitted = resolution.split('x')
        if len(splitted) == 2:
            width = int(splitted[0])
            height = int(splitted[1])

            return width, height

    return None, None


def parse_bitrate_ffprobe(output):
    groups = re.findall('([0-9]+) kb/s', output)
    if groups and len(groups) > 0:
        bitrate = groups[0]
        return bitrate

    return None


def process_file(input_path, output_path, mode='cut', **kwargs):
    if mode == 'cut':
        ffmpeg_cmd = shlex.split(
            cut_tpl.format(
                input_path=input_path,
                output_path=output_path,
                **kwargs,
            )
        )
    elif mode == 'resize':
        # Need to get bitrate and resolution
        if kwargs.get('resolution') == 'original' or kwargs.get('maxrate') == 'original':
            ffprobe_cmd = ['ffprobe', '-hide_banner', '-i', input_path]
            ffprobe = subprocess.Popen(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = ffprobe.stdout.read().decode()
            ffprobe.communicate()
            w, h = parse_resolution_ffprobe(output)
            bitrate = parse_bitrate_ffprobe(output)
            print(f'Detected resolution={w}x{h} and bitrate={bitrate} kb/s')

            if kwargs.get('resolution') == 'original':
                kwargs['resolution'] = f'{w}x{h}'
            if kwargs.get('maxrate') == 'original':
                kwargs['maxrate'] = f'{bitrate}k'

        ffmpeg_cmd = shlex.split(
            resize_tpl.format(
                input_path=input_path,
                output_path=output_path,
                **kwargs,
            )
        )
    print(f"Running {ffmpeg_cmd}")
    pipe = subprocess.Popen(ffmpeg_cmd)
    pipe.communicate()


def get_output_filename(input_path, mode='cut'):
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    # Insert "%d" in file template
    if mode == 'cut':
        return f'{name}%d{ext}'
    else:
        return f'{name}{ext}'


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
        output_name = get_output_filename(args.input_path, mode=args.mode)
        output_path = os.path.join(args.output_dir, output_name)
        process_file(
            args.input_path,
            output_path,
            mode=args.mode,
            duration=args.duration,
            resolution=args.resize_res,
            maxrate=args.resize_maxrate,
        )
    elif os.path.isdir(args.input_path):
        files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if is_video_file(f)]
        if not files:
            raise RuntimeError('Please specify input source to file or directory containing video.')

        for f in files:
            output_name = get_output_filename(f, mode=args.mode)
            output_path = os.path.join(args.output_dir, output_name)
            process_file(
                f,
                output_path,
                mode=args.mode,
                duration=args.duration,
                resolution=args.resize_res,
                maxrate=args.resize_maxrate,
            )
    else:
        raise RuntimeError('Please specify input source to file or directory containing video.')
