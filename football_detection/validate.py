import argparse
import pickle

import cv2
from ml_serving.drivers import driver

from football_detection import detection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output')

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.preds, 'rb') as f:
        preds = pickle.load(f)

    preds_i = 0
    vc = cv2.VideoCapture(args.video)
    frame_count = -1
    fps = vc.get(cv2.CAP_PROP_FPS)

    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_format = video.get(cv2.CAP_PROP_FORMAT)
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps,
            frameSize=(width, height)
        )

    # if args.show:
    #     cv2.namedWindow('Video')

    while True:
        frame_count += 1
        ret, frame = vc.read()
        if not ret:
            break

        if frame_count > preds[preds_i][0]:
            if preds_i < len(preds) - 1:
                preds_i += 1

        is_action = preds[preds_i][1] == 0

        if is_action:
            cv2.putText(
                frame,
                'ACTION',
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 250, 0), thickness=2, lineType=cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame,
                'NONE',
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 250), thickness=2, lineType=cv2.LINE_AA
            )

        # cv2.imshow('Video', frame)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break

        if args.output:
            video_writer.write(frame)

        if frame_count % 100 == 0:
            print(f'Processed {frame_count} frames.')

    if args.output:
        video_writer.release()
    cv2.destroyAllWindows()
    vc.release()


if __name__ == '__main__':
    main()
