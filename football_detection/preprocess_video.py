import argparse
import pickle

import cv2
from ml_serving.drivers import driver

from football_detection import detection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--each-frame', type=int, default=5)
    parser.add_argument('--output', default='index.pkl')
    parser.add_argument('--show', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()
    obj_det = driver.load_driver('auto')().load_model(args.detection)

    vc = cv2.VideoCapture(args.video)
    frame_count = -1
    frame_processed = 0
    frame_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS)

    # if args.show:
    #     cv2.namedWindow('Video')

    all_boxes = []
    while True:
        frame_count += 1
        is_process = frame_count % args.each_frame == 0
        if is_process:
            ret, frame = vc.read()
            if not ret:
                break
        else:
            vc.grab()
            continue

        # Frame processing
        if obj_det.driver_name == 'tensorflow':
            boxes = detection.detect_bboxes_tensorflow(obj_det, frame, only_classes=[1, 37])
        else:
            raise RuntimeError('unrecognized driver')

        for box in boxes:
            color = (0, 250, 0) if int(box[5]) == 37 else (0, 0, 250)
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, thickness=2, lineType=cv2.LINE_AA
            )

        all_boxes.append((frame_count, boxes))

        if args.show:
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        frame_processed += 1
        if frame_processed % 100 == 0:
            print(f'Processed {frame_processed} frames.')

    cv2.destroyAllWindows()
    vc.release()

    with open(args.output, 'wb') as f:
        pickle.dump(all_boxes, f)
    print(f'Index saved to {args.output}.')


if __name__ == '__main__':
    main()
