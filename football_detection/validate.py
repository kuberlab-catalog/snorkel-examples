import argparse
import pickle

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output')
    parser.add_argument('--index')
    parser.add_argument('--mode', default='validate', choices=['validate', 'label'])

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.preds, 'rb') as f:
        preds = pickle.load(f)

    all_boxes = None
    if args.index:
        with open(args.index, 'rb') as f:
            all_boxes = pickle.load(f)

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

    gt_preds = [(0, 0) for i in range(len(preds))]
    while True:
        frame_count += 1
        ret, frame = vc.read()
        if not ret:
            break

        if frame_count > preds[preds_i][0]:
            if preds_i < len(preds) - 1:
                preds_i += 1

        is_action = preds[preds_i][1] == 0
        probability = preds[preds_i][2][0]

        if is_action:
            cv2.putText(
                frame,
                f'ACTION {int(probability * 100)}%',
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
        if all_boxes is not None:
            _, boxes = all_boxes[preds_i]
            for box in boxes:
                color = (0, 0, 250) if box[5] == 1 else (0, 250, 0)
                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        # cv2.imshow('Video', frame)
        if args.mode == 'label':
            txt = "Press 'Space' to mark 'Action' or 'N' to mark 'None'"
            (x_size, y_size), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
            cv2.putText(
                frame,
                txt,
                (frame.shape[1] // 2 - x_size // 2, int(frame.shape[0] * 0.9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame,
                txt,
                (frame.shape[1] // 2 - x_size // 2, int(frame.shape[0] * 0.9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (250, 250, 250), thickness=1, lineType=cv2.LINE_AA
            )
            end = None
            cv2.imshow('Video', frame)

            # if not is_action:
            #     gt_preds[preds_i] = (preds[preds_i][0], -1)
            #     continue
            while True:
                key = cv2.waitKey(0)
                if key in {ord('n'), ord('N')}:
                    gt_preds[preds_i] = (preds[preds_i][0], -1)
                    break
                elif key == 32:
                    gt_preds[preds_i] = (preds[preds_i][0], 0)
                    break
                elif key == 27:
                    end = True
                    break

            if end:
                break

        if args.output:
            video_writer.write(frame)

        if frame_count % 100 == 0:
            print(f'Processed {frame_count} frames.')

    if args.mode == 'label':
        with open('labels.pkl', 'wb') as f:
            pickle.dump(gt_preds, f)
        print(f'Labels are saved to labels.pkl.')

    if args.output:
        video_writer.release()
    cv2.destroyAllWindows()
    vc.release()


if __name__ == '__main__':
    main()
