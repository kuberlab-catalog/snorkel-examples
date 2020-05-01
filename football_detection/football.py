import argparse
import pickle

import numpy as np
from snorkel import labeling
from snorkel.labeling.model import baselines
from snorkel.labeling.model import label_model

from football_detection import utils


ABSTAIN = -1
NOT_ACTION = -1
ACTION = 0


PERSON_CLASS = 1
BALL_CLASS = 37


@labeling.labeling_function()
def lf_contains_ball(frame_boxes):
    frame, boxes = frame_boxes
    any_ball = any([b[5] == BALL_CLASS for b in boxes])
    return ACTION if any_ball else NOT_ACTION


@labeling.labeling_function()
def lf_contains_person(frame_boxes):
    frame, boxes = frame_boxes
    any_person = any([b[5] == PERSON_CLASS for b in boxes])
    return ACTION if any_person else NOT_ACTION


@labeling.labeling_function()
def lf_contains_person_ball(frame_boxes):
    frame, boxes = frame_boxes
    any_person = any([b[5] == PERSON_CLASS for b in boxes])
    any_ball = any([b[5] == BALL_CLASS for b in boxes])
    return ACTION if any_person and any_ball else NOT_ACTION


@labeling.labeling_function()
def lf_ball_person_intersects(frame_boxes):
    frame, boxes = frame_boxes
    persons = [b for b in boxes if b[5] == PERSON_CLASS]
    balls = [b for b in boxes if b[5] == BALL_CLASS]
    for person in persons:
        for ball in balls:
            intersection = utils.box_intersection(person, ball)
            if intersection >= 0.03:
                return ACTION
    return NOT_ACTION


@labeling.labeling_function()
def lf_ball_at_bottom(frame_boxes):
    frame, boxes = frame_boxes
    persons = [b for b in boxes if b[5] == PERSON_CLASS]
    balls = [b for b in boxes if b[5] == BALL_CLASS]
    for person in persons:
        for ball in balls:
            person_height = person[3] - person[1]
            ball_center_y = (ball[3] + ball[1]) / 2
            if abs(person[3] - ball_center_y) <= person_height / 8:
                # print(f'ball={ball[3]}, person={person[1]},{person[3]}')
                return ACTION
    return NOT_ACTION


@labeling.labeling_function()
def lf_ball_left_right_position(frame_boxes):
    frame, boxes = frame_boxes
    persons = [b for b in boxes if b[5] == PERSON_CLASS]
    balls = [b for b in boxes if b[5] == BALL_CLASS]
    for person in persons:
        for ball in balls:
            person_width = person[2] - person[0]
            ball_center_x = (ball[2] + ball[0]) / 2
            if abs(person[0] - ball_center_x) < person_width / 8:
                return ACTION
            if abs(person[2] - ball_center_x) < person_width / 8:
                return ACTION

    return NOT_ACTION


@labeling.labeling_function()
def lf_ball_in_kick_box(frame_boxes):
    frame, boxes = frame_boxes
    persons = [b for b in boxes if b[5] == PERSON_CLASS]
    balls = [b for b in boxes if b[5] == BALL_CLASS]
    for person in persons:
        for ball in balls:
            person_width = person[2] - person[0]
            person_height = person[3] - person[1]
            ball_center_x = (ball[2] + ball[0]) / 2
            ball_center_y = (ball[3] + ball[1]) / 2
            in_box_x = person[0] - person_width / 4 <= ball_center_x <= person[2] + person_width / 4
            in_box_y = person[3] - person_height / 8 <= ball_center_y <= person[3] + person_height / 8
            if in_box_x and in_box_y:
                return ACTION

    return NOT_ACTION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--val-data')
    parser.add_argument('--val-labels')

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.index, 'rb') as f:
        all_boxes = pickle.load(f)

    need_validate = args.val_data and args.val_labels
    if need_validate:
        with open(args.val_data, 'rb') as f:
            val_boxes = pickle.load(f)
        with open(args.val_labels, 'rb') as f:
            val_labels_raw = pickle.load(f)
            val_labels = []
            for _, lbl in val_labels_raw:
                val_labels.append(lbl)
            val_labels = np.array(val_labels)

    lfs = [
        # lf_contains_ball,
        # lf_contains_person,
        # lf_contains_person_ball,
        lf_ball_person_intersects,
        lf_ball_at_bottom,
        lf_ball_left_right_position,
        lf_ball_in_kick_box,
    ]

    applier = labeling.LFApplier(lfs=lfs)
    L_train = applier.apply(all_boxes)
    if need_validate:
        L_test = applier.apply(val_boxes)

    summary = labeling.LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(summary)

    lm = label_model.LabelModel(cardinality=2, verbose=True)
    lm.fit(L_train=L_train, n_epochs=500, log_freq=100)

    if need_validate:
        majority_model = baselines.MajorityLabelVoter()
        majority_metrics = majority_model.score(
            L=L_test, Y=val_labels,
            tie_break_policy="random", metrics=['accuracy', 'f1', 'precision', 'recall']
        )

        print(f"{'Majority Vote Accuracy:':<25} {majority_metrics['accuracy'] * 100:.1f}%")
        print(f"{'Majority Vote F1:':<25} {majority_metrics['f1'] * 100:.1f}%")
        print(f"{'Majority Vote precision:':<25} {majority_metrics['precision'] * 100:.1f}%")
        print(f"{'Majority Vote recall:':<25} {majority_metrics['recall'] * 100:.1f}%")

        label_model_metrics = lm.score(
            L=L_test, Y=val_labels,
            tie_break_policy="random",
            metrics=['accuracy', 'f1', 'precision', 'recall']
        )
        print(f"{'Label Model Accuracy:':<25} {label_model_metrics['accuracy'] * 100:.1f}%")
        print(f"{'Label Model F1:':<25} {label_model_metrics['f1'] * 100:.1f}%")
        print(f"{'Label Model Precision:':<25} {label_model_metrics['precision'] * 100:.1f}%")
        print(f"{'Label Model Recall:':<25} {label_model_metrics['recall'] * 100:.1f}%")
    # lm.save('label_model.pkl')
    preds, probs = lm.predict(L_train, return_probs=True)
    preds_frames = []
    for i, (frame, _) in enumerate(all_boxes):
        preds_frames.append((frame, preds[i], probs[i]))

    name = 'preds.pkl'
    with open(name, 'wb') as f:
        pickle.dump(preds_frames, f)

    print(f'Saved to {name}.')


if __name__ == '__main__':
    main()
