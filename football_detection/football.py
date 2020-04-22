import argparse
import pickle

import cv2
from snorkel import labeling
from snorkel.labeling.model import baselines
from snorkel.labeling.model import label_model

from football_detection import detection


ACTION = 0
ABSTAIN = -1

PERSON_CLASS = 1
BALL_CLASS = 37


def box_intersection(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    d = float(box_a_area + box_b_area - inter_area)
    if d == 0:
        return 0
    iou = inter_area / d
    return iou


@labeling.labeling_function()
def lf_contains_ball(frame_boxes):
    frame, boxes = frame_boxes
    any_ball = any([b[5] == BALL_CLASS for b in boxes])
    return ACTION if any_ball else ABSTAIN


@labeling.labeling_function()
def lf_contains_person(frame_boxes):
    frame, boxes = frame_boxes
    any_person = any([b[5] == PERSON_CLASS for b in boxes])
    return ACTION if any_person else ABSTAIN


@labeling.labeling_function()
def lf_contains_person_ball(frame_boxes):
    frame, boxes = frame_boxes
    any_person = any([b[5] == PERSON_CLASS for b in boxes])
    any_ball = any([b[5] == BALL_CLASS for b in boxes])
    return ACTION if any_person and any_ball else ABSTAIN


@labeling.labeling_function()
def lf_ball_person_intersects(frame_boxes):
    frame, boxes = frame_boxes
    persons = [b for b in boxes if b[5] == PERSON_CLASS]
    balls = [b for b in boxes if b[5] == BALL_CLASS]
    for person in persons:
        for ball in balls:
            intersection = box_intersection(person, ball)
            if intersection >= 0.03:
                return ACTION
    return ABSTAIN


@labeling.labeling_function()
def lf_ball_at_bottom(frame_boxes):
    frame, boxes = frame_boxes
    persons = [b for b in boxes if b[5] == PERSON_CLASS]
    balls = [b for b in boxes if b[5] == BALL_CLASS]
    for person in persons:
        for ball in balls:
            intersection = box_intersection(person, ball)
            if intersection >= 0:
                person_height = person[3] - person[1]
                ball_center_y = (ball[3] + ball[1]) / 2
                if abs(person[3] - ball_center_y) <= person_height / 4:
                    # print(f'ball={ball[3]}, person={person[1]},{person[3]}')
                    return ACTION
    return ABSTAIN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.index, 'rb') as f:
        all_boxes = pickle.load(f)

    lfs = [
        lf_contains_ball,
        # lf_contains_person,
        lf_contains_person_ball,
        lf_ball_person_intersects,
        lf_ball_at_bottom
    ]

    applier = labeling.LFApplier(lfs=lfs)
    L_train = applier.apply(all_boxes)

    summary = labeling.LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(summary)

    majority_model = baselines.MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)

    preds = []
    for i, (frame, _) in enumerate(all_boxes):
        preds.append((frame, preds_train[i]))

    name = 'preds.pkl'
    with open(name, 'wb') as f:
        pickle.dump(preds, f)

    print(f'Saved to {name}.')
    # lm = label_model.LabelModel(cardinality=2, verbose=True)
    # lm.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
    # majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
    # print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

    # label_model_acc = lm.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
    # print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")


if __name__ == '__main__':
    main()
