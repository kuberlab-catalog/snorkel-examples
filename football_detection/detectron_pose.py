import logging

import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np
import torch

from football_detection import utils


LOG = logging.getLogger(__name__)


class DetectronPose(object):
    def __init__(self, model_path):
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = 'cpu'

        cfg.MODEL.WEIGHTS = model_path
        self.cfg = cfg
        LOG.info(f'Loading detectron pose model from {model_path}...')
        self.predictor = DefaultPredictor(cfg)
        LOG.info('Done.')

    def predict(self, img):
        outputs = self.predictor(img)
        fields = outputs['instances'].get_fields()
        for k, v in fields.items():
            if hasattr(v, 'tensor'):
                fields[k] = v.tensor.detach().cpu().numpy()
            else:
                fields[k] = v.detach().cpu().numpy()

        return fields

    def draw_predictions(self, img_rgb, predictions):
        v = Visualizer(img_rgb, MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        labels = [str(p) for p in predictions.get('scores', [])]
        v.overlay_instances(
            boxes=predictions.get('pred_boxes'),
            labels=labels if labels else None,
            keypoints=predictions.get('pred_keypoints'),
        )
        return v.get_output().get_image()

    @staticmethod
    def get_leg_bounds(keypoints):
        right_leg = [13, 15]
        left_leg = [14, 16]
        pairs = [left_leg, right_leg]
        if isinstance(keypoints, dict):
            keypoints = keypoints['pred_keypoints']

        boxes = []
        for instance in keypoints:
            for pair in pairs:
                point1_xy = instance[pair[0]]
                point2_xy = instance[pair[1]]
                if point1_xy[0] > point2_xy[0]:
                    point1_xy[0], point2_xy[0] = point2_xy[0], point1_xy[0]
                if point1_xy[1] > point2_xy[1]:
                    point1_xy[1], point2_xy[1] = point2_xy[1], point1_xy[1]
                boxes.append((point1_xy[0], point1_xy[1], point2_xy[0], point2_xy[1]))

        return np.stack(boxes).astype(np.int)

    @staticmethod
    def expand_box(box):
        w = abs(box[2] - box[0])
        h = abs(box[3] - box[1])
        max_dim = max(w, h)
        new_box = [box[0] - max_dim, box[1] - max_dim, box[2] + max_dim, box[3] + max_dim]
        return np.stack(new_box)

    @classmethod
    def intersects_with_expanded_legs(cls, keypoints, target_box):
        leg_boxes = cls.get_leg_bounds(keypoints)
        for box in leg_boxes:
            expanded_box = cls.expand_box(box)
            if utils.box_intersection(expanded_box, target_box) > 0:
                return True

        return False


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    img_path = sys.argv[2]
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)
    det = DetectronPose(path)
    im = cv2.imread(img_path)
    LOG.info('Run predict...')
    detections = det.predict(im)
    LOG.info('Done predict.')
    new_im = det.draw_predictions(im[:, :, ::-1], detections)
    ball_box = [68, 495, 145, 569]
    intersects = det.intersects_with_expanded_legs(detections, ball_box)
    leg_boxes = det.get_leg_bounds(detections)
    for box in leg_boxes:
        cv2.rectangle(new_im, (box[0], box[1]), (box[2], box[3]), (250, 0, 0), lineType=cv2.LINE_AA)

    cv2.rectangle(new_im, (ball_box[0], ball_box[1]), (ball_box[2], ball_box[3]), (0, 250, 0), lineType=cv2.LINE_AA)
    print(f'Ball intersects with legs: {"yes" if intersects else "no"}')
    cv2.imshow('Img', new_im[:, :, ::-1])
    cv2.waitKey(0)
