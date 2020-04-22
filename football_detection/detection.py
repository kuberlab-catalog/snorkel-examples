from ml_serving.drivers import driver
import numpy as np


def detect_bboxes_tensorflow(drv: driver.ServingDriver, frame: np.ndarray,
                             threshold: float = 0.5, offset=(0, 0), only_classes=None):
    input_name, input_shape = list(drv.inputs.items())[0]
    inference_frame = np.expand_dims(frame, axis=0)
    outputs = drv.predict({input_name: inference_frame})
    boxes = outputs["detection_boxes"].copy().reshape([-1, 4])
    scores = outputs["detection_scores"].copy().reshape([-1])
    scores = scores[np.where(scores > threshold)]
    boxes = boxes[:len(scores)]
    classes = np.int32((outputs["detection_classes"].copy())).reshape([-1])
    classes = classes[:len(scores)]
    if only_classes is not None:
        indices = np.isin(classes, np.array(only_classes))
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]
    boxes[:, 0] *= frame.shape[0] + offset[0]
    boxes[:, 2] *= frame.shape[0] + offset[0]
    boxes[:, 1] *= frame.shape[1] + offset[1]
    boxes[:, 3] *= frame.shape[1] + offset[1]
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # .astype(int)

    # add probabilities
    confidence = np.expand_dims(scores, axis=0).transpose()
    classes = np.expand_dims(classes, axis=0).transpose()
    boxes = np.concatenate((boxes, confidence, classes), axis=1)

    return boxes
