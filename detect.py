from datetime import datetime

import tensorflow as tf
import numpy
import warnings
import pool_detector
from sort import *

warnings.filterwarnings('ignore')
from PIL import Image
import cv2

from object_detection.utils import label_map_util


PATH_TO_SAVED_MODEL = "graph5/saved_model"
print('Loading model...', end='')

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')



cap = cv2.VideoCapture('vids/2tr.mp4')
mot_tracker = Sort(max_age=70,
                   min_hits=1,
                   iou_threshold=0.2)


def detect(frame):
    lines, start_row, end_row = pool_detector.detect_objects(frame)
    image_np = numpy.array(frame)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] > 80) & (hsv[:, :, 0] < 130)
    hsv[mask, 0] -= 50
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    image_np = result
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)


    dets = detections["detection_boxes"][0][:]
    norm_dets = []
    for idx, det in enumerate(dets):
        if detections["detection_scores"][0][idx] > 0.6:
            norm_dets.append([float(det[0]) * 700, float(det[1]) * 1400, float(det[2]) * 700, float(det[3]) * 1400, 1])

    num_detections = int(detections.pop('num_detections'))
    norm_dets = ([i for i in norm_dets if i[2] > 400])
    norm_dets = numpy.array(norm_dets)

    if (len(norm_dets) != 0):
        trackers = mot_tracker.update(norm_dets)

        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections, trackers, lines, start_row, end_row
