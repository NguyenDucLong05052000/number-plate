import cv2
import numpy as np
import tensorflow as tf
import os
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import draw_bbox, load_yolo_weights, detect_image
from yolov3.configs import *


def detect_number_plate(image_path):
    if YOLO_TYPE == "yolov4":
        Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
    if YOLO_TYPE == "yolov3":
        Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
    load_yolo_weights(yolo, Darknet_weights) # use Darknet weights

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights

    image = detect_image(yolo, image_path, "", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_crop = detect_image(yolo, image_path, "", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))[1]
    image_crop = cv2.cvtColor(image_crop,cv2.COLOR_BGR2RGB)
    
    #plt.imshow(image_crop)
    #plt.show()
    return image_crop