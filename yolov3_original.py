# Based on : https://github.com/mystic123/tensorflow-yolo-v3/blob/master/demo.py
# -*- coding: utf-8 -*-

import os
import sys
import socket
import json

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_video', '', 'Input video')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    
    # img = Image.open(FLAGS.input_img)
    # img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
    # img_resized = img_resized.astype(np.float32)
    classes = load_coco_names(FLAGS.class_names)

    # if FLAGS.frozen_model:

    t0 = time.time()
    frozenGraph = load_graph(FLAGS.frozen_model)
    print("Loaded graph in {:.2f}s".format(time.time()-t0))

    boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

    ### Start inference on Video
    cap = cv2.VideoCapture(FLAGS.input_video)
    cap.open(FLAGS.input_video)
    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
        
    with tf.Session(graph=frozenGraph, config=config) as sess:
        while cap.isOpened():
            flag, img = cap.read()
            if not flag:
                break
            key_pressed = cv2.waitKey(27)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Image.open(FLAGS.input_video)
            # convert from cv2 image to PIL image
            img = Image.fromarray(img)
            img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
            img_resized = img_resized.astype(np.float32)
            classes = load_coco_names(FLAGS.class_names)
            
            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})
            infer_time = time.time() - t0

            filtered_boxes = non_max_suppression(detected_boxes,
                                                confidence_threshold=FLAGS.conf_threshold,
                                                iou_threshold=FLAGS.iou_threshold)

            draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)

            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.putText(img,"infer time= "+str('{:.1f}'.format(infer_time*1000)) +" ms", (80,40), 0, 0.5, (250,0,0),1)

            
            ### Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(img)
            sys.stdout.flush()
            
            # Break if escape key pressed
            if key_pressed == 27:
                break

    # Release the out capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
