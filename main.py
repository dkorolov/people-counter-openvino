"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from yoloparams import YoloParams, parse_yolo_region, intersection_over_union

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-o", "--output_type", type=str, default="yolo",
                        help="Output type - yolo, yolo_tiny or ssd"
                        "(yolo by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def statistic(total_count, current_count, prev_count, prev_in_duration, duration, mqtt):
    '''
    Calculate statistics and publish it to MQTT
    '''
    # Number of seconds threshold that we count that person won't leave view
    # Use constant here, but possible to add argument if need
    leave_threshold = 2
    
    in_duration = prev_in_duration
    if ((duration - prev_in_duration) < leave_threshold):
        if(current_count > prev_count):
            return total_count, prev_count, prev_in_duration
    
        elif((current_count < prev_count)):
            # update current_count if see less people
            mqtt.publish("person", json.dumps({"count": current_count}))
            return total_count, current_count, duration

    # Check for new people in view
    if(current_count > prev_count):
        in_duration = duration
        # increase total counter
        total_count = total_count + current_count - prev_count
        mqtt.publish("person", json.dumps({"total": total_count}))


    # Check if people left the view
    elif(current_count < prev_count):
        leave_duration = duration
        # how many time people was in view
        duration_in_view = leave_duration - prev_in_duration
        mqtt.publish("person/duration", json.dumps({"duration": duration_in_view}))

    mqtt.publish("person", json.dumps({"count": current_count}))
    return total_count, current_count, in_duration


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    global prob_threshold, output_type
    prob_threshold = args.prob_threshold
    output_type = args.output_type
    assert (output_type in ["yolo", "yolo_tiny", "ssd"]), "Wrong output_type - {}".format(output_type)
    
    image_mode = False
    cur_request_id = 0
    last_count = 0
    total_count = 0
    prev_count = 0
    prev_in_duration = 0
    frame_count = 0
    start_time = 0
    cur_frame = 0
    total_inf_time = 0
    
    ### Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 3,
                                          cur_request_id,
                                          args.cpu_extension)[1]
    
    ### Handle the input stream ###
    # live stream input
    if args.input == 'CAM':
        input_stream = 0

    # image input
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        image_mode = True
        input_stream = args.input

    # video file input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist {}".format(args.input)

    cap = cv2.VideoCapture(input_stream)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    global initial_w, initial_h
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    
    duration = 0
    person_frames = 0
    
    ### Loop until stream is over ###
    while cap.isOpened():
        
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### Pre-process the image as needed ###
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        ### Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(cur_request_id, image)
        
        ### Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            inf_time = time.time() - inf_start
            
            frame_count += 1
            cur_frame += 1
            total_inf_time += inf_time
            
            ### Get the results of the inference request ###
            result = infer_network.get_output(cur_request_id, output_type)
            
            if output_type == "yolo":
                frame, current_count = yolo_out(result, infer_network.net, frame, image, False)
            elif output_type == "yolo_tiny":
                frame, current_count = yolo_out(result, infer_network.net, frame, image, True)
            elif output_type == "ssd":
                frame, current_count = ssd_out(frame, result)
                
            
            ### Extract any desired stats from the results ###
            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
             
            total_count, prev_count, prev_in_duration = statistic(total_count, current_count, prev_count, prev_in_duration, frame_count/FPS, client)

            # Print information on screen
            inf_time_string = "Inference time: {:.3f}ms"\
                               .format(inf_time * 1000)
            cv2.putText(frame, inf_time_string, (75, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, "Total Count : %d" %total_count, (75, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 10, 10), 1)

            
            last_count = current_count
            
            if key_pressed == 27:
                break
            
            
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### Write an output image if `image_mode` ###
        if image_mode:
            cv2.imwrite('output_image.jpg', frame)
                
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.cleanup()

    # Print average inf_time for statistics
    #average_inf_time = total_inf_time/cur_frame
    #print ("average inf time for frame : {}".format(average_inf_time))
    
def ssd_out(frame, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            current_count = current_count + 1
    return frame, current_count



def yolo_out(result, network, frame, in_frame, is_tiny):
    objects = list()
    for layer_name, out_blob in result.items():
        out_blob = out_blob.reshape(network.layers[network.layers[layer_name].parents[0]].shape)
        layer_params = YoloParams(network.layers[layer_name].params, out_blob.shape[2], is_tiny)
        objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                        frame.shape[:-1], layer_params, prob_threshold)
    

    # Set default for Intersection over union threshold for overlapping (iou_threshold)
    # It possible to to put this in paraneters if need
    iou_threshold = 0.4
    
    # Filtering overlapping boxes with respect to the iou_threshold parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    # Filter objects with respect to the --prob_threshold CLI parameter
    # AND filter objects if their size out of original frame size
    origin_im_size = frame.shape[:-1]
    bboxes = [obj for obj in objects if obj['confidence'] >= prob_threshold and
                    (obj['xmax'] <= origin_im_size[1] or
                     obj['ymax'] <= origin_im_size[0] or
                     obj['xmin'] >= 0 or
                     obj['ymin'] >= 0)]

    #Draw bounding boxes and inference time onto the frame.
    current_count = 0
    for box in bboxes: # Output shape is 1x1x100x7
        conf = box['confidence']
        # class 0 means person
        if  conf >= prob_threshold and box['class_id'] == 0:
            xmin = box['xmin']
            ymin =  box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            current_count += 1
      
    return frame ,current_count

    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
