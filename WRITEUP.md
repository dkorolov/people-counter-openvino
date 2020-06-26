# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research


In investigating potential people counter models, I tried and succesfuly convert for IR format each of the following three models:


### Model 1 - YOLO v3

A deep CNN model for real-time object detection that detects 80 different classes. A little bigger than YOLOv2 but still very fast. As accurate as SSD but 3 times faster.

I get model from this YOLO v3 [repository](https://github.com/mystic123/tensorflow-yolo-v3). Conversion was based on Model Optimizer Developer Guide
[instructions](https://docs.openvinotoolkit.org/2020.3/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html)
  - #####Build instructions:


  
```
# Clone the YOLO v3 repository
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
# Instructions base on commit ed60b90, so I checkout to it
cd tensorflow-yolo-v3
git checkout ed60b90
	
# Download COCO class names file
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
	
#Download binary file with weights (exist 3 option, I use only first)
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://pjreddie.com/media/files/yolov3-spp.weights
	
	
# I will use only yolov3 in next steps
# Convert .weights file to a .pb file
python convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
	
# Convert model to IR format using Model Optimizer
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --batch 1
	
```
During .weights to a .pb convertion I get an several errors like:
	
```
AttributeError: module 'tensorflow.python.ops.nn' has no attribute 'leaky_relu'
and
No module PIL
```
This fixed by installing tensorflow v1.12 and Pillow:
	
```
pip install tensorflow==1.12
pip install numpy --upgrade
pip install Pillow
```
	
	
Was generated .xml and .bin files:
	
```
frozen_darknet_yolov3_model.xml
frozen_darknet_yolov3_model.bin
```


### Model 2 - ssd mobilenet v2 coco

Single Stage Detector: real-time CNN for object detection that detects 80 different classes.

I get original model from Tensorflow [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
#####Build instructions:

```
# Download model from Tensorflow detection model zoo and extract it
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

# Convert model to IR format using Model Optimizer
cd ssd_mobilenet_v2_coco_2018_03_29
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

```
Was generated .xml and .bin files:

```
frozen_inference_graph.xml
frozen_inference_graph.bin
```


### Model 3 - faster rcnn inception v2 coco

This is a real-time neural network for object instance segmentation that detects 80 different classes. Extends Faster R-CNN as each of the 300 elected ROIs go through 3 parallel branches of the network: label prediction, bounding box prediction and mask prediction.

I get original model from Tensorflow [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
#####Build instructions:

```
# Download model from Tensorflow detection model zoo and extract it
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

# Convert model to IR format using Model Optimizer
cd faster_rcnn_inception_v2_coco_2018_01_28
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

```
Was generated .xml and .bin files:

```
frozen_inference_graph.xml
frozen_inference_graph.bin
```

