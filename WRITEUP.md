# Project Write-Up

This is Write-Up for "People Counter App at the Edge using OpenVINO" project for my Intel® Edge AI for IoT Developers Nanodegree Program at Udacity.

## Explaining Custom Layers

Custom layers are layers that are not included in the list of known layers. If the topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom. 

In OpenVINO we can register new custom layers as extensions to the Model Optimizer. Or there are several other options specific to the Caffe, TensorFlow, MXNet, and ONIX libraries. This is described in [Custom Layers Guide](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html) on OpenVINO site

Some of the potential reasons for handling custom layers are:

 - implement missing layers new for cutting-edge topologies
 - to implement them in last layers as regular post-processing in application


Tutorial explains the flow and provides examples for custom layers you can find [here](https://software.intel.com/content/www/us/en/develop/articles/openvino-custom-layers-support-in-inference-engine.html)


## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app :

 - To precisely control the number of people inside indoors. Actual during the pandemic for stores
 - To measure people's interest in advertising screens. You can sell advertising to a client for real views.
 - It can be applied in security systems. To control a person’s presence in a specific area
 - In retail to control attention to specific products

Each of these use cases will be useful, because it is an "automatic" work that may be needed, but people don’t really like it. People need to have opportunity for more creative job.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. To get better results, consider the following factors:

 - Lighting: Bad lighting condition can gives significant impact on model accuracy. In most cases you need good light to get good accuracy. Idealy test data distributions, training data distributions and end user lighting condition shoud be rather close.
 - Model Accuracy: in many cases edge models is need to run in realtime. So usually you need to find balance speed/accuracy. High accurate model can requier a lot computer power. some times is good to test different IR precision FP32 or FP16 - this directly affects performance.
 - Camera focal Length and Image Size: Models give the best results if the camera image is close to the image parameters from the data set. Image size should be the same or higher than the resolution of the internal processing. Smaller image sizes degradate image and can give worse results. Too large image size requires more processing power to reduce. Make sure to use the input image mean/scale parameters (--scale and –mean_values) with the Model Optimizer when you need pre-processing. It allows the tool to bake the pre-processing into the IR to get accelerated by the Inference Engine. 
 - Camera position: In addition, the location and viewing angle of the camera can affect performance. If it is located too different from training data cameras, for example too high - you may get poor results

So main recommendatio is to test model on real user condition and real data. And re-traim model if need.


## Model Research


In investigating potential people counter models, I tried and succesfuly convert for IR format each of the following three models:


### Model 1 - YOLO v3

A deep CNN model for real-time object detection that detects 80 different classes. A little bigger than YOLOv2 but still very fast. As accurate as SSD but 3 times faster.

I get model from this YOLO v3 [repository](https://github.com/mystic123/tensorflow-yolo-v3). Conversion was based on Model Optimizer Developer Guide
[instructions](https://docs.openvinotoolkit.org/2020.3/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html)

#####Build instructions:

  
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

