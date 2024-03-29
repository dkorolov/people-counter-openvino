#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        
    def cleanup(self):
        del self.net_plugin
        del self.plugin
        del self.net

    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension=None, plugin=None):
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device
        # and load extensions library if specified
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IEPlugin(device=device)
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)

        log.info("Reading IR...")
        self.net = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR to the plugin...")
        
        ### Check for supported layers ###
        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.net)
    
        
            ### Add any necessary extensions ###
            not_supported_layers = \
                [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by "
                          "the plugin for specified device {}:\n {}".
                          format(self.plugin.device,
                                 ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)
        
        ### Return the loaded inference plugin ###
        if num_requests == 0:
            # Loads network read from IR to the plugin
            self.net_plugin = self.plugin.load(network=self.net)
        else:
            self.net_plugin = self.plugin.load(network=self.net, num_requests=num_requests)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
               
        ### Note: You may need to update the function parameters. ###
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        ### Start an asynchronous request ###
        self.infer_request_handle = self.net_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame})
        
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.net_plugin

    def wait(self, request_id):
        ### Wait for the request to be complete. ###
        wait_process = self.net_plugin.requests[request_id].wait(-1)
        
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return wait_process

    def get_output(self, request_id, output_type):
        ### Extract and return the output results
        if output_type in ["yolo", "yolo_tiny"]:
            res = self.net_plugin.requests[0].outputs
        elif output_type == "ssd":
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        else:
            res = self.infer_request_handle.outputs[output]
            
        ### Note: You may need to update the function parameters. ###
        return res

