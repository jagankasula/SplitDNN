# server_right_cpu.py

The provided code is a Python script that sets up a web server using the Tornado framework. The server includes several request handlers to handle different HTTP requests. Let's break down the code into its main components:

### Import Statements:

The code begins by importing the necessary modules and libraries, including http.client, unittest.result, tornado, json, numpy, concurrent, torch, torchvision, PIL, and warnings.
It also imports specific classes and functions from some modules.

### Custom JSON Encoder:
The code defines a custom JSONEncoder class called NumpyArrayEncoder. It overrides the default method to handle numpy arrays by converting them to lists before serializing them into JSON.
Flatten Module:

The code defines a custom module called Flatten, which is a subclass of nn.Module.
This module is used to flatten the input tensor by reshaping it to have a size of (batch_size, -1).

### Loading Pretrained ResNet Model:

* The code loads a pretrained ResNet-50 model from torchvision.models.
* It appends the layers of the ResNet model to a list called net1, excluding the last fully connected layer.
* It then creates an instance of nn.Sequential using the layers in net1, and assigns it to the variable resnet_right.
* The Flatten module is appended after resnet_right to flatten the output before the last layer.
* The device is set to 'cpu:1' (a specific CPU device) using torch.distributed.rpc.

### Request Handlers:

The code defines three request handler classes that inherit from tornado.web.RequestHandler.
* `MainHandler` handles the root path / and responds with "Hello, world" when a GET request is made.
ComputeHandler handles requests to the /compute path. When a POST request is made, it receives JSON data, calculates the square of the input number, and responds with the result as JSON.
* `ModelHandler` handles requests to the /model path. When a POST request is made, it receives JSON data, and asynchronously passes it to the model_right function for processing. The response is written back to the client.
`model_right` 

* Function:

     * This function processes the input data received from the client.
    * It converts the input data to a NumPy array and then to a PyTorch tensor.
    * It passes the tensor through the resnet_right model (previously defined) and obtains the output.
    * The output tensor is converted to a NumPy array and then serialized to JSON using the NumpyArrayEncoder class.
    * The resulting JSON data is returned.


### App Initialization and Server Start:

* The make_app function creates a Tornado web application by mapping the defined request handlers to specific paths.
* In the if __name__ == "__main__": block, the app is created, and the server starts listening on port 8881 using app.listen.
* The server starts running and waits for incoming requests.

This code sets up a web server that can handle HTTP requests for computing the square of a number (/compute endpoint) and processing data through a pre-trained ResNet model (/model endpoint) using Tornado and PyTorch.




# client_lift_cpu.py

### Import Statements:

* The code begins by importing various modules and libraries required for the application, such as datetime, tornado, json, numpy, torch, torchvision, cv2, sys, and warnings.
* It also imports specific classes and functions from some modules.

### Queue Initialization:

* The code initializes a Queue object named q with a maximum size of 2.

* The purpose of this queue is to hold processed video frames before they are consumed by the consumer function.
NVIDIA ConvNets Processing Utils:

### Steps: 
* 
* The code uses the torch.hub module to load the "nvidia_convnets_processing_utils" package from the NVIDIA DeepLearningExamples repository.
* This package provides utility functions for working with NVIDIA's convolutional neural network models.
Custom JSON Encoder:

* The code defines a custom JSONEncoder class called NumpyArrayEncoder. It overrides the default method to handle numpy arrays by converting them to lists before serializing them into JSON.
Device Selection:

* The code sets the variable device to 'cpu', indicating that the computation will be performed on the CPU.
Note that this code does not involve GPU utilization.
Loading Pretrained ResNet Model:

* The code loads a pretrained ResNet-50 model from torchvision.models.
* It appends the layers of the ResNet model to a list called net1.
* The layers are extracted using net1.children().
* The code then creates an instance of nn.Sequential using the layers in net1, and assigns it to the variable resnet_left.
* The resnet_left model is set to evaluation mode (.eval()) and moved to the specified device (CPU).

### Consumer Function:

* The consumer function is defined as an asynchronous function.
* It uses the httpclient.AsyncHTTPClient to make asynchronous HTTP requests to a server.
* The function loops through items in the q queue, which holds processed video frames.
* It converts the item to a NumPy array and constructs a JSON payload with additional data.
* The payload is sent to the server using an HTTP POST request.
* The response from the server is processed and the result is printed.

### Producer Function:

* The producer_video_left function takes an RGB image as input and processes it using the resnet_left model.
* The image is converted to grayscale, resized, converted to a tensor, and passed through the resnet_left model.
* The output of the model is returned.

### Main Runner:

* The main_runner function is defined as an asynchronous function.
* It adds the consumer function to the IOLoop for execution.
* It opens a video file using OpenCV's cv2.VideoCapture.
* It loops through the video frames, calling the producer_video_left function to process each frame.
* The processed output is put into the q queue for the consumer to consume.
* The loop continues until all frames have been processed.
* Finally, the IOLoop is started.

### IOLoop and Application Execution:

In the __main__ block, the IOLoop is created, and the main_runner function is added to the IOLoop for execution# SplitDNN
