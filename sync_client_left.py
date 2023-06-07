import cv2
import requests
import json
import numpy as np
import torch
import datetime
from json import JSONEncoder
from PIL import Image
from torch.nn.modules.container import Sequential
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import csv
import sys

# Decides number of layers to run on client and server
split_point = 21

# URL for calling the server.
url = "http://KC-SCER-FCCT0M3:8881/"

# Get pytorch utilities.
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

# Define the device type to run the model on. (cpu or cuda)
device = 'cpu'

# Load a pretrained ResNet-50 model.
r_50 = models.resnet101(pretrained=True)

# Add r_50 model in to a list of models.
r_models = [r_50]

# Append all the modules of the ResNet models in to a single list called net1.
for r in r_models:
    net1 = []

    for module in r.children():
        if isinstance(module, Sequential):
            for m in module.children():
                net1.append(m)
        else:
            net1.append(module)

# Create a new sequential container by combining all the extracted modules from all models.
net1 = nn.Sequential(*net1)

def get_resnet_left(split_point):
    # Split the container in to left/right for running on the client.
    left = list(net1.children())[:split_point]

    # Create a new sequential container for the left portion of the modules.
    resnet_left = nn.Sequential(*left)

    # Set the sequential container to evaluation mode and move it to a specific device.
    resnet_left = resnet_left.eval().to(device) 

    return resnet_left

# Define a class for serializing NumPy arrays.
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Execution 1: This is the function called by the driver code.
def main_runner():

    for split_point in range(0, 1):

        print('SPLIT: ' + str(split_point))
        # Get resnet model after splitting
        resnet_left = get_resnet_left(split_point)

        # Request JSON.
        request_json = get_request_body(None, 0, split_point)
        send_request(request_json, 'split_point')

        #Initialize total time taken for split_point
        total_time_for_split_point = 0

        # Frame number.
        count = 0

        # Capture the start time of the execution.
        start_time = datetime.datetime.now()

        # Read the input file.
        cam = cv2.VideoCapture('hdvideo.mp4')

        # Process all the frames in the video. (cam.isOpened(), returns True as long as the video capture is open and there are more frames to read)
        while cam.isOpened() and count < 50:
            # ret: Boolean, is the frame successfully read.
            ret, frame = cam.read()
            if ret:
                # Increment the frame number.
                count += 1

                # Process the tensor in left modules.
                left_output = producer_video_left(frame, resnet_left)

                # Request JSON.
                request_json = get_request_body(left_output, count, split_point)

                # Send request to server for processing.
                response = send_request(request_json, 'model')

                # Extract data from server response.
                response_data = response.json()

                # Get the result of the processed frame.
                result = response_data['result']

                # Create a numpy array of the result for converting it into tensor in later steps.
                result_np = np.asarray(result)

                # Convert result_np into tensor.
                result_tensor = torch.Tensor(result_np)

                # Remove gradients and perform softmax on the result_tensor.
                with torch.no_grad():
                    output = torch.nn.functional.softmax(result_tensor, dim=1)

                #final_results = utils.pick_n_best(predictions=output, n=1)

                #print(f"Final Result for Image number {count}: {final_results}")

                end_time = datetime.datetime.now()

                total_time_for_split_point = end_time - start_time

                print(f"Total processing time: {end_time - start_time}")

            else:
                break
            
        write_to_csv('output.csv', split_point, (total_time_for_split_point) * 8, sys.getsizeof(request_json))

        cam.release()
        cv2.destroyAllWindows()


# Execution 2: This function is called to convert image to tensor and process the tensor in left modules.
def producer_video_left(image, resnet_left):
    # Convert image to RGB form.
    image_rgb = Image.fromarray(image).convert('RGB')

    # Resize the image.
    resize = transforms.Resize([224, 224])
    image_rgb = resize(image_rgb)

    # Transform the image into a tensor.
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image_rgb)

    # Add a new dimension of size 1 to tensor. (Align tensor dimension to pass to the resnet model.)
    tensor = tensor.unsqueeze(0)

    # Process the tensor through left layers.
    left_output = resnet_left(tensor)

    # Convert tensor into CPU format. (Default PyTorch tensor is of CPU type. No effect of .cpu if the tensor is already in CPU format.)
    left_output = left_output.cpu()

    return left_output


# Returns JSON body for sending to server.
def get_request_body(left_output, count, split_point):
    # Convert the tensor to numpy array for sending it to the server.
    left_output_np = None
    if(left_output != None):
        left_output_np = left_output.detach().numpy()

    # Request JSON.
    request_body = {'data': left_output_np, 'count': count, 'split_point': split_point}

    return request_body


# Send HTTP request to server.
def send_request(request_body, endpoint):
    # JSON serialize request body.
    request_body_serialized = json.dumps(request_body, cls=NumpyArrayEncoder)

    # Set the Content-Type header to indicate JSON data.
    headers = {'Content-Type': 'application/json'}

    # Send POST request to the server.
    response = requests.post(url + endpoint, headers=headers, data=request_body_serialized)

    return response


def write_to_csv(filename, data1, data2, size):
    # Check if the file exists
    file_exists = False
    try:
        with open(filename, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in the appropriate mode
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        # Write a new line if the file is empty
        if not file_exists:
            writer.writerow(['Split no.', 'Total time taken', 'size'])  # Example column headers

        # Write the data to the file
        writer.writerow([data1, data2, size])


# Driver code.
if __name__ == '__main__':
    main_runner()
