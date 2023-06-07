from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import datetime
import numpy as np
import torch.nn as nn
from torch.nn.modules.container import Sequential
from json import JSONEncoder


# Define the device type to run the model on. (cpu or cuda)
device = 'cuda:1'

# Load a pretrained ResNet-50 model.
r_50 = models.resnet50(pretrained=True)

# Add r_50 model into a list of models.
r_models = [r_50]

# Append all the modules of the ResNet models into a single list called net1.
for r in r_models:
    net1=[]

    for module in r.children():
        if isinstance(module, Sequential):
            for m in module.children():
                net1.append(m)
        else:
            net1.append(module)

# Create a new sequential container by combining all the extracted modules from all models.
net1 = nn.Sequential(*net1)

resnet_right = None

# Define a class for serializing NumPy arrays.
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Request handler class
class MyRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        if self.path.startswith('/model'):            
            json_dump_return_data = model_right(data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json_dump_return_data.encode('utf-8'))

        if self.path.startswith('/split_point'):
            set_resnet_right(int(data['split_point']))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write('Model split successful'.encode('utf-8'))
            

def set_resnet_right(split_point):
    global resnet_right
    # Split the container into left/right for running on the client.
    right = list(net1.children())[split_point:-1]

    # Create a new sequential container for the right portion of the modules.
    resnet_right = nn.Sequential(*right)

    # Set the sequential container to evaluation mode and move it to a specific device.
    resnet_right = resnet_right.eval().to(device)

def model_right(data):
    x = data['data']
    count = data['count']
    result_left_np = np.asarray(x)
    result_left = torch.Tensor(result_left_np)
    print(f"Sending data to the right side of the model: {count}")

    out_right = resnet_right(result_left.to('cuda'))
    out_right = out_right.cpu()
    print(f"Receiving data from right side of the model: {count}")
    result_right_np = out_right.detach().numpy()
    return_data = {'result': result_right_np, 'count': count}
    print(f"Transferring data back to the client: {count}")
    json_dump_return_data = json.dumps(return_data, cls=NumpyArrayEncoder)
    return json_dump_return_data

# Create an HTTP server with the request handler
server_address = ('', 8881)
httpd = HTTPServer(server_address, MyRequestHandler)

# Start the server
print('Server started')
httpd.serve_forever()