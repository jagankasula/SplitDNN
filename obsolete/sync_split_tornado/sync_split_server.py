from http import client
import tornado.ioloop
import tornado.web
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


# Decides number of layers to run on client and server
split_point = 21

# Define the device type to run the model on. (cpu or cuda)
device = 'cpu:1'

# Load a pretrained ResNet-50 model.
r_50 = models.resnet50(pretrained=True)

# Add r_50 model in to a list of models.
r_models = [r_50]

# Append all the modules of the ResNet models in to a single list called net1.
for r in r_models:

    net1=[]

    for module in r.children():
        
        if isinstance(module,Sequential):
            for m in module.children():
                net1.append(m)
        else:
            net1.append(module) 


# Create a new sequential container by combining all the extracted modules from all models.
net1 = nn.Sequential(*net1)

# Split the container in to left/right for running on the client.
right = list(net1.children())[split_point:-1]

# Create a new sequential container for the right portion of the modules.
resnet_right = nn.Sequential(*right)

# Set the sequential container to evaluation mode and move it to a specific device.
resnet_right = resnet_right.eval().to(device)

# Define a class for serializing NumPy arrays.
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# API request handler.
class ProcessHandler(tornado.web.RequestHandler):

    # This method will be executed when the POST request is made by the client.
    def post(self):

        # Extract data from client request.
        data = json.loads(self.request.body)

        json_dump_return_data = model_right(data)        

        self.write(json_dump_return_data)

def model_right(data):
    x = data['data']
    count = data['count']
    #print(f"Receiving data from client: {count}")
    result_left_np=np.asarray(x)
    result_left = torch.Tensor(result_left_np)
    #print(result_left)
    print(f"Sending data to the right side of the model: {count}")


    out_right=resnet_right(result_left)
    out_right=out_right.cpu()
    print(f"Receiving data from right side of the model: {count}")
    result_right_np = out_right.detach().numpy()
    return_data = {'result':result_right_np, 'count':count}
    print(f"Transferring data back to the client: {count}")
    json_dump_return_data = json.dumps(return_data, cls = NumpyArrayEncoder)

    #print(out_right)
    return json_dump_return_data


def make_app():
    return tornado.web.Application([
        (r"/model", ProcessHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8881)
    print("Server started")
    tornado.ioloop.IOLoop.current().start()
