from http import client
from unittest import result
import tornado.ioloop
import tornado.web
import json
import numpy as np

import concurrent
from tornado import gen, httpclient, ioloop, queues

import json
from json import JSONEncoder

import os
import torch.distributed.rpc as rpc
import torch

import torchvision.transforms as transforms
from PIL import Image
from torch.nn.modules.container import Sequential
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig

# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

import warnings
warnings.filterwarnings('ignore')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_model = vit_model.eval().to(device)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

class ComputeHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Computation done")
    def post(self):
        data = json.loads(self.request.body)
        #print(data['data'])
        x = data['data']
        y = int(x)**2
        return_data = {'result':y}
        json_dump_return_data = json.dumps(return_data)
        self.write(json_dump_return_data)

io_loop = ioloop.IOLoop.current()

class ModelHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Model done")
    async def post(self):
        #data = json.loads(self.request.body)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            data = await io_loop.run_in_executor(pool, json.loads, self.request.body)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            json_dump_return_data = await io_loop.run_in_executor(pool, model_right, data)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await io_loop.run_in_executor(pool, self.write, json_dump_return_data)

def model_right(data):
    x = data['data']
    count = data['count']
    #print(f"Receiving data from client: {count}")
    result_left_np = np.asarray(x)
    result_left = torch.Tensor(result_left_np).unsqueeze(0).to(device)
    #print(result_left)
    print(f"Sending data to the right side of the model: {count}")

    with torch.no_grad():
        out_right = vit_model(result_left)

    out_right = out_right.cpu()
    print(f"Receiving data from right side of the model: {count}")
    result_right_np = out_right.detach().numpy()
    return_data = {'result':result_right_np, 'count':count}
    print(f"Transferring data back to the client: {count}")
    json_dump_return_data = json.dumps(return_data, cls=NumpyArrayEncoder)

    #print(out_right)
    return json_dump_return_data


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/compute", ComputeHandler),
        (r"/model", ModelHandler)
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8881)
    print("Got a Call")
    tornado.ioloop.IOLoop.current().start()
