import datetime
from tornado.concurrent import Future
import concurrent
from tornado import gen, httpclient, ioloop, queues
import json
import numpy as np
from threading import *   
from tornado.queues import Queue
from json import JSONEncoder
# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.modules.container import Sequential
import torch.nn as nn
import torchvision.models as models
import json
from json import JSONEncoder

import cv2    
import sys
import warnings
import base64
warnings.filterwarnings('ignore')


q = Queue(maxsize = 2)  

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

device = 'cpu'
r_50=models.resnet50(pretrained=True)
r_models=[r_50]
for r in r_models:
    net1=[]
    for module in r.children():
        
        if isinstance(module,Sequential):
            for m in module.children():
                net1.append(m)
        else:
            net1.append(module) 

split_point=0
net1=nn.Sequential(*net1)
left=list(net1.children())[:split_point]
resnet_left=nn.Sequential(*left)

resnet_left=resnet_left.eval().to(device)


async def consumer():
    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 100000000.0 ,request_timeout=100000000000.0))
    count = 0

    startTime = datetime.datetime.now()

    async for item in q:
        try:
            count += 1
            #print(item)
            item_np = item.detach().numpy()

            post_data = {'data': item_np, 'count': count}
            #print(f"Transferring data to the server: {count}")
            body = json.dumps(post_data, cls=NumpyArrayEncoder)
            a = sys.getsizeof(body)
            print(a)
            response = await http_client.fetch("http://localhost:8881/model",  method='POST', headers=None, body=body)
            load_data = json.loads(response.body)
            result = load_data['result']
            count_return = load_data['count']
            #print(f"Receiving result from Server: {count_return}")
            result_right_np = np.asarray(result)
            result_right = torch.Tensor(result_right_np)
            with torch.no_grad():
                output = torch.nn.functional.softmax(result_right, dim=1)
            
            #print(f"Final Result: {result_right}")
            results = utils.pick_n_best(predictions=output, n=1)
            #print(f"Final Result: {results}")
            print(f"Final Result for Image number {count}: {results}")
            endTime = datetime.datetime.now()
            print(f"Total processing time: {endTime - startTime}")

        finally:
            q.task_done()
            



def producer_video_left(img_rbg):

    count = 0
    
    img = Image.fromarray(img_rbg).convert('RGB')
    count+=1
    resize = transforms.Resize([224, 224])
    img = resize(img)

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)
    tensor = tensor.unsqueeze(0)
    tensor=tensor.to(device)
    out_left=tensor.cpu()

    return out_left

async def main_runner():
    ioloop.IOLoop.current().add_callback(consumer)    
    
    time_start = datetime.datetime.now()
    print('Timestamp Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    cam = cv2.VideoCapture('hdvideo.mp4')
    while True:
        ret, img_rbg = cam.read()   
        if ret: 
            with concurrent.futures.ThreadPoolExecutor() as pool:
                out_left = await io_loop.run_in_executor(pool, producer_video_left, img_rbg)
            await q.put(out_left)
        else:
            time_finish = datetime.datetime.now()
            print('Timestamp Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            break
    print(f'Timestamp difference: {time_finish - time_start}')
    cam.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    print("1")
    io_loop = ioloop.IOLoop.current()
    print("2")
    io_loop.add_callback(main_runner)     # Wait for producer to put all tasks.
    print("3")
    q.join() 
    print("4")
    io_loop.start()
    print("5")
    