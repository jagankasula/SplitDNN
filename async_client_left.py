import concurrent
import datetime
import json
import sys
import warnings
import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
import torchvision.transforms as transforms

from json import JSONEncoder
from threading import Thread
from torch.nn.modules.container import Sequential
from PIL import Image
from json import JSONEncoder
from tornado import httpclient, ioloop
from tornado.concurrent import Future
from tornado.queues import Queue
from utils import Config, NumpyArrayEncoder, Logger, my_models

warnings.filterwarnings('ignore')

#Read configurations
config = Config.get_config()

device = config['client_device']
model_name = config['model']
split_point = config['split_point']
url = config['url']
frames_to_process = config['frames_to_process']

q = Queue(maxsize=2)  


startTime = datetime.datetime.now()
count = 0


torch_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

my_model = my_models.get(model_name, lambda: print(f"Model not present in my_models: {model_name}"))(pretrained=True)

sub_models=[my_model]

for sub_model in sub_models:
    all_modules = []

    for module in sub_model.children():
        if isinstance(module,Sequential):
            for m in module.children():
                all_modules.append(m)
        else:
            all_modules.append(module) 


all_modules = nn.Sequential(*all_modules)
new_list = list(all_modules.children())
Logger.log(f"******** size: {len(new_list)}" )
left_modules = list(all_modules.children())[:split_point]
Logger.log(f"******** size: {len(left_modules)}" )
resnet_left = nn.Sequential(*left_modules)

resnet_left = resnet_left.eval().to(device)

def handle_response(response):

    # Process the response here
    load_data = json.loads(response.body)
    result = load_data['result']
    count_return = load_data['count']
    result_right_np = np.asarray(result)
    result_right = torch.Tensor(result_right_np)

    with torch.no_grad():
        output = torch.nn.functional.softmax(result_right, dim=1)

    #results = torch_utils.pick_n_best(predictions=output, n=1)
    #print(f"Final Result: {results}")
    #print(f"Final Result for Image number {count}: {results}")
    endTime = datetime.datetime.now()
    time = (endTime - startTime) * 8
    Logger.log(f"Total processing time: {count_return} time {time}")


async def consumer():

    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    global count
    count = 0

    global startTime
    startTime = datetime.datetime.now()

    async for item in q:

        try:            
            count += 1
            Logger.log("consumer: " + str(count))
            item_np = item.detach().numpy()

            post_data = {'data': item_np, 'count': count}
            body = json.dumps(post_data, cls=NumpyArrayEncoder)
            a = sys.getsizeof(body)
            response =  http_client.fetch(url,  method='POST', headers=None, body=body)
            response.add_done_callback(lambda f: handle_response(f.result()))            

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

    out_left=resnet_left(tensor)
    out_left=out_left.cpu()
    
    return out_left

async def main_runner():
    ioloop.IOLoop.current().add_callback(consumer)    
    
    time_start = datetime.datetime.now()
    print('Timestamp Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    cam = cv2.VideoCapture('hdvideo.mp4')
    frame_count = 0
    while frame_count < frames_to_process:        
        frame_count += 1
        Logger.log('main_runner: ' + str(frame_count))
        ret, img_rbg = cam.read()   
        if ret: 
            with concurrent.futures.ThreadPoolExecutor() as pool:
                out_left = await io_loop.run_in_executor(pool, producer_video_left, img_rbg)
            await q.put(out_left)
        else:
            time_finish = datetime.datetime.now()
            print('Timestamp Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            break
    time_finish = datetime.datetime.now()
    print('Timestamp Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    print(f'Timestamp difference: {time_finish - time_start}')
    cam.release()
    cv2.destroyAllWindows()


if __name__=='__main__':

    print('Initialize IOLoop')
    io_loop = ioloop.IOLoop.current()

    print('Add main_runner to Tornado event loop as call back')
    io_loop.add_callback(main_runner)     # Wait for producer to put all tasks.

    print('Join the queue')
    q.join() 

    print('Start IOLoop')
    io_loop.start()

    print('After start IOLoop')
    