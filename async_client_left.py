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
from tornado import httpclient, ioloop
from tornado.concurrent import Future
from tornado.queues import Queue
from utils import Config, NumpyArrayEncoder, Logger, my_models


# Ignore the warnings generated by the code.
warnings.filterwarnings('ignore')

# Read the configurations from the config file.
config = Config.get_config()

# Assign the configurations to the global variables.
device = config['client_device']
model_name = config['model']
split_point = config['split_point']
url = config['url']
frames_to_process = config['frames_to_process']

# Initialize queue for storing the output of the frames that were processed on the client (left) side.
q = Queue(maxsize=2)  

# Initialize the start time to None. This value will be set in main_runner when it is initialized.
start_time = None

# Initialize consumer start time to None. This value will be set in consumer when http request is sent to server.
consumer_startTime = None

# Initialize frame_count. This variable will be used for tracking the number of frames processed.
frame_count = 1

# Consumer frame count. This value might be different from frame_count as the consumer runs asynchronously.
consumer_frame_count = 1

# Track total responses handled.
total_handled_responses = 0


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
left_modules = list(all_modules.children())[:split_point]
resnet_left = nn.Sequential(*left_modules)

resnet_left = resnet_left.eval().to(device)

def handle_response(response):

    global total_handled_responses

    total_handled_responses += 1

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
    end_time = datetime.datetime.now()
    time = (end_time - start_time) * 8

    Logger.log(f'[Inside handle_response] [FRAME: {count_return}] Total processing time: {time}')

    if total_handled_responses == frames_to_process:
        Logger.log(f'TOTAL TIME FOR PROCESSING:: {time}')


async def consumer():

    global consumer_frame_count

    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    
    consumer_startTime = datetime.datetime.now()

    Logger.log(f'[Inside consumer] CONSUMER START TIME.')

    async for item in q:

        try:            
            Logger.log(f'[Inside consumer] [Frame: {consumer_frame_count}] Preparing body to send request to server.')
            item_np = item.detach().numpy()

            post_data = {'data': item_np, 'count': consumer_frame_count}
            body = json.dumps(post_data, cls = NumpyArrayEncoder)

            # Sending HTTP request to server.
            Logger.log(f'[Inside consumer] [Frame: {consumer_frame_count}] Send HTTP request to server.')
            response =  http_client.fetch(url,  method = 'POST', headers = None, body = body)

            consumer_frame_count += 1

            response.add_done_callback(lambda f: handle_response(f.result()))            

        finally:
            q.task_done()

    consumer_endTime = datetime.datetime.now()

    Logger.log(f'[Inside consumer] CONSUMER END TIME.')

    Logger.log(f'[Inside consumer] TOTAL TIME TAKEN BY CONSUMER:: {consumer_endTime - consumer_startTime}')
            

def producer_video_left(img):
    
    tensor = convert_image_to_tensor(img)

    local_start_time = datetime.datetime.now()

    Logger.log(f'[Inside producer_video_left] [FRAME: {frame_count}] Start time of frame processing in left side.')

    out_left = resnet_left(tensor)
    out_left = out_left.cpu()

    local_end_time = datetime.datetime.now()

    Logger.log(f'[Inside producer_video_left] [FRAME: {frame_count}] End time of frame processing in left side.')

    Logger.log(f'[Inside producer_video_left] [FRAME: {frame_count}] Total time taken to process frame in left side:: {local_end_time - local_start_time}')
    
    return out_left

def convert_image_to_tensor(img):

    Logger.log(f'[Inside convert_image_to_tensor]:: start line')
    local_start_time = datetime.datetime.now()

    img_rgb = Image.fromarray(img).convert('RGB')
    resize = transforms.Resize([224, 224])
    img_rgb = resize(img_rgb)

    Logger.log(f'[Inside convert_image_to_tensor]:: after resize')
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img_rgb)
    Logger.log(f'[Inside convert_image_to_tensor]:: after to_tensor')
    tensor = tensor.unsqueeze(0)
    Logger.log(f'[Inside convert_image_to_tensor]:: after unsqueeze')
    tensor = tensor.to(device)
    Logger.log(f'[Inside convert_image_to_tensor]:: after to(device)')
    local_end_time = datetime.datetime.now()

    Logger.log(f'[Inside convert_image_to_tensor] [FRAME: {frame_count}] Time taken to convert frame to tensor and transfer it to device:: {local_end_time - local_start_time} ')

    return tensor


async def main_runner():

    global start_time
    global frame_count
      
    # This is the start of the video processing. Initialize the start time.
    start_time = datetime.datetime.now()

    Logger.log(f'[Inside main_runner] Start time of video processing.')

    # Read the input from the file.
    cam = cv2.VideoCapture('hdvideo.mp4')


    while frame_count <= frames_to_process:

        Logger.log(f'[Inside main_runner] [FRAME: {frame_count}] Current reading frame')

        # Reading next frame from the input.       
        ret, img_rbg = cam.read()   

        # If the frame exists
        if ret: 

            with concurrent.futures.ThreadPoolExecutor() as pool:

                # Send the frame for left processing.

                Logger.log(f'[Inside main_runner] [FRAME: {frame_count}] Sending frame for left processing')

                out_left = await io_loop.run_in_executor(pool, producer_video_left, img_rbg)

                Logger.log(f'[Inside main_runner] [FRAME: {frame_count}] Received frame after left processing')

            await q.put(out_left)

        # Increment frame count after left processing.    
        frame_count += 1
    
    # This is the end of the left processing. Set the end time of left video processing.
    end_time = datetime.datetime.now()

    print(f'[Inside main_runner] TOTAL TIME TAKEN FOR LEFT PROCESSING {frames_to_process} frames:: {end_time - start_time}')

    cam.release()
    cv2.destroyAllWindows()


if __name__=='__main__':

    print('Initialize IOLoop')
    io_loop = ioloop.IOLoop.current()

    print('Add main_runner and consumer to Tornado event loop as call back')
    io_loop.add_callback(main_runner)
    io_loop.add_callback(consumer) 

    print('Join the queue')
    q.join()                # Block until all the items in the queue are processed.

    print('Start IOLoop')
    io_loop.start()

    print('After start IOLoop')
    