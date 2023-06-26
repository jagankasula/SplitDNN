import datetime
import warnings
import cv2
import numpy as np
import pickle
import torch.nn as nn
import torch
import torchvision.transforms as transforms

from PIL import Image
from tornado import httpclient, ioloop
from tornado.queues import Queue
from utils import Config, get_model, get_left_modules, get_test_data, vocab, tokenizer
from flops_profiler.profiler import get_model_profile


# Read the configurations from the config file.
config = Config.get_config()

# Assign the configurations to the global variables.
device = config['client_device']
model_file_path = config['model_file_path']
split_point = config['split_point']
url = config['url']


def data_process(raw_text):
  """Converts raw text into a flat Tensor."""
  data = torch.tensor(vocab(tokenizer(raw_text)), dtype=torch.long) # convert to tensor
  """Remove tensors of size 0, convert data to tuple, concatenate tensors along a specified dimension (defualt dimension = 0)."""
  return torch.cat(data)

model = get_model()

# Model splitting.
left_modules = get_left_modules(split_point, model)
left_model = left_modules.eval().to(device)

flops, macs, params = get_model_profile(
    left_model,
    input_shape= tuple([1, 3, 224, 224]),
    print_profile=True,
    detailed=True,
    as_string=True,
)

def handle_response(response):

    global total_handled_responses

    total_handled_responses += 1

    # Process the response here
    load_data = pickle.loads(response.body)
    result = load_data['result']
    count_return = load_data['count']

    Logger.log(f'[Inside handle_response] [Frame: {count_return}] Server response timestamp')

    result_right_np = np.asarray(result)
    result_right = torch.Tensor(result_right_np)

    with torch.no_grad():
        output = torch.nn.functional.softmax(result_right, dim=1)
     


async def consumer():

    global consumer_frame_count
    global total_encoding_time
    global detached_numpy_size

    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    
    consumer_startTime = datetime.datetime.now()

    Logger.log(f'[Inside consumer] CONSUMER START TIME.')

    async for item in q:

        try:            
            Logger.log(f'[Inside consumer] [Frame: {consumer_frame_count}] Preparing body to send request to server.')
            item_np = item.detach().numpy()

            if consumer_frame_count == 1:
                detached_numpy_size = item_np.size

            post_data = {'data': item_np, 'count': consumer_frame_count}

            encoding_time_start = datetime.datetime.now()
            #body = json.dumps(post_data, cls = NumpyArrayEncoder)
            body = pickle.dumps(post_data)
            encoding_time_end = datetime.datetime.now()

            total_encoding_time += (encoding_time_end - encoding_time_start).total_seconds()
            
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

    global initial_tensor_size
    global left_output_size
    
    tensor = convert_image_to_tensor(img)

    local_start_time = datetime.datetime.now()

    Logger.log(f'[Inside producer_video_left] [FRAME: {frame_count}] Start time of frame processing in left side.')

    out_left = resnet_left(tensor)
    out_left = out_left.cpu()

    if frame_count == 1:
        initial_tensor_size = tensor.size()
        left_output_size = out_left.size()

    local_end_time = datetime.datetime.now()

    Logger.log(f'[Inside producer_video_left] [FRAME: {frame_count}] End time of frame processing in left side.')

    Logger.log(f'[Inside producer_video_left] [FRAME: {frame_count}] Total time taken to process frame in left side:: {local_end_time - local_start_time}')
    
    return out_left


async def main_runner():

    test_raw = get_test_data()
      
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

                # Send the frame for left processing.

            Logger.log(f'[Inside main_runner] [FRAME: {frame_count}] Sending frame for left processing')

            out_left = producer_video_left(img_rbg)

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
    