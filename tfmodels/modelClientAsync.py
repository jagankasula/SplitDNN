import cv2
import datetime
import pickle
import tensorflow as tf
import tornado.locks
import numpy as np
import pandas as pd
import warnings
import io

from tensorflow import keras
from tornado import httpclient, ioloop
from tornado.queues import Queue
from PIL import Image
from modelUtils import Config, Logger, write_to_csv, get_flops, dry_run_left_model, get_model, convert_image_to_tensor, input_size, my_split_points
from model_profiler import model_profiler

io.StringIO
warnings.filterwarnings('ignore')

all_responses_loop_event = tornado.locks.Event()
right_model_time_loop_event = tornado.locks.Event()

# Read the configurations from the config file.
config = Config.get_config()
metrics_headers = ['frames_to_process', 'split_no', 'flops', 'total_processing_time', 'single_frame_time', 'left_output_size', 'avg_consec_inference_gap', 'total_left_model_time', 'total_right_model_time']

# Assign the configurations to the global variables.
device = config['client_device']
url = config['url']
model_name = config['model']
frames_to_process = config['frames_to_process']

# Initialize queue for storing the output of the frames that were processed on the client (left) side.
q = Queue(maxsize=2)

# Initialize the start time to None. This value will be set in main_runner when it is initialized.
start_time = None

# Track total responses handled.
total_handled_responses = 0

# Total inference gap. Sum of the time gaps between two consecutive inferences.
total_inference_gap = 0
total_left_model_time = 0
total_right_model_time  = 0
prev_frame_end_time = None
left_output_size = 0
flops = 0

with tf.device(device):
  model = get_model(model_name)
  num_layers = len(model.layers)  
  split_points = my_split_points.get(model_name)
  print('*************************************************')
  print(tf.config.list_physical_devices(device_type=None))
  print('**************************************************')  

def handle_response(response):
    global total_handled_responses
    global total_inference_gap
    global prev_frame_end_time

    # Process the response here
    load_data = pickle.loads(response.body)
    result = load_data['result']
    frame_seq_no = load_data['frame_seq_no']

    Logger.log(f'Processed frame # {frame_seq_no}')
    print(result.shape)
    total_handled_responses += 1
    # First frame that is processed. Record its end time. 
    if total_handled_responses == 1:
        prev_frame_end_time = datetime.datetime.now()
    else:
        curr_frame_end_time = datetime.datetime.now()
        total_inference_gap += (curr_frame_end_time - prev_frame_end_time).total_seconds()
        prev_frame_end_time = curr_frame_end_time

    if total_handled_responses == frames_to_process:
        # Calculate total time taken to process all 50 frames.
        end_time = datetime.datetime.now()
        time = (end_time - start_time).total_seconds()
        single_frame_time = time/frames_to_process
        # Calculate average inference gap between two consequtive frames
        avg_consec_inference_gap = total_inference_gap/(frames_to_process - 1)
        # Reset to zero for next loop.
        total_inference_gap = 0
        request_total_right_model_time_from_server()
        log_metrics(split_point, flops, time, single_frame_time, left_output_size, avg_consec_inference_gap, total_left_model_time)

def log_metrics(split_point, flops, time, single_frame_time, left_output_size, avg_consec_inference_gap, total_left_model_time):
    right_model_time_loop_event.wait()
    write_to_csv(model_name + '_async' + '.csv', metrics_headers, [frames_to_process, split_point, flops, time, single_frame_time, left_output_size, avg_consec_inference_gap, total_left_model_time, total_right_model_time])
    Logger.log(f'CONSECUTIVE INFERENCE GAP BETWEEN TWO FRAMES:: {avg_consec_inference_gap}')
    Logger.log(f'PROCESSING TIME FOR SINGLE FRAME:: {single_frame_time} sec')
    Logger.log(f'TOTAL LEFT PROCESSING TIME:: {total_left_model_time}')
    Logger.log(f'TOTAL RIGHT PROCESSING TIME:: {total_right_model_time}')
    Logger.log(f'TOTAL PROCESSING TIME:: {time} sec')
    right_model_time_loop_event.clear()
    all_responses_loop_event.set()
        
async def consumer():
    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    
    async for item in q:
        try:   
            Logger.log(f'[Inside consumer] Frame # {item[1]}: Preparing body to send request to server.')
            post_data = {'data': item[0], 'frame_seq_no': item[1]} # item[0] = out_left, item[1] = frame_seq_no
            body = pickle.dumps(post_data)            
            # Sending HTTP request to server.
            response =  http_client.fetch(url + 'model',  method = 'POST', headers = None, body = body)
            response.add_done_callback(lambda f: handle_response(f.result()))            

        finally:
            q.task_done()

def request_total_right_model_time_from_server():
    request_json = get_request_body(None, 0, split_point)
    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    body = pickle.dumps(request_json) 
    response = http_client.fetch(url + 'right_model_time',  method = 'POST', headers = None, body = body)
    response.add_done_callback(lambda f: set_total_right_model_time(f.result()))

def set_total_right_model_time(response):
    global total_right_model_time
    response_body = pickle.loads(response.body)
    total_right_model_time = response_body['total_right_model_time']
    right_model_time_loop_event.set()

def producer_video_left(img, left_model):
    size = input_size.get(model_name)
    tensor = convert_image_to_tensor(img, size)
    out_left = left_model(tensor)    
    return out_left

def get_left_model(split_point):    
    split_layer = model.layers[split_point]
    print(split_layer.name)  
    left_model = keras.Model(inputs=model.input, outputs=split_layer.output)
    return left_model

# Returns JSON body for sending to server.
def get_request_body(left_output, frame_seq_no, split_point):
    # Request JSON.
    request_body = {'data': left_output, 'frame_seq_no': frame_seq_no, 'split_point': split_point}
    return request_body

# Send HTTP request to server.
async def send_request(request_body, endpoint):
    global client_request_time
    global server_response_time

    client_request_time = datetime.datetime.now()
    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    body = pickle.dumps(request_body) 
    response = None   

    if endpoint == 'split_point' or 'right_model_time':
        # Wait for the split point to be set in the server.
        response =  await http_client.fetch(url + endpoint,  method = 'POST', headers = None, body = body)
    else:
        response =  http_client.fetch(url + endpoint,  method = 'POST', headers = None, body = body)

    server_response_time = datetime.datetime.now()
    return response

async def main_runner():
    global split_point
    global flops

    for split_point in split_points:
        print('SPLIT: ' + str(split_point))
        left_model = get_left_model(split_point)
        dry_run_left_model(left_model, input_size.get(model_name))
        profile = model_profiler(left_model, frames_to_process)
        flops = get_flops(profile)

        # Request JSON.
        request_json = get_request_body(None, 0, split_point)
        response = await send_request(request_json, 'split_point')
        Logger.log(f'SPLIT POINT IS SET TO {split_point} IN SERVER')

        frame_seq_no = 1
        global start_time
        global left_output_size
        global total_handled_responses
        global total_left_model_time    
    
        # Read the input from the file.
        cam = cv2.VideoCapture('hdvideo.mp4')
        # This is the start of the video processing. Initialize the start time.
        start_time = datetime.datetime.now()
    
        while frame_seq_no < frames_to_process + 1:
            # Reading next frame from the input.       
            ret, img_rbg = cam.read()   
            # If the frame exists
            if ret:                 
                Logger.log(f'[Inside main_runner] Frame # {frame_seq_no}: Send for left processing.')    
                # Send the frame for left processing.
                out_left = producer_video_left(img_rbg, left_model)    
                await q.put([out_left, frame_seq_no])   

            # Increment frame count after left processing.    
            frame_seq_no += 1    

        # This is the end of the left processing. Set the end time of left video processing.
        end_time = datetime.datetime.now()    
        left_output_size = tf.size(out_left).numpy()
        total_left_model_time = (end_time - start_time).total_seconds()    
        Logger.log(f'[Inside main_runner] TOTAL TIME TAKEN FOR LEFT PROCESSING {frames_to_process} frames:: {total_left_model_time}')
    
        cam.release()
        cv2.destroyAllWindows()

        # Wait until all the responses for this split point are received and processed.
        await all_responses_loop_event.wait()
        total_handled_responses = 0
        all_responses_loop_event.clear()


with tf.device(device):
    if __name__=='__main__':
        Logger.log('Initialize IOLoop')
        io_loop = ioloop.IOLoop.current()

        Logger.log('Add main_runner and consumer to Tornado event loop as call back')
        io_loop.add_callback(main_runner)
        io_loop.add_callback(consumer) 

        Logger.log('Join the queue')
        q.join()                # Block until all the items in the queue are processed.

        Logger.log('Start IOLoop')
        io_loop.start()

        Logger.log('After start IOLoop')
