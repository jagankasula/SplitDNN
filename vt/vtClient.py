from vit_keras import vit
from tensorflow import keras
from tornado import httpclient, ioloop
from tornado.queues import Queue
from PIL import Image
from vtutils import Config, Logger, write_to_csv
from model_profiler import model_profiler

import cv2
import datetime
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
import io

io.StringIO
warnings.filterwarnings('ignore')

# Read the configurations from the config file.
config = Config.get_config()

metrics_headers = ['split no.', 'flops', 'total_processing_time', 'single_frame_time', 'left_output_size']


# Assign the configurations to the global variables.
device = config['client_device']
split_point = config['split_point']
url = config['url']
frames_to_process = config['frames_to_process']

# Initialize queue for storing the output of the frames that were processed on the client (left) side.
q = Queue(maxsize=2)

# Initialize the start time to None. This value will be set in main_runner when it is initialized.
start_time = None

# Track total responses handled.
total_handled_responses = 0

left_output_size = 0

with tf.device(device):
  model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=19,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)
  
  print('*****************************')
  print(tf.config.list_physical_devices('GPU'))
  
  split_layer = model.layers[split_point]

  print(split_layer.name)
  
  left_model = keras.Model(inputs=model.input, outputs=split_layer.output)

  profile = model_profiler(left_model, frames_to_process)

  df = pd.read_csv(io.StringIO(profile), sep='|', skiprows=0, skipinitialspace=True)

  df.columns = df.columns.str.strip()

  flops = df["Value"].values[2]


def handle_response(response):

    global total_handled_responses

    # Process the response here
    load_data = pickle.loads(response.body)
    result = load_data['result']
    frame_seq_no = load_data['frame_seq_no']

    Logger.log(f'Processed frame # {frame_seq_no}')
    print(result.shape)

    total_handled_responses += 1

    if total_handled_responses == frames_to_process:
        end_time = datetime.datetime.now()
        time = (end_time - start_time).total_seconds()
        single_frame_time = time/frames_to_process
        write_to_csv('vt.csv', metrics_headers, [split_point, flops, time, single_frame_time, left_output_size])
        Logger.log(f'TOTAL TIME FOR PROCESSING:: {time} sec')
        Logger.log(f'TIME TAKEN FOR SINGLE FRAME:: {single_frame_time} sec')
        
      
async def consumer():

    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    
    async for item in q:

        try:   
            Logger.log(f'[Inside consumer] Frame # {item[1]}: Preparing body to send request to server.')

            post_data = {'data': item[0], 'frame_seq_no': item[1]} # item[0] = out_left, item[1] = frame_seq_no

            body = pickle.dumps(post_data)
            
            # Sending HTTP request to server.
            response =  http_client.fetch(url,  method = 'POST', headers = None, body = body)

            response.add_done_callback(lambda f: handle_response(f.result()))            

        finally:
            q.task_done()


def producer_video_left(img):

    tensor = convert_image_to_tensor(img)

    out_left = left_model(tensor)
    
    return out_left

def convert_image_to_tensor(img):

    img_rgb = Image.fromarray(img).convert('RGB')
    tensor = tf.image.resize(img_rgb, [224, 224]) 
    tensor  = tf.expand_dims(tensor, axis=0)

    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    # with strategy.scope():
    #     gpu_tensor = tf.constant(tensor)

    return tensor

async def main_runner():

    frame_seq_no = 1

    global start_time
    global frame_count
    global left_output_size
      
    # This is the start of the video processing. Initialize the start time.
    start_time = datetime.datetime.now()

    # Read the input from the file.
    cam = cv2.VideoCapture('hdvideo.mp4')


    while frame_seq_no < frames_to_process + 1:

        # Reading next frame from the input.       
        ret, img_rbg = cam.read()   

        # If the frame exists
        if ret: 
            
            Logger.log(f'[Inside main_runner] Frame # {frame_seq_no}: Send for left processing.')

            # Send the frame for left processing.
            out_left = producer_video_left(img_rbg)

            await q.put([out_left, frame_seq_no])

        # Increment frame count after left processing.    
        frame_seq_no += 1

    # This is the end of the left processing. Set the end time of left video processing.
    end_time = datetime.datetime.now()

    left_output_size = tf.size(out_left).numpy()

    Logger.log(f'[Inside main_runner] TOTAL TIME TAKEN FOR LEFT PROCESSING {frames_to_process} frames:: {end_time - start_time}')

    cam.release()
    cv2.destroyAllWindows()


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
