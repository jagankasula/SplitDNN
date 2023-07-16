# Run modelServerSync before running this code.

import cv2
import datetime
import pickle
import tensorflow as tf
import warnings
import io

from tensorflow import keras
from tornado import httpclient
from tornado.queues import Queue
from PIL import Image
from modelUtils import Config, Logger, write_to_csv, get_flops, dry_run_left_model, get_model, my_split_points
from model_profiler import model_profiler

io.StringIO
warnings.filterwarnings('ignore')

# Read the configurations from the config file.
config = Config.get_config()

#metrics_headers = ['split_no', 'flops', 'total_processing_time', 'single_frame_time', 'left_output_size', 'total_communication_time', 'single_frame_communication_time', 'avg_consec_inference_gap']
metrics_headers = ['model_name','split_no','flops','total_processing_time','single_frame_time','left_output_size','total_communication_time','single_frame_communication_time','avg_consec_inference_gap']

# Assign the configurations to the global variables.
device = config['client_device']
model_name = config['model']
url = config['url']
frames_to_process = config['frames_to_process']

# Initialize queue for storing the output of the frames that were processed on the client (left) side.
q = Queue(maxsize=2)

# Initialize the start time to None. This value will be set in main_runner when it is initialized.
start_time = None

client_request_time = None
server_response_time= None
total_communication_time = 0

# Total inference gap. Sum of the time gap between two consecutive inferences.
total_inference_gap = 0
prev_frame_end_time = None

# Track total responses handled.
total_handled_responses = 0
left_output_size = 0
flops = 0

with tf.device(device):
  model = get_model(model_name)
  
  split_points = my_split_points.get(model_name)
  print('*************************************************')
  print(tf.config.list_physical_devices(device_type=None))
  print('**************************************************')

        
def producer_video_left(img, left_model):
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
def send_request(request_body, endpoint):
    global client_request_time
    global server_response_time
    client_request_time = datetime.datetime.now()
    http_client = httpclient.HTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    body = pickle.dumps(request_body)    
    response =  http_client.fetch(url + endpoint,  method = 'POST', headers = None, body = body)
    server_response_time = datetime.datetime.now()
    return response

def handle_response(response):            
    # Process the response here
    response_data = pickle.loads(response.body)
    result = response_data['result']
    frame_seq_no = response_data['frame_seq_no']
    server_processing_time = response_data['server_processing_time']
    set_total_communication_time(server_processing_time)    
    Logger.log(f'Processed frame # {frame_seq_no}')
    print(result.shape)

def set_total_communication_time(server_processing_time):
    global total_communication_time
    communication_time = (server_response_time - client_request_time).total_seconds() - server_processing_time
    total_communication_time += communication_time

def main_runner():
    global split_point
    global flops

    for split_point in split_points:
        print('SPLIT: ' + str(split_point))
        left_model = get_left_model(split_point)
        dry_run_left_model(left_model)
        profile = model_profiler(left_model, frames_to_process)
        flops = get_flops(profile)
        # Request JSON.
        request_json = get_request_body(None, 0, split_point)
        response = send_request(request_json, 'split_point') 

        #Initialize total time taken for split_point
        total_time_for_split_point = 0

        frame_seq_no = 1    
        global start_time
        global frame_count
        global left_output_size
        global total_communication_time
        global prev_frame_end_time
        global total_inference_gap        
    
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
                # Request JSON.
                request_json = get_request_body(out_left, frame_seq_no, split_point)
                # Send request to server for processing.
                response = send_request(request_json, 'model')
                handle_response(response)

                # First frame that is processed. Record its end time. 
                if frame_seq_no == 1:
                    prev_frame_end_time = datetime.datetime.now()
                else:
                    curr_frame_end_time = datetime.datetime.now()
                    total_inference_gap += (curr_frame_end_time - prev_frame_end_time).total_seconds()
                    prev_frame_end_time = curr_frame_end_time
                    
            # Increment frame count after left processing.    
            frame_seq_no += 1
    
        left_output_size = tf.size(out_left).numpy()

        # Calculate average inference gap between two consequtive frames
        avg_consec_inference_gap = total_inference_gap/(frames_to_process - 1)
        # Reset to zero for next loop.
        total_inference_gap = 0

        # This is the end of the left processing. Set the end time of left video processing.
        end_time = datetime.datetime.now()        
        time = (end_time - start_time).total_seconds()
        single_frame_time = time/frames_to_process
        single_frame_communication_time = total_communication_time/frames_to_process

        #write_to_csv(model_name + '_sync.csv', metrics_headers, [split_point, flops, time, single_frame_time, left_output_size, total_communication_time, single_frame_communication_time, avg_consec_inference_gap])
        write_to_csv('gpu_time_all_models' + '_sync.csv', metrics_headers, [model_name, split_point, flops, time, single_frame_time, left_output_size, total_communication_time, single_frame_communication_time, avg_consec_inference_gap])
        print('-------------------------------------------------------------------------------------------')
        Logger.log(f'TOTAL COMMUNICATION TIME FOR {frames_to_process} frames:: {total_communication_time}')
        Logger.log(f'COMMUNICATION TIME FOT SINGLE FRAME:: {single_frame_communication_time}')
        Logger.log(f'CONSECUTIVE INFERENCE GAP BETWEEN TWO FRAMES:: {avg_consec_inference_gap}')
        Logger.log(f'TOTAL TIME FOR PROCESSING {frames_to_process} frames:: {time} sec')
        Logger.log(f'TIME TAKEN TO PROCESS SINGLE FRAME:: {single_frame_time} sec')
        print('-------------------------------------------------------------------------------------------')

        total_communication_time = 0

        cam.release()
        cv2.destroyAllWindows()


with tf.device(device):
    # Driver code.
    if __name__ == '__main__':
        main_runner()