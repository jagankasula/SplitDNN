from vit_keras import vit
from tensorflow import keras
from tornado import httpclient
from tornado.queues import Queue
from PIL import Image
from vtutils import Config, Logger, write_to_csv

import cv2
import datetime
import pickle
import tensorflow as tf
import warnings
import io

io.StringIO
warnings.filterwarnings('ignore')

# Read the configurations from the config file.
config = Config.get_config()

metrics_headers = ['split no.', 'flops', 'total_processing_time', 'single_frame_time', 'left_output_size', 'total_communication_time']


# Assign the configurations to the global variables.
device = config['client_device']
split_point = config['split_point']
url = "http://localhost:8881/"
frames_to_process = config['frames_to_process']

# Initialize queue for storing the output of the frames that were processed on the client (left) side.
q = Queue(maxsize=2)

# Initialize the start time to None. This value will be set in main_runner when it is initialized.
start_time = None

client_request_time = None
server_response_time= None
total_communication_time = 0

# Track total responses handled.
total_handled_responses = 0

left_output_size = 0

with tf.device(device):
  model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=19,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)
  
  
  split_layer = model.layers[split_point]

  print(split_layer.name)
  
  left_model = keras.Model(inputs=model.input, outputs=split_layer.output)

#   profile = model_profiler(left_model, frames_to_process)

#   df = pd.read_csv(io.StringIO(profile), sep='|', skiprows=0, skipinitialspace=True)

#   df.columns = df.columns.str.strip()

  flops = 0

        
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

    communication_time = (server_response_time - client_request_time) + server_processing_time

    total_communication_time += communication_time



def main_runner():

    for split_point in range(1, 22):

        print('SPLIT: ' + str(split_point))

        left_model = get_left_model(split_point)

        # Request JSON.
        request_json = get_request_body(None, 0, split_point)

        response = send_request(request_json, 'split_point')        

        #Initialize total time taken for split_point
        total_time_for_split_point = 0

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
                out_left = producer_video_left(img_rbg, left_model)

                # Request JSON.
                request_json = get_request_body(out_left, frame_seq_no, split_point)

                # Send request to server for processing.
                response = send_request(request_json, 'model')

                handle_response(response)
                    
            # Increment frame count after left processing.    
            frame_seq_no += 1
    
        # This is the end of the left processing. Set the end time of left video processing.
        end_time = datetime.datetime.now()
        
        time = (end_time - start_time).total_seconds()
        single_frame_time = time/frames_to_process

        write_to_csv('vtSync.csv', metrics_headers, [split_point, flops, time, single_frame_time, left_output_size, total_communication_time])
        print('-------------------------------------------------------------------------------------------')
        Logger.log(f'TOTAL COMMUNICATION TIME FOR {frames_to_process} frames:: {total_communication_time}')
        Logger.log(f'COMMUNICATION TIME FOT SINGLE FRAME:: {total_communication_time/frames_to_process}')
        Logger.log(f'TOTAL TIME FOR PROCESSING {frames_to_process} frames:: {time} sec')
        Logger.log(f'TIME TAKEN TO PROCESS SINGLE FRAME:: {single_frame_time} sec')
        print('-------------------------------------------------------------------------------------------')

        cam.release()
        cv2.destroyAllWindows()


with tf.device(device):

    # Driver code.
    if __name__ == '__main__':
        main_runner()
