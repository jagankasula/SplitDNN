from vit_keras import vit
from tensorflow import keras
from tornado import httpclient, ioloop
from tornado.queues import Queue
from PIL import Image

import cv2
import pickle
import tensorflow as tf
import numpy as np
import warnings


warnings.filterwarnings('ignore')

device = '/cpu:0'
url = 'http://localhost:8881/model'

# Initialize queue for storing the output of the frames that were processed on the client (left) side.
q = Queue(maxsize=2)

with tf.device(device):
  model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=12,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)



split_point = 5
split_layer = model.layers[split_point]
next_layer = model.layers[split_point + 1]

print(split_layer.name)
print(next_layer.name)

left_model = keras.Model(inputs=model.input, outputs=split_layer.output)
right_model = keras.Model(inputs=next_layer.input, outputs=model.output)

frame_count = 0

def handle_response(response):

    # Process the response here
    load_data = pickle.loads(response.body)
    result = load_data['result']
    count_return = load_data['count']
    print(result)
    print(f'Processed frame # : {count_return}')
      

async def consumer():

    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout = 10000000.0 ,request_timeout=100000000000.0))
    
    async for item in q:

        try:            
            post_data = {'data': item, 'count': frame_count}

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

    return tensor

async def main_runner():

    global frame_count

    # Read the input from the file.
    cam = cv2.VideoCapture('hdvideo.mp4')


    while frame_count < 10:

        # Reading next frame from the input.       
        ret, img_rbg = cam.read()   

        # If the frame exists
        if ret: 

            # Send the frame for left processing.
            out_left = producer_video_left(img_rbg)

            await q.put(out_left)

        # Increment frame count after left processing.    
        frame_count += 1


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