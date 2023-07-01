from vit_keras import vit
from tensorflow import keras
from PIL import Image
from vtutils import Logger

import tensorflow as tf
import numpy as np
import warnings
import cv2
import datetime

warnings.filterwarnings('ignore')

device = '/cpu:0'  
frames_to_process = 50

def convert_image_to_tensor(img):

    img_rgb = Image.fromarray(img).convert('RGB')
    tensor = tf.image.resize(img_rgb, [224, 224]) 
    tensor  = tf.expand_dims(tensor, axis=0)

    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    # with strategy.scope():
    #     gpu_tensor = tf.constant(tensor)

    return tensor

with tf.device(device):
  model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=19,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)
  
  model.summary()

  physical_devices = tf.config.list_physical_devices()

  print("----------------------------------------------------------------------------------------")
  print(physical_devices)
  print("----------------------------------------------------------------------------------------")

  # layers = model.layers

  # for layer in layers:
  #   print(layer)

  # split_point = 3
  # split_layer = model.layers[split_point]
  # next_layer = model.layers[split_point + 1]
  
  # print('----------------------- SPLIT LAYER OUTPUT SHAPE -----------------------')
  # print(split_layer.output_shape)
  
  # print('----------------------- NEXT LAYER INPUT SHAPE -----------------------')
  # print(next_layer.input_shape)
  
  # print(split_layer.name)
  # print(next_layer.name)
  
  # left_model = keras.Model(inputs=model.input, outputs=split_layer.output)
  # right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
  
  # print('----------------------- LEFT MODEL -----------------------')
  # print(left_model)
  # print('----------------------- RIGHT MODEL -----------------------')
  # print(right_model)

  frame_seq_no = 1

  start_time = datetime.datetime.now()
    
  # Read the input from the file.
  cam = cv2.VideoCapture('hdvideo.mp4')

  while frame_seq_no < frames_to_process + 1:

    # Reading next frame from the input.       
    ret, img = cam.read()

    if ret: 

      tensor = convert_image_to_tensor(img)

      output = model(tensor)

      Logger.log(f'Processed frame # {frame_seq_no}')
      print(output.shape)

  end_time = datetime.datetime.now()

  time = (end_time - start_time).total_seconds()
  single_frame_time = time/frames_to_process

  Logger.log(f'TOTAL TIME FOR PROCESSING:: {time} sec')
  Logger.log(f'TIME TAKEN FOR SINGLE FRAME:: {single_frame_time} sec')
