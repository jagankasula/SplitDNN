import tensorflow as tf
import pandas as pd 
import io
import cv2
import datetime

#from model_profiler import model_profiler
from modelUtils import Config, convert_image_to_tensor, write_to_csv, my_models

#Metrics
metrics_headers = ['Model', 'flops', 'num_layers', 'total_processing_time', 'single_frame_time']

# Read the configurations from the config file.
config = Config.get_config()

batch_size = config['frames_to_process']

# Model name
model_name = config['model']

# Get the model.
model = None

device = '/CPU:0'

with tf.device(device):
    if model_name in {'resnet50', 'resnet101'}:
        model = my_models.get(model_name, lambda: print(f"Model not present in my_models: {model_name}"))(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000,)
    else:
        model = my_models.get(model_name, lambda: print(f"Model not present in my_models: {model_name}"))(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax",)

    # Get layers from the model.
    layers = model.layers

    # Get layer count.
    num_layers = len(layers)

    # Profile the model.
    #profile = model_profiler(model, batch_size)

    # Get flops from the profile.
    #flops = get_flops(profile)
    flops = 0

    # Read video input.
    cam = cv2.VideoCapture('hdvideo.mp4')

    # No. of frames processed.
    count = 1

    # Processing start time
    processing_start_time = datetime.datetime.now()

    while count < batch_size + 1:
        ret, img = cam.read() 
        if ret:
            tensor = convert_image_to_tensor(img)
            output = model(tensor)
            print(f'Processed frame # {count}')
            print(output.shape)
            count += 1

    # Processing start time.
    processing_end_time = datetime.datetime.now()

    total_processing_time  = (processing_end_time - processing_start_time).total_seconds()
    single_frame_time = total_processing_time/batch_size

    # Write metrics to csv file.
    write_to_csv('model_cpu.csv', metrics_headers, [model_name, flops, num_layers, total_processing_time, single_frame_time])

