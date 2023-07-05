import csv
import json
import datetime
import pandas as pd
import io
import tensorflow as tf

from PIL import Image

my_models = {
    'resnet50': tf.keras.applications.ResNet50,
    'resnet101': tf.keras.applications.ResNet101,
    'VGG16': tf.keras.applications.VGG16,
    'VGG19': tf.keras.applications.VGG19,
    'DenseNet121': tf.keras.applications.DenseNet121,
    'DenseNet169': tf.keras.applications.DenseNet169,
}

my_split_points = {
    'resnet50': [0],
    'resnet101': [0],
    'VGG16': [0],
    'VGG19': [0],
    'DenseNet121': [0],
    'DenseNet169': [0]
}

class Config:
    @staticmethod
    def get_config():
        with open('modelConfig.json') as config_file:
            config = json.load(config_file)
            config['url'] = config['url'].replace("{{server}}", config['server'])
        return config


class Logger():
    def log(message):
        time = datetime.datetime.now()
        print(f"{time}::{message}")

def write_to_csv(filename, field_names, data):
    # Check if the file exists
    file_exists = False
    try:
        with open(filename, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in the appropriate mode
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        # Write a new line if the file is empty
        if not file_exists:
            writer.writerow(field_names)  # Example column headers

        # Write the data to the file
        writer.writerow(data)

def get_flops(profile):

    df = pd.read_csv(io.StringIO(profile), sep='|', skiprows=0, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df["Value"].values[2].strip()


def convert_image_to_tensor(img):

    img_rgb = Image.fromarray(img).convert('RGB')
    tensor = tf.image.resize(img_rgb, [224, 224]) 
    tensor  = tf.expand_dims(tensor, axis=0)
    return tensor