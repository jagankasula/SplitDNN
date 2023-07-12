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
    'EfficientNetV2B0': tf.keras.applications.EfficientNetV2B0
}

my_split_points = {
    'resnet50': [1, 2, 3, 4, 5, 6, 12, 17, 18, 27, 28, 37, 38, 49, 50, 59, 60, 69, 70, 79, 80, 91, 92, 101, 102, 111, 112, 121, 122, 131, 132, 141, 142, 153, 154, 163, 164, 173, 174, 175],
    'resnet101': [1, 2, 3, 4, 5, 6, 12, 17, 18, 27, 28, 37, 38, 49, 50, 59, 60, 69, 70, 79, 80, 91, 92, 101, 102, 111, 112, 121, 122, 131, 132, 141, 142, 151, 152, 161, 162, 171, 172, 181, 182, 191, 192, 201, 202, 211, 212, 221, 222, 231, 232, 241, 242, 251, 252, 261, 262, 271, 272, 281, 282, 291, 292, 301, 302, 311, 312, 323, 324, 333, 334, 343, 344, 345],
    'VGG16': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    'VGG19': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'DenseNet121': [1, 2, 3, 4, 5, 6, 13, 20, 27, 34, 41, 48, 49, 50, 51, 52, 59, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 137, 138, 139, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 309, 310, 311, 312, 319, 326, 333, 340, 347, 354, 361, 368, 375, 382, 389, 396, 403, 410, 417, 424, 425, 426, 427],
    'DenseNet169': [34, 41, 48, 49, 50, 51, 52, 59, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 137, 138, 139, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 343, 350, 357, 364, 365, 366, 367, 368, 375, 382, 389, 396, 403, 410, 417, 424, 431, 438, 445, 452, 459, 466, 473, 480, 487, 494, 501, 508, 515, 522, 529, 536, 543, 550, 557, 564, 571, 578, 585, 592, 593, 594, 595],
    'EfficientNetV2B0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 43, 44, 45, 60, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 103, 118, 133, 148, 149, 150, 151, 152, 153, 154, 159, 160, 161, 176, 191, 206, 221, 236, 251, 266, 267, 268, 269, 270, 271]
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