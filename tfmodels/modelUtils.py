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
    'EfficientNetV2B0': tf.keras.applications.EfficientNetV2B0,
    'EfficientNetV2L': tf.keras.applications.EfficientNetV2L,
    'NASNetMobile': tf.keras.applications.NASNetMobile,
    'InceptionV3': tf.keras.applications.InceptionV3,
    'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
    'Xception': tf.keras.applications.Xception,
}

my_split_points = {
    'resnet50': [27, 50],
    'resnet101': [1, 2, 3, 4, 5, 6, 12, 17, 18, 27, 28, 37, 38, 49, 50, 59, 60, 69, 70, 79, 80, 91, 92, 101, 102, 111, 112, 121, 122, 131, 132, 141, 142, 151, 152, 161, 162, 171, 172, 181, 182, 191, 192, 201, 202, 211, 212, 221, 222, 231, 232, 241, 242, 251, 252, 261, 262, 271, 272, 281, 282, 291, 292, 301, 302, 311, 312, 323, 324, 333, 334, 343, 344, 345],
    'VGG16': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    'VGG19': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'DenseNet121': [27, 50],
    'DenseNet169': [34, 41, 48, 49, 50, 51, 52, 59, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 137, 138, 139, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 343, 350, 357, 364, 365, 366, 367, 368, 375, 382, 389, 396, 403, 410, 417, 424, 431, 438, 445, 452, 459, 466, 473, 480, 487, 494, 501, 508, 515, 522, 529, 536, 543, 550, 557, 564, 571, 578, 585, 592, 593, 594, 595],
    'EfficientNetV2B0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 43, 44, 45, 60, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 103, 118, 133, 148, 149, 150, 151, 152, 153, 154, 159, 160, 161, 176, 191, 206, 221, 236, 251, 266, 267, 268, 269, 270, 271],
    'EfficientNetV2L': [1, 2, 3, 4, 8, 13, 18, 23, 24, 25, 26, 27, 28, 35, 42, 49, 56, 63, 70, 71, 72, 73, 74, 75, 82, 89, 96, 103, 110, 117, 118, 119, 120, 121, 122, 123, 128, 129, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 266, 267, 268, 269, 270, 271, 276, 277, 278, 293, 308, 323, 338, 353, 368, 383, 398, 413, 428, 443, 458, 473, 488, 503, 518, 533, 548, 549, 550, 551, 552, 553, 554, 559, 560, 561, 576, 591, 606, 621, 636, 651, 666, 681, 696, 711, 726, 741, 756, 771, 786, 801, 816, 831, 846, 861, 876, 891, 906, 921, 922, 923, 924, 925, 926, 927, 932, 933, 934, 949, 964, 979, 994, 1009, 1024, 1025, 1026, 1027, 1028, 1029],
    'NASNetMobile': [1, 2, 767, 768, 769],
    'InceptionV3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 40, 50, 63, 73, 86, 100, 119, 132, 151, 164, 183, 196, 215, 228, 248, 262, 279, 293, 310, 311],
    'InceptionResNetV2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 40, 61, 62, 83, 84, 105, 106, 127, 128, 149, 150, 171, 172, 193, 194, 215, 216, 237, 238, 259, 260, 274, 289, 290, 305, 306, 321, 322, 337, 338, 353, 354, 369, 370, 385, 386, 401, 402, 417, 418, 433, 434, 449, 450, 465, 466, 481, 482, 497, 498, 513, 514, 529, 530, 545, 546, 561, 562, 577, 578, 593, 594, 617, 632, 633, 648, 649, 664, 665, 680, 681, 696, 697, 712, 713, 728, 729, 744, 745, 760, 761, 776, 777, 778, 779, 780],
    'Xception': [1, 2, 3, 4, 5, 6, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 126, 127, 128, 129, 130, 131, 132]
}

input_size = {
    'resnet50': 224,
    'resnet101': 224,
    'VGG16': 224,
    'VGG19': 224,
    'DenseNet121': 224,
    'DenseNet169': 224,
    'EfficientNetV2B0': 224,
    'EfficientNetV2L': 480,
    'NASNetMobile':224,
    'InceptionV3': 299,
    'InceptionResNetV2': 299,
    'Xception': 299
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


def convert_image_to_tensor(img, input_size):
    img_rgb = Image.fromarray(img).convert('RGB')
    tensor = tf.image.resize(img_rgb, [input_size, input_size]) 
    tensor  = tf.expand_dims(tensor, axis=0)

    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    # with strategy.scope():
    #     gpu_tensor = tf.constant(tensor)
    return tensor

def dry_run_left_model(model, input_size):
    input = tf.random.uniform(shape=(1, input_size, input_size, 3))
    output = model(input)
    print('Dry run for LEFT model is completed.')
    print(output.shape)
    return output

def dry_run_right_model(left_model, right_model, input_size):
    left_output = dry_run_left_model(left_model, input_size)
    output = right_model(left_output)
    print('Dry run for RIGHT model is completed.')
    print(output.shape)
    return output

def get_model(model_name):
    if model_name in {'resnet50', 'resnet101'}:
        model = my_models.get(model_name, lambda: print(f"Model not present in my_models: {model_name}"))(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000,)
    elif model_name in {'EfficientNetV2B0', 'EfficientNetV2L'}:
        model = my_models.get(model_name, lambda: print(f"Model not present in my_models: {model_name}"))(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", include_preprocessing=True,)
    else:
        model = my_models.get(model_name, lambda: print(f"Model not present in my_models: {model_name}"))(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax",)
    return model
  
