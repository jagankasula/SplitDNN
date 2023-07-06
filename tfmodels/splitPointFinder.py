import tensorflow as tf

from PIL import Image
from model_profiler import model_profiler
import pandas as pd 
import io
from modelUtils import get_flops

model = tf.keras.applications.EfficientNetV2S(include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,)

profile = model_profiler(model, 50)

print(profile)

print(f'FLOPS::::: {get_flops(profile)}')

layers = model.layers

def convert_image_to_tensor(img):

    img_rgb = Image.fromarray(img).convert('RGB')
    tensor = tf.image.resize(img_rgb, [224, 224]) 
    tensor  = tf.expand_dims(tensor, axis=0)
    return tensor

print(len(layers))

candidate_layers = []
print('*****************')
#print(candidate_layers)
print('*****************')

print(model_profiler(model, 50))

for split_point in range(1, len(layers) - 1):
    split_layer = model.layers[split_point]
    next_layer = model.layers[split_point + 1]

    # print(split_layer.name)
    # print(next_layer.name)    

    try: 
        left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
        right_model = tf.keras.Model(inputs=next_layer.input, outputs=model.output)

        input = tf.random.uniform(shape=(None, 384, 384, 3))
        left_output = left_model(input)
        right_output = right_model(left_output)
        print(f'Success at split point # {split_point}, Name: {split_layer.name}')
        model_profiler(left_model, 50)
        model_profiler(right_model, 50)
        #print(model_profiler(left_model, 50))       
        # print(input.shape)
        # print(left_output.shape)
        # print(right_output.shape)
        candidate_layers.append(split_point)
    except ValueError as vrr:
        print(f'------- Failed at split point # {split_point}, Name: {split_layer.name}')
        print(vrr)
    except Exception as e:
        print(f'------- EXCEPTION -- Failed at split point # {split_point}, Name: {split_layer.name}')

print(candidate_layers)

# print(model.layers)

# print(model.summary())

# print(model_profiler(left_model, 50))

# print(model_profiler(right_model, 50))

# input = tf.random.uniform(shape=(1, 224, 224, 3))
# output = model(input)
# print(output.shape)
