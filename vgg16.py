import tensorflow as tf

from utils import get_video_input
from PIL import Image
from model_profiler import model_profiler

video_input = get_video_input()

count = 0

model = tf.keras.applications.VGG16()

layers = model.layers

def convert_image_to_tensor(img):

    img_rgb = Image.fromarray(img).convert('RGB')
    tensor = tf.image.resize(img_rgb, [224, 224]) 
    tensor  = tf.expand_dims(tensor, axis=0)
    return tensor

print(len(layers))

split_point = 15
split_layer = model.layers[split_point]
next_layer = model.layers[split_point + 1]

print(split_layer.name)
print(next_layer.name)

left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
right_model = tf.keras.Model(inputs=next_layer.input, outputs=model.output)

# while count < 10:
#     ret, img = video_input.read()

#     if ret:
#         input  = convert_image_to_tensor(img)

#         left_output = left_model(input)
#         right_output = right_model(left_output)
#         print(input.shape)
#         print(left_output.shape)
#         print(right_output.shape)
#         count += 1

# print(model.layers)

# print(model.summary())

print(model_profiler(left_model, 50))

print(model_profiler(right_model, 50))

# input = tf.random.uniform(shape=(1, 224, 224, 3))
# output = model(input)
# print(output.shape)
