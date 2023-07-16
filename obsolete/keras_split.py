import tensorflow as tf
import cv2
import pickle

from PIL import Image


def get_input():
    cam = cv2.VideoCapture('hdvideo.mp4')
    ret, img_rbg = cam.read()

    count = 0

    while count < 1:

        print('in while')

        # Reading next frame from the input.       
        ret, img = cam.read()   

        if ret:
            img_rgb = Image.fromarray(img).convert('RGB')
            tensor = tf.image.resize(img_rgb, [224, 224])            
            count += 1
            # print(tensor)
            # print(type(tensor))
            tensor  = tf.expand_dims(tensor, axis=0)
            return tensor

model = tf.keras.applications.VGG16()

layers = model.layers

print(len(layers))

split_point = 21
split_layer = model.layers[split_point]
next_layer = model.layers[split_point + 1]

print(split_layer.name)
print(next_layer.name)

left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
right_model = tf.keras.Model(inputs=next_layer.input, outputs=model.output)

#input = tf.random.uniform(shape=(1, 224, 224, 3))
input = get_input()


# print('NUMPY INPUT ------------')
# print(input_np)
# print('NUMPY INPUT ------------')

output = model(input)
print(output.shape)

left_output = left_model(input)

post_data = {'data': left_output, 'count': 0}
body = pickle.dumps(post_data)

res_body = pickle.loads(body)

print(res_body['data'])
right_output = right_model(res_body['data'])

# left_output_np = input.numpy()
# left_output_tensor = tf.convert_to_tensor(left_output_np)
# shape = tf.shape(left_output)
# print(f'required shape--------> {shape}')
# print(type(shape))
# left_output_tensor = tf.reshape(left_output_tensor, tf.shape(left_output))

# right_output = right_model(left_output)

print(input.shape)
print(left_output.shape)
print(right_output.shape)
print(left_output)