import cv2
import tensorflow as tf

from PIL import Image
from vit_keras import vit

device = '/cpu:0'

print(tf.config.list_physical_devices(device_type=None))

with tf.device(device):
  model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=12,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)
  
print(model.input)

# def convert_image_to_tensor():

#      # Read the input from the file.
#     cam = cv2.VideoCapture('hdvideo.mp4')

#     print('out while')

#     count = 0

#     while count < 1:

#         print('in while')

#         # Reading next frame from the input.       
#         ret, img = cam.read()   

#         if ret:
#             img_rgb = Image.fromarray(img).convert('RGB')
#             img_rgb = tf.image.resize(img_rgb, [224, 224])            
#             count += 1
#             print(img_rgb)
#             print(type(img_rgb))


# convert_image_to_tensor()