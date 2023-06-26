from vit_keras import vit
import numpy as np

input_shape = (224, 224, 3)  # Input image shape
num_classes = 1000  # Number of output classes
patch_size = 16  # Patch size for splitting the image

model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=12,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)
layers = model.layers

for layer in layers:
  print(layer)

split_point = 3
split_layer = model.layers[split_point]
next_layer = model.layers[split_point + 1]

print('----------------------- SPLIT LAYER OUTPUT SHAPE -----------------------')
print(split_layer.output_shape)

print('----------------------- NEXT LAYER INPUT SHAPE -----------------------')
print(next_layer.input_shape)

print(split_layer.name)
print(next_layer.name)

left_model = keras.Model(inputs=model.input, outputs=split_layer.output)
right_model = keras.Model(inputs=next_layer.input, outputs=model.output)

print('----------------------- LEFT MODEL -----------------------')
print(left_model)
print('----------------------- RIGHT MODEL -----------------------')
print(right_model)

input_image = np.random.rand(1, 224, 224, 3)  # Shape: (batch_size, image_size, image_size, num_channels)

# Pass the input image to the model for inference
left_output = left_model(input_image)

right_input = None

if split_point < 5:
  right_input = left_output
else:
  right_input = left_output[0]

output = right_model(right_input)

# The output will be the predictions or representations depending on the model configuration
print(output)