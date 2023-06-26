import tensorflow as tf

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

input = tf.random.uniform(shape=(1, 224, 224, 3))
output = model(input)
print(output.shape)

left_output = left_model(input)
right_output = right_model(left_output)
print(input.shape)
print(left_output.shape)
print(right_output.shape)