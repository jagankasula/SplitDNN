import tensorflow as tf
import time

# Load the ConvNeXtTiny model
model = tf.keras.applications.ConvNeXtTiny(
    model_name="convnext_tiny",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# Assuming you have a list of 50 frames (images), you need to preprocess them as needed
# Replace 'frames' with your actual input data
frame = tf.random.uniform(shape=(1, 224, 224, 3))  # Replace this with your 50 frames

# Prepare the input data (assuming it's a NumPy array or a list of arrays)
input_data = tf.keras.applications.convnext.preprocess_input(frame)

# Create a batch of 50 frames for profiling
batch_size = 50
input_batch = tf.concat([input_data] * batch_size, axis=0)

# Run a forward pass once to build the graph (optional, for more accurate profiling)
_ = model(input_batch[:1])

# Profile the model for batch size of 50 frames
start_time = time.time()
predictions = model(input_batch)
end_time = time.time()

# Calculate the total execution time for the batch of 50 frames
total_time = end_time - start_time
average_time_per_frame = total_time / batch_size

print(f"Total execution time for batch size of {batch_size}: {total_time:.2f} seconds.")
print(f"Average time per frame: {average_time_per_frame:.4f} seconds.")

#%%
import tensorflow as tf

# Load your TensorFlow model here
model = tf.keras.applications.MobileNetV2()

# Create a synthetic input batch with shape (50, input_height, input_width, num_channels)
input_batch = tf.random.uniform((50, 224, 224, 3))

# Run the profiler to get profiling information
with tf.profiler.experimental.Profile('/tmp/profiler'):
    model(input_batch)

# Read the profiling information
profile_log_path = '/tmp/profiler'
profile_ctx = tf.profiler.experimental.Profiler(profile_log_path)
profile_ctx.parse()

# Calculate the FLOPs from the profiling results
flops = tf.profiler.profile(profile_ctx, cmd='op', options=tf.profiler.ProfileOptionBuilder.float_operation())
print("Estimated FLOPs for 50 inputs:", flops.total_float_ops)
# %%
