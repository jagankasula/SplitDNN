import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils

# Load the ConvNeXtSmall model
convnext_model = tf.keras.applications.ConvNeXtSmall(
    model_name="convnext_small",
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
frames = tf.random.uniform(shape=(50, 224, 224, 3))  # Replace this with your 50 frames

# Prepare the input data (assuming it's a NumPy array or a list of arrays)
input_data = tf.keras.applications.convnext.preprocess_input(frames)

# Run a forward pass once to build the graph and compute FLOPs
@tf.function
def compute_flops():
    return convnext_model(input_data[:1])

# Use TensorFlow Profiler to get FLOPs information
tf.profiler.experimental.start(logdir=None)
compute_flops()
tf.profiler.experimental.stop()

# Get the FLOPs from the profiler
flops = tf_utils.model_flops(convnext_model)

# Compute the FLOPs for 50 frames (since it's the same for all frames)
num_frames = 50
total_flops = flops * num_frames

print(f"Total FLOPs for {num_frames} frames: {total_flops:.2f}")
print(f"FLOPs per frame: {total_flops/num_frames:.2f}")