import tensorflow as tf
import numpy as np

# Load model architecture
json_path = r"/Users/ganeshtalluri/PycharmProjects/MicroCT/.venv/BP-Model1.json"
try:
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
except Exception as e:
    print(f"Error reading JSON file: {e}")
    raise

# Deserialize model from JSON
try:
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
except Exception as e:
    print(f"Error deserializing model from JSON: {e}")
    raise

# Load model weights
weights_path = r"/Users/ganeshtalluri/PycharmProjects/MicroCT/.venv/BP-Model1.hdf5"
try:
    loaded_model.load_weights(weights_path)
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise


# Function to convert 2D layers to 3D layers
@keras.saving.register_keras_serializable()
class CustomConv2DLayer(tf.keras.layers.Layer):
    def convert_2d_to_3d(model):
        new3D_model = tf.keras.Sequential()
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                new_layer = tf.keras.layers.Conv3D(
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size[0], layer.kernel_size[1], 3),
                    strides=(layer.strides[0], layer.strides[1], 1),
                    padding=layer.padding,
                    activation=layer.activation,
                    name=layer.name
                )
                new3D_model.add(new_layer)
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                new_layer = tf.keras.layers.MaxPooling3D(
                    pool_size=(layer.pool_size[0], layer.pool_size[1], 2),
                    strides=(layer.strides[0], layer.strides[1], 2),
                    padding=layer.padding,
                    name=layer.name
                )
                new3D_model.add(new_layer)
            elif isinstance(layer, tf.keras.layers.UpSampling2D):
                new_layer = tf.keras.layers.UpSampling3D(
                    size=(layer.size[0], layer.size[1], 2),
                    name=layer.name
                )
                new3D_model.add(new_layer)
            else:
                new3D_model.add(layer)
        return new3D_model


# Convert the model
new3D_model = convert_2d_to_3d(loaded_model)


# Function to copy weights from 2D layers to 3D layers
def copy_weights_2d_to_3d(old_model, new_model):
    for old_layer, new_layer in zip(old_model.layers, new_model.layers):
        if isinstance(old_layer, tf.keras.layers.Conv2D):
            weights_2d, bias = old_layer.get_weights()
            depth = 3
            weights_3d = np.repeat(weights_2d[:, :, np.newaxis, :], depth, axis=2)
            new_layer.set_weights([weights_3d, bias])
        elif isinstance(old_layer, tf.keras.layers.MaxPooling2D) or isinstance(old_layer, tf.keras.layers.UpSampling2D):
            # No weights to copy for pooling and upsampling layers
            continue
        else:
            new_layer.set_weights(old_layer.get_weights())


# Copy the weights
copy_weights_2d_to_3d(loaded_model, new3D_model)

# Print the new model summary to verify
new3D_model.summary()
