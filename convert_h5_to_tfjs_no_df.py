import tensorflow as tf
import tensorflowjs as tfjs

# Load your Keras model
model = tf.keras.models.load_model("C:/Users/DELL/OneDrive/Desktop/dataset/vin_model.h5")

# Save in TensorFlow.js format
tfjs.converters.save_keras_model(model, "C:/Users/DELL/OneDrive/Desktop/dataset/vin_model_tfjs")

print("Model converted to TensorFlow.js format successfully!")
