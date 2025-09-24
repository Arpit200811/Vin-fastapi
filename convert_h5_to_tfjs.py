import tensorflow as tf

# Load your Keras .h5 model
model = tf.keras.models.load_model("C:/Users/DELL/OneDrive/Desktop/dataset/vin_model.h5")

# Save it in TensorFlow SavedModel format
tf.saved_model.save(model, "C:/Users/DELL/OneDrive/Desktop/dataset/vin_model_saved")

print("Model SavedModel format me save ho gaya!")
