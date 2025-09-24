import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Step 1: Dataset Paths
# -----------------------------
IMG_SIZE = 128
base_path = r"C:\Users\DELL\OneDrive\Desktop\dataset"

train_real = os.path.join(base_path, "train", "real")
train_fake = os.path.join(base_path, "train", "fake")
test_real = os.path.join(base_path, "test", "real")
test_fake = os.path.join(base_path, "test", "fake")

# -----------------------------
# Step 2: Load Images
# -----------------------------
def load_images(path, label):
    images, labels = [], []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)
    return images, labels

# Training data
train_images_real, train_labels_real = load_images(train_real, 1)
train_images_fake, train_labels_fake = load_images(train_fake, 0)
X_train = np.array(train_images_real + train_images_fake)
y_train = np.array(train_labels_real + train_labels_fake)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Testing data
test_images_real, test_labels_real = load_images(test_real, 1)
test_images_fake, test_labels_fake = load_images(test_fake, 0)
X_test = np.array(test_images_real + test_images_fake)
y_test = np.array(test_labels_real + test_labels_fake)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print("Training dataset shape:", X_train.shape, y_train.shape)
print("Testing dataset shape:", X_test.shape, y_test.shape)

# -----------------------------
# Step 3: Data Augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# -----------------------------
# Step 4: Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_test, y_test)
)

# -----------------------------
# Step 6: Save Model
# -----------------------------
model_save_path = os.path.join(base_path, "vin_model.h5")
model.save(model_save_path)
print(f"✅ Model saved at: {model_save_path}")
