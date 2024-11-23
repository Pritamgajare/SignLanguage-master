import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model's layers so they won't be retrained
base_model.trainable = False

# Define the model architecture
model = models.Sequential([
    base_model,  # Feature extractor
    layers.GlobalAveragePooling2D(),  # Convert features to a single vector per image
    layers.Dense(128, activation='relu'),  # Add a fully connected layer
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(4, activation='softmax')  # Output layer (adjust number of classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load your dataset and preprocess it
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = data_gen.flow_from_directory(
    r'C:\Users\HP\PycharmProjects\SignLanguage\Data', # Directory with subdirectories for each class
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_data = data_gen.flow_from_directory(
    r'C:\Users\HP\PycharmProjects\SignLanguage\validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(train_data, epochs=10, validation_data=val_data)

# Save the model
model.save('keras_model.h5')