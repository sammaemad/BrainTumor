import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.layers import GlobalAveragePooling2D , Dense, Dropout
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.applications import MobileNetV2

# Config
data_dir = "brain_tumor_dataset"
img_size = (150, 150)
batch_size = 32
epochs = 15

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# Train and Validation Generators
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False  
)

# Preview Images
images, labels = next(train_generator)
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title("Tumor" if labels[i] == 1 else "No Tumor")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Transfer Learning Model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  

# Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Save Model and Training History
model.save("brain_tumor_model.h5")
print("Model saved as 'brain_tumor_model.h5'")

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("Training history saved as 'history.pkl'")

# Evaluate Model
val_preds = model.predict(val_generator)
val_preds = np.round(val_preds).astype(int)
true_labels = val_generator.classes

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, val_preds))

print("\nClassification Report:")
print(classification_report(true_labels, val_preds, target_names=["No Tumor", "Tumor"]))
