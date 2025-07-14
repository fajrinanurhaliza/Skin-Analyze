import os
import json
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Matikan warning dan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Konfigurasi ===
DATASET_DIR = os.path.join('dataset', 'before')  # Struktur: dataset/before/[acne|dry|normal|oily]
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
MODEL_PATH = 'models/skin_analysis_cnn.h5'
LABELS_PATH = 'models/labels.json'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# === Image Augmentasi ===
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Hitung Class Weights ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))

# === Model: Transfer Learning (MobileNetV2) ===
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === Training ===
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# === Simpan model dan label ===
model.save(MODEL_PATH)
with open(LABELS_PATH, 'w') as f:
    json.dump(train_gen.class_indices, f)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Label saved to {LABELS_PATH}")
