import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers

# ==========================
# Set Correct Local Paths
# ==========================

BASE_DIR = r"C:\Users\User\OneDrive\Pictures\OneDrive\Desktop\mini_project\dataset_new"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

print("Train:", TRAIN_DIR)
print("Test:", TEST_DIR)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

train_gen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.15,
    zoom_range=0.25,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

train_batches = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_batches = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

test_batches = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# Build EfficientNet Model
# ==========================

base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

for layer in base.layers[:-25]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = BatchNormalization()(x)
x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
output = Dense(4, activation="softmax")(x)

model = Model(base.input, output)

model.compile(
    optimizer=Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# ==========================
# Callbacks
# ==========================

callbacks = [
    EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# ==========================
# Train Model
# ==========================

history = model.fit(
    train_batches,
    validation_data=val_batches,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================
# Save Model
# ==========================
model.save("best_model.h5")
print("\n Model Saved ✓")
