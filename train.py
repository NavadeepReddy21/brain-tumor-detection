"""
train.py
Trains the VGG16-based 4-class brain tumor classifier.

This is meant to be run separately (e.g. on Colab/Kaggle with GPU),
NOT as part of the deployed app. Run this once, then copy the
resulting BrainTumor_VGG16_Final.h5 into the app folder.

Dataset expected structure:
    Training/
        glioma/
        meningioma/
        notumor/
        pituitary/
    Testing/
        glioma/
        meningioma/
        notumor/
        pituitary/
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================== CONFIG =====================
DATA_DIR = "Training"
TEST_DIR = "Testing"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
MODEL_OUT = "BrainTumor_VGG16_Final.h5"

AUTOTUNE = tf.data.AUTOTUNE
tf.keras.mixed_precision.set_global_policy("mixed_float16")


def load_datasets():
    train_ds = image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    val_ds = image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    test_ds = image_dataset_from_directory(
        TEST_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print("Class Order:", class_names)

    def preprocess(x, y):
        return preprocess_input(x), y

    train_ds = train_ds.map(preprocess).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(AUTOTUNE)
    test_ds = test_ds.map(preprocess).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model():
    base_model = VGG16(include_top=False, weights="imagenet",
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in base_model.layers[:11]:
        layer.trainable = False
    for layer in base_model.layers[11:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation="softmax", dtype="float32")(x)

    model = Model(base_model.input, output)
    model.compile(optimizer=Adam(2e-5), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def evaluate(model, test_ds, class_names):
    loss, acc = model.evaluate(test_ds)
    print(f"Final Test Accuracy: {acc * 100:.2f}%")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=2, factor=0.3)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              callbacks=callbacks)

    evaluate(model, test_ds, class_names)

    model.save(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
