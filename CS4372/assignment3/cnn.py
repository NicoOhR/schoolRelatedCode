import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import shuffle

_URL = "https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification"
path_to_zip = tf.keras.utils.get_file("intel.zip", origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), "intel_extracted")
train_dir = os.path.join(PATH, "seg_train/seg_train")
validation_dir = os.path.join(PATH, "seg_test/seg_test")
test_dir = os.path.join(PATH, "seg_pred/")

BATCH_SIZE = 32
IMG_SIZE = (150, 150)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

preprocess_input = tf.keras.applications.resnet50.preprocess_input


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

IMG_SHAPE = IMG_SIZE + (3,)

base_model = tf.keras.applications.ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet",
)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(6, activation="softmax")
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.summary()
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)

history = model.fit(
    train_dataset, epochs=initial_epochs, validation_data=validation_dataset
)

history.history

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")
plt.show()

plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 10.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

## Fine Tuning
import itertools

base_model.trainable = True

len(base_model.layers)

fine_tune_at = [100 + (10 * x) for x in range(3)]
learning_rate_divider = [10**x for x in range(1, 4)]

combinations = itertools.product(fine_tune_at, learning_rate_divider)

curr_best = [None, 0]
histories = []

for f, r in combinations:
    for layer in base_model.layers[:f]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / r),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_epochs = 1

    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=len(history.epoch),
        validation_data=validation_dataset,
    )
    histories.append(history_fine)

    if curr_best[1] < history_fine.history["val_accuracy"][-1]:
        curr_best[0] = (f, r)
        curr_best[1] = history_fine.history["val_accuracy"][-1]

best_f, best_r = curr_best[0]

for layer in base_model.layers[:best_f]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / best_r),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=len(history.epoch),
    validation_data=validation_dataset,
)

acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]

loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.ylim([0.8, 1])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylim([0, 1.0])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
)
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()
